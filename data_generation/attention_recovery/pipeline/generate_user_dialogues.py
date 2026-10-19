#!/usr/bin/env python3
"""Stage 2 — User-dialogue stretching for the Attention Recovery dimension.

Converts Stage 1 interrupt-trigger blueprints into pure user-side dialogue templates.
Structure per dialogue (4-section, 5-7 turns total):
  [Thematic Connection ×2-4] → [Constraint ×1] → [Interruption Placeholder ×1] → [Trigger ×1]

Key design: "Template-Payload Decoupling"
  - The interruption_placeholder turn only contains the bridge prefix
  - No actual math/code payload is inserted at this stage
  - This 1×N design enables dynamic pairing with any external problem later

No assistant utterances are generated at this stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time as time_module
from pathlib import Path

from tqdm import tqdm

PIPELINE_DIR = Path(__file__).resolve().parent
DIM_DIR = PIPELINE_DIR.parent
GEN_ROOT = DIM_DIR.parent
sys.path.insert(0, str(GEN_ROOT))
sys.path.insert(0, str(DIM_DIR))

from shared.json_utils import extract_json_object, save_json
from shared.llm_client import complete
from shared.time_utils import parse_iso
from seeds.blueprints import ATTENTION_RECOVERY_SEEDS
from seeds.user_dialogues import SEED_USER_DIALOGUES

# ── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
for _noisy in ("azure", "azure.identity", "azure.core", "openai", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════
#  Time Gap Parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_time_gap_minutes(gap_str: str) -> int | None:
    """Parse strings like '3 minutes', '65 minutes', '2 hours' → integer minutes."""
    if not gap_str:
        return None
    gap_str = gap_str.strip().lower()
    m = re.match(r"(\d+)\s*(minutes?|mins?|hours?|hrs?)", gap_str)
    if m:
        val = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("hour") or unit.startswith("hr"):
            return val * 60
        return val
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Blueprint Stripping (remove fields that could leak info to LLM)
# ═══════════════════════════════════════════════════════════════════════════

def strip_blueprint_for_prompt(bp: dict) -> dict:
    """Strip blueprint to only fields the dialogue-generation LLM should see.

    Removes: trigger.overloaded, gold_labels.intent
    """
    story = bp.get("story", {})
    gl = bp.get("gold_labels", {})

    # Extract trigger.clean only
    trigger = story.get("trigger", "")
    if isinstance(trigger, dict):
        trigger_clean = trigger.get("clean", "")
    else:
        trigger_clean = trigger

    return {
        "task_id": bp.get("task_id", ""),
        "meta": bp.get("meta", {}),
        "story": {
            "background": story.get("background", ""),
            "task_a_start": story.get("task_a_start", ""),
            "interruption_bridge_prefix": story.get("interruption_bridge_prefix", ""),
            "time_gap_between_task_a_and_interruption": story.get(
                "time_gap_between_task_a_and_interruption", ""
            ),
            "time_gap_between_interruption_and_trigger": story.get(
                "time_gap_between_interruption_and_trigger", ""
            ),
            "trigger_clean": trigger_clean,
        },
        "gold_labels": {
            "slots": gl.get("slots", {}),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Seed Example for ICL
# ═══════════════════════════════════════════════════════════════════════════

def get_seed_example() -> tuple:
    """Return (stripped_seed_001_blueprint, seed_001_dialogue) for ICL."""
    seed_bp = None
    for s in ATTENTION_RECOVERY_SEEDS:
        if s["task_id"] == "interrupt_seed_001":
            seed_bp = s
            break
    assert seed_bp is not None, "interrupt_seed_001 not found in ATTENTION_RECOVERY_SEEDS"

    stripped = strip_blueprint_for_prompt(seed_bp)
    dlg = SEED_USER_DIALOGUES.get("interrupt_seed_001")
    assert dlg is not None, "interrupt_seed_001 not found in SEED_USER_DIALOGUES"

    return stripped, dlg


# ═══════════════════════════════════════════════════════════════════════════
#  Thematic Turn Count Sampling
# ═══════════════════════════════════════════════════════════════════════════

def sample_thematic_count(rng: random.Random) -> int:
    """Sample number of thematic_connection turns (2-4)."""
    return rng.randint(2, 4)


def sample_start_time(rng: random.Random) -> str:
    """Sample a diverse start time for the first turn.

    Returns an ISO string like '2026-03-17T14:23'.
    Covers the full range 06:00-17:00 with non-round minutes.
    Hour capped at 17 to leave room for time gaps without hitting night hours.
    """
    # Random date in 2026, months 1-5
    month = rng.randint(1, 5)
    day = rng.randint(1, 28)  # safe for all months
    # Random hour: 6-17 (leave headroom for time gaps)
    hour = rng.randint(6, 17)
    # Non-round minute: avoid 0, 15, 30, 45
    minute = rng.choice([m for m in range(1, 60) if m % 15 != 0])
    return "2026-{:02d}-{:02d}T{:02d}:{:02d}".format(month, day, hour, minute)


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt Construction
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert dialogue writer for **TriggerBench**, a benchmark that tests whether \
LLM agents can proactively remind users to resume interrupted tasks.

This is the Attention Recovery dimension (also called Interrupt-Trigger).

Your job: given a scenario **blueprint**, generate a realistic user-only dialogue \
(no assistant responses) as a JSON object.

────────────────────────────────────────────────────
DIALOGUE STRUCTURE  (STRICT — 4-section sequence, 5-7 turns total)
────────────────────────────────────────────────────
[1. Thematic Connection (2-4 turns)] → [2. Constraint (exactly 1 turn)] → \
[3. Interruption Placeholder (exactly 1 turn)] → [4. Trigger (exactly 1 turn, LAST)]

The EXACT number of thematic connection turns is specified in the user prompt \
under "REQUIRED TURN STRUCTURE". You MUST follow that count exactly.

────────────────────────────────────────────────────
TURN FUNCTIONS (enum values)
────────────────────────────────────────────────────
- "thematic_connection": Domain-aligned professional questions/tasks that are \
thematically related to the blueprint's background/task but have ZERO connection \
to the actual interrupted task details (task_a_start) or the interruption domain \
(math/code/physics/chemistry/biology/puzzle). These are warm-up turns.
- "constraint": Exactly 1 turn. The user states what they are doing and what they \
are about to do next (i.e., another task that interrupts Task A). This MUST faithfully reproduce \
the blueprint's `task_a_start` text as natural first-person speech.
- "interruption_placeholder": Exactly 1 turn. The user is suddenly interrupted. \
This MUST contain the blueprint's `interruption_bridge_prefix` text EXACTLY \
(including the trailing newline if present). Do NOT add any actual math/code problem.
- "trigger": Exactly 1 turn (LAST). The external task is done. This MUST use the \
blueprint's `trigger_clean` text EXACTLY.

────────────────────────────────────────────────────
ABSOLUTE RULES
────────────────────────────────────────────────────
1. **CONTEXT-FREE thematic turns**
   Every thematic_connection turn must be a task/question an LLM can answer purely \
from its internal parametric knowledge. NO references to external environments, \
private files, calendars, emails, or real-time data.
   GOOD examples:
     • "Draft a polite text to my teacher saying I might be 5 mins late." (text drafting)
     • "Can you summarize the core principles of Stoicism?" (concept explanation)
     • "Introduce the history of classical music." (knowledge introduction)
     • "Give me some suggestions on how to care for my cello." (advice request)
     • "Who is the most famous cellist of the 21st century?" (factual question)
   BAD examples:
     • "Read my morning emails." (no mailbox)
     • "What's the weather today?" (no real-time info)
     • "Check my calendar for tomorrow." (no calendar)

2. **THEMATIC COHERENCE**
   Thematic turns MUST be grounded in the blueprint's background topics. They should \
feel like a natural part of the user's current activity/situation — NOT random or \
disconnected. Derive them from the user's occupation, interests, or domain context.

3. **STRICT INFORMATION ISOLATION**
   Thematic turns must have:
   - ZERO semantic connection to the constraint's specific task details (task_a_start)
   - ZERO connection to the interruption domain (math, code, physics, chemistry, \
biology, puzzle, competition, algorithm, equation)
   - NO discussion of upcoming interruptions or reminders
   They are purely professional/contextual warm-up.

4. **CONSTRAINT TURN (task_a_start)**
   MUST faithfully reproduce the blueprint's `task_a_start` as the user's utterance. \
Keep it as close to verbatim as possible. It must read as natural first-person speech \
where the user states what they're currently doing and what they're about to do next.
   CRITICAL: The generated text MUST contain the key entity words from the \
blueprint's `slot_values` (at least one alias per required slot).

5. **INTERRUPTION PLACEHOLDER**
   MUST contain the blueprint's `interruption_bridge_prefix` text EXACTLY. \
Do NOT add any additional content (no math problem, no code snippet). \
The placeholder ends with the bridge prefix text.

6. **TRIGGER (trigger_clean)**
   MUST use the blueprint's `trigger_clean` text EXACTLY as-is. \
Do NOT modify, expand, or rephrase it.

7. **TIME FORMATTING & ORDERING**
   - All times in ISO format: YYYY-MM-DDTHH:MM
   - All turn times strictly monotonically increasing.
   - Default spacing between thematic turns: ~3-8 minutes (varies naturally).
   - The gap between constraint and interruption_placeholder must EXACTLY equal \
the blueprint's `time_gap_between_task_a_and_interruption`.
   - The gap between interruption_placeholder and trigger must EXACTLY equal \
the blueprint's `time_gap_between_interruption_and_trigger`.
   - Do NOT place any turn between 01:00 and 05:00 AM.
   - Use non-round times (e.g., 10:13, 14:47) — avoid :00, :15, :30, :45.
   - The first turn MUST start at the exact time specified in the user prompt under \"MANDATORY\".\n\

────────────────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────────────────
Return ONLY a valid JSON object:
{"user_turns": [<turn>, <turn>, ...]}

Each turn: {"turn_id": <int starting from 1>, "time": "<ISO>", \
"function": "<thematic_connection|constraint|interruption_placeholder|trigger>", \
"text": "<utterance>"}

No markdown fences, no comments, no extra text.\
"""


def build_user_prompt(
    blueprint: dict,
    seed_bp: dict,
    seed_dlg: list,
    n_thematic: int,
    start_time: str = None,
) -> str:
    """Assemble the user prompt for one dialogue generation call."""

    stripped_bp = strip_blueprint_for_prompt(blueprint)
    bp_json = json.dumps(stripped_bp, indent=2, ensure_ascii=False)

    gap_task_to_interrupt = blueprint["story"].get(
        "time_gap_between_task_a_and_interruption", ""
    )
    gap_interrupt_to_trigger = blueprint["story"].get(
        "time_gap_between_interruption_and_trigger", ""
    )
    total_turns = n_thematic + 3  # thematic + constraint + interruption + trigger

    # Seed example
    seed_section = (
        "═══ SEED EXAMPLE (for format & quality reference — do NOT duplicate content) ═══\n\n"
        "Input Blueprint:\n"
        "{seed_bp_json}\n\n"
        "Output Dialogue:\n"
        "{seed_dlg_json}"
    ).format(
        seed_bp_json=json.dumps(seed_bp, indent=2, ensure_ascii=False),
        seed_dlg_json=json.dumps({"user_turns": seed_dlg}, indent=2, ensure_ascii=False),
    )

    prompt = (
        "Convert the following blueprint into a base user dialogue.\n\n"
        "═══ BLUEPRINT TO CONVERT ═══\n"
        "{bp_json}\n\n"
        "═══ TIME GAP INFO ═══\n"
        "- time_gap_between_task_a_and_interruption: {gap1}\n"
        "  → The constraint turn and interruption_placeholder turn timestamps must be "
        "EXACTLY this far apart.\n"
        "- time_gap_between_interruption_and_trigger: {gap2}\n"
        "  → The interruption_placeholder and trigger turn timestamps must be "
        "EXACTLY this far apart.\n"
        "- **MANDATORY**: Your first turn MUST start at exactly {start_time}.\n"
        "- Space thematic turns naturally: ~3-8 minutes between consecutive turns.\n\n"
        "{seed_section}\n\n"
        "═══ REQUIRED TURN STRUCTURE ═══\n"
        "- Thematic Connection: exactly {n_thematic} turns\n"
        "- Constraint turn: exactly 1 turn\n"
        "- Interruption Placeholder: exactly 1 turn\n"
        "- Trigger turn: exactly 1 turn (LAST turn)\n"
        "- Total: {total_turns} turns\n\n"
        "═══ SLOT VALUES TO PRESERVE ═══\n"
        "The constraint turn text MUST contain at least one alias from each required slot:\n"
        "{slot_info}\n\n"
        "═══ FIXED TEXTS ═══\n"
        "These texts must appear EXACTLY in the dialogue (copy verbatim):\n"
        "- constraint (task_a_start): \"{task_a_start}\"\n"
        "- interruption_placeholder (bridge prefix): \"{bridge_prefix}\"\n"
        "- trigger (trigger_clean): \"{trigger_clean}\"\n\n"
        "═══ REMINDERS ═══\n"
        "- All thematic_connection turns must be context-free (LLM-answerable, no external tools)\n"
        "- Thematic turns MUST be grounded in the blueprint's background topics\n"
        "(the objects, next step, or slot aliases). Those belong only in the constraint turn.\n"
        "- ZERO overlap between thematic turns and the interruption domain "
        "(math/code/physics/chemistry/biology/puzzle)\n"
        "- Constraint turn: faithfully reproduce task_a_start as first-person speech\n"
        "- Interruption placeholder: ONLY the bridge prefix text, nothing more\n"
        "- Trigger turn: EXACT copy of trigger_clean\n"
        "- All times strictly increasing, in YYYY-MM-DDTHH:MM format\n"
        "- Do NOT place any turn between 01:00 and 05:00 AM\n"
        "- Use non-round times (e.g., 09:13, 14:47) — avoid :00, :15, :30, :45\n"
        "- Return ONLY {{\"user_turns\": [...]}} as valid JSON"
    ).format(
        bp_json=bp_json,
        gap1=gap_task_to_interrupt,
        gap2=gap_interrupt_to_trigger,
        seed_section=seed_section,
        n_thematic=n_thematic,
        total_turns=total_turns,
        start_time=start_time or "2025-04-14T09:13",
        slot_info=json.dumps(
            blueprint.get("gold_labels", {}).get("slots", {}),
            indent=2, ensure_ascii=False,
        ),
        task_a_start=blueprint["story"].get("task_a_start", ""),
        bridge_prefix=blueprint["story"].get("interruption_bridge_prefix", "").rstrip(),
        trigger_clean=_get_trigger_clean(blueprint),
    )
    return prompt


def _get_trigger_clean(bp: dict) -> str:
    """Extract trigger.clean from blueprint (handles both dict and str formats)."""
    trigger = bp["story"].get("trigger", "")
    if isinstance(trigger, dict):
        return trigger.get("clean", "")
    return trigger


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════

VALID_FUNCTIONS = {"thematic_connection", "constraint", "interruption_placeholder", "trigger"}

# Expected function sequence pattern: thematic{2,4} + constraint + interruption_placeholder + trigger
SEQUENCE_PATTERN = re.compile(
    r"^(thematic_connection){2,4}(constraint)(interruption_placeholder)(trigger)$"
)


def _text_contains(haystack: str, needle: str, threshold: float = 0.85) -> bool:
    """Check if needle is substantially contained in haystack.

    Uses substring check first, then falls back to overlap ratio.
    """
    h = haystack.strip().lower()
    n = needle.strip().lower()

    # Exact substring
    if n in h or h in n:
        return True

    # Word overlap ratio
    h_words = set(h.split())
    n_words = set(n.split())
    if not n_words:
        return True
    overlap = len(h_words & n_words) / len(n_words)
    return overlap >= threshold


def validate_dialogue(turns: list, blueprint: dict) -> tuple:
    """Validate a generated dialogue against its blueprint.

    Returns (is_valid, error_list).
    """
    errors = []

    # ── 1. Basic structure ──────────────────────────────────────────────────
    if not isinstance(turns, list):
        errors.append("user_turns must be a list")
        return False, errors

    if len(turns) < 5:
        errors.append("Too few turns: {} (minimum 5: 2+1+1+1)".format(len(turns)))
    if len(turns) > 7:
        errors.append("Too many turns: {} (maximum 7: 4+1+1+1)".format(len(turns)))

    # ── 2. Per-turn field validation ────────────────────────────────────────
    for i, turn in enumerate(turns):
        for key in ("turn_id", "time", "function", "text"):
            if key not in turn:
                errors.append("Turn {}: missing field '{}'".format(i + 1, key))
        if "function" in turn and turn["function"] not in VALID_FUNCTIONS:
            errors.append("Turn {}: invalid function '{}'".format(i + 1, turn["function"]))
    if errors:
        return False, errors

    # ── 3. Function sequence check ──────────────────────────────────────────
    functions = [t["function"] for t in turns]
    seq_str = "".join(functions)

    if not SEQUENCE_PATTERN.match(seq_str):
        errors.append(
            "Function sequence invalid: {}. "
            "Expected: thematic_connection{{2,4}} + constraint + "
            "interruption_placeholder + trigger".format(" → ".join(functions))
        )
        return False, errors

    # Identify positions
    constraint_idx = functions.index("constraint")
    interrupt_idx = functions.index("interruption_placeholder")
    trigger_idx = functions.index("trigger")

    # ── 4. turn_id monotonicity ─────────────────────────────────────────────
    for i, turn in enumerate(turns):
        if turn.get("turn_id") != i + 1:
            errors.append(
                "Turn {}: turn_id should be {}, got {}".format(i + 1, i + 1, turn.get("turn_id"))
            )
            break

    # ── 5. Time monotonicity ────────────────────────────────────────────────
    parsed_times = []
    for i, turn in enumerate(turns):
        dt = parse_iso(turn["time"])
        if dt is None:
            errors.append("Turn {}: invalid time format '{}'".format(i + 1, turn["time"]))
        else:
            parsed_times.append((i, dt))

    for j in range(1, len(parsed_times)):
        if parsed_times[j][1] <= parsed_times[j - 1][1]:
            idx_a = parsed_times[j - 1][0]
            idx_b = parsed_times[j][0]
            errors.append(
                "Time not monotonically increasing: turn {} ({}) >= turn {} ({})".format(
                    idx_a + 1, turns[idx_a]["time"], idx_b + 1, turns[idx_b]["time"]
                )
            )
            break

    if errors:
        return False, errors

    # ── 6. Text fidelity: constraint (task_a_start) ─────────────────────────
    task_a_start = blueprint["story"].get("task_a_start", "")
    constraint_text = turns[constraint_idx]["text"]

    if not _text_contains(constraint_text, task_a_start):
        errors.append(
            "Constraint text doesn't faithfully reproduce task_a_start. "
            "Expected content like: \"{}...\"".format(task_a_start[:80])
        )

    # ── 7. Text fidelity: interruption_placeholder (bridge prefix) ──────────
    bridge_prefix = blueprint["story"].get("interruption_bridge_prefix", "").strip()
    interrupt_text = turns[interrupt_idx]["text"].strip()

    if bridge_prefix not in interrupt_text:
        # Also check without trailing whitespace differences
        bridge_clean = bridge_prefix.rstrip()
        interrupt_clean = interrupt_text.rstrip()
        if bridge_clean not in interrupt_clean:
            errors.append(
                "Interruption placeholder doesn't contain the exact bridge prefix. "
                "Expected to contain: \"{}\"".format(bridge_prefix[:80])
            )

    # ── 8. Text fidelity: trigger (trigger_clean) ───────────────────────────
    trigger_clean = _get_trigger_clean(blueprint).strip()
    trigger_text = turns[trigger_idx]["text"].strip()

    if not _text_contains(trigger_text, trigger_clean):
        errors.append(
            "Trigger text doesn't match trigger_clean. "
            "Expected: \"{}\"".format(trigger_clean[:80])
        )

    # ── 9. Slot grounding check ─────────────────────────────────────────────
    constraint_text_lower = constraint_text.lower()
    slots = blueprint.get("gold_labels", {}).get("slots", {})
    required_slots = slots.get("required_slots", [])
    slot_values = slots.get("slot_values", {})

    for slot_name in required_slots:
        aliases = slot_values.get(slot_name, [])
        if not aliases:
            continue
        found = any(alias.lower() in constraint_text_lower for alias in aliases)
        if not found:
            errors.append(
                "Slot '{}' not grounded: none of {} found in constraint text".format(
                    slot_name, aliases
                )
            )

    # ── 10. Time gap check: constraint → interruption (±1 min tolerance) ────
    gap1_str = blueprint["story"].get("time_gap_between_task_a_and_interruption", "")
    expected_gap1 = parse_time_gap_minutes(gap1_str)

    ct_dt = parse_iso(turns[constraint_idx]["time"])
    int_dt = parse_iso(turns[interrupt_idx]["time"])

    if ct_dt and int_dt and expected_gap1 is not None:
        actual_gap1 = (int_dt - ct_dt).total_seconds() / 60
        if abs(actual_gap1 - expected_gap1) > 1:
            errors.append(
                "Time gap constraint→interruption: actual={:.0f}min, "
                "expected={:.0f}min (tolerance ±1min)".format(actual_gap1, expected_gap1)
            )

    # ── 11. Time gap check: interruption → trigger (±2 min tolerance) ───────
    gap2_str = blueprint["story"].get("time_gap_between_interruption_and_trigger", "")
    expected_gap2 = parse_time_gap_minutes(gap2_str)

    trig_dt = parse_iso(turns[trigger_idx]["time"])

    if int_dt and trig_dt and expected_gap2 is not None:
        actual_gap2 = (trig_dt - int_dt).total_seconds() / 60
        if abs(actual_gap2 - expected_gap2) > 2:
            errors.append(
                "Time gap interruption→trigger: actual={:.0f}min, "
                "expected={:.0f}min (tolerance ±2min)".format(actual_gap2, expected_gap2)
            )

    # ── 12. No night hours (01:00-05:00) ────────────────────────────────────
    for i, turn in enumerate(turns):
        dt = parse_iso(turn["time"])
        if dt and 1 <= dt.hour < 5:
            errors.append(
                "Turn {} time '{}' is between 01:00-05:00 AM (forbidden)".format(
                    i + 1, turn["time"]
                )
            )
            break

    return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
#  Postprocess: force-replace fixed-text turns
# ═══════════════════════════════════════════════════════════════════════════

def postprocess_dialogue(turns: list, blueprint: dict) -> list:
    """Force-replace interruption_placeholder and trigger turn texts.

    Even if the LLM's output passed validation, it may have appended extra
    content after the bridge prefix (e.g., a partial math problem).  This
    postprocess step guarantees the template is clean:
      - interruption_placeholder.text = story.interruption_bridge_prefix (exact)
      - trigger.text = story.trigger.clean (exact)
    """
    bridge_prefix = blueprint["story"].get("interruption_bridge_prefix", "")
    trigger_clean = _get_trigger_clean(blueprint)

    for turn in turns:
        if turn["function"] == "interruption_placeholder":
            turn["text"] = bridge_prefix
        elif turn["function"] == "trigger":
            turn["text"] = trigger_clean

    return turns


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage 2: Generate base user dialogues from Attention Recovery blueprints"
    )
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--blueprint-file", type=str, required=True,
                    help="Path to Stage 1 blueprint JSON file")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to existing output JSON to resume from")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-seeds", action="store_true",
                    help="Also generate dialogues for seed blueprints from seeds/blueprints.py")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="Seconds to sleep between items (rate limiting)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    def call_llm(sys_p, usr_p):
        return complete(sys_p, usr_p, model=args.model, temperature=args.temperature)

    bp_path = Path(args.blueprint_file)
    if not bp_path.exists():
        log.error("Blueprint file not found: %s", bp_path)
        sys.exit(1)

    with open(bp_path, encoding="utf-8") as f:
        blueprints = json.load(f)

    log.info("Loaded %d blueprints from %s", len(blueprints), bp_path)

    if args.include_seeds:
        bp_ids = {b["task_id"] for b in blueprints}
        new_seeds = [s for s in ATTENTION_RECOVERY_SEEDS if s["task_id"] not in bp_ids]
        blueprints = new_seeds + blueprints
        log.info("--include-seeds: added %d seed blueprints (total now %d)",
                 len(new_seeds), len(blueprints))

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else DIM_DIR / "outputs" / "stage2_user_dialogues"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "user_dialogues_{}.json".format(args.model)

    all_dialogues = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as f:
                all_dialogues = json.load(f)
            log.info("Resumed %d dialogues from %s", len(all_dialogues), resume_path)

    seed_bp, seed_dlg = get_seed_example()
    log.info("Seed example loaded: interrupt_seed_001 (%d turns)", len(seed_dlg))

    total = len(blueprints)
    done = 0
    failed = 0

    log.info(
        "Target: %d dialogues | Model=%s, T=%.2f, max_retries=%d",
        total, args.model, args.temperature, args.max_retries,
    )

    for bp_idx, blueprint in enumerate(tqdm(blueprints, desc="Stage 2 (Attention Recovery)")):
        if args.sleep:
            time_module.sleep(args.sleep)
        task_id = blueprint.get("task_id", "unknown_{}".format(bp_idx))

        if task_id in all_dialogues:
            log.info("[%d/%d] %s — already done, skipping", bp_idx + 1, total, task_id)
            done += 1
            continue

        log.info("[%d/%d] Generating dialogue for %s ...", bp_idx + 1, total, task_id)

        n_thematic = sample_thematic_count(rng)
        start_time = sample_start_time(rng)

        user_prompt = build_user_prompt(blueprint, seed_bp, seed_dlg, n_thematic, start_time)
        base_user_prompt = user_prompt
        success = False

        for retry in range(args.max_retries):
            try:
                raw = call_llm(SYSTEM_PROMPT, user_prompt)
                result = extract_json_object(raw)

                turns = result.get("user_turns", [])
                valid, errs = validate_dialogue(turns, blueprint)

                if valid:
                    turns = postprocess_dialogue(turns, blueprint)
                    all_dialogues[task_id] = turns
                    done += 1
                    log.info(
                        "  OK (attempt %d/%d)  %d turns",
                        retry + 1, args.max_retries, len(turns),
                    )
                    success = True
                    save_json(all_dialogues, out_file)
                    break
                else:
                    log.warning(
                        "  Validation failed (attempt %d/%d):",
                        retry + 1, args.max_retries,
                    )
                    for e in errs:
                        log.warning("    - %s", e)

                    if retry < args.max_retries - 1:
                        err_block = "\n".join("- {}".format(e) for e in errs)
                        user_prompt = (
                            base_user_prompt
                            + "\n\n=== YOUR PREVIOUS ATTEMPT FAILED VALIDATION ===\n"
                            + "Errors:\n{}\n".format(err_block)
                            + "Fix ALL issues above and generate a new valid dialogue."
                        )

            except json.JSONDecodeError as e:
                log.warning(
                    "  JSON parse error (attempt %d/%d): %s",
                    retry + 1, args.max_retries, e,
                )
            except Exception as e:
                log.warning(
                    "  Error (attempt %d/%d): %s",
                    retry + 1, args.max_retries, e,
                )

        if not success:
            failed += 1
            log.error(
                "  FAILED after %d retries — skipping %s",
                args.max_retries, task_id,
            )

    save_json(all_dialogues, out_file)

    log.info("=" * 70)
    log.info("Done!  %d / %d dialogues generated (%d failed).",
             done, total, failed)
    log.info("Saved -> %s", out_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 5 — Negative-Clean variant for Logical Adherence.

Generates "negative clean" dialogues by replacing the critical user turn with
its negative counterpart from the blueprint:

  - Type A (Forward_If-Then):
      Replace condition_constraint user turn → blueprint condition_constraint.negative
      Regenerate the immediately-following assistant_reply (1 LLM call per dialogue)

  - Type B (Mutual_Exclusion / Only-After):
      Replace trigger user turn (last turn) → blueprint trigger.negative
      Zero LLM calls (pure text substitution)

Input:  Stage 3 Positive-Clean dialogues + Stage 1 blueprints
Output: Same format as Stage 3, with dialogue_id suffix "_negative_clean"
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
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

from shared.json_utils import save_json
from shared.llm_client import chat
from seeds.blueprints import LOGICAL_ADHERENCE_SEEDS

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
#  System Prompt — Same as Stage 3 (neutral assistant)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a professional, naturally conversational, and knowledgeable AI assistant.

CRITICAL INSTRUCTION REGARDING TIME:
The timestamp prefixing each user message (e.g., [2026-04-19T13:30]) represents \
the EXACT current system clock time at the moment the user sends that message. \
You must strictly acknowledge this timeline if time is relevant, but DO NOT \
proactively read the clock for the user unless explicitly asked.

CRITICAL PERSONA REQUIREMENT:
You assume the user is a fully capable adult. Unless the user EXPLICITLY asks \
you for safety advice or EXPLICITLY requests you to set a reminder, you must:
- NEVER proactively offer unsolicited advice or warnings.
- NEVER say things like "remember to do X", "don't forget about Y", or \
"since you mentioned [rule], you should...".
- NEVER track, infer, or remind the user about conditional logic, personal \
rules, budgets, constraints, or prerequisites they mentioned earlier.
- NEVER connect dots between a user's earlier statement and a later action \
(e.g., "Earlier you said if X then Y — this applies now!").

When the user shares a personal plan, budget, rule, or condition:
- Provide brief, neutral acknowledgement or moral support ("Sounds like a plan!", \
"Got it.", "That makes sense.").
- Do NOT elaborate on the logical implications of their rule.
- Do NOT offer to help track or monitor the condition.
- Do NOT ask follow-up questions about the rule's enforcement.

When the user casually reports a status update or condition being met:
- Respond naturally to the surface content only.
- Do NOT connect it back to any earlier stated rule or plan.

Maintain proper boundaries — do NOT be a preachy, over-helpful, or \
pattern-matching assistant.

Prioritize answering the user's explicitly stated question or task thoroughly. \
Do NOT over-intervene on the user's casually mentioned daily behaviors or states.

Length guideline: Keep responses professional and concise, typically under \
500 words. If the user requests a detailed explanation, summary, or report, \
provide a thorough answer, up to 1000-2000 words.\
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Blacklist — Constraint-subset (for condition_constraint assistant reply)
# ═══════════════════════════════════════════════════════════════════════════

BLACKLIST_PATTERNS_CONSTRAINT = [
    r"i('ll| will) (track|monitor|keep track|watch).{0,20}(that|this|for you)",
    r"i('ll| will) remind",
    r"i('ll| will) let you know when",
    r"i('ll| will) alert you",
    r"i('ll| will) notify you",
    r"want me to (track|monitor|remind|flag|alert)",
    r"shall i (remind|alert|notify|flag|track|monitor)",
    r"set (a |an )?(reminder|alarm|timer|countdown)",
    r"i('ll| will) (set|start) (a |an )?(timer|alarm|countdown|reminder)",
    r"i('ll| will) (flag|note) (this|that) for (later|you|when)",
    r"(that|this) (means|implies|triggers) (you should|that you)",
    r"so when .{0,30}(happens|comes|occurs).{0,20}(you should|i('ll| will))",
]

_BLACKLIST_RE_CONSTRAINT = [re.compile(p, re.IGNORECASE) for p in BLACKLIST_PATTERNS_CONSTRAINT]


def check_blacklist(text: str) -> list:
    """Return list of matched blacklist patterns (empty = clean)."""
    return [p for p, rx in zip(BLACKLIST_PATTERNS_CONSTRAINT, _BLACKLIST_RE_CONSTRAINT)
            if rx.search(text)]


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

_TIMESTAMP_PREFIX_RE = re.compile(r"^\s*\[?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}\]?\s*")


def strip_timestamp_prefix(text: str) -> str:
    return _TIMESTAMP_PREFIX_RE.sub("", text).strip()


# ═══════════════════════════════════════════════════════════════════════════
#  Blueprint Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_blueprint_map(blueprint_file: str) -> dict:
    """Load blueprints keyed by task_id (seed + LLM-generated)."""
    bp_map = {}

    # Load seed blueprints first
    for bp in LOGICAL_ADHERENCE_SEEDS:
        bp_map[bp["task_id"]] = bp
    log.info("Loaded %d seed blueprints", len(bp_map))

    # Load LLM-generated blueprints (override seeds if same task_id)
    bp_path = Path(blueprint_file)
    if not bp_path.exists():
        log.error("Blueprint file not found: %s", bp_path)
        sys.exit(1)

    with open(bp_path, encoding="utf-8") as f:
        blueprints = json.load(f)

    for bp in blueprints:
        bp_map[bp["task_id"]] = bp

    log.info("Loaded %d total blueprints (seed + LLM-generated)", len(bp_map))
    return bp_map


# ═══════════════════════════════════════════════════════════════════════════
#  Type A: Regenerate assistant reply after condition_constraint
# ═══════════════════════════════════════════════════════════════════════════

def build_messages_for_condition_reply(turns: list, condition_idx: int) -> list:
    """Build ChatML messages for regenerating the assistant reply after condition_constraint.

    Provides full dialogue history up to and including the (new) condition_constraint
    user turn, then a timestamp prefill for the assistant.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # All user-assistant pairs before condition_constraint
    for i in range(condition_idx):
        t = turns[i]
        messages.append({
            "role": t["role"],
            "content": "[{}] {}".format(t["time"], t["text"]),
        })

    # The (new) condition_constraint user turn
    cond_turn = turns[condition_idx]
    messages.append({
        "role": "user",
        "content": "[{}] {}".format(cond_turn["time"], cond_turn["text"]),
    })

    # Assistant timestamp prefill
    ast_turn = turns[condition_idx + 1]  # the assistant_reply we're regenerating
    messages.append({
        "role": "assistant",
        "content": "[{}]".format(ast_turn["time"]),
    })

    return messages


def regenerate_condition_reply(
    turns: list,
    condition_idx: int,
    model: str,
    temperature: float,
    max_retries: int,
) -> str:
    """Regenerate assistant reply after the (modified) condition_constraint turn."""
    messages = build_messages_for_condition_reply(turns, condition_idx)

    for attempt in range(1, max_retries + 1):
        try:
            raw = chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=512,
            )
            text = strip_timestamp_prefix(raw.strip())

            if len(text) < 5:
                log.warning("  Attempt %d/%d: too short (%d chars)",
                            attempt, max_retries, len(text))
                continue

            # Blacklist check
            hits = check_blacklist(text)
            if hits:
                log.warning("  Attempt %d/%d: blacklist hit: %s",
                            attempt, max_retries, hits)
                if attempt < max_retries:
                    continue
                # Accept on final attempt with warning
                log.warning("  Accepting on final attempt despite blacklist hit")

            return text
        except Exception as e:
            log.warning("  Attempt %d/%d API error: %s", attempt, max_retries, e)
            time_module.sleep(2)

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Save helper
# ═══════════════════════════════════════════════════════════════════════════

def save_output(results: list, path: Path):
    save_json(results, path)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage 5: Generate negative clean variant dialogues (Logical Adherence)"
    )
    ap.add_argument("--dialogue-input", required=True,
                    help="Path to Stage 3 Positive-Clean dialogues JSON")
    ap.add_argument("--blueprint-input", required=True,
                    help="Path to Stage 1 blueprint JSON")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to existing output JSON to resume from")
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    log.info("Loading dialogues from: %s", args.dialogue_input)
    with open(args.dialogue_input, encoding="utf-8") as f:
        dialogues = json.load(f)
    log.info("  -> %d dialogues", len(dialogues))

    bp_map = load_blueprint_map(args.blueprint_input)

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else DIM_DIR / "outputs" / "stage5_negative_clean"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dialogues_negative_clean_{}.json".format(args.model)

    completed = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as f:
                loaded = json.load(f)
            for item in loaded:
                completed[item["dialogue_id"]] = item
            log.info("Resumed %d dialogues from %s", len(completed), resume_path)

    results = list(completed.values())
    success = len(results)
    fail = 0
    type_a_count = 0
    type_b_count = 0

    for dlg in tqdm(dialogues, desc="Stage 5 Negative Clean"):
        if args.sleep:
            time_module.sleep(args.sleep)

        dialogue_id = dlg["dialogue_id"]
        task_id = dialogue_id
        for suffix in ("_positive_clean", "_standard"):
            if task_id.endswith(suffix):
                task_id = task_id[: -len(suffix)]
                break
        negative_id = "{}_negative_clean".format(task_id)

        # Skip if already done
        if negative_id in completed:
            continue

        # Skip incomplete dialogues
        if dlg.get("status") != "complete":
            log.warning("  %s — status '%s', skipping", dialogue_id, dlg.get("status"))
            fail += 1
            continue

        # Look up blueprint
        bp = bp_map.get(task_id)
        if bp is None:
            log.warning("  %s — no blueprint found, skipping", task_id)
            fail += 1
            continue

        logic_type = bp["meta"]["logic_type"]
        story = bp.get("story", {})
        turns = dlg["turns"]

        # ── Determine replacement strategy ──────────────────────────────────
        if logic_type == "Forward_If-Then":
            # Type A: replace condition_constraint + regenerate assistant reply
            negative_text = story.get("condition_constraint", {}).get("negative")
            if not negative_text:
                log.warning("  %s — no condition_constraint.negative in blueprint", task_id)
                fail += 1
                continue

            # Find condition_constraint user turn
            cond_idx = None
            for i, t in enumerate(turns):
                if t["role"] == "user" and t["function"] == "condition_constraint":
                    cond_idx = i
                    break

            if cond_idx is None:
                log.warning("  %s — condition_constraint turn not found", task_id)
                fail += 1
                continue

            # Verify next turn is assistant_reply
            if cond_idx + 1 >= len(turns) or turns[cond_idx + 1]["role"] != "assistant":
                log.warning("  %s — no assistant_reply after condition_constraint", task_id)
                fail += 1
                continue

            # Deep copy and replace
            new_dlg = copy.deepcopy(dlg)
            new_turns = new_dlg["turns"]
            new_turns[cond_idx]["text"] = negative_text

            # Regenerate assistant reply
            new_reply = regenerate_condition_reply(
                turns=new_turns,
                condition_idx=cond_idx,
                model=args.model,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
            if new_reply is None:
                log.error("  %s — FAILED to regenerate assistant reply", task_id)
                fail += 1
                continue

            new_turns[cond_idx + 1]["text"] = new_reply
            new_dlg["dialogue_id"] = negative_id
            new_dlg["_variant"] = "negative_clean"
            new_dlg["_logic_type"] = logic_type
            new_dlg["_replaced_turn"] = cond_idx + 1  # 1-indexed turn_id
            new_dlg["_replaced_function"] = "condition_constraint"

            results.append(new_dlg)
            completed[negative_id] = new_dlg
            success += 1
            type_a_count += 1

            log.info("  %s [Type A] — done: condition='%s' | reply='%s'",
                     task_id, negative_text[:50], new_reply[:50])

        elif logic_type in ("Mutual_Exclusion", "Only-After"):
            # Type B: replace trigger (last turn) — zero LLM calls
            negative_text = story.get("trigger", {}).get("negative")
            if not negative_text:
                log.warning("  %s — no trigger.negative in blueprint", task_id)
                fail += 1
                continue

            # Find trigger turn (should be last)
            trigger_idx = None
            for i in range(len(turns) - 1, -1, -1):
                if turns[i]["role"] == "user" and turns[i]["function"] == "trigger":
                    trigger_idx = i
                    break

            if trigger_idx is None:
                log.warning("  %s — trigger turn not found", task_id)
                fail += 1
                continue

            # Deep copy and replace
            new_dlg = copy.deepcopy(dlg)
            new_dlg["turns"][trigger_idx]["text"] = negative_text
            new_dlg["dialogue_id"] = negative_id
            new_dlg["_variant"] = "negative_clean"
            new_dlg["_logic_type"] = logic_type
            new_dlg["_replaced_turn"] = trigger_idx + 1  # 1-indexed turn_id
            new_dlg["_replaced_function"] = "trigger"

            results.append(new_dlg)
            completed[negative_id] = new_dlg
            success += 1
            type_b_count += 1

            log.info("  %s [Type B] — done (no LLM): trigger='%s'",
                     task_id, negative_text[:60])

        else:
            log.warning("  %s — unknown logic_type '%s', skipping", task_id, logic_type)
            fail += 1
            continue

        # Incremental save
        save_output(results, out_file)

    # ── Final save ──────────────────────────────────────────────────────────
    save_output(results, out_file)

    log.info("=" * 70)
    log.info("Done! %d / %d negative-clean dialogues (%d failed)",
             success, len(dialogues), fail)
    log.info("  Type A (condition replaced + reply regenerated): %d", type_a_count)
    log.info("  Type B (trigger replaced, no LLM): %d", type_b_count)
    log.info("Saved -> %s", out_file)


if __name__ == "__main__":
    main()

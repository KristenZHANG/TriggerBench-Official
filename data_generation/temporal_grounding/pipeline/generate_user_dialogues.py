#!/usr/bin/env python3
"""Stage 2 — User-dialogue stretching for the Temporal Grounding dimension.

Converts Stage 1 blueprints into user-only dialogues:

  [Thematic Connection ×2-5] → [Constraint ×1]
  → [Thematic Connection ×2-5] → [Trigger ×1]

Constraint and trigger timestamps must match the pre-computed anchors exactly.
Assistant replies are generated in Stage 3.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time as time_module
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

PIPELINE_DIR = Path(__file__).resolve().parent
DIM_DIR = PIPELINE_DIR.parent
GEN_ROOT = DIM_DIR.parent
sys.path.insert(0, str(GEN_ROOT))
sys.path.insert(0, str(DIM_DIR))

from shared.json_utils import extract_json_object, save_json
from shared.llm_client import complete
from shared.time_utils import TIME_FMT, parse_iso
from seeds.blueprints import TEMPORAL_GROUNDING_SEEDS
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
#  Seed ICL Pairing (time_logic_type × trigger_pattern → seed task_id)
# ═══════════════════════════════════════════════════════════════════════════

SEED_KEY_MAP = {
    ("Absolute_Time", "Time_Blind_Action"): "seed_01_absolute_blind_action",
    ("Relative_Time", "Time_Query"):        "seed_02_relative_time_query",
    ("Absolute_Time", "Time_Query"):        "seed_03_absolute_time_query",
    ("Relative_Time", "Time_Blind_Action"): "seed_04_relative_blind_action",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Blueprint Stripping (remove fields that could leak info to LLM)
# ═══════════════════════════════════════════════════════════════════════════

def strip_blueprint_for_prompt(bp: dict) -> dict:
    """Strip a blueprint to only the fields the dialogue-generation LLM should see.

    Removes: hidden_vulnerability, variations, gold_labels.intent
    """
    story = bp.get("story", {})
    ta = story.get("timeline_anchors", {})
    gl = bp.get("gold_labels", {})

    return {
        "task_id": bp.get("task_id", ""),
        "meta": bp.get("meta", {}),
        "story": {
            "background": story.get("background", ""),
            "timeline_anchors": {
                "constraint_talking_time": ta.get("constraint_talking_time"),
                "duration_time": ta.get("duration_time"),
                "exact_target_action_time": ta.get("exact_target_action_time"),
                "trigger_talking_time": ta.get("trigger_talking_time"),
            },
            "constraint": story.get("constraint", ""),
            "trigger": story.get("trigger", ""),
        },
        "gold_labels": {
            "slots": gl.get("slots", {}),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Seed Example for ICL
# ═══════════════════════════════════════════════════════════════════════════

def get_seed_example(time_logic_type: str, trigger_pattern: str) -> tuple:
    """Return (stripped_seed_blueprint, seed_dialogue) for the matching cell.

    Uses SEED_KEY_MAP to find the right seed based on (time_logic_type, trigger_pattern).
    Returns (None, None) if no matching seed found.
    """
    key = SEED_KEY_MAP.get((time_logic_type, trigger_pattern))
    if not key:
        return None, None

    seed_bp = None
    for s in TEMPORAL_GROUNDING_SEEDS:
        if s["task_id"] == key:
            seed_bp = s
            break
    if seed_bp is None:
        return None, None

    seed_dlg = SEED_USER_DIALOGUES.get(key)
    if seed_dlg is None:
        return None, None

    stripped = strip_blueprint_for_prompt(seed_bp)
    return stripped, seed_dlg


# ═══════════════════════════════════════════════════════════════════════════
#  Thematic Connection Turn Count Sampling
# ═══════════════════════════════════════════════════════════════════════════

def sample_thematic_split(rng: random.Random) -> tuple:
    """Sample (n_section1, n_section2).

    Section 1 (pre-constraint thematic): 2-5 turns
    Section 2 (post-constraint thematic): 2-5 turns
    Total turns: n1 + 1 (constraint) + n2 + 1 (trigger) = 7-12
    If both are 2, randomly bump one to 3 to guarantee >= 7 turns.
    """
    n1 = rng.randint(2, 5)
    n2 = rng.randint(2, 5)
    if n1 == 2 and n2 == 2:
        if rng.random() < 0.5:
            n1 = 3
        else:
            n2 = 3
    return n1, n2


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt Construction
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert dialogue writer for **TriggerBench**, a benchmark that tests whether \
LLM agents can proactively detect hidden time-sensitive risks in conversation.

Your job: given a scenario **blueprint**, generate a realistic user-only dialogue \
(no assistant responses) as a JSON object.

────────────────────────────────────────────────────
DIALOGUE STRUCTURE  (STRICT — 4-step sequence)
────────────────────────────────────────────────────
[1. Thematic Connection (Section 1: 2-5 turns)] → [2. Constraint (exactly 1 turn)] → \
[3. Thematic Connection (Section 2: 2-5 turns)] → [4. Trigger (exactly 1 turn, LAST)]

The EXACT number of thematic connection turns for each section is specified in \
the user prompt under "REQUIRED TURN STRUCTURE". You MUST follow those counts exactly.

────────────────────────────────────────────────────
TURN FUNCTIONS (enum values)
────────────────────────────────────────────────────
- "thematic_connection": Domain-aligned filler turns. High cognitive-load tasks/questions/discussions \
that are thematically related to the blueprint's (background) topics but have ZERO connection to \
the constraint or trigger or any time-sensitive events.
- "constraint": Exactly 1 turn that states the constraint. Natural, casual, \
first-person speech. NO "remind me", "warn me", "don't forget", etc.
- "trigger": The LAST turn. Natural first-person restatement of the blueprint's trigger.

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

2. **THEMATIC COHERENCE** (strongly required)
   Thematic turns must be grounded in the blueprint's topics. Derive them from the \
user's current situation, occupation, interests, or surroundings. They should feel \
like a natural continuation of the user's day — NOT random or disconnected.
   The FIRST thematic turn should naturally set the scene (e.g., mention the user's \
current activity or situation) combined with a thematic question.

3. **ZERO LEAKAGE**
   Thematic turns must have absolutely NO semantic connection to the constraint's \
objects, states, time-sensitive events, or risks. 100% logically orthogonal.
   *Absolutely NO questions about risks, safety, hazards, dangers, warnings, \
precautions, time management, alarms, timers, or reminders* — even if phrased generically.

4. **CONSTRAINT TURN** (CRITICAL for Time-Trigger)
   Restate the blueprint's constraint faithfully as natural first-person speech. \
It must remain IMPLICIT — no "remind me", "warn me", "don't forget", \
"set a reminder", "TODO" etc. The user just casually mentions a time-bound event.
   CRITICAL TIME RULES:
   - If time_logic_type is "Absolute_Time": The constraint text MUST explicitly mention \
the exact_target_action_time in some recognizable time format (e.g., "8:00 PM", "20:00", \
"midnight", "4:45 PM"). The user directly states the future moment.
   - If time_logic_type is "Relative_Time": The constraint text MUST explicitly mention \
the duration_time string (e.g., "45 minutes", "2 hours"). The user expresses the time \
as a relative offset. ABSOLUTELY FORBIDDEN to state the computed final target time!
   - The generated text MUST contain the key entity words from the blueprint's slot_values \
(at least one alias per required slot).

5. **CLEAN TRIGGER**
   The trigger turn must faithfully restate the blueprint's trigger text as natural \
first-person speech. Strip [timestamp] prefixes. Convert third-person to first-person \
if needed. Do NOT merge in any distracting content.

6. **FIXED TIMESTAMPS** (MUST match exactly)
   - The constraint turn's "time" field MUST exactly equal the blueprint's \
timeline_anchors.constraint_talking_time.
   - The trigger turn's "time" field MUST exactly equal the blueprint's \
timeline_anchors.trigger_talking_time.

7. **TIME FORMATTING & ORDERING**
   - All times in ISO format: YYYY-MM-DDTHH:MM
   - All turn times strictly monotonically increasing.
   - Section 1 thematic turns: all times BEFORE constraint_talking_time.
   - Section 2 thematic turns: all times BETWEEN constraint_talking_time and \
trigger_talking_time.
   - Space turns naturally based on the available time window.
   - Use non-round times (e.g., 10:13, 14:47) — avoid :00, :15, :30, :45 \
for thematic turns.
   - Do NOT place any turn between 01:00 and 05:00 AM.

────────────────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────────────────
Return ONLY a valid JSON object:
{"user_turns": [<turn>, <turn>, ...]}

Each turn: {"turn_id": <int starting from 1>, "time": "<ISO>", \
"function": "<thematic_connection|constraint|trigger>", "text": "<utterance>"}

No markdown fences, no comments, no extra text.\
"""


def build_user_prompt(
    blueprint: dict,
    seed_bp: dict | None,
    seed_dlg: list | None,
    n_thematic1: int,
    n_thematic2: int,
) -> str:
    """Assemble the user prompt for one dialogue generation call."""

    stripped_bp = strip_blueprint_for_prompt(blueprint)
    bp_json = json.dumps(stripped_bp, indent=2, ensure_ascii=False)

    ta = blueprint["story"]["timeline_anchors"]
    constraint_time = ta["constraint_talking_time"]
    trigger_time = ta["trigger_talking_time"]
    time_logic = blueprint["meta"]["time_logic_type"]

    # Compute time gap for guidance
    ct_dt = datetime.strptime(constraint_time, TIME_FMT)
    tt_dt = datetime.strptime(trigger_time, TIME_FMT)
    gap_minutes = int((tt_dt - ct_dt).total_seconds() / 60)
    if gap_minutes >= 60:
        gap_h = gap_minutes // 60
        gap_m = gap_minutes % 60
        gap_human = "~{}h{}m".format(gap_h, gap_m) if gap_m else "~{}h".format(gap_h)
    else:
        gap_human = "~{}min".format(gap_minutes)

    # Total turns
    total_turns = n_thematic1 + 1 + n_thematic2 + 1

    # Time logic specific instructions
    if time_logic == "Absolute_Time":
        target_time = ta.get("exact_target_action_time", "")
        time_logic_instruction = (
            "TIME LOGIC: Absolute_Time\n"
            "- The constraint MUST explicitly mention the exact target time: {}\n"
            "- Use any recognizable format: '8:00 PM', '20:00', 'midnight', etc.\n"
            "- The user directly states the exact moment when the event happens."
        ).format(target_time)
    else:  # Relative_Time
        duration = ta.get("duration_time", "")
        target_time = ta.get("exact_target_action_time", "")
        time_logic_instruction = (
            "TIME LOGIC: Relative_Time\n"
            "- The constraint MUST explicitly mention the duration: \"{}\"\n"
            "- The user describes a process that takes this specific duration.\n"
            "- ABSOLUTELY FORBIDDEN: Do NOT reveal or compute the final target time ({})!"
        ).format(duration, target_time)

    # Seed example section
    if seed_bp and seed_dlg:
        seed_section = (
            "═══ SEED EXAMPLE (for format & quality reference — do NOT duplicate content) ═══\n\n"
            "Input Blueprint:\n"
            "{}\n\n"
            "Output Dialogue:\n"
            "{}"
        ).format(
            json.dumps(seed_bp, indent=2, ensure_ascii=False),
            json.dumps({"user_turns": seed_dlg}, indent=2, ensure_ascii=False),
        )
    else:
        seed_section = "(No seed example available — follow the schema and rules above.)"

    prompt = (
        "Convert the following blueprint into a base user dialogue.\n\n"
        "═══ BLUEPRINT TO CONVERT ═══\n"
        "{bp_json}\n\n"
        "═══ KEY TIMESTAMPS (USE EXACTLY) ═══\n"
        "- constraint_talking_time: {constraint_time}  (constraint turn MUST use this exact time)\n"
        "- trigger_talking_time: {trigger_time}  (trigger turn MUST use this exact time)\n"
        "- Time gap between constraint and trigger: {gap_human} ({gap_minutes} minutes)\n"
        "- Section 1 turns: place within a reasonable window BEFORE {constraint_time} "
        "(e.g., 20 min to a few hours before, depending on background)\n"
        "- Section 2 turns: spread naturally BETWEEN {constraint_time} and {trigger_time}\n\n"
        "═══ {time_logic_instruction} ═══\n\n"
        "{seed_section}\n\n"
        "═══ REQUIRED TURN STRUCTURE ═══\n"
        "- Thematic Connection Section 1: exactly {n_thematic1} turns\n"
        "- Constraint turn: exactly 1 turn\n"
        "- Thematic Connection Section 2: exactly {n_thematic2} turns\n"
        "- Trigger turn: exactly 1 turn (LAST turn)\n"
        "- Total: {total_turns} turns\n\n"
        "═══ SLOT VALUES TO PRESERVE ═══\n"
        "The constraint turn text MUST contain at least one alias from each required slot:\n"
        "{slot_info}\n\n"
        "═══ REMINDERS ═══\n"
        "- All thematic_connection turns must be context-free (LLM-answerable, no external tools)\n"
        "- Thematic turns should be grounded in the scenario background topics (domain-aligned)\n"
        "- ZERO semantic overlap between thematic turns and the constraint/trigger topic\n"
        "- NO questions about risks, safety, hazards, warnings, time management, or precautions\n"
        "- Constraint turn: natural first-person speech, IMPLICIT (no 'remind me' etc.)\n"
        "- Trigger turn: faithful first-person restatement of the blueprint trigger "
        "(strip [timestamp] prefix)\n"
        "- All times strictly increasing, in YYYY-MM-DDTHH:MM format\n"
        "- constraint time MUST be exactly {constraint_time}\n"
        "- trigger time MUST be exactly {trigger_time}\n"
        "- Do NOT place any turn between 01:00 and 05:00 AM\n"
        "- Use non-round times for thematic turns (e.g., 09:13, 14:47)\n"
        "- Return ONLY {{\"user_turns\": [...]}} as valid JSON"
    ).format(
        bp_json=bp_json,
        constraint_time=constraint_time,
        trigger_time=trigger_time,
        gap_human=gap_human,
        gap_minutes=gap_minutes,
        time_logic_instruction=time_logic_instruction,
        seed_section=seed_section,
        n_thematic1=n_thematic1,
        n_thematic2=n_thematic2,
        total_turns=total_turns,
        slot_info=json.dumps(
            blueprint.get("gold_labels", {}).get("slots", {}),
            indent=2, ensure_ascii=False,
        ),
    )
    return prompt


# JSON extraction is provided by shared.json_utils.extract_json_object


# ═══════════════════════════════════════════════════════════════════════════
#  Validation Helpers
# ═══════════════════════════════════════════════════════════════════════════

EXPLICIT_KEYWORDS = [
    "remind me", "warn me", "alert me", "notify me",
    "don't let me forget", "do not let me forget",
    "take a note", "set a reminder", "set an alarm",
    "make sure i", "remember to", "todo",
    "don't forget", "do not forget", "keep in mind",
    "heads up later", "ping me", "flag this",
    "note to self",
]

VALID_FUNCTIONS = {"thematic_connection", "constraint", "trigger"}


def _time_appears_in_text(time_iso: str, text: str) -> bool:
    """Check if a target time appears in text in any common format.

    Handles: 12-hour, 24-hour, with/without :00, midnight, noon, etc.
    """
    text_lower = text.lower()

    try:
        dt = datetime.strptime(time_iso.strip(), TIME_FMT)
    except ValueError:
        return False

    # Special cases
    if dt.hour == 0 and dt.minute == 0:
        if "midnight" in text_lower:
            return True
    if dt.hour == 12 and dt.minute == 0:
        if "noon" in text_lower:
            return True

    h = dt.hour
    m = dt.minute

    # Generate all common time formats
    variants = set()

    # 24-hour formats
    variants.add("{:02d}:{:02d}".format(h, m))  # "20:00"
    if m == 0:
        variants.add("{:02d}:00".format(h))

    # 12-hour formats
    h12 = h % 12 if h % 12 != 0 else 12
    ampm = "am" if h < 12 else "pm"
    AMPM = "AM" if h < 12 else "PM"

    if m == 0:
        variants.add("{} {}".format(h12, ampm))         # "8 pm"
        variants.add("{} {}".format(h12, AMPM))         # "8 PM"
        variants.add("{}{}".format(h12, ampm))          # "8pm"
        variants.add("{}{}".format(h12, AMPM))          # "8PM"
        variants.add("{}:00 {}".format(h12, ampm))      # "8:00 pm"
        variants.add("{}:00 {}".format(h12, AMPM))      # "8:00 PM"
        variants.add("{}:00{}".format(h12, ampm))       # "8:00pm"
        variants.add("{}:00{}".format(h12, AMPM))       # "8:00PM"
        variants.add("{} o'clock".format(h12))          # "8 o'clock"
    else:
        variants.add("{}:{:02d} {}".format(h12, m, ampm))   # "4:45 pm"
        variants.add("{}:{:02d} {}".format(h12, m, AMPM))   # "4:45 PM"
        variants.add("{}:{:02d}{}".format(h12, m, ampm))    # "4:45pm"
        variants.add("{}:{:02d}{}".format(h12, m, AMPM))    # "4:45PM"

    for v in variants:
        if v.lower() in text_lower:
            return True
    return False


NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight",
    9: "nine", 10: "ten", 12: "twelve", 15: "fifteen",
    20: "twenty", 30: "thirty", 45: "forty-five",
}


def _parse_duration_to_minutes(duration_str: str) -> int | None:
    """Parse a human-readable duration string to total minutes."""
    s = duration_str.lower().strip()
    total = 0
    h_match = re.search(r"(\d+)\s*hours?", s)
    m_match = re.search(r"(\d+)\s*minutes?", s)
    if h_match:
        total += int(h_match.group(1)) * 60
    if m_match:
        total += int(m_match.group(1))
    return total if total > 0 else None


def _duration_appears_in_text(duration_str: str, text: str) -> bool:
    """Check if a duration appears in text in any common format.

    Handles: exact match, unit conversion, half-hour variants, number-word variants.
    """
    text_lower = text.lower()

    # Direct match
    if duration_str.lower() in text_lower:
        return True

    minutes = _parse_duration_to_minutes(duration_str)
    if minutes is None:
        return False

    alts = set()

    # Pure minutes
    alts.add("{} minutes".format(minutes))
    alts.add("{} min".format(minutes))
    if minutes in NUMBER_WORDS:
        alts.add("{} minutes".format(NUMBER_WORDS[minutes]))

    # Hours + minutes
    if minutes >= 60:
        h = minutes // 60
        m = minutes % 60
        if m == 0:
            alts.add("{} hour{}".format(h, "s" if h > 1 else ""))
            if h in NUMBER_WORDS:
                alts.add("{} hour{}".format(NUMBER_WORDS[h], "s" if h > 1 else ""))
        else:
            alts.add("{} hour and {} minute".format(h, m))
            alts.add("{} hours and {} minutes".format(h, m))
            alts.add("{}h{}m".format(h, m))
            if m == 30:
                alts.add("{}.5 hours".format(h))
                alts.add("{} and a half hours".format(h))
                if h == 1:
                    alts.add("an hour and a half")
    elif minutes == 30:
        alts.add("half an hour")
        alts.add("half hour")
    elif minutes == 90:
        alts.add("an hour and a half")
        alts.add("1.5 hours")
        alts.add("one and a half hours")

    alts.discard("")
    for alt in alts:
        if alt.lower() in text_lower:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_dialogue(
    turns: list,
    blueprint: dict,
) -> tuple:
    """Validate a generated dialogue against its blueprint.

    Returns (is_valid, error_list).
    """
    errors = []
    time_logic = blueprint["meta"]["time_logic_type"]
    ta = blueprint["story"]["timeline_anchors"]

    # ── 1. Basic structure ──────────────────────────────────────────────────
    if not isinstance(turns, list):
        errors.append("user_turns must be a list")
        return False, errors

    if len(turns) < 6:
        errors.append("Too few turns: {} (minimum 6: 2+1+2+1)".format(len(turns)))
    if len(turns) > 12:
        errors.append("Too many turns: {} (maximum 12)".format(len(turns)))

    # ── 2. Per-turn field validation ────────────────────────────────────────
    for i, turn in enumerate(turns):
        for key in ("turn_id", "time", "function", "text"):
            if key not in turn:
                errors.append("Turn {}: missing field '{}'".format(i + 1, key))
        if "function" in turn and turn["function"] not in VALID_FUNCTIONS:
            errors.append("Turn {}: invalid function '{}'".format(i + 1, turn["function"]))
    if errors:
        return False, errors

    # ── 3. Function sequence ────────────────────────────────────────────────
    functions = [t["function"] for t in turns]

    constraint_positions = [i for i, f in enumerate(functions) if f == "constraint"]
    trigger_positions = [i for i, f in enumerate(functions) if f == "trigger"]

    if len(constraint_positions) != 1:
        errors.append(
            "Expected exactly 1 constraint turn, got {}".format(len(constraint_positions))
        )
    if len(trigger_positions) != 1:
        errors.append(
            "Expected exactly 1 trigger turn, got {}".format(len(trigger_positions))
        )
    if errors:
        return False, errors

    c_pos = constraint_positions[0]
    t_pos = trigger_positions[0]

    if t_pos <= c_pos:
        errors.append("Trigger must come after constraint")
        return False, errors

    # Trigger must be the last turn
    if t_pos != len(turns) - 1:
        errors.append("Trigger must be the last turn")

    # All turns before constraint must be thematic_connection
    section1 = functions[:c_pos]
    section2 = functions[c_pos + 1:t_pos]

    if not all(f == "thematic_connection" for f in section1):
        errors.append("All turns before constraint must be 'thematic_connection'")
    if not all(f == "thematic_connection" for f in section2):
        errors.append("All turns between constraint and trigger must be 'thematic_connection'")

    # ── 4. Section counts ───────────────────────────────────────────────────
    n_thematic1 = len(section1)
    n_thematic2 = len(section2)

    if n_thematic1 < 2:
        errors.append("Thematic Section 1 has {} turns (need >= 2)".format(n_thematic1))
    if n_thematic2 < 2:
        errors.append("Thematic Section 2 has {} turns (need >= 2)".format(n_thematic2))
    if n_thematic1 > 5:
        errors.append("Thematic Section 1 has {} turns (max 5)".format(n_thematic1))
    if n_thematic2 > 5:
        errors.append("Thematic Section 2 has {} turns (max 5)".format(n_thematic2))

    # ── 5. turn_id monotonicity ─────────────────────────────────────────────
    for i, turn in enumerate(turns):
        if turn.get("turn_id") != i + 1:
            errors.append(
                "Turn {}: turn_id should be {}, got {}".format(i + 1, i + 1, turn.get("turn_id"))
            )
            break

    # ── 6. Time monotonicity ────────────────────────────────────────────────
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

    # ── 7. Fixed timestamp matching (CRITICAL) ─────────────────────────────
    expected_ct = ta["constraint_talking_time"]
    expected_tt = ta["trigger_talking_time"]

    actual_ct = turns[c_pos]["time"].strip()
    actual_tt = turns[t_pos]["time"].strip()

    if actual_ct != expected_ct:
        errors.append(
            "Constraint time mismatch: got '{}', expected '{}'".format(
                actual_ct, expected_ct
            )
        )
    if actual_tt != expected_tt:
        errors.append(
            "Trigger time mismatch: got '{}', expected '{}'".format(
                actual_tt, expected_tt
            )
        )

    # ── 8. Boundary checks (section times in correct ranges) ────────────────
    ct_dt = parse_iso(expected_ct)
    tt_dt = parse_iso(expected_tt)

    if ct_dt and tt_dt:
        # Section 1: all times < constraint_talking_time
        for i in range(c_pos):
            dt = parse_iso(turns[i]["time"])
            if dt and dt >= ct_dt:
                errors.append(
                    "Turn {} (thematic section 1) time '{}' must be BEFORE "
                    "constraint time '{}'".format(i + 1, turns[i]["time"], expected_ct)
                )
                break

        # Section 2: constraint_talking_time < time < trigger_talking_time
        for i in range(c_pos + 1, t_pos):
            dt = parse_iso(turns[i]["time"])
            if dt:
                if dt <= ct_dt:
                    errors.append(
                        "Turn {} (thematic section 2) time '{}' must be AFTER "
                        "constraint time '{}'".format(i + 1, turns[i]["time"], expected_ct)
                    )
                    break
                if dt >= tt_dt:
                    errors.append(
                        "Turn {} (thematic section 2) time '{}' must be BEFORE "
                        "trigger time '{}'".format(i + 1, turns[i]["time"], expected_tt)
                    )
                    break

    # ── 9. Slot Grounding Check ─────────────────────────────────────────────
    constraint_text = turns[c_pos].get("text", "")
    constraint_lower = constraint_text.lower()
    slots = blueprint.get("gold_labels", {}).get("slots", {})
    required_slots = slots.get("required_slots", [])
    slot_values = slots.get("slot_values", {})

    for slot_name in required_slots:
        aliases = slot_values.get(slot_name, [])
        if not aliases:
            continue
        found = any(alias.lower() in constraint_lower for alias in aliases)
        if not found:
            errors.append(
                "Slot '{}' not grounded: none of {} found in constraint text".format(
                    slot_name, aliases
                )
            )

    # ── 10. Constraint must be implicit (no explicit keywords) ──────────────
    for kw in EXPLICIT_KEYWORDS:
        if kw in constraint_lower:
            errors.append("Constraint text contains explicit keyword: '{}'".format(kw))
            break

    # ── 11. Time Logic Compliance Check (CRITICAL) ──────────────────────────
    if time_logic == "Relative_Time":
        duration_str = ta.get("duration_time", "")
        if duration_str and not _duration_appears_in_text(duration_str, constraint_text):
            errors.append(
                "Relative_Time: duration '{}' must appear in constraint text "
                "(or a recognized variant)".format(duration_str)
            )
        # Check that the computed target time does NOT appear
        target_time = ta.get("exact_target_action_time", "")
        if target_time and _time_appears_in_text(target_time, constraint_text):
            errors.append(
                "Relative_Time: computed target time '{}' must NOT appear in "
                "constraint text (only the duration should be stated)".format(target_time)
            )
    elif time_logic == "Absolute_Time":
        target_time = ta.get("exact_target_action_time", "")
        if target_time and not _time_appears_in_text(target_time, constraint_text):
            errors.append(
                "Absolute_Time: target time '{}' must appear in constraint text "
                "in some recognizable format".format(target_time)
            )

    return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage 2: Stretch Temporal Grounding blueprints into user-only dialogues"
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
                    help="Also generate dialogues for the 4 human-authored seed blueprints")
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
        new_seeds = [s for s in TEMPORAL_GROUNDING_SEEDS if s["task_id"] not in bp_ids]
        blueprints = new_seeds + blueprints
        log.info("--include-seeds: added %d seed blueprints (total now %d)",
                 len(new_seeds), len(blueprints))

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else DIM_DIR / "outputs" / "stage2_user_dialogues"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "user_dialogues_{}.json".format(args.model)

    # ── Resume support ──────────────────────────────────────────────────────
    all_dialogues = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as f:
                all_dialogues = json.load(f)
            log.info("Resumed %d dialogues from %s", len(all_dialogues), resume_path)

    # ── Stats ───────────────────────────────────────────────────────────────
    total = len(blueprints)
    done = len(all_dialogues)
    failed = 0

    log.info(
        "Target: %d dialogues | Model=%s, T=%.2f, max_retries=%d",
        total, args.model, args.temperature, args.max_retries,
    )

    # ═══════════════════════════════════════════════════════════════════════
    #  Generation loop
    # ═══════════════════════════════════════════════════════════════════════
    for bp_idx, blueprint in enumerate(tqdm(blueprints, desc="Stage 2 Temporal Grounding")):
        if args.sleep:
            time_module.sleep(args.sleep)

        task_id = blueprint.get("task_id", "unknown_{}".format(bp_idx))

        # Skip if already done (resume)
        if task_id in all_dialogues:
            log.info("[%d/%d] %s — already done, skipping", bp_idx + 1, total, task_id)
            continue

        time_logic = blueprint["meta"]["time_logic_type"]
        trigger_pattern = blueprint["meta"]["trigger_pattern"]

        log.info("[%d/%d] Generating dialogue for %s  (%s × %s)",
                 bp_idx + 1, total, task_id, time_logic, trigger_pattern)

        # Get matching seed example for ICL
        seed_bp, seed_dlg = get_seed_example(time_logic, trigger_pattern)

        # Sample thematic turn counts
        n_thematic1, n_thematic2 = sample_thematic_split(rng)

        user_prompt = build_user_prompt(
            blueprint, seed_bp, seed_dlg, n_thematic1, n_thematic2
        )
        base_user_prompt = user_prompt
        success = False

        for retry in range(args.max_retries):
            try:
                raw = call_llm(SYSTEM_PROMPT, user_prompt)
                result = extract_json_object(raw)

                turns = result.get("user_turns", [])
                valid, errs = validate_dialogue(turns, blueprint)

                if valid:
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

    # Per time_logic × trigger_pattern breakdown
    stats = {}
    for tid in all_dialogues:
        for bp in blueprints:
            if bp["task_id"] == tid:
                meta = bp.get("meta", {})
                key = "{}_×_{}".format(
                    meta.get("time_logic_type", "?"),
                    meta.get("trigger_pattern", "?"),
                )
                stats[key] = stats.get(key, 0) + 1
                break
    log.info("Per-cell counts:")
    for key in sorted(stats):
        log.info("  %-45s  %d", key, stats[key])


if __name__ == "__main__":
    main()

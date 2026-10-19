#!/usr/bin/env python3
"""Stage 2 — User-dialogue stretching for the Logical Adherence dimension.

Converts Stage 1 blueprints into user-only dialogues.

Two dialogue structure types based on logic_type:

Type A (Forward_If-Then) — 6-section structure:
  [Thematic ×2-5] → [rule_constraint ×1] → [Thematic ×2-5]
  → [condition_constraint ×1] → [Thematic ×2-5] → [trigger ×1]

Type B (Mutual_Exclusion, Only-After) — 4-section structure:
  [Thematic ×2-5] → [rule_constraint ×1] → [Thematic ×2-5] → [trigger ×1]

Key design decisions:
  - Only positive variants are used (negative info stripped from prompt).
  - Slot grounding for Type A merges rule_constraint + condition_constraint text.
  - Time gap validation uses parse_time_gap_minutes with ±tolerance.

No assistant utterances are generated at this stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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
from shared.time_utils import parse_iso, parse_time_gap_minutes
from seeds.blueprints import LOGICAL_ADHERENCE_SEEDS
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
#  Blueprint Stripping (remove irrelevant/leaking fields from prompt)
# ═══════════════════════════════════════════════════════════════════════════

def strip_blueprint_for_prompt(bp: dict) -> dict:
    """Strip a blueprint to only the fields the dialogue-generation LLM should see.

    Removes: negative variants, trigger_distracting_task, negative_gold.
    For Type A: keeps only condition_constraint.positive.
    For Type B: keeps only trigger.positive.
    """
    story = bp.get("story", {})
    meta = bp.get("meta", {})
    gl = bp.get("gold_labels", {})
    logic_type = meta.get("logic_type", "")

    stripped_story = {
        "background": story.get("background", ""),
        "rule_constraint": story.get("rule_constraint", ""),
    }

    if logic_type == "Forward_If-Then":
        # Type A: condition_constraint positive only, trigger is plain string
        cond = story.get("condition_constraint", {})
        stripped_story["condition_constraint"] = cond.get("positive", "") if isinstance(cond, dict) else cond
        stripped_story["time_gap_rule_to_condition"] = story.get("time_gap_rule_to_condition", "")
        stripped_story["time_gap_condition_to_trigger"] = story.get("time_gap_condition_to_trigger", "")
        stripped_story["trigger"] = story.get("trigger", "")
    else:
        # Type B: trigger positive only
        trig = story.get("trigger", {})
        stripped_story["time_gap_between_constraint_and_trigger"] = story.get(
            "time_gap_between_constraint_and_trigger", ""
        )
        stripped_story["trigger"] = trig.get("positive", "") if isinstance(trig, dict) else trig

    return {
        "task_id": bp.get("task_id", ""),
        "meta": meta,
        "story": stripped_story,
        "gold_labels": {
            "positive_gold": {
                "slots": gl.get("positive_gold", {}).get("slots", {}),
            }
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Seed Example for ICL
# ═══════════════════════════════════════════════════════════════════════════

# Map logic_type to seed task_id for ICL example
SEED_KEY_MAP = {
    "Forward_If-Then": "seed_ifthen_001",
    "Mutual_Exclusion": "seed_exclusion_001",
    "Only-After": "seed_onlyafter_001",
}


def get_seed_example(logic_type: str) -> tuple:
    """Return (stripped_seed_blueprint, seed_dialogue) for ICL based on logic_type."""
    seed_task_id = SEED_KEY_MAP.get(logic_type)
    if not seed_task_id:
        return None, None

    seed_bp = None
    for s in LOGICAL_ADHERENCE_SEEDS:
        if s["task_id"] == seed_task_id:
            seed_bp = s
            break
    if seed_bp is None:
        return None, None

    seed_dlg = SEED_USER_DIALOGUES.get(seed_task_id)
    if seed_dlg is None:
        return None, None

    stripped = strip_blueprint_for_prompt(seed_bp)
    return stripped, seed_dlg


# ═══════════════════════════════════════════════════════════════════════════
#  Thematic Connection Turn Count Sampling
# ═══════════════════════════════════════════════════════════════════════════

def sample_thematic_split_type_a(rng: random.Random) -> tuple:
    """Sample (n_section1, n_section2, n_section3) for Type A (6-section).

    Each section: 2-5 thematic turns.
    Total: n1 + 1(rule) + n2 + 1(condition) + n3 + 1(trigger) = n1+n2+n3+3
    Range: 9-18 turns.
    """
    n1 = rng.randint(2, 5)
    n2 = rng.randint(2, 5)
    n3 = rng.randint(2, 5)
    # Ensure minimum total of ~9 turns
    if n1 + n2 + n3 < 7:
        # Bump one randomly
        choice = rng.choice([0, 1, 2])
        if choice == 0:
            n1 = max(n1, 3)
        elif choice == 1:
            n2 = max(n2, 3)
        else:
            n3 = max(n3, 3)
    return n1, n2, n3


def sample_thematic_split_type_b(rng: random.Random) -> tuple:
    """Sample (n_section1, n_section2) for Type B (4-section).

    Section 1 (pre-constraint): 2-5 turns
    Section 2 (post-constraint): 2-5 turns
    Total: n1 + 1(rule) + n2 + 1(trigger) = n1+n2+2
    Range: 6-12 turns.
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
#  System Prompts (Type A & Type B)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TYPE_A = """\
You are an expert dialogue writer for **TriggerBench**, a benchmark that tests whether \
LLM agents can proactively detect and enforce user-defined conditional rules in conversation.

Your job: given a scenario **blueprint**, generate a realistic user-only dialogue \
(no assistant responses) as a JSON object.

────────────────────────────────────────────────────
DIALOGUE STRUCTURE  (STRICT — 6-section sequence for Forward_If-Then)
────────────────────────────────────────────────────
[1. Thematic Connection Section 1] → [2. rule_constraint (exactly 1 turn)] → \
[3. Thematic Connection Section 2] → [4. condition_constraint (exactly 1 turn)] → \
[5. Thematic Connection Section 3] → [6. trigger (exactly 1 turn, LAST)]

The EXACT number of thematic connection turns for each section is specified in \
the user prompt under "REQUIRED TURN STRUCTURE". You MUST follow those counts exactly.

────────────────────────────────────────────────────
TURN FUNCTIONS (enum values)
────────────────────────────────────────────────────
- "thematic_connection": Domain-aligned filler turns. High cognitive-load tasks/questions/discussions \
that are thematically related to the blueprint's topics but have ZERO connection to \
the rule_constraint, condition_constraint, or trigger.
- "rule_constraint": Exactly 1 turn where user IMPLICITLY states the If-Then rule. \
Natural, casual, first-person speech. NO "remind me", "warn me", "don't forget", etc.  Strictly include all information from `rule_constraint`.
- "condition_constraint": Exactly 1 turn where the user casually reports that the condition \
has been met. Natural first-person speech. Strictly include all information from `condition_constraint`.
- "trigger": The LAST turn. Natural first-person restatement of the blueprint's trigger. Strictly include all information from `trigger`.

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
   Thematic turns must have absolutely NO semantic connection to the rule_constraint, \
the condition_constraint, or the trigger. 100% logically orthogonal.
   *Absolutely NO questions about risks, safety, hazards, dangers, warnings, \
precautions, or any protective/preventive measures* — even if phrased generically.

4. **RULE_CONSTRAINT TURN** (CRITICAL)
   Restate the blueprint's rule_constraint faithfully and completely as natural first-person speech. \
It must remain IMPLICIT — no "remind me", "warn me", "don't forget", \
"set a reminder", "TODO" etc. The user just casually states their personal rule/intent.
   CRITICAL: The generated text MUST contain the key entity words from the \
blueprint's slot_values (at least one alias per required slot).

5. **CONDITION_CONSTRAINT TURN**
   Restate the blueprint's condition_constraint faithfully and completely as natural first-person \
speech. The user casually reports that the If-condition has been met. Keep it natural \
and conversational. The text MUST contain relevant entity words that establish the condition.

6. **CLEAN TRIGGER**
   The trigger turn must faithfully and completely restate the blueprint's trigger text as natural \
first-person speech. Do NOT merge in any distracting content.

7. **TIME FORMATTING & ORDERING**
   - All times in ISO format: YYYY-MM-DDTHH:MM
   - All turn times strictly monotonically increasing.
   - Default spacing between consecutive thematic turns: ~2-10 minutes (varies naturally).
   - The gap between rule_constraint and condition_constraint must EXACTLY match \
the blueprint's `time_gap_rule_to_condition`.
   - The gap between condition_constraint and trigger must EXACTLY match \
the blueprint's `time_gap_condition_to_trigger`.
   - Do NOT place any turn between 01:00 and 05:00 AM.
   - Use non-round times (e.g., 10:13, 14:47).

────────────────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────────────────
Return ONLY a valid JSON object:
{"user_turns": [<turn>, <turn>, ...]}

Each turn: {"turn_id": <int starting from 1>, "time": "<ISO>", \
"function": "<thematic_connection|rule_constraint|condition_constraint|trigger>", "text": "<utterance>"}

No markdown fences, no comments, no extra text.\
"""

SYSTEM_PROMPT_TYPE_B = """\
You are an expert dialogue writer for **TriggerBench**, a benchmark that tests whether \
LLM agents can proactively detect and enforce user-defined conditional rules in conversation.

Your job: given a scenario **blueprint**, generate a realistic user-only dialogue \
(no assistant responses) as a JSON object.

────────────────────────────────────────────────────
DIALOGUE STRUCTURE  (STRICT — 4-section sequence for Mutual_Exclusion / Only-After)
────────────────────────────────────────────────────
[1. Thematic Connection Section 1] → [2. rule_constraint (exactly 1 turn)] → \
[3. Thematic Connection Section 2] → [4. trigger (exactly 1 turn, LAST)]

The EXACT number of thematic connection turns for each section is specified in \
the user prompt under "REQUIRED TURN STRUCTURE". You MUST follow those counts exactly.

────────────────────────────────────────────────────
TURN FUNCTIONS (enum values)
────────────────────────────────────────────────────
- "thematic_connection": Domain-aligned filler turns. High cognitive-load tasks/questions/discussions \
that are thematically related to the blueprint's topics but have ZERO connection to \
the rule_constraint or trigger.
- "rule_constraint": Exactly 1 turn that IMPLICITLY states the user's rule/red-line/prerequisite. \
Natural, casual, first-person speech. NO "remind me", "warn me", "don't forget", etc. Strictly include all information from `rule_constraint`.
- "trigger": The LAST turn. Natural first-person restatement of the blueprint's trigger. Strictly include all information from `trigger`. 

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
   Thematic turns must be grounded in the story's topics. Derive them from the \
user's current situation, occupation, interests, or surroundings. They should feel \
like a natural continuation of the user's day — NOT random or disconnected.
   The FIRST thematic turn should naturally set the scene (e.g., mention the user's \
current activity or situation) combined with a thematic question.

3. **ZERO LEAKAGE**
   Thematic turns must have absolutely NO semantic connection to the rule_constraint's \
objects, thresholds, red-lines, or the trigger content. 100% logically orthogonal.
   *Absolutely NO questions about risks, safety, hazards, dangers, warnings, \
precautions, or any protective/preventive measures* — even if phrased generically.

4. **RULE_CONSTRAINT TURN** (CRITICAL)
   Restate the blueprint's rule_constraint faithfully as natural first-person speech. \
It must remain IMPLICIT — no "remind me", "warn me", "don't forget", \
"set a reminder", "TODO" etc. The user just casually states their personal rule/limit. 
   CRITICAL: The generated text MUST contain the key entity words from the \
blueprint's slot_values (at least one alias per required slot).

5. **CLEAN TRIGGER**
   The trigger turn must faithfully restate the blueprint's trigger \
as natural first-person speech. Do NOT merge in any distracting content.
    
6. **TIME FORMATTING & ORDERING**
   - All times in ISO format: YYYY-MM-DDTHH:MM
   - All turn times strictly monotonically increasing.
   - Default spacing between consecutive thematic turns: ~2-10 minutes (varies naturally).
   - The gap between rule_constraint and trigger must EXACTLY match \
the blueprint's time_gap_between_constraint_and_trigger.
   - Do NOT place any turn between 01:00 and 05:00 AM.
   - Use non-round times (e.g., 10:13, 14:47) — avoid :00, :15, :30, :45.

────────────────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────────────────
Return ONLY a valid JSON object:
{"user_turns": [<turn>, <turn>, ...]}

Each turn: {"turn_id": <int starting from 1>, "time": "<ISO>", \
"function": "<thematic_connection|rule_constraint|trigger>", "text": "<utterance>"}

No markdown fences, no comments, no extra text.\
"""


# ═══════════════════════════════════════════════════════════════════════════
#  User Prompt Construction
# ═══════════════════════════════════════════════════════════════════════════

def build_user_prompt_type_a(
    blueprint: dict,
    seed_bp: dict | None,
    seed_dlg: list | None,
    n_thematic1: int,
    n_thematic2: int,
    n_thematic3: int,
) -> str:
    """Build user prompt for Type A (Forward_If-Then) — 6-section structure."""

    stripped_bp = strip_blueprint_for_prompt(blueprint)
    bp_json = json.dumps(stripped_bp, indent=2, ensure_ascii=False)

    time_gap_rc = blueprint["story"].get("time_gap_rule_to_condition", "")
    time_gap_ct = blueprint["story"].get("time_gap_condition_to_trigger", "")
    total_turns = n_thematic1 + 1 + n_thematic2 + 1 + n_thematic3 + 1

    # Seed example section
    seed_section = ""
    if seed_bp and seed_dlg:
        seed_section = (
            "═══ SEED EXAMPLE (for format & quality reference — do NOT duplicate content) ═══\n\n"
            "Input Blueprint:\n"
            "{seed_bp_json}\n\n"
            "Output Dialogue:\n"
            "{seed_dlg_json}\n\n"
        ).format(
            seed_bp_json=json.dumps(seed_bp, indent=2, ensure_ascii=False),
            seed_dlg_json=json.dumps({"user_turns": seed_dlg}, indent=2, ensure_ascii=False),
        )

    # Slot info
    slots = blueprint.get("gold_labels", {}).get("positive_gold", {}).get("slots", {})
    slot_info = json.dumps(slots, indent=2, ensure_ascii=False)

    prompt = (
        "Convert the following blueprint into a base user dialogue.\n\n"
        "═══ BLUEPRINT TO CONVERT ═══\n"
        "{bp_json}\n\n"
        "═══ TIME GAP INFO (Type A: Forward_If-Then) ═══\n"
        "- time_gap_rule_to_condition: {time_gap_rc}\n"
        "  → The rule_constraint and condition_constraint turn timestamps must be EXACTLY this far apart.\n"
        "- time_gap_condition_to_trigger: {time_gap_ct}\n"
        "  → The condition_constraint and trigger turn timestamps must be EXACTLY this far apart.\n"
        "- Choose a plausible start time for the day (e.g., between 06:00 and 22:00).\n"
        "- Space thematic turns naturally: ~2-10 minutes between consecutive turns.\n\n"
        "{seed_section}"
        "═══ REQUIRED TURN STRUCTURE ═══\n"
        "- Thematic Connection Section 1: exactly {n_thematic1} turns\n"
        "- rule_constraint turn: exactly 1 turn\n"
        "- Thematic Connection Section 2: exactly {n_thematic2} turns\n"
        "- condition_constraint turn: exactly 1 turn\n"
        "- Thematic Connection Section 3: exactly {n_thematic3} turns\n"
        "- trigger turn: exactly 1 turn (LAST turn)\n"
        "- Total: {total_turns} turns\n\n"
        "═══ SLOT VALUES TO PRESERVE ═══\n"
        "The rule_constraint and/or condition_constraint turn text MUST contain at least one alias from each required slot:\n"
        "{slot_info}\n\n"
        "═══ REMINDERS ═══\n"
        "- All thematic_connection turns must be context-free (LLM-answerable, no external tools)\n"
        "- Thematic turns should be grounded in the scenario background topics (domain-aligned)\n"
        "- ZERO semantic overlap between thematic turns and the constraints/trigger topic\n"
        "- NO questions about risks, safety, hazards, warnings, or precautions\n"
        "- rule_constraint turn: natural first-person restatement, IMPLICIT (no 'remind me' etc.)\n"
        "- condition_constraint turn: natural first-person report that the condition has been met\n"
        "- trigger turn: faithful first-person restatement of the blueprint trigger\n"
        "- constraint/condition utterance must express ALL information from the blueprint\n"
        "- All times strictly increasing, in YYYY-MM-DDTHH:MM format\n"
        "- Do NOT place any turn between 01:00 and 05:00 AM\n"
        "- Use non-round times (e.g., 09:13, 14:47) — avoid :00, :15, :30, :45\n"
        "- Make dialogue natural, engaging, and realistic\n"
        "- Return ONLY {{\"user_turns\": [...]}} as valid JSON"
    ).format(
        bp_json=bp_json,
        time_gap_rc=time_gap_rc,
        time_gap_ct=time_gap_ct,
        seed_section=seed_section,
        n_thematic1=n_thematic1,
        n_thematic2=n_thematic2,
        n_thematic3=n_thematic3,
        total_turns=total_turns,
        slot_info=slot_info,
    )
    return prompt


def build_user_prompt_type_b(
    blueprint: dict,
    seed_bp: dict | None,
    seed_dlg: list | None,
    n_thematic1: int,
    n_thematic2: int,
) -> str:
    """Build user prompt for Type B (Mutual_Exclusion / Only-After) — 4-section structure."""

    stripped_bp = strip_blueprint_for_prompt(blueprint)
    bp_json = json.dumps(stripped_bp, indent=2, ensure_ascii=False)

    time_gap = blueprint["story"].get("time_gap_between_constraint_and_trigger", "")
    total_turns = n_thematic1 + 1 + n_thematic2 + 1

    # Seed example section
    seed_section = ""
    if seed_bp and seed_dlg:
        seed_section = (
            "═══ SEED EXAMPLE (for format & quality reference — do NOT duplicate content) ═══\n\n"
            "Input Blueprint:\n"
            "{seed_bp_json}\n\n"
            "Output Dialogue:\n"
            "{seed_dlg_json}\n\n"
        ).format(
            seed_bp_json=json.dumps(seed_bp, indent=2, ensure_ascii=False),
            seed_dlg_json=json.dumps({"user_turns": seed_dlg}, indent=2, ensure_ascii=False),
        )

    # Slot info
    slots = blueprint.get("gold_labels", {}).get("positive_gold", {}).get("slots", {})
    slot_info = json.dumps(slots, indent=2, ensure_ascii=False)

    prompt = (
        "Convert the following blueprint into a base user dialogue.\n\n"
        "═══ BLUEPRINT TO CONVERT ═══\n"
        "{bp_json}\n\n"
        "═══ TIME GAP INFO (Type B: {logic_type}) ═══\n"
        "- time_gap_between_constraint_and_trigger: {time_gap}\n"
        "  → The rule_constraint and trigger turn timestamps must be EXACTLY this far apart.\n"
        "- Choose a plausible start time for the day (e.g., between 06:00 and 22:00).\n"
        "- Space thematic turns naturally: ~2-10 minutes between consecutive turns.\n\n"
        "{seed_section}"
        "═══ REQUIRED TURN STRUCTURE ═══\n"
        "- Thematic Connection Section 1: exactly {n_thematic1} turns\n"
        "- rule_constraint turn: exactly 1 turn\n"
        "- Thematic Connection Section 2: exactly {n_thematic2} turns\n"
        "- trigger turn: exactly 1 turn (LAST turn)\n"
        "- Total: {total_turns} turns\n\n"
        "═══ SLOT VALUES TO PRESERVE ═══\n"
        "The rule_constraint turn text MUST contain at least one alias from each required slot:\n"
        "{slot_info}\n\n"
        "═══ REMINDERS ═══\n"
        "- All thematic_connection turns must be context-free (LLM-answerable, no external tools)\n"
        "- Thematic turns should be grounded in the scenario background topics (domain-aligned)\n"
        "- ZERO semantic overlap between thematic turns and the constraint/trigger topic\n"
        "- NO questions about risks, safety, hazards, warnings, or precautions\n"
        "- rule_constraint turn: natural first-person restatement, IMPLICIT (no 'remind me' etc.)\n"
        "- trigger turn: faithful first-person restatement of the blueprint trigger (positive)\n"
        "- constraint utterance must express ALL information from the blueprint\n"
        "- All times strictly increasing, in YYYY-MM-DDTHH:MM format\n"
        "- Do NOT place any turn between 01:00 and 05:00 AM\n"
        "- Use non-round times (e.g., 09:13, 14:47) — avoid :00, :15, :30, :45\n"
        "- Make dialogue natural, engaging, and realistic\n"
        "- Return ONLY {{\"user_turns\": [...]}} as valid JSON"
    ).format(
        bp_json=bp_json,
        logic_type=blueprint.get("meta", {}).get("logic_type", ""),
        time_gap=time_gap,
        seed_section=seed_section,
        n_thematic1=n_thematic1,
        n_thematic2=n_thematic2,
        total_turns=total_turns,
        slot_info=slot_info,
    )
    return prompt


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

VALID_FUNCTIONS_TYPE_A = {"thematic_connection", "rule_constraint", "condition_constraint", "trigger"}
VALID_FUNCTIONS_TYPE_B = {"thematic_connection", "rule_constraint", "trigger"}


# ═══════════════════════════════════════════════════════════════════════════
#  Validation — Type A (Forward_If-Then)
# ═══════════════════════════════════════════════════════════════════════════

def validate_dialogue_type_a(turns: list, blueprint: dict) -> tuple:
    """Validate a Type A dialogue (Forward_If-Then) against its blueprint.

    Structure: thematic{2-5} → rule_constraint → thematic{2-5} → condition_constraint → thematic{2-5} → trigger

    Returns (is_valid, error_list).
    """
    errors = []

    # ── 1. Basic structure ──────────────────────────────────────────────────
    if not isinstance(turns, list):
        errors.append("user_turns must be a list")
        return False, errors

    if len(turns) < 9:
        errors.append("Too few turns: {} (minimum 9 for Type A: 2+1+2+1+2+1)".format(len(turns)))
    if len(turns) > 18:
        errors.append("Too many turns: {} (maximum 18 for Type A: 5+1+5+1+5+1)".format(len(turns)))

    # ── 2. Per-turn field validation ────────────────────────────────────────
    for i, turn in enumerate(turns):
        for key in ("turn_id", "time", "function", "text"):
            if key not in turn:
                errors.append("Turn {}: missing field '{}'".format(i + 1, key))
        if "function" in turn and turn["function"] not in VALID_FUNCTIONS_TYPE_A:
            errors.append("Turn {}: invalid function '{}' (expected one of {})".format(
                i + 1, turn["function"], VALID_FUNCTIONS_TYPE_A))
    if errors:
        return False, errors

    # ── 3. Function sequence validation ─────────────────────────────────────
    functions = [t["function"] for t in turns]

    rule_positions = [i for i, f in enumerate(functions) if f == "rule_constraint"]
    cond_positions = [i for i, f in enumerate(functions) if f == "condition_constraint"]
    trigger_positions = [i for i, f in enumerate(functions) if f == "trigger"]

    if len(rule_positions) != 1:
        errors.append("Expected exactly 1 rule_constraint turn, got {}".format(len(rule_positions)))
    if len(cond_positions) != 1:
        errors.append("Expected exactly 1 condition_constraint turn, got {}".format(len(cond_positions)))
    if len(trigger_positions) != 1:
        errors.append("Expected exactly 1 trigger turn, got {}".format(len(trigger_positions)))
    if errors:
        return False, errors

    r_pos = rule_positions[0]
    c_pos = cond_positions[0]
    t_pos = trigger_positions[0]

    # Order: rule < condition < trigger
    if not (r_pos < c_pos < t_pos):
        errors.append("Ordering violated: rule_constraint(pos={}) must come before condition_constraint(pos={}) which must come before trigger(pos={})".format(r_pos, c_pos, t_pos))
        return False, errors

    # Trigger must be last
    if t_pos != len(turns) - 1:
        errors.append("Trigger must be the last turn (got pos={}, len={})".format(t_pos, len(turns)))

    # Validate section labels
    section1 = functions[:r_pos]
    section2 = functions[r_pos + 1:c_pos]
    section3 = functions[c_pos + 1:t_pos]

    if not all(f == "thematic_connection" for f in section1):
        errors.append("All turns before rule_constraint must be 'thematic_connection'")
    if not all(f == "thematic_connection" for f in section2):
        errors.append("All turns between rule_constraint and condition_constraint must be 'thematic_connection'")
    if not all(f == "thematic_connection" for f in section3):
        errors.append("All turns between condition_constraint and trigger must be 'thematic_connection'")

    # ── 4. Section counts ───────────────────────────────────────────────────
    n1 = len(section1)
    n2 = len(section2)
    n3 = len(section3)

    if n1 < 2:
        errors.append("Thematic Section 1 has {} turns (need >= 2)".format(n1))
    if n2 < 2:
        errors.append("Thematic Section 2 has {} turns (need >= 2)".format(n2))
    if n3 < 2:
        errors.append("Thematic Section 3 has {} turns (need >= 2)".format(n3))
    if n1 > 5:
        errors.append("Thematic Section 1 has {} turns (max 5)".format(n1))
    if n2 > 5:
        errors.append("Thematic Section 2 has {} turns (max 5)".format(n2))
    if n3 > 5:
        errors.append("Thematic Section 3 has {} turns (max 5)".format(n3))

    # ── 5. turn_id monotonicity ─────────────────────────────────────────────
    for i, turn in enumerate(turns):
        if turn.get("turn_id") != i + 1:
            errors.append("Turn {}: turn_id should be {}, got {}".format(i + 1, i + 1, turn.get("turn_id")))
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
                    idx_a + 1, turns[idx_a]["time"], idx_b + 1, turns[idx_b]["time"]))
            break

    # ── 7. Time gap checks ──────────────────────────────────────────────────
    # Gap 1: rule_constraint → condition_constraint
    time_gap_rc_str = blueprint["story"].get("time_gap_rule_to_condition", "")
    expected_gap_rc = parse_time_gap_minutes(time_gap_rc_str)

    r_dt = parse_iso(turns[r_pos]["time"])
    c_dt = parse_iso(turns[c_pos]["time"])

    if r_dt and c_dt and expected_gap_rc is not None:
        actual_gap_rc = (c_dt - r_dt).total_seconds() / 60
        tolerance = min(expected_gap_rc * 0.20, 10)
        if actual_gap_rc < expected_gap_rc - tolerance:
            errors.append(
                "Time gap (rule→condition) too short: actual={:.0f}min, expected~{:.0f}min (±{:.0f}min)".format(
                    actual_gap_rc, expected_gap_rc, tolerance))
        if actual_gap_rc > expected_gap_rc + tolerance:
            errors.append(
                "Time gap (rule→condition) too long: actual={:.0f}min, expected~{:.0f}min (±{:.0f}min)".format(
                    actual_gap_rc, expected_gap_rc, tolerance))

    # Gap 2: condition_constraint → trigger
    time_gap_ct_str = blueprint["story"].get("time_gap_condition_to_trigger", "")
    expected_gap_ct = parse_time_gap_minutes(time_gap_ct_str)

    t_dt = parse_iso(turns[t_pos]["time"])

    if c_dt and t_dt and expected_gap_ct is not None:
        actual_gap_ct = (t_dt - c_dt).total_seconds() / 60
        tolerance = min(expected_gap_ct * 0.20, 10)
        if actual_gap_ct < expected_gap_ct - tolerance:
            errors.append(
                "Time gap (condition→trigger) too short: actual={:.0f}min, expected~{:.0f}min (±{:.0f}min)".format(
                    actual_gap_ct, expected_gap_ct, tolerance))
        if actual_gap_ct > expected_gap_ct + tolerance:
            errors.append(
                "Time gap (condition→trigger) too long: actual={:.0f}min, expected~{:.0f}min (±{:.0f}min)".format(
                    actual_gap_ct, expected_gap_ct, tolerance))

    # ── 8. Slot Grounding Check (merged rule + condition text) ──────────────
    rule_text = turns[r_pos].get("text", "").lower()
    cond_text = turns[c_pos].get("text", "").lower()
    merged_text = rule_text + " " + cond_text

    slots = blueprint.get("gold_labels", {}).get("positive_gold", {}).get("slots", {})
    required_slots = slots.get("required_slots", [])
    slot_values = slots.get("slot_values", {})

    for slot_name in required_slots:
        aliases = slot_values.get(slot_name, [])
        if not aliases:
            continue
        found = any(alias.lower() in merged_text for alias in aliases)
        if not found:
            errors.append(
                "Slot '{}' not grounded: none of {} found in rule_constraint + condition_constraint text".format(
                    slot_name, aliases))

    # ── 9. Implicit language check (rule_constraint only) ───────────────────
    for kw in EXPLICIT_KEYWORDS:
        if kw in rule_text:
            errors.append("rule_constraint text contains explicit keyword: '{}'".format(kw))
            break

    # # ── 10. No night hours (01:00-05:00) ────────────────────────────────────
    # for i, turn in enumerate(turns):
    #     dt = parse_iso(turn["time"])
    #     if dt and 1 <= dt.hour < 5:
    #         errors.append("Turn {} time '{}' is between 01:00-05:00 AM (forbidden)".format(
    #             i + 1, turn["time"]))
    #         break

    return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
#  Validation — Type B (Mutual_Exclusion / Only-After)
# ═══════════════════════════════════════════════════════════════════════════

def validate_dialogue_type_b(turns: list, blueprint: dict) -> tuple:
    """Validate a Type B dialogue (Mutual_Exclusion / Only-After) against its blueprint.

    Structure: thematic{2-5} → rule_constraint → thematic{2-5} → trigger

    Returns (is_valid, error_list).
    """
    errors = []

    # ── 1. Basic structure ──────────────────────────────────────────────────
    if not isinstance(turns, list):
        errors.append("user_turns must be a list")
        return False, errors

    if len(turns) < 6:
        errors.append("Too few turns: {} (minimum 6 for Type B: 2+1+2+1)".format(len(turns)))
    if len(turns) > 12:
        errors.append("Too many turns: {} (maximum 12 for Type B: 5+1+5+1)".format(len(turns)))

    # ── 2. Per-turn field validation ────────────────────────────────────────
    for i, turn in enumerate(turns):
        for key in ("turn_id", "time", "function", "text"):
            if key not in turn:
                errors.append("Turn {}: missing field '{}'".format(i + 1, key))
        if "function" in turn and turn["function"] not in VALID_FUNCTIONS_TYPE_B:
            errors.append("Turn {}: invalid function '{}' (expected one of {})".format(
                i + 1, turn["function"], VALID_FUNCTIONS_TYPE_B))
    if errors:
        return False, errors

    # ── 3. Function sequence validation ─────────────────────────────────────
    functions = [t["function"] for t in turns]

    rule_positions = [i for i, f in enumerate(functions) if f == "rule_constraint"]
    trigger_positions = [i for i, f in enumerate(functions) if f == "trigger"]

    if len(rule_positions) != 1:
        errors.append("Expected exactly 1 rule_constraint turn, got {}".format(len(rule_positions)))
    if len(trigger_positions) != 1:
        errors.append("Expected exactly 1 trigger turn, got {}".format(len(trigger_positions)))
    if errors:
        return False, errors

    r_pos = rule_positions[0]
    t_pos = trigger_positions[0]

    if t_pos <= r_pos:
        errors.append("Trigger must come after rule_constraint")
        return False, errors

    # Trigger must be last
    if t_pos != len(turns) - 1:
        errors.append("Trigger must be the last turn (got pos={}, len={})".format(t_pos, len(turns)))

    # Validate section labels
    section1 = functions[:r_pos]
    section2 = functions[r_pos + 1:t_pos]

    if not all(f == "thematic_connection" for f in section1):
        errors.append("All turns before rule_constraint must be 'thematic_connection'")
    if not all(f == "thematic_connection" for f in section2):
        errors.append("All turns between rule_constraint and trigger must be 'thematic_connection'")

    # ── 4. Section counts ───────────────────────────────────────────────────
    n1 = len(section1)
    n2 = len(section2)

    if n1 < 2:
        errors.append("Thematic Section 1 has {} turns (need >= 2)".format(n1))
    if n2 < 2:
        errors.append("Thematic Section 2 has {} turns (need >= 2)".format(n2))
    if n1 > 5:
        errors.append("Thematic Section 1 has {} turns (max 5)".format(n1))
    if n2 > 5:
        errors.append("Thematic Section 2 has {} turns (max 5)".format(n2))

    # ── 5. turn_id monotonicity ─────────────────────────────────────────────
    for i, turn in enumerate(turns):
        if turn.get("turn_id") != i + 1:
            errors.append("Turn {}: turn_id should be {}, got {}".format(i + 1, i + 1, turn.get("turn_id")))
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
                    idx_a + 1, turns[idx_a]["time"], idx_b + 1, turns[idx_b]["time"]))
            break

    # ── 7. Time gap check ───────────────────────────────────────────────────
    time_gap_str = blueprint["story"].get("time_gap_between_constraint_and_trigger", "")
    expected_gap = parse_time_gap_minutes(time_gap_str)

    r_dt = parse_iso(turns[r_pos]["time"])
    t_dt = parse_iso(turns[t_pos]["time"])

    if r_dt and t_dt and expected_gap is not None:
        actual_gap = (t_dt - r_dt).total_seconds() / 60
        tolerance = min(expected_gap * 0.20, 10)
        if actual_gap < expected_gap - tolerance:
            errors.append(
                "Time gap (rule→trigger) too short: actual={:.0f}min, expected~{:.0f}min (±{:.0f}min)".format(
                    actual_gap, expected_gap, tolerance))
        if actual_gap > expected_gap + tolerance:
            errors.append(
                "Time gap (rule→trigger) too long: actual={:.0f}min, expected~{:.0f}min (±{:.0f}min)".format(
                    actual_gap, expected_gap, tolerance))

    # ── 8. Slot Grounding Check (rule_constraint text only) ─────────────────
    rule_text = turns[r_pos].get("text", "").lower()

    slots = blueprint.get("gold_labels", {}).get("positive_gold", {}).get("slots", {})
    required_slots = slots.get("required_slots", [])
    slot_values = slots.get("slot_values", {})

    for slot_name in required_slots:
        aliases = slot_values.get(slot_name, [])
        if not aliases:
            continue
        found = any(alias.lower() in rule_text for alias in aliases)
        if not found:
            errors.append(
                "Slot '{}' not grounded: none of {} found in rule_constraint text".format(
                    slot_name, aliases))

    # ── 9. Implicit language check ──────────────────────────────────────────
    for kw in EXPLICIT_KEYWORDS:
        if kw in rule_text:
            errors.append("rule_constraint text contains explicit keyword: '{}'".format(kw))
            break

    # # ── 10. No night hours (01:00-05:00) ────────────────────────────────────
    # for i, turn in enumerate(turns):
    #     dt = parse_iso(turn["time"])
    #     if dt and 1 <= dt.hour < 5:
    #         errors.append("Turn {} time '{}' is between 01:00-05:00 AM (forbidden)".format(
    #             i + 1, turn["time"]))
    #         break

    return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
#  Unified Validation Dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def validate_logical_adherence_dialogue(turns: list, blueprint: dict) -> tuple:
    """Dispatch validation based on logic_type."""
    logic_type = blueprint.get("meta", {}).get("logic_type", "")
    if logic_type == "Forward_If-Then":
        return validate_dialogue_type_a(turns, blueprint)
    else:
        return validate_dialogue_type_b(turns, blueprint)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage 2: Stretch Logical Adherence blueprints into user-only dialogues"
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
                    help="Also generate dialogues for the 6 human-authored seed blueprints")
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
        new_seeds = [s for s in LOGICAL_ADHERENCE_SEEDS if s["task_id"] not in bp_ids]
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

    seed_examples = {}  # logic_type -> (stripped_bp, dialogue)
    for lt in ["Forward_If-Then", "Mutual_Exclusion", "Only-After"]:
        sbp, sdlg = get_seed_example(lt)
        if sbp and sdlg:
            seed_examples[lt] = (sbp, sdlg)
            log.info("Seed example loaded for %s: %d turns", lt, len(sdlg))
        else:
            log.warning("No seed example found for logic_type=%s", lt)

    total = len(blueprints)
    done = 0
    failed = 0

    log.info(
        "Target: %d dialogues | Model=%s, T=%.2f, max_retries=%d",
        total, args.model, args.temperature, args.max_retries,
    )

    for bp_idx, blueprint in enumerate(tqdm(blueprints, desc="Stage 2 (Logical Adherence)")):
        if args.sleep:
            time_module.sleep(args.sleep)
        task_id = blueprint.get("task_id", "unknown_{}".format(bp_idx))
        logic_type = blueprint.get("meta", {}).get("logic_type", "")

        if task_id in all_dialogues:
            log.info("[%d/%d] %s — already done, skipping", bp_idx + 1, total, task_id)
            done += 1
            continue

        log.info("[%d/%d] Generating dialogue for %s (logic=%s) ...",
                 bp_idx + 1, total, task_id, logic_type)

        seed_bp, seed_dlg = seed_examples.get(logic_type, (None, None))

        if logic_type == "Forward_If-Then":
            n1, n2, n3 = sample_thematic_split_type_a(rng)
            system_prompt = SYSTEM_PROMPT_TYPE_A
            user_prompt = build_user_prompt_type_a(
                blueprint, seed_bp, seed_dlg, n1, n2, n3)
        else:
            n1, n2 = sample_thematic_split_type_b(rng)
            system_prompt = SYSTEM_PROMPT_TYPE_B
            user_prompt = build_user_prompt_type_b(
                blueprint, seed_bp, seed_dlg, n1, n2)

        base_user_prompt = user_prompt
        success = False

        for retry in range(args.max_retries):
            try:
                raw = call_llm(system_prompt, user_prompt)
                result = extract_json_object(raw)

                turns = result.get("user_turns", [])
                valid, errs = validate_logical_adherence_dialogue(turns, blueprint)

                if valid:
                    all_dialogues[task_id] = turns
                    done += 1
                    log.info("  OK (attempt %d/%d)  %d turns",
                             retry + 1, args.max_retries, len(turns))
                    success = True
                    save_json(all_dialogues, out_file)
                    break
                else:
                    log.warning("  Validation failed (attempt %d/%d):",
                                retry + 1, args.max_retries)
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
                log.warning("  JSON parse error (attempt %d/%d): %s",
                            retry + 1, args.max_retries, e)
            except Exception as e:
                log.warning("  Error (attempt %d/%d): %s",
                            retry + 1, args.max_retries, e)

        if not success:
            failed += 1
            log.error("  FAILED after %d retries — skipping %s",
                      args.max_retries, task_id)

    save_json(all_dialogues, out_file)

    log.info("=" * 70)
    log.info("Done!  %d / %d dialogues generated (%d failed).",
             done, total, failed)
    log.info("Saved -> %s", out_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 1 — Blueprint design for the Temporal Grounding dimension.

Generates scenario blueprints in which:
  - a user casually mentions a strict point-in-time commitment (Constraint);
  - a later utterance shows they are about to miss that exact moment (Trigger);
  - a capable agent should intervene without being asked.

Python pre-computes all timestamps; the LLM only writes the scenario text.

Loop structure:
  L1: 7 domains
    L2: 2 time-logic types (Absolute_Time, Relative_Time)
      L3: 2 trigger patterns (Time_Blind_Action, Time_Query)
        L4: N samples per cell (default = 2)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
DIM_DIR = PIPELINE_DIR.parent
GEN_ROOT = DIM_DIR.parent
sys.path.insert(0, str(GEN_ROOT))
sys.path.insert(0, str(DIM_DIR))

from shared.json_utils import extract_json_object, save_json
from shared.llm_client import complete
from shared.time_utils import format_iso
from seeds.blueprints import TEMPORAL_GROUNDING_SEEDS

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
#  Constants — 7 natural domains (no forced risk_type)
# ═══════════════════════════════════════════════════════════════════════════

V2_DOMAINS = [
    {
        "domain": "Social",
        "prompt_description": (
            "Ritual/Ceremony (仪式与约见): Scenarios where a social act MUST happen at "
            "an exact moment — a midnight birthday call pact, a New Year countdown toast, "
            "a precise anniversary-minute surprise. Doing it early is meaningless; doing it "
            "late is hurtful. The agent must wait until the exact moment to remind."
        ),
        "archetypes": "midnight birthday first-call, New Year countdown, anniversary minute, surprise party reveal",
    },
    {
        "domain": "Science_Lab",
        "prompt_description": (
            "Chemical/Biological Reaction (生化反应): Lab protocols that require action at "
            "a precise point — cell incubation ending, centrifuge spin completing, reagent "
            "reaching gel state, culture needing harvest at exactly N minutes. Acting early "
            "ruins the sample; acting late destroys it or causes safety hazards."
        ),
        "archetypes": "cell lysis buffer timer, centrifuge spin, gel polymerization, PCR cycle completion, culture harvest",
    },
    {
        "domain": "Home",
        "prompt_description": (
            "Precision Time Schedule (精确的时间窗口): Household events with a narrow exact "
            "window — a one-time-use temporary door code valid for exactly 5 minutes, "
            "a scheduled pest control fumigation start requiring windows sealed at the "
            "exact minute, a timed smart-home appliance cycle that must be stopped at "
            "the precise completion moment."
        ),
        "archetypes": "temporary access code window, appliance cycle completion, timed delivery pickup, pet care service arrival window",
    },
    {
        "domain": "Nature",
        "prompt_description": (
            "Transient Phenomena (瞬态自然现象): One-shot natural events that occur at a "
            "precise moment — solar eclipse totality, ISS overhead pass, peak meteor "
            "shower radiant, exact high tide for reef photography, rare bird migration "
            "flyover at dawn. The window is seconds to minutes; miss it and wait months/years.\n"
            "IMPORTANT: The constraint must clearly express the user's COMMITMENT to act — "
            "not just describe the event conditionally. The user should sound excited and "
            "determined, e.g.:\n"
            "  ✓ 'The aurora substorm breaks at exactly 9:50 PM tonight, and I will definitely "
            "not miss it!'\n"
            "  ✓ 'I want to see the Iridium flare — tonight's ephemeris says it peaks at "
            "exactly 11:45 PM here. I'm going to enjoy that flash!'\n"
            "  ✗ BAD: '...that's the moment to stand on the boardwalk if I want to catch it' "
            "(too conditional, sounds optional)"
        ),
        "archetypes": "solar eclipse totality, ISS pass, peak tide moment, meteor shower peak, rare astronomical alignment, Iridium flare",
    },
    {
        "domain": "Digital_Entertainment",
        "prompt_description": (
            "Sniping & Broadcasting (秒杀与首播): Time-locked digital events — limited-edition "
            "sneaker drop at exactly 10:00 AM, concert ticket release, live TV premiere, "
            "flash sale countdown, online auction final seconds. The system opens/closes "
            "at the precise second; early clicks fail, late clicks miss out."
        ),
        "archetypes": "sneaker drop, concert ticket release, TV premiere, flash sale opening, auction closing second",
    },
    {
        "domain": "Work",
        "prompt_description": (
            "Just-in-time Execution (卡点执行): Professional scenarios requiring action at "
            "an exact timestamp — global product launch GO button at synchronized time, "
            "stock market opening bell operation, scheduled meeting start where presenter "
            "must be ready, server migration switchover at maintenance window start."
        ),
        "archetypes": "global launch sync, market opening operation, presentation start, server switchover, coordinated release",
    },
    {
        "domain": "Health",
        "prompt_description": (
            "Medical Precision Timing (精准医疗时刻): Health scenarios where an action MUST "
            "happen at a precise moment — insulin injection timed to a meal schedule, "
            "medication dose that must be taken at an exact hour to maintain blood levels, "
            "a doctor's appointment with a strict check-in window, dialysis session start, "
            "or a scheduled medical procedure requiring fasting-period compliance at the "
            "exact minute. Acting early disrupts the protocol; acting late causes health risk."
        ),
        "archetypes": "insulin injection timing, scheduled medication dose, doctor appointment check-in, dialysis session start, pre-surgery fasting endpoint",
    },
]

TIME_LOGIC_TYPES = ["Absolute_Time", "Relative_Time"]
TRIGGER_PATTERNS = ["Time_Query", "Time_Blind_Action"]

# Keywords that must NOT appear in constraint text
EXPLICIT_KEYWORDS = [
    "remind me", "warn me", "alert me", "notify me",
    "don't let me forget", "do not let me forget",
    "take a note", "set a reminder", "set an alarm",
    "make sure i", "remember to", "todo",
    "don't forget", "do not forget", "keep in mind",
    "heads up later", "ping me", "flag this",
]

# Keywords that signal a flexible deadline (FORBIDDEN in V2)
FLEXIBLE_DEADLINE_KEYWORDS = [
    "by ", "before ", "no later than", "at the latest",
    "must be done by", "needs to be finished by",
    "has to be completed by", "should be done by",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Time Generation (Python pre-computes everything)
# ═══════════════════════════════════════════════════════════════════════════

def sample_duration_minutes(rng: random.Random, time_logic: str = "Absolute_Time") -> int:
    """Sample duration in minutes per the spec distribution.
    60% -> 5-8 hours (300-480 min)
    30% -> 2-5 hours (120-300 min)
    10% -> 30 min-2 hours (30-120 min)
    """
    r = rng.random()
    if r < 0.10:
        lo, hi = 30, 120
    elif r < 0.40:
        lo, hi = 120, 300
    else:
        lo, hi = 300, 480

    if time_logic == "Relative_Time":
        # Always round to multiple of 30 for natural language
        lo30 = (lo + 29) // 30 * 30
        hi30 = hi // 30 * 30
        return rng.randrange(lo30, hi30 + 1, 30)
    else:
        # Absolute_Time: always round to multiple of 5
        lo5 = (lo + 4) // 5 * 5
        hi5 = hi // 5 * 5
        return rng.randrange(lo5, hi5 + 1, 5)


def format_duration(minutes: int) -> str:
    """Convert minutes to a human-readable duration string."""
    if minutes >= 60 and minutes % 60 == 0:
        h = minutes // 60
        return f"{h} hour{'s' if h > 1 else ''}"
    elif minutes >= 60:
        h = minutes // 60
        m = minutes % 60
        return f"{h} hour{'s' if h > 1 else ''} and {m} minute{'s' if m > 1 else ''}"
    else:
        return f"{minutes} minutes"


def generate_random_base_time(rng: random.Random) -> datetime:
    """Generate a random base datetime in 2026 (Jan-Apr), hours 6-22, any minute."""
    month = rng.randint(1, 4)
    day = rng.randint(1, 28)
    hour = rng.randint(6, 22)
    minute = rng.randint(0, 59)
    return datetime(2026, month, day, hour, minute)


def generate_timeline_anchors(time_logic: str, rng: random.Random) -> dict:
    """Pre-compute all timeline anchors for one sample.

    V2 Schema:
      - constraint_talking_time
      - duration_time (only for Relative_Time, null for Absolute_Time)
      - exact_target_action_time (replaces deadline_time)
      - trigger_talking_time

    No more action_start_time.
    """
    constraint_talking_time = generate_random_base_time(rng)
    duration_min = sample_duration_minutes(rng, time_logic)
    trigger_offset_min = rng.randint(2, 10)
    exact_target_action_time = constraint_talking_time + timedelta(minutes=duration_min)

    # For Absolute_Time, round exact_target_action_time to the nearest 5-minute mark.
    # Because constraint_talking_time can have any minute (0-59), the sum may not be
    # a multiple of 5 even though duration_min is. We randomly round up or down.
    if time_logic == "Absolute_Time":
        remainder = exact_target_action_time.minute % 5
        if remainder != 0:
            if rng.random() < 0.5:
                # Round down
                exact_target_action_time = exact_target_action_time - timedelta(minutes=remainder)
            else:
                # Round up
                exact_target_action_time = exact_target_action_time + timedelta(minutes=(5 - remainder))

    # If target falls in 1:00 AM – 4:59 AM, shift constraint_talking_time earlier
    # so target lands in late evening (22:00–00:00) instead.
    if 1 <= exact_target_action_time.hour < 5:
        minutes_past_1am = (exact_target_action_time.hour - 1) * 60 + exact_target_action_time.minute
        # Shift back by (distance to 01:00) + (1-3 extra hours) → target lands 22:00–00:00
        shift_back = minutes_past_1am + rng.randint(60, 180)
        constraint_talking_time = constraint_talking_time - timedelta(minutes=shift_back)
        exact_target_action_time = constraint_talking_time + timedelta(minutes=duration_min)
        # Re-round after shift for Absolute_Time
        if time_logic == "Absolute_Time":
            remainder = exact_target_action_time.minute % 5
            if remainder != 0:
                if rng.random() < 0.5:
                    exact_target_action_time = exact_target_action_time - timedelta(minutes=remainder)
                else:
                    exact_target_action_time = exact_target_action_time + timedelta(minutes=(5 - remainder))

    trigger_talking_time = exact_target_action_time - timedelta(minutes=trigger_offset_min)

    if time_logic == "Relative_Time":
        return {
            "constraint_talking_time": format_iso(constraint_talking_time),
            "duration_time": format_duration(duration_min),
            "exact_target_action_time": format_iso(exact_target_action_time),
            "trigger_talking_time": format_iso(trigger_talking_time),
        }
    else:  # Absolute_Time
        return {
            "constraint_talking_time": format_iso(constraint_talking_time),
            "duration_time": None,
            "exact_target_action_time": format_iso(exact_target_action_time),
            "trigger_talking_time": format_iso(trigger_talking_time),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Seed Selection
# ═══════════════════════════════════════════════════════════════════════════

def get_seed_for_cell(time_logic: str, trigger_pattern: str) -> dict | None:
    """Return the seed example matching the given time_logic and trigger_pattern."""
    for s in TEMPORAL_GROUNDING_SEEDS:
        meta = s.get("meta", {})
        if (meta.get("time_logic_type") == time_logic and
                meta.get("trigger_pattern") == trigger_pattern):
            return s
    return None


def strip_seed_for_prompt(seed: dict) -> dict:
    """Remove task_id and meta from seed — LLM doesn't output those."""
    out = deepcopy(seed)
    out.pop("task_id", None)
    out.pop("meta", None)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  History Formatting
# ═══════════════════════════════════════════════════════════════════════════

def format_history_compact(items: list) -> str:
    """Build a concise history string showing only constraint + trigger."""
    if not items:
        return "(none yet)"
    parts = []
    for i, item in enumerate(items, 1):
        story = item.get("story", {})
        c = story.get("constraint", "")
        t = story.get("trigger", "")
        parts.append(f"  --- #{i} ---\n  Constraint: {c}\n  Trigger:    {t}")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt Construction
# ═══════════════════════════════════════════════════════════════════════════

TIME_LOGIC_DESCRIPTIONS = {
    "Absolute_Time": {
        "label": "Absolute_Time (绝对时间点型)",
        "description": (
            "The constraint directly states a specific future time point as the exact "
            "moment of action (e.g., 'the drop goes live at exactly 10:00 AM', "
            "'the call must be at midnight sharp'). The agent needs to compare the "
            "current time against this exact target.\n"
            "RULES:\n"
            "  - The constraint sentence MUST explicitly mention the exact_target_action_time.\n"
            "  - timeline_anchors.duration_time = null\n"
            "  - The constraint must NOT mention any duration — only the exact target time.\n"
            "  - The event CAN ONLY happen at that precise moment — NOT before, NOT after."
        ),
    },
    "Relative_Time": {
        "label": "Relative_Time (相对时间偏移型)",
        "description": (
            "The constraint describes a process/event that requires a specific duration "
            "from the current moment (e.g., 'the reaction needs exactly 45 minutes', "
            "'the server needs to be shut down after exactly 2 hours from now').\n"
            "RULES:\n"
            "  - The constraint sentence MUST mention the duration (e.g., '2 hours', "
            "'45 minutes') but must NEVER directly state the computed target time.\n"
            "  - timeline_anchors.duration_time = the duration string\n"
            "  - The process completes at exactly that duration — not before, not after."
        ),
    },
}

TRIGGER_PATTERN_DESCRIPTIONS = {
    "Time_Blind_Action": {
        "label": "Time_Blind_Action (时间盲目行为)",
        "description": (
            "The user does NOT ask what time it is. Instead, the user announces they are "
            "about to do another time-consuming activity (e.g., 'I plan to take a long bath', "
            "'I'm heading out for a run', 'I'll go shopping with my friends!'). Make the description diverse and fluent. The user is completely oblivious to the approaching "
            "exact moment.\n"
            "RULES:\n"
            "  - The trigger must describe a planned action or current intention — NOT a time query.\n"
            "  - The trigger action should be something that would cause the user to miss "
            "the exact target moment."
        ),
    },
    "Time_Query": {
        "label": "Time_Query (时间询问)",
        "description": (
            "The user has been deeply absorbed in some activity and casually asks 'What time "
            "is it now?'. This is used to test if agent can answer the time AND proactively warn about the "
            "approaching exact moment during evaluation.\n"
            "RULES:\n"
            "  - The trigger MUST contain a 'what time is it' style question. Use diverse "
            "paraphrasing: 'Tell me the time', 'What's the time now?', 'May I know the "
            "current time?', etc.\n"
            "  - {time_query_variant}\n"
            "  - The user's described activity (if any) must be semantically unrelated to "
            "the constraint. Make it diverse and natural."
        ),
    },
}


SYSTEM_PROMPT = """\
You are an expert benchmark-data creator for **TriggerBench**, a benchmark that evaluates \
whether LLM agents can *proactively* remind users of time-sensitive actions **without ever \
being explicitly asked**.

Your job: generate ONE new "blueprint" (scenario skeleton) as a single JSON object.

────────────────────────────────────────────────────
CORE CONCEPT — Strict Point-in-Time (V2)
────────────────────────────────────────────────────
• This is NOT about flexible deadlines ("do it before 8 PM").
• Every scenario MUST be a **strict point-in-time** event:
  - The action CANNOT be done early (physically impossible, socially meaningless, or system-locked).
  - The action CANNOT be done late (opportunity vanishes, sample ruined, event over).
  - The action MUST happen at the EXACT moment (or within a very narrow window of 1-5 minutes).

Examples of valid "strict point-in-time" scenarios:
  ✓ Midnight birthday call pact (calling at 11:50 PM is NOT "being first at midnight")
  ✓ Lab sample reaching gel state at exactly 45 minutes (removing early = liquid, late = hardened)
  ✓ Sneaker drop going live at 10:00 AM sharp (clicking at 9:55 does nothing)
  ✓ ISS overhead pass at 21:34 (looking at 21:20 sees nothing)

Examples of FORBIDDEN "flexible deadline" scenarios:
  ✗ "The oven must be turned off by 8 PM" (turning it off at 7 PM is fine → flexible)
  ✗ "Tiles must be rinsed before 9:25 AM" (rinsing at 9:00 is equally good → flexible)
  ✗ "Close windows before the fumes come" (closing early is perfectly fine → flexible)

• An *implicit constraint* is casually mentioned by the user as a factual statement. \
There is NEVER any explicit request to be reminded — no "remind me", "warn me", etc.
• The constraint must NOT explicitly state catastrophic risk in the user's own words. \
Leave the danger analysis for `hidden_vulnerability`.
• Later in the conversation, a *trigger* occurs: the user describes their current state \
or intention. The trigger is an ordinary, innocent utterance.
• A smart agent should detect the hidden time conflict and **proactively** intervene.

────────────────────────────────────────────────────
ABSOLUTE RULES (violations = rejection)
────────────────────────────────────────────────────
1. **STRICT POINT-IN-TIME ONLY (core rule)**
   The scenario MUST be "cannot do early, cannot do late". The constraint must describe \
an event/action that is only valid at the exact target moment.
   - FORBIDDEN: "by [Time]", "before [Time]", "no later than", "must be done by"
   - REQUIRED: "at exactly [Time]", "goes live at [Time]", "takes exactly [Duration]", \
"needs precisely [Duration]"

2. **IMPLICIT CONSTRAINTS — WITH CLEAR STAKES BUT NO EXPLICIT CATASTROPHE**
   The constraint must read like a natural, offhand factual remark. No "remind me", \
"warn me", "TODO", "don't forget", etc. Do NOT have the user explicitly state the \
catastrophic consequence (e.g., do NOT write "or my face will burn"). The consequence \
is analyzed in `hidden_vulnerability`.
   **The user must clearly express their INTENTION/COMMITMENT to act** — not just \
describe the event conditionally. The constraint should convey that the user WANTS to \
do something at that moment.

3. **ZERO SURFACE-MATCH between constraint and trigger**
   No shared domain-specific keywords between the trigger text and constraint text. \
The connection can only be discovered through temporal reasoning.

4. **distracting_task ISOLATION**
   The distracting_task must relate to the trigger context, NOT the constraint.

5. **SLOT GROUNDING RULE**
   - Define 1-2 slots in gold_labels.slots.required_slots (e.g., "event"/"action"). Most cases use 1 slot is fine.
   - Each slot must have a list of 2-4 diverse aliases in slot_values.
   - For EACH slot, at least ONE alias must appear as an exact substring (case-insensitive) \
in the constraint text (after removing the timestamp prefix).
   - Slot values capture what the agent should remind about (the key event/action).

6. **DIVERSITY**
   Vary backgrounds, settings, vulnerability types. Never repeat seeds or history.

────────────────────────────────────────────────────
OUTPUT
────────────────────────────────────────────────────
Return ONLY a single, valid JSON object. No markdown fences, no comments, no extra text.
You MUST make your output blueprint natural and realistic for the assigned timeline_anchors.\
"""


def build_user_prompt(
    domain_info: dict,
    time_logic: str,
    trigger_pattern: str,
    timeline_anchors: dict,
    seed_example: dict | None,
    history: list,
    time_query_variant: str = "time_with_state",
) -> str:
    """Assemble the user prompt for one generation call."""

    domain = domain_info["domain"]
    prompt_description = domain_info["prompt_description"]

    tl_info = TIME_LOGIC_DESCRIPTIONS[time_logic]

    # Handle Time_Query variant instruction (always time_with_state)
    tp_info_raw = TRIGGER_PATTERN_DESCRIPTIONS[trigger_pattern]
    if trigger_pattern == "Time_Query":
        variant_text = (
            "The trigger MUST ask for the time AND include a brief description of "
            "the user's current distracted state/activity (e.g., 'I've been coding "
            "all day, what time is it?')."
        )
        tp_desc = tp_info_raw["description"].replace("{time_query_variant}", variant_text)
        tp_label = tp_info_raw["label"]
    else:
        tp_desc = tp_info_raw["description"].replace("{time_query_variant}", "")
        tp_label = tp_info_raw["label"]

    if seed_example:
        stripped = strip_seed_for_prompt(seed_example)
        seed_text = json.dumps(stripped, indent=2, ensure_ascii=False)
    else:
        seed_text = "(No matching seed available — follow the schema description below.)"

    history_text = format_history_compact(history)
    anchors_json = json.dumps(timeline_anchors, indent=2, ensure_ascii=False)

    # Compute the implicit time gap for temporal-realism guidance
    ct = datetime.strptime(timeline_anchors["constraint_talking_time"], "%Y-%m-%dT%H:%M")
    tat = datetime.strptime(timeline_anchors["exact_target_action_time"], "%Y-%m-%dT%H:%M")
    ttt = datetime.strptime(timeline_anchors["trigger_talking_time"], "%Y-%m-%dT%H:%M")
    gap_minutes = int((tat - ct).total_seconds() / 60)
    gap_human = format_duration(gap_minutes)
    # Time left from trigger to target (the offset the user will experience)
    time_left_minutes = int((tat - ttt).total_seconds() / 60)

    # Pre-compute intent prefix string (applies to ALL trigger patterns)
    intent_prefix_str = f"[Time left to target action: {time_left_minutes} minutes] "
    intent_prefix_rule = (
        f"\nINTENT PREFIX RULE:\n"
        f"- The gold_labels.intent MUST begin with exactly:\n"
        f"  \"[Time left to target action: {time_left_minutes} minutes] \"\n"
        f"  followed by the proactive reminder text.\n"
        f"- This prefix is pre-computed (exact_target_action_time - trigger_talking_time = {time_left_minutes} minutes).\n"
        f"  Do NOT change the number.\n"
    )

    if time_logic == "Relative_Time":
        schema_note = (
            'For Relative_Time, the output timeline_anchors MUST include:\n'
            '  "duration_time": (the duration string, e.g., "2 hours"),\n'
            '  "exact_target_action_time": (pre-computed, keep as-is),\n'
            'The constraint text MUST mention the duration but NEVER the computed target time.\n'
            'The process/reaction completes at EXACTLY that duration — not before, not after.'
        )
        realism_note = (
            f"\n═══ TEMPORAL REALISM (CRITICAL) ═══\n"
            f"The duration from constraint time to target time is "
            f"**{gap_human}** ({gap_minutes} minutes).\n"
            f"Your scenario MUST involve a process that genuinely takes EXACTLY this long.\n"
            f"Think carefully: what real-world process naturally requires exactly ~{gap_human}?\n"
            f"Examples: chemical reactions, incubation periods, curing/drying times, "
            f"fermentation, slow-cooking, code validity windows, etc.\n"
        )
    else:
        schema_note = (
            'For Absolute_Time, the output timeline_anchors MUST include:\n'
            '  "duration_time": null,\n'
            '  "exact_target_action_time": (pre-computed, keep as-is),\n'
            'The constraint text MUST directly mention the exact_target_action_time but NEVER any duration.\n'
            'The event ONLY happens at that exact time — not before, not after.'
        )
        realism_note = (
            f"\n═══ TEMPORAL REALISM (CRITICAL) ═══\n"
            f"The time gap between constraint_talking_time and exact_target_action_time is "
            f"**{gap_human}** ({gap_minutes} minutes).\n"
            f"Your scenario MUST involve a situation where the user mentions the exact "
            f"target time naturally and the gap is physically realistic.\n"
            f"Think carefully: what real-world event is locked to this exact time point "
            f"and where the user would naturally mention it ~{gap_human} in advance?\n"
        )

    prompt = f"""\
Generate ONE new blueprint for:
  - domain = {domain}
  - time_logic_type = {time_logic}
  - trigger_pattern = {trigger_pattern}

═══ DOMAIN & SCENARIO DESCRIPTION ═══
{prompt_description}

Scenario archetypes for inspiration: {domain_info.get('archetypes', '')}

═══ TIME LOGIC: {tl_info['label']} ═══
{tl_info['description']}

═══ TRIGGER PATTERN: {tp_label} ═══
{tp_desc}

═══ PRE-COMPUTED TIMELINE ANCHORS (USE EXACTLY, DO NOT MODIFY) ═══
{anchors_json}

{schema_note}
{realism_note}
IMPORTANT TIME RULES:
- The constraint text must start with "[{timeline_anchors['constraint_talking_time']}] ..."
- The trigger text must start with "[{timeline_anchors['trigger_talking_time']}] ..."
- All times in timeline_anchors must be EXACTLY as provided above — do NOT change them.
- trigger_talking_time is {timeline_anchors['trigger_talking_time']} (already 2-10 min before target).

═══ STRICT POINT-IN-TIME REMINDER (CRITICAL) ═══
Your scenario MUST be "strict point-in-time":
  - The action CANNOT be done early (the button isn't active, the reaction hasn't completed, \
the ceremony hasn't started, the system hasn't opened).
  - The action CANNOT be done late (the window closes, the sample is ruined, the moment passes).
  - FORBIDDEN phrases in constraint: "by [Time]", "before [Time]", "no later than"
  - REQUIRED pattern: the event only becomes actionable AT the exact target moment.

═══ SEED EXAMPLE (for format & quality reference — do NOT duplicate content) ═══
{seed_text}

═══ ALREADY GENERATED FOR THIS DOMAIN (avoid similar scenarios) ═══
{history_text}

═══ REQUIRED OUTPUT JSON SCHEMA ═══
{{
  "story": {{
    "background": "<1-2 sentence scene setting>",
    "timeline_anchors": {{
      "constraint_talking_time": "{timeline_anchors['constraint_talking_time']}",
      "duration_time": {json.dumps(timeline_anchors['duration_time'])},
      "exact_target_action_time": "{timeline_anchors['exact_target_action_time']}",
      "trigger_talking_time": "{timeline_anchors['trigger_talking_time']}"
    }},
    "constraint": "[{timeline_anchors['constraint_talking_time']}] <implicit constraint sentence — strict point-in-time>",
    "trigger": "[{timeline_anchors['trigger_talking_time']}] <trigger sentence>",
    "hidden_vulnerability": "<why the trigger creates a time conflict; what happens if the exact moment is missed>",
    "variations": {{
      "distracting_task": "<a question related to the trigger context, NOT the constraint>"
    }}
  }},
  "gold_labels": {{
    "intent": "{intent_prefix_str}<what the ideal agent should proactively say at the trigger moment>",
    "slots": {{
      "required_slots": ["<slot_name_1>"],
      "slot_values": {{
        "<slot_name_1>": ["alias1", "alias2", "alias3"]
      }}
    }}
  }}
}}

SLOT GROUNDING RULES:
- 1-2 slots (e.g., "event"/"action"), each with 2-4 aliases. Most cases only need 1 slot.
- For EACH slot, at least ONE alias must literally appear in the constraint text \
(exact substring, case-insensitive, ignoring the [TIMESTAMP] prefix).
- Slot values should capture what the agent needs to remind about.
{intent_prefix_rule}

Final reminders:
- Constraint must be IMPLICIT — no "remind me / warn me / don't forget / TODO" etc.
- Constraint must NOT explicitly state catastrophic consequences in the user's words.
- Trigger must follow the {trigger_pattern} pattern described above.
- ZERO keyword overlap between constraint and trigger text.
- distracting_task must relate to the trigger context, NEVER to the constraint.
- Be creative and diverse — different from seeds & history.
- STRICT POINT-IN-TIME: The action can ONLY happen at the exact target moment.\
"""
    return prompt


# JSON extraction is provided by shared.json_utils.extract_json_object


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_time(t: str) -> str:
    """Normalize an ISO time string for comparison."""
    return t.strip().replace(" ", "")


def _time_appears_in_text(time_iso: str, text: str) -> bool:
    """Check if a time appears in text in any common format."""
    if time_iso in text:
        return True
    try:
        dt = datetime.strptime(time_iso.strip(), "%Y-%m-%dT%H:%M")
    except ValueError:
        return False

    h12 = dt.strftime("%-I:%M %p")       # "8:00 PM"
    h12_no_min = dt.strftime("%-I %p")    # "8 PM"
    h24 = dt.strftime("%H:%M")            # "20:00"
    h12_nospace = h12.replace(" ", "")     # "8:00PM"

    for fmt in [h12, h12_no_min, h24, h12_nospace,
                h12.lower(), h12_no_min.lower(),
                h12.replace(":00 ", " ")]:
        if fmt in text:
            return True
    return False


def validate_blueprint(
    item: dict,
    time_logic: str,
    trigger_pattern: str,
    expected_anchors: dict,
) -> tuple[bool, list[str]]:
    """Validate a generated blueprint (V2). Returns (is_valid, errors)."""
    errors = []

    # ── 1. Top-level keys ───────────────────────────────────────────────────
    if "story" not in item:
        errors.append("Missing top-level key: 'story'")
    if "gold_labels" not in item:
        errors.append("Missing top-level key: 'gold_labels'")
    if errors:
        return False, errors

    story = item["story"]
    gl = item["gold_labels"]

    # ── 2. Story fields ─────────────────────────────────────────────────────
    for key in ("background", "timeline_anchors", "constraint", "trigger",
                "hidden_vulnerability", "variations"):
        if key not in story:
            errors.append(f"Missing story.{key}")
    if errors:
        return False, errors

    # ── 3. timeline_anchors match pre-assigned (V2 schema) ──────────────────
    ta = story["timeline_anchors"]
    for key in ("constraint_talking_time", "exact_target_action_time", "trigger_talking_time"):
        if key not in ta:
            errors.append(f"Missing timeline_anchors.{key}")
        elif _normalize_time(str(ta[key])) != _normalize_time(str(expected_anchors[key])):
            errors.append(
                f"timeline_anchors.{key} mismatch: "
                f"got '{ta[key]}', expected '{expected_anchors[key]}'"
            )

    if time_logic == "Relative_Time":
        if "duration_time" not in ta or ta["duration_time"] is None:
            errors.append("Relative_Time requires non-null duration_time")
        elif ta["duration_time"].strip().lower() != expected_anchors["duration_time"].strip().lower():
            errors.append(
                f"timeline_anchors.duration_time mismatch: "
                f"got '{ta['duration_time']}', expected '{expected_anchors['duration_time']}'"
            )
    else:  # Absolute_Time
        if ta.get("duration_time") is not None:
            errors.append("Absolute_Time requires duration_time = null")

    # ── 4. Constraint timestamp prefix ──────────────────────────────────────
    constraint_text = story.get("constraint", "")
    expected_prefix = f"[{expected_anchors['constraint_talking_time']}]"
    if not constraint_text.startswith(expected_prefix):
        errors.append(
            f"Constraint must start with '{expected_prefix}', "
            f"got: '{constraint_text[:60]}...'"
        )

    # ── 5. Trigger timestamp prefix ─────────────────────────────────────────
    trigger_text = story.get("trigger", "")
    expected_trigger_prefix = f"[{expected_anchors['trigger_talking_time']}]"
    if not trigger_text.startswith(expected_trigger_prefix):
        errors.append(
            f"Trigger must start with '{expected_trigger_prefix}', "
            f"got: '{trigger_text[:60]}...'"
        )

    # ── 6. Time logic specific checks ──────────────────────────────────────
    if time_logic == "Relative_Time":
        duration_str = expected_anchors["duration_time"]
        if duration_str.lower() not in constraint_text.lower():
            errors.append(
                f"Relative_Time: duration '{duration_str}' must appear in constraint text"
            )
        target_str = expected_anchors["exact_target_action_time"]
        if _time_appears_in_text(target_str, constraint_text):
            errors.append(
                "Relative_Time: computed exact_target_action_time must NOT appear in constraint "
                "(only the duration should be mentioned)"
            )
    else:  # Absolute_Time
        target_str = expected_anchors["exact_target_action_time"]
        if not _time_appears_in_text(target_str, constraint_text):
            errors.append(
                f"Absolute_Time: exact_target_action_time (derived from '{target_str}') "
                f"must appear in constraint text in some recognizable format"
            )

    # ── 7. Trigger pattern check ────────────────────────────────────────────
    trigger_lower = trigger_text.lower()
    if trigger_pattern == "Time_Query":
        time_query_phrases = [
            "what time", "what's the time", "do you know the time",
            "what hour", "how late is it", "check the time",
            "tell me the time", "know what time", "current time",
            "time is it", "may i know the time",
        ]
        if not any(p in trigger_lower for p in time_query_phrases):
            errors.append(
                "Time_Query pattern requires the trigger to contain a time-asking phrase"
            )
    elif trigger_pattern == "Time_Blind_Action":
        time_query_phrases = ["what time", "what's the time", "do you know the time",
                              "tell me the time", "time is it"]
        if any(p in trigger_lower for p in time_query_phrases):
            errors.append(
                "Time_Blind_Action pattern: trigger should NOT contain a time query"
            )

    # ── 8. Implicit constraint check ────────────────────────────────────────
    c_lower = constraint_text.lower()
    for kw in EXPLICIT_KEYWORDS:
        if kw in c_lower:
            errors.append(f"Constraint contains explicit keyword '{kw}'")
            break

    # ── 9. Flexible deadline check (V2 CRITICAL) ───────────────────────────
    # Remove the timestamp prefix before checking for forbidden phrases
    constraint_body = constraint_text
    if constraint_body.startswith("["):
        bracket_end = constraint_body.find("]")
        if bracket_end != -1:
            constraint_body = constraint_body[bracket_end + 1:].strip()
    c_body_lower = constraint_body.lower()
    for kw in FLEXIBLE_DEADLINE_KEYWORDS:
        if kw in c_body_lower:
            errors.append(
                f"V2 VIOLATION: Constraint contains flexible-deadline phrase '{kw}'. "
                f"Must use strict point-in-time language (at exactly, goes live at, etc.)"
            )
            break

    # ── 10. gold_labels validation ──────────────────────────────────────────
    if "intent" not in gl:
        errors.append("Missing gold_labels.intent")
    if "slots" not in gl:
        errors.append("Missing gold_labels.slots")
    else:
        slots = gl["slots"]
        if "required_slots" not in slots:
            errors.append("Missing gold_labels.slots.required_slots")
        if "slot_values" not in slots:
            errors.append("Missing gold_labels.slots.slot_values")
        elif "required_slots" in slots:
            req_slots = slots["required_slots"]
            if not (1 <= len(req_slots) <= 2):
                errors.append(
                    f"required_slots must have 1-2 items, got {len(req_slots)}"
                )
            for s in req_slots:
                if s not in slots["slot_values"]:
                    errors.append(f"required_slot '{s}' missing from slot_values")
                elif len(slots["slot_values"][s]) < 2:
                    errors.append(f"slot_values['{s}'] needs >= 2 aliases")

            # Slot grounding: each slot must have at least one alias in constraint
            constraint_body_lower = constraint_text.lower()
            # Strip timestamp prefix for matching
            if constraint_body_lower.startswith("["):
                bracket_end = constraint_body_lower.find("]")
                if bracket_end != -1:
                    constraint_body_lower = constraint_body_lower[bracket_end + 1:]
            for s in req_slots:
                if s in slots.get("slot_values", {}):
                    aliases = slots["slot_values"][s]
                    if not any(alias.lower() in constraint_body_lower for alias in aliases):
                        errors.append(
                            f"Slot grounding failed for '{s}': none of the aliases "
                            f"{aliases} appear in the constraint text (case-insensitive)"
                        )

    # ── 11. variations check ────────────────────────────────────────────────
    variations = story.get("variations", {})
    if "distracting_task" not in variations:
        errors.append("Missing story.variations.distracting_task")

    # ── 12. Intent prefix check ─────────────────────────────────────────────
    # Intent must start with "[Time left to target action: X minutes]"
    intent_text = gl.get("intent", "")
    try:
        _tat = datetime.strptime(expected_anchors["exact_target_action_time"], "%Y-%m-%dT%H:%M")
        _ttt = datetime.strptime(expected_anchors["trigger_talking_time"], "%Y-%m-%dT%H:%M")
        expected_time_left = int((_tat - _ttt).total_seconds() / 60)
        expected_prefix = f"[Time left to target action: {expected_time_left} minutes]"
        if not intent_text.startswith(expected_prefix):
            errors.append(
                f"Intent must start with '{expected_prefix}', "
                f"got: '{intent_text[:80]}...'"
            )
    except (ValueError, KeyError):
        pass  # Skip if anchors can't be parsed

    # ── 13. Constraint-trigger keyword overlap check ────────────────────────
    # Enforce ABSOLUTE RULE 3: Zero Surface-Match between constraint and trigger
    if constraint_text and trigger_text:
        # Strip timestamp prefixes
        c_body = constraint_text
        if c_body.startswith("["):
            bracket_end = c_body.find("]")
            if bracket_end != -1:
                c_body = c_body[bracket_end + 1:].strip()
        t_body = trigger_text
        if t_body.startswith("["):
            bracket_end = t_body.find("]")
            if bracket_end != -1:
                t_body = t_body[bracket_end + 1:].strip()

        # Extract significant words from constraint (>= 4 chars, not common stop words)
        stop_words = {
            "just", "about", "that", "this", "with", "from", "have", "been",
            "will", "would", "could", "should", "going", "getting", "doing",
            "being", "into", "onto", "they", "them", "their", "what", "when",
            "where", "which", "there", "here", "some", "other", "more", "most",
            "very", "really", "then", "than", "also", "only", "still", "already",
            "before", "after", "while", "during", "through", "between",
            "started", "finished", "ready", "need", "start", "time", "know",
            "like", "come", "back", "want", "take", "make", "right",
            "up", "down", "over", "under", "again", "first", "last", "next",
            "every", "any", "and", "but", "or", "if", "because", "as", "so", "although",
            "can", "cannot", "must", "should", "could", "would", "may", "might",
            "do", "don't", "does", "doesn't", "did", "didn't", "has", "hasn't",
            "have", "haven't", "had", "hadn't", "is", "isn't", "are", "aren't", "was", "wasn't", "were",
            "weren't", "be", "been", "being",
        }
        constraint_words = set()
        for word in re.findall(r"[a-zA-Z]{4,}", c_body.lower()):
            if word not in stop_words:
                constraint_words.add(word)

        t_body_lower = t_body.lower()
        overlap = [w for w in constraint_words if w in t_body_lower]
        if overlap:
            errors.append(
                f"Constraint-trigger keyword overlap detected: {overlap[:5]}. "
                f"Rule 3 requires ZERO surface-match between constraint and trigger."
            )

    return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Generate Temporal Grounding blueprints")
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--n-per-cell", type=int, default=2,
                    help="Samples per (domain x time_logic x trigger_pattern) cell")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to existing output JSON to resume from")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    def call_llm(sys_p, usr_p):
        return complete(sys_p, usr_p, model=args.model, temperature=args.temperature)

    out_dir = Path(args.output_dir) if args.output_dir else (DIM_DIR / "outputs" / "stage1_blueprints")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"blueprints_temporal_grounding_{args.model}.json"

    # ── resume support ──────────────────────────────────────────────────────
    all_items: list[dict] = []
    per_domain_history: dict[str, list] = {
        d["domain"]: [] for d in V2_DOMAINS
    }

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as f:
                all_items = json.load(f)
            for item in all_items:
                meta = item.get("meta", {})
                dom = meta.get("domain", "")
                if dom in per_domain_history:
                    per_domain_history[dom].append(item)
            log.info("Resumed %d items from %s", len(all_items), resume_path)

    global_idx = len(all_items)

    # ── compute totals ──────────────────────────────────────────────────────
    total_cells = len(V2_DOMAINS) * len(TIME_LOGIC_TYPES) * len(TRIGGER_PATTERNS)
    total_target = total_cells * args.n_per_cell
    log.info("Target: %d domains x %d time_logic x %d trigger_pattern x %d per_cell = %d blueprints",
             len(V2_DOMAINS), len(TIME_LOGIC_TYPES),
             len(TRIGGER_PATTERNS), args.n_per_cell, total_target)
    log.info("Model=%s, T=%.2f, max_retries=%d", args.model, args.temperature, args.max_retries)

    # ── build per-cell count from resumed items ───────────────────────────
    per_cell_count: dict[str, int] = {}
    for item in all_items:
        meta = item.get("meta", {})
        _d = meta.get("domain", "")
        _tl = meta.get("time_logic_type", "")
        _tp = meta.get("trigger_pattern", "")
        _key = f"{_d}__{_tl}__{_tp}"
        per_cell_count[_key] = per_cell_count.get(_key, 0) + 1

    # ═══════════════════════════════════════════════════════════════════════
    #  Generation loop:  L1 domain → L2 time_logic → L3 trigger_pattern → L4 N
    # ═══════════════════════════════════════════════════════════════════════
    for domain_idx, domain_info in enumerate(V2_DOMAINS):
        domain = domain_info["domain"]
        history = per_domain_history[domain]

        log.info("=" * 70)
        log.info("L1 [%d/%d]: %s  (history=%d)",
                 domain_idx + 1, len(V2_DOMAINS), domain, len(history))

        for time_logic in TIME_LOGIC_TYPES:
            for trigger_pattern in TRIGGER_PATTERNS:
                cell_tag = f"{domain}__{time_logic}__{trigger_pattern}"
                seed_example = get_seed_for_cell(time_logic, trigger_pattern)

                existing_count = per_cell_count.get(cell_tag, 0)
                remaining = args.n_per_cell - existing_count

                log.info("-" * 50)
                log.info("  Cell: %s x %s  (seed=%s, existing=%d/%d)",
                         time_logic, trigger_pattern,
                         "yes" if seed_example else "no",
                         existing_count, args.n_per_cell)

                if remaining <= 0:
                    log.info("    ⏭ Already complete — skipping.")
                    continue

                for n_idx in range(remaining):
                    sample_tag = f"{cell_tag}__#{existing_count + n_idx + 1}"
                    log.info("    Generating %s ...", sample_tag)

                    anchors = generate_timeline_anchors(time_logic, rng)

                    # Time_Query always uses time_with_state
                    time_query_variant = "time_with_state"

                    user_prompt = build_user_prompt(
                        domain_info, time_logic, trigger_pattern,
                        anchors, seed_example, history,
                        time_query_variant=time_query_variant,
                    )
                    success = False

                    for retry in range(args.max_retries):
                        try:
                            raw = call_llm(SYSTEM_PROMPT, user_prompt)
                            item = extract_json_object(raw)
                            valid, errs = validate_blueprint(
                                item, time_logic, trigger_pattern, anchors,
                            )

                            if valid:
                                global_idx += 1
                                item["task_id"] = f"temporal_grounding_{global_idx:03d}"
                                item["meta"] = {
                                    "domain": domain,
                                    "trigger_type": "time-based",
                                    "time_logic_type": time_logic,
                                    "trigger_pattern": trigger_pattern,
                                }

                                all_items.append(item)
                                history.append(item)
                                log.info("    ✓ OK (attempt %d/%d)  task_id=%s",
                                         retry + 1, args.max_retries,
                                         item["task_id"])

                                success = True
                                save_json(all_items, out_file)
                                break

                            else:
                                log.warning("    ✗ Validation failed (attempt %d/%d):",
                                            retry + 1, args.max_retries)
                                for e in errs:
                                    log.warning("      - %s", e)

                                if retry < args.max_retries - 1:
                                    err_block = "\n".join(f"- {e}" for e in errs)
                                    user_prompt = (
                                        build_user_prompt(
                                            domain_info, time_logic, trigger_pattern,
                                            anchors, seed_example, history,
                                            time_query_variant=time_query_variant,
                                        )
                                        + f"\n\n=== YOUR PREVIOUS ATTEMPT FAILED VALIDATION ===\n"
                                          f"Errors:\n{err_block}\n"
                                          f"Fix ALL issues above and generate a NEW valid blueprint."
                                    )

                        except json.JSONDecodeError as e:
                            log.warning("    ✗ JSON parse error (attempt %d/%d): %s",
                                        retry + 1, args.max_retries, e)
                        except Exception as e:
                            log.warning("    ✗ Error (attempt %d/%d): %s",
                                        retry + 1, args.max_retries, e)

                    if not success:
                        log.error("    ✗ FAILED after %d retries — skipping %s",
                                  args.max_retries, sample_tag)

    save_json(all_items, out_file)

    log.info("=" * 70)
    log.info("Done!  %d / %d blueprints generated.", len(all_items), total_target)
    log.info("Saved -> %s", out_file)

    stats: dict[str, int] = {}
    for item in all_items:
        meta = item.get("meta", {})
        key = (f"{meta.get('domain', '?')}__"
               f"{meta.get('time_logic_type', '?')}__{meta.get('trigger_pattern', '?')}")
        stats[key] = stats.get(key, 0) + 1
    log.info("Per-cell counts:")
    for key in sorted(stats):
        log.info("  %-60s  %d", key, stats[key])


if __name__ == "__main__":
    main()

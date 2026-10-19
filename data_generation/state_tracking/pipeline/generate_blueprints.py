#!/usr/bin/env python3
"""Stage 1 — Blueprint design for the State-Tracking dimension.

Generates scenario blueprints in which:
  - a user casually mentions a temporary physical/logical state change (Constraint);
  - a later action creates a hidden conflict (Trigger);
  - a capable agent should intervene without being asked.

Loop structure:
  L1: 9 domain × risk-type combinations
    L2: per-combo count defined in COMBO_ITEM_COUNT
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from copy import deepcopy
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
DIM_DIR = PIPELINE_DIR.parent
GEN_ROOT = DIM_DIR.parent
sys.path.insert(0, str(GEN_ROOT))
sys.path.insert(0, str(DIM_DIR))

from shared.json_utils import extract_json_object, save_json
from shared.llm_client import complete
from shared.time_utils import parse_time_gap_minutes
from seeds.blueprints import STATE_TRACKING_SEEDS

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
#  Constants — 9 domain × risk-type combinations
# ═══════════════════════════════════════════════════════════════════════════

STATE_TRIGGER_COMBOS = [

    # ──────────── Transit & Travel ────────────────────────────────────────
    {
        "domain": "Transit/Travel",
        "risk_type": "Object_Loss",
        "description": (
            "Scenario archetype: Spatial separation. The user places a large or valuable item in a "
            "non-body compartment (trunk, overhead rack, hotel safe) while using public or hired transport or in hotel rooms or travel places.\n"
            "Trigger mechanism: Upon arriving at the destination, the user focuses on the transition, "
            "announcing payment or exiting the vehicle while only accounting for carry-on items.\n"
            "Examples:\n"
            "1. Cello placed in the Uber trunk because it won't fit on the seat → (Trigger) Driver stops "
            "at the venue; user says they're scanning the QR code to pay and hopping out.\n"
            "2. Passport and cash locked in the hotel room safe → (Trigger) User hands the room key "
            "back to the front desk to complete check-out."
        )
    },
    {
        "domain": "Transit/Travel",
        "risk_type": "Task_Failure",
        "description": (
            "Scenario archetype: Device setting changes during transit. The user "
            "temporarily alters a critical device setting to adapt to the transport "
            "environment (no signal, battery saving, noise isolation).\n"
            "Trigger mechanism: The user exits the vehicle and enters a new situation "
            "that critically requires the device to function normally.\n"
            "Examples:\n"
            "1. Put phone on Airplane Mode in the subway to save battery (no signal "
            "anyway) → (Trigger) Walks out of the station, says they're waiting for "
            "the hospital to call back with urgent test results.\n"
            "2. Wearing strong noise-canceling headphones watching a video in the "
            "departure lounge → (Trigger) Mentions their flight boards in 10 minutes "
            "and plans to close their eyes and rest."
        ),
    },

    # ──────────── Home & Domestic ─────────────────────────────────────────
    {
        "domain": "Home/Domestic",
        "risk_type": "Object_Damage_or_Forgetting",
        "description": (
            "Scenario archetype: Unusual or counter-intuitive storage. The user temporarily places an "
            "item in a bizarre or dangerous location due to child-proofing, space constraints, or being "
            "flustered.\n"
            "Trigger mechanism: The user announces starting a routine household chore or leaving the "
            "house, which directly conflicts with the item's hidden location.\n"
            "Examples:\n"
            "1. Hiding a Nintendo Switch in the washing machine drum to keep it from a nephew → "
            "(Trigger) User gathers dirty clothes and prepares to start the wash cycle.\n"
            "2. Placing car keys on a fridge shelf while struggling with heavy grocery bags → "
            "(Trigger) User puts on their shoes and heads to the garage to pick up their child."
        )
    },
    {
        "domain": "Home/Domestic",
        "risk_type": "Safety_Hazard",
        "description": (
            "Scenario archetype: Persistent environmental state. The user alters a household appliance "
            "or environment for a specific maintenance or cooking task (e.g., simmering, defrosting, airing).\n"
            "Trigger mechanism: After being distracted, the user announces an action that involves leaving "
            "the environment for a long period (e.g., napping, bathing, leaving) without restoring safety.\n"
            "Examples:\n"
            "1. Setting the stove to the lowest heat to slowly melt chocolate → (Trigger) User feels "
            "exhausted and decides to go upstairs for a 45-minute bubble bath.\n"
            "2. Propping the front door open with a heavy box to air out the hallway → (Trigger) User "
            "receives a phone call and mentions they are heading to the bedroom to sleep."
        )
    },

    # ──────────── Work & Professional ─────────────────────────────────────
    {
        "domain": "Work/Professional",
        "risk_type": "Privacy_Leak_or_Security",
        "description": (
            "Scenario archetype: Digital state exposure in a professional setting. The "
            "user enables screen sharing during a meeting or grants elevated access "
            "privileges to an external account for testing/debugging.\n"
            "Trigger mechanism: The work task ends and the user immediately switches to "
            "a personal task or signs off, forgetting to revoke the exposed permission "
            "or sharing state.\n"
            "Examples:\n"
            "1. Shared entire desktop on Zoom to show a client the slide deck → "
            "(Trigger) Presentation wraps up; user is about to open their personal "
            "medical test results on the same screen.\n"
        ),
    },
    {
        "domain": "Work/Professional",
        "risk_type": "Object_Forgetting",
        "description": (
            "Scenario archetype: Accessories left at temporary workstations. The user "
            "plugs a small but critical accessory into a wall outlet or shared device "
            "in a conference room, co-working space, or someone else's desk.\n"
            "Trigger mechanism: The meeting or interaction ends; the user closes their "
            "laptop and announces they are leaving the room.\n"
            "Examples:\n"
            "1. Laptop running low—plugged the charger into the wall behind the "
            "conference room projector → (Trigger) Meeting ends; user carries their "
            "laptop back to their own desk.\n"
            "2. Inserted a USB drive with important project files into the podium "
            "computer for a presentation → (Trigger) Talk goes great; user walks off "
            "stage to shake hands with sponsors."
        ),
    },
    {
        "domain": "Work/Professional",
        "risk_type": "Task_Failure",
        "description": (
            "Scenario archetype: Interrupted digital workflow. The user pauses a critical "
            "background sync, auto-save, or system process to free up bandwidth or "
            "prevent interference during a task.\n"
            "Trigger mechanism: The user disconnects from the network or physically "
            "leaves the office environment, stranding un-synced work.\n"
            "Examples:\n"
            "1. Paused local code folder's cloud sync because the video call was "
            "lagging → (Trigger) Call ends; user closes their laptop and heads to the "
            "airport to edit code on a Wi-Fi-less flight.\n"
        ),
    },

    # ──────────── Maintenance & DIY ───────────────────────────────────────
    {
        "domain": "Maintenance/DIY",
        "risk_type": "Property_Damage",
        "description": (
            "Scenario archetype: Forced physical system intervention. The user physically "
            "blocks equipment from closing or operating normally—for defrosting, cleaning, "
            "or repair—by propping a door, disconnecting power, or leaving a valve open.\n"
            "Trigger mechanism: Maintenance is interrupted or completed; the user announces "
            "leaving the house or starting a task that conflicts with the open/disabled "
            "system.\n"
            "Examples:\n"
            "1. Unplugged the chest freezer and propped its door open with a chair to "
            "defrost it → (Trigger) Friend calls about a pickup game; user grabs their "
            "sports bag and heads out for the whole afternoon.\n"
            "2. Opened the garden faucet and attached a hose (no shutoff at the other "
            "end) to wash the car → (Trigger) Boss calls demanding an urgent return to "
            "the office; user drops the hose and runs to the garage."
        ),
    },
    {
        "domain": "Maintenance/DIY",
        "risk_type": "Security_Breach",
        "description": (
            "Scenario archetype: Temporary physical security downgrade. The user disables "
            "a lock, removes a child safety gate, or turns off a perimeter alarm to let "
            "a repair worker in, move large furniture, or reduce noise.\n"
            "Trigger mechanism: The task is done and the environment no longer needs to be "
            "open, but the user jumps straight into rest or caregiving without restoring "
            "security.\n"
            "Examples:\n"
            "1. Disabled the doorbell and front door auto-lock so as not to wake the "
            "sleeping baby → (Trigger) Exhausted after a long day, the user turns off "
            "the lights and goes to bed.\n"
            "2. Removed the child safety gate at the top of the stairs to fit a new "
            "sofa through → (Trigger) Sofa's in place; user heads to the kitchen to "
            "make a bottle for their 2-year-old toddler."
        ),
    }
]

# ═══════════════════════════════════════════════════════════════════════════
#  Per-Combo Item Count  (how many blueprints to generate for each combo)
# ═══════════════════════════════════════════════════════════════════════════

COMBO_ITEM_COUNT: dict[str, int] = {
    "Transit/Travel__Object_Loss":                  15,
    "Transit/Travel__Task_Failure":                  10,
    "Home/Domestic__Object_Damage_or_Forgetting":    15,
    "Home/Domestic__Safety_Hazard":                  10,
    "Work/Professional__Privacy_Leak_or_Security":   10,
    "Work/Professional__Object_Forgetting":          15,
    "Work/Professional__Task_Failure":               10,
    "Maintenance/DIY__Property_Damage":              10,
    "Maintenance/DIY__Security_Breach":              10,
}

# Keywords that must NOT appear in constraint text (explicit reminder language)
EXPLICIT_KEYWORDS = [
    "remind me", "warn me", "alert me", "notify me",
    "don't let me forget", "do not let me forget",
    "take a note", "set a reminder", "set an alarm",
    "make sure i", "remember to", "todo",
    "don't forget", "do not forget", "keep in mind",
    "heads up later", "ping me", "flag this",
    "note to self",
]

# Keywords that are too explicit about catastrophic consequences in constraint
CATASTROPHE_KEYWORDS = [
    "will break", "will destroy", "will be ruined", "will be damaged",
    "will burn", "will explode", "will die", "will get stolen",
    "dangerous if", "deadly if", "fatal if",
    "or else", "otherwise i'll",
]


MIN_TIME_GAP_MINUTES = 30


# ═══════════════════════════════════════════════════════════════════════════
#  Seed Selection
# ═══════════════════════════════════════════════════════════════════════════

def select_seed_examples_for_combo(combo_info: dict) -> list[dict]:
    """Return seed examples whose domain and risk_type match the given combo.

    Falls back to same-domain seeds if no exact match, then to all seeds.
    """
    domain = combo_info["domain"]
    risk_type = combo_info["risk_type"]

    # Exact match: same domain AND risk_type
    exact = [
        s for s in STATE_TRACKING_SEEDS
        if s.get("meta", {}).get("domain", "") == domain
        and s.get("meta", {}).get("risk_type", "") == risk_type
    ]
    if exact:
        return exact

    # Fallback: same domain only
    domain_only = [
        s for s in STATE_TRACKING_SEEDS
        if s.get("meta", {}).get("domain", "") == domain
    ]
    if domain_only:
        return domain_only[:2]

    return STATE_TRACKING_SEEDS[:2]


def strip_seed_for_prompt(seed: dict) -> dict:
    """Remove task_id and meta from seed, normalize 'variation' → 'variations'."""
    out = deepcopy(seed)
    out.pop("task_id", None)
    out.pop("meta", None)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  History Formatting
# ═══════════════════════════════════════════════════════════════════════════

def format_history_compact(items: list) -> str:
    """Build a concise history string showing constraint + trigger for dedup."""
    if not items:
        return "(none yet)"
    parts = []
    for i, item in enumerate(items, 1):
        story = item.get("story", {})
        c = story.get("constraint", "")
        t = story.get("trigger", "")
        parts.append(f"  --- #{i} ---\n  Constraint: {c}\n  Trigger:    {t}")
    return "\n".join(parts)


SYSTEM_PROMPT = """\
You are an expert benchmark-data creator for **TriggerBench**, a benchmark that evaluates \
whether LLM agents can *proactively* remind users about hidden risks **without ever being \
explicitly asked**.

Your job: generate ONE new state-change "blueprint" (scenario skeleton) as a single JSON object.

────────────────────────────────────────────────────
CORE CONCEPT — Implicit Prospective Memory
────────────────────────────────────────────────────
In real life, the most dangerous mistakes happen when **the user themselves doesn't realize \
they are about to cause a problem**. A truly intelligent agent must:
  1. Track implicit state changes mentioned casually in conversation
  2. Detect when a new user action would conflict with that earlier state
  3. **Proactively intervene** — without ever being asked to "remind me"

This benchmark tests exactly this: can the agent connect an implicit constraint with \
a later trigger and speak up at the critical moment?

────────────────────────────────────────────────────
THREE-PART SCENARIO STRUCTURE
────────────────────────────────────────────────────
Every scenario follows the "Implicit Constraint → Context Shift → Hidden Vulnerability" pattern:

1. **Constraint (Implicit State Mutation):**
   The user casually mentions a TEMPORARY physical or logical change — placing an item in \
an unusual location, disabling a safety feature, removing a personal essential, etc. \
This is stated as a natural, matter-of-fact remark. **NEVER** includes "remind me", \
"don't forget", "TODO", or any explicit request for reminding. The user does NOT \
describe catastrophic consequences in their own words.

2. **Trigger (Context-Shift Action):**
   Later, the user announces they are about to do something new — leaving a location, \
starting a routine task, transitioning to a new activity. This action is completely \
innocent on its own, but when combined with the forgotten constraint, creates a hidden \
conflict. The trigger must have **ZERO keyword overlap** with the constraint.

3. **Hidden Vulnerability (What goes wrong if Agent stays silent):**
   If the agent doesn't speak up, the user's trigger action will cause: item loss, \
property damage, safety hazard, task failure, or social embarrassment. The agent \
should proactively remind the user about the constraint before they proceed.

────────────────────────────────────────────────────
ABSOLUTE RULES (violations = rejection)
────────────────────────────────────────────────────
1. **IMPLICIT CONSTRAINT (core rule)**
   The constraint must be a NATURAL, CASUAL factual statement. It describes what happened \
or what the user did, NOT what the user wants the agent to do.
   - FORBIDDEN in constraint: "remind me", "don't forget", "make sure I", "TODO", \
"take a note", "heads up", "note to self", "keep in mind"
   - FORBIDDEN in constraint: explicit catastrophe language — "or it will break", \
"otherwise I'll die", "which is dangerous". Leave consequences for hidden_vulnerability.
   - The constraint should read like something a person says in passing without \
thinking about it again.

2. **ZERO SURFACE-MATCH between constraint and trigger**
   No shared domain-specific keywords. The trigger must describe a completely different \
activity/context from the constraint. The connection is discovered only through \
REASONING about physical states and consequences.

3. **PHYSICAL/LOGICAL PLAUSIBILITY & GENUINE RISK**
   Both the constraint and trigger must be realistic. The time gap must be **at least 30 minutes**.
   - The hidden vulnerability must involve **genuine consequences**: valuable/important item \
loss, property damage, safety hazard, critical task failure, or serious social embarrassment.
   - The user must be able to physically carry out the trigger WITHOUT immediately noticing \
the problem. Ask yourself: "Would a real person actually overlook this?" If the answer is no, \
the scenario is invalid.
   REJECTED EXAMPLES (do NOT generate anything like these):
   - Forgetting shoes → you literally cannot walk away without noticing you're barefoot.
   - Forgetting lunch → low-stakes; you just buy another one.
   - Forgetting a pen → trivially replaceable, no real consequence.
   - Leaving your phone on the table you're sitting at → you'd notice within seconds.
   GOOD EXAMPLES (genuine hidden risk):
   - Passport left in hotel safe → can't board international flight.
   - Laptop charger left plugged in conference room → critical device dies before presentation.
   - Back door unlocked for plumber → security vulnerability overnight.
   - Engagement ring in coat pocket going to dry cleaner → irreplaceable sentimental loss.
   - Contact lenses removed for nap → dangerous to drive without them.

4. **distracting_task ISOLATION**
   The distracting_task must relate to the trigger context, NOT the constraint. It must \
be answerable using general knowledge (no external API, no user-specific data needed).

5. **resolution_for_negative REQUIRED**
   Must provide a realistic way the constraint gets resolved WITHOUT agent intervention \
(user remembers, someone else reminds, etc.). Used for negative/false-alarm testing.

6. **SLOT VALUES (free-form keys)**
   You define the slot names yourself based on the scenario. Common slots include \
"object", "location", "mechanism", "state", "action", etc. — but use whatever fits.
   - There must be 1-3 slots (usually 1-2).
   - Each slot in required_slots must have a corresponding list of 2-4 diverse aliases \
in slot_values.
   - **GROUNDING RULE**: For each slot's alias list, at least one alias must be a word \
or phrase that literally appears in the constraint text. This ensures slots are \
grounded in the actual scenario content.
     Example: constraint mentions "cello" and "trunk" → \
slot_values = {{"object": ["cello", "instrument"], "location": ["trunk", "taxi"]}}
     Here "cello" appears in the constraint (grounding "object") and "trunk" appears \
in the constraint (grounding "location").

7. **WITHIN-COMBO DIVERSITY**
   Each generation call targets a specific domain × risk-type combination. Stay within \
the designated risk type, but vary the specific objects, device settings, backgrounds, \
characters, and trigger scenarios. Do NOT reuse the same item/setting or the same trigger \
pattern from the seed examples or from previously generated scenarios in the history.

────────────────────────────────────────────────────
OUTPUT
────────────────────────────────────────────────────
Return ONLY a single, valid JSON object. No markdown fences, no comments, no extra text.\
"""


def build_user_prompt(
    combo_info: dict,
    seed_examples: list[dict],
    history: list,
) -> str:
    """Assemble the user prompt for one generation call."""

    domain = combo_info["domain"]
    risk_type = combo_info["risk_type"]
    description = combo_info["description"]

    # Format seed examples
    seed_texts = []
    for i, seed in enumerate(seed_examples, 1):
        stripped = strip_seed_for_prompt(seed)
        seed_texts.append(
            f"--- Example {i} ---\n"
            f"{json.dumps(stripped, indent=2, ensure_ascii=False)}"
        )
    seeds_block = "\n\n".join(seed_texts)

    history_text = format_history_compact(history)

    prompt = f"""\
Generate ONE new state-change blueprint for:
  - domain = {domain}
  - risk_type = {risk_type}

═══ DOMAIN × RISK-TYPE DESCRIPTION ═══
{description}

═══ WHY THIS MATTERS (Benchmark Purpose) ═══
We are building TriggerBench to test whether LLM agents have **implicit prospective memory** — \
the ability to track casually-mentioned state changes and proactively intervene when a later \
user action would create a conflict. The constraint is NEVER an explicit "remind me" request. \
Instead, the user just states a fact ("I put my keys on the fridge shelf") and later \
announces an unrelated action ("I'm heading out to pick up my kid"). A smart agent should \
connect these and say: "Wait — your keys are still in the fridge!"

═══ SEED EXAMPLES (for format & quality — do NOT duplicate) ═══
{seeds_block}

═══ ALREADY GENERATED FOR THIS COMBO (avoid repetition) ═══
{history_text}

═══ REQUIRED OUTPUT JSON SCHEMA ═══
{{
  "story": {{
    "background": "<1-2 sentence scene setting — who is the user, what is the general context>",
    "constraint": "<User's natural, casual first-person statement describing a temporary \
state change. Must be IMPLICIT — no 'remind me', no explicit danger warnings. Just a \
factual remark about what they did.>",
    "trigger": "<User's later statement announcing a new action or context switch. Must \
have ZERO keyword overlap with constraint. The action seems completely innocent on its own.>",
    "time_gap_between_constraint_and_trigger": "<MUST be >= 30 minutes. e.g., '30 minutes', '2 hours', '1 day'>",
    "hidden_vulnerability": "<Analysis: why the trigger + constraint creates a hidden \
problem. What will go wrong if the agent stays silent?>",
    "variations": {{
      "distracting_task": "<A question related to the TRIGGER context (not constraint). \
Must be answerable with general knowledge only.>",
      "resolution_for_negative": "<How the constraint gets naturally resolved without \
agent help — user remembers, someone else reminds, etc.>"
    }}
  }},
  "gold_labels": {{
    "intent": "<What the ideal agent should proactively say>",
    "slots": {{
      "required_slots": ["<slot_name_1>", "<slot_name_2>", ...],
      "slot_values": {{
        "<slot_name_1>": ["alias1", "alias2"],
        "<slot_name_2>": ["alias1", "alias2"]
      }}
    }}
  }}

Note on slots: YOU choose the slot names that best describe the key entities in your \
scenario. Common slots: "object", "location", "mechanism", "state", "device", \
"person", "action". Use 1-3 slots. Each must have 2-4 aliases in slot_values.
IMPORTANT: For each slot, at least one alias in slot_values MUST literally appear \
in the constraint text (exact substring match, case-insensitive).
}}

Final reminders:
- Constraint must be IMPLICIT: natural, casual, no "remind me / warn me / don't forget / TODO".
- Constraint must NOT describe catastrophic consequences in the user's own words.
- ZERO keyword overlap between constraint and trigger.
- distracting_task relates to trigger context, NEVER to constraint.
- resolution_for_negative must be realistic and natural.
- Be creative — different items, locations, and risk patterns from seeds & history.
- Think about what makes a scenario truly tricky: the state change should be easy to forget, \
and the trigger action should feel completely routine.
- The hidden vulnerability must involve GENUINE consequences (valuable item loss, property \
damage, safety hazard, critical task failure). Do NOT generate trivial risks (forgetting a pen, \
forgetting lunch) or physically impossible oversights (forgetting shoes you need to walk).
- Focus on the specified risk type ({risk_type}). Vary specific items, settings, and \
characters, but stay within the designated scenario archetype.\
- Make the generated blueprint distinct and diverse from the history and seed examples. Include different scenario, objects, location etc.
"""
    return prompt




def validate_blueprint(item: dict) -> tuple[bool, list[str]]:
    """Validate a generated state-change blueprint. Returns (is_valid, errors)."""
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
    required_story_keys = [
        "background", "constraint", "trigger",
        "time_gap_between_constraint_and_trigger",
        "hidden_vulnerability",
    ]
    for key in required_story_keys:
        if key not in story:
            errors.append(f"Missing story.{key}")
    # Accept both 'variation' and 'variations'
    if "variations" not in story and "variation" not in story:
        errors.append("Missing story.variations (or story.variation)")
    if errors:
        return False, errors

    # ── 3. Constraint must not be empty ─────────────────────────────────────
    constraint_text = story.get("constraint", "").strip()
    if len(constraint_text) < 20:
        errors.append(f"Constraint too short ({len(constraint_text)} chars), likely incomplete")

    trigger_text = story.get("trigger", "").strip()
    if len(trigger_text) < 15:
        errors.append(f"Trigger too short ({len(trigger_text)} chars), likely incomplete")

    # ── 4. Implicit constraint check — no explicit reminder language ────────
    c_lower = constraint_text.lower()
    for kw in EXPLICIT_KEYWORDS:
        if kw in c_lower:
            errors.append(f"Constraint contains FORBIDDEN explicit keyword: '{kw}'")
            break

    # ── 5. No explicit catastrophe language in constraint ───────────────────
    for kw in CATASTROPHE_KEYWORDS:
        if kw in c_lower:
            errors.append(f"Constraint contains explicit catastrophe language: '{kw}'")
            break

    # ── 6. Variations completeness (accept both 'variation' and 'variations') ─
    variations = story.get("variations") or story.get("variation", {})
    if "distracting_task" not in variations:
        errors.append("Missing story.variations.distracting_task")
    elif len(variations["distracting_task"].strip()) < 10:
        errors.append("distracting_task is too short")

    if "resolution_for_negative" not in variations:
        errors.append("Missing story.variations.resolution_for_negative")
    elif len(variations["resolution_for_negative"].strip()) < 10:
        errors.append("resolution_for_negative is too short")

    # ── 7. gold_labels validation ───────────────────────────────────────────
    if "intent" not in gl:
        errors.append("Missing gold_labels.intent")
    elif len(gl["intent"].strip()) < 10:
        errors.append("gold_labels.intent is too short")

    if "slots" not in gl:
        errors.append("Missing gold_labels.slots")
    else:
        slots = gl["slots"]
        if "required_slots" not in slots:
            errors.append("Missing gold_labels.slots.required_slots")
        elif not isinstance(slots["required_slots"], list) or len(slots["required_slots"]) < 1:
            errors.append("gold_labels.slots.required_slots must be a non-empty list")
        if "slot_values" not in slots:
            errors.append("Missing gold_labels.slots.slot_values")
        elif not isinstance(slots["slot_values"], dict) or len(slots["slot_values"]) < 1:
            errors.append("gold_labels.slots.slot_values must be a non-empty dict")
        # Cross-check: every required_slot must have aliases in slot_values
        if (isinstance(slots.get("required_slots"), list)
                and isinstance(slots.get("slot_values"), dict)):
            # Max 3 slots
            if len(slots["required_slots"]) > 3:
                errors.append(
                    f"Too many slots: {len(slots['required_slots'])} "
                    f"(max 3). required_slots={slots['required_slots']}"
                )
            for s in slots["required_slots"]:
                if s not in slots["slot_values"]:
                    errors.append(f"required_slot '{s}' missing from slot_values")
                elif not isinstance(slots["slot_values"][s], list):
                    errors.append(f"slot_values['{s}'] must be a list")
                elif len(slots["slot_values"][s]) < 2:
                    errors.append(f"slot_values['{s}'] needs >= 2 aliases")

            # Grounding check: each slot must have ≥1 alias in constraint
            c_lower = constraint_text.lower()
            for s in slots["required_slots"]:
                aliases = slots.get("slot_values", {}).get(s, [])
                if isinstance(aliases, list) and len(aliases) >= 1:
                    if not any(a.lower() in c_lower for a in aliases):
                        errors.append(
                            f"slot '{s}': none of {aliases} appears in the "
                            f"constraint text (grounding rule violated)"
                        )

    # ── 8. time_gap: must be parseable AND >= 30 minutes ─────────────────────
    time_gap = story.get("time_gap_between_constraint_and_trigger", "")
    if not time_gap or len(time_gap.strip()) < 3:
        errors.append("time_gap_between_constraint_and_trigger is missing or too short")
    else:
        parsed_mins = parse_time_gap_minutes(time_gap)
        if parsed_mins is None:
            errors.append(
                f"time_gap '{time_gap}' could not be parsed into minutes — "
                "use a clear format like '30 minutes', '2 hours', '1 day'"
            )
        elif parsed_mins < MIN_TIME_GAP_MINUTES:
            errors.append(
                f"time_gap '{time_gap}' → {parsed_mins:.0f} min, "
                f"which is below the minimum of {MIN_TIME_GAP_MINUTES} minutes"
            )

    # ── 9. hidden_vulnerability check ───────────────────────────────────────
    hv = story.get("hidden_vulnerability", "")
    if len(hv.strip()) < 20:
        errors.append("hidden_vulnerability is too short — needs meaningful analysis")

    return len(errors) == 0, errors


def main():
    ap = argparse.ArgumentParser(
        description="Generate State-Tracking blueprints (dimension scaling)"
    )
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to existing output JSON to resume from")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    def call_llm(sys_p, usr_p):
        return complete(sys_p, usr_p, model=args.model, temperature=args.temperature)

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else DIM_DIR / "outputs" / "stage1_blueprints"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"blueprints_state_tracking_{args.model}.json"

    # ── resume support ──────────────────────────────────────────────────────
    all_items: list[dict] = []
    per_combo_history: dict[str, list] = {
        f"{c['domain']}__{c['risk_type']}": [] for c in STATE_TRIGGER_COMBOS
    }

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as f:
                all_items = json.load(f)
            for item in all_items:
                meta = item.get("meta", {})
                combo_key = f"{meta.get('domain', '')}__{meta.get('risk_type', '')}"
                if combo_key in per_combo_history:
                    per_combo_history[combo_key].append(item)
            log.info("Resumed %d items from %s", len(all_items), resume_path)

    global_idx = len(all_items)

    total_target = sum(
        COMBO_ITEM_COUNT.get(f"{c['domain']}__{c['risk_type']}", 0)
        for c in STATE_TRIGGER_COMBOS
    )
    log.info("Target: %d blueprints across %d combos",
             total_target, len(STATE_TRIGGER_COMBOS))
    for ck, cv in COMBO_ITEM_COUNT.items():
        log.info("  %-50s  %d", ck, cv)
    log.info("Model=%s, T=%.2f, max_retries=%d",
             args.model, args.temperature, args.max_retries)

    # ═══════════════════════════════════════════════════════════════════════
    #  Generation loop:  L1 combo → L2 K samples
    # ═══════════════════════════════════════════════════════════════════════
    for combo_idx, combo_info in enumerate(STATE_TRIGGER_COMBOS):
        domain = combo_info["domain"]
        risk_type = combo_info["risk_type"]
        combo_key = f"{domain}__{risk_type}"
        history = per_combo_history[combo_key]
        k_target = COMBO_ITEM_COUNT.get(combo_key, 0)

        if k_target == 0:
            log.info(
                "L1 [%d/%d]: %s — target=0, skipping",
                combo_idx + 1, len(STATE_TRIGGER_COMBOS), combo_key,
            )
            continue

        # Skip combos already completed via resume
        already_done = len(history)
        if already_done >= k_target:
            log.info(
                "L1 [%d/%d]: %s — already has %d/%d, skipping",
                combo_idx + 1, len(STATE_TRIGGER_COMBOS),
                combo_key, already_done, k_target,
            )
            continue

        log.info("=" * 70)
        log.info(
            "L1 [%d/%d]: %s  (history=%d, generating %d more)",
            combo_idx + 1, len(STATE_TRIGGER_COMBOS),
            combo_key, already_done, k_target - already_done,
        )

        for k_idx in range(already_done, k_target):
            sample_tag = f"{combo_key}__#{k_idx + 1}"
            log.info("  Generating %s ...", sample_tag)

            seed_examples = select_seed_examples_for_combo(combo_info)

            user_prompt = build_user_prompt(combo_info, seed_examples, history)
            success = False

            for retry in range(args.max_retries):
                try:
                    raw = call_llm(SYSTEM_PROMPT, user_prompt)
                    item = extract_json_object(raw)
                    valid, errs = validate_blueprint(item)

                    if valid:
                        global_idx += 1
                        item["task_id"] = f"state_tracking_{global_idx:03d}"
                        item["meta"] = {
                            "domain": domain,
                            "risk_type": risk_type,
                            "trigger_type": "state-change",
                        }

                        all_items.append(item)
                        history.append(item)
                        log.info(
                            "  ✓ OK (attempt %d/%d)  task_id=%s",
                            retry + 1, args.max_retries, item["task_id"],
                        )

                        success = True

                        # incremental save
                        save_json(all_items, out_file)
                        break

                    else:
                        log.warning(
                            "  ✗ Validation failed (attempt %d/%d):",
                            retry + 1, args.max_retries,
                        )
                        for e in errs:
                            log.warning("    - %s", e)

                        if retry < args.max_retries - 1:
                            err_block = "\n".join(f"- {e}" for e in errs)
                            user_prompt = (
                                build_user_prompt(
                                    combo_info, seed_examples, history,
                                )
                                + f"\n\n=== YOUR PREVIOUS ATTEMPT FAILED VALIDATION ===\n"
                                  f"Errors:\n{err_block}\n"
                                  f"Fix ALL issues above and generate a NEW valid blueprint."
                            )

                except json.JSONDecodeError as e:
                    log.warning(
                        "  ✗ JSON parse error (attempt %d/%d): %s",
                        retry + 1, args.max_retries, e,
                    )
                except Exception as e:
                    log.warning(
                        "  ✗ Error (attempt %d/%d): %s",
                        retry + 1, args.max_retries, e,
                    )

            if not success:
                log.error(
                    "  ✗ FAILED after %d retries — skipping %s",
                    args.max_retries, sample_tag,
                )

    save_json(all_items, out_file)

    log.info("=" * 70)
    log.info("Done!  %d / %d blueprints generated.", len(all_items), total_target)
    log.info("Saved -> %s", out_file)

    stats: dict[str, int] = {}
    for item in all_items:
        meta = item.get("meta", {})
        key = f"{meta.get('domain', '?')}__{meta.get('risk_type', '?')}"
        stats[key] = stats.get(key, 0) + 1
    log.info("Per-combo counts:")
    for key in sorted(stats):
        log.info("  %-50s  %d", key, stats[key])


if __name__ == "__main__":
    main()
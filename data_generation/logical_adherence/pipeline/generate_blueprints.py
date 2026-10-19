#!/usr/bin/env python3
"""Stage 1 — Blueprint design for the Logical Adherence dimension.

Generates scenario blueprints evaluating whether an LLM can follow a
user-defined conditional rule (Dynamic Rule Adherence).

Three logic types:
  - Forward_If-Then  : multi-hop condition tracking
  - Mutual_Exclusion : red-line / threshold blocking
  - Only-After       : prerequisite unlock

Loop structure:
  L1: 7 domains
    L2: 3 logic types per domain  →  7 × 3 = 21 combos
      L3: K items per combo
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
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
#  Constants — 7 Domains
# ═══════════════════════════════════════════════════════════════════════════

CONDITION_TRIGGER_DOMAINS = [
    {
        "domain": "Work/Professional",
        "base_description": (
            "Workplace scenarios involving cross-department meetings, clock-in/clock-out, "
            "report submissions, invoice reimbursement approvals, client data anonymization, "
            "and confidential communications, etc."
        ),
        "logic_inspirations": {
            "Forward_If-Then": (
                "Example: 'If the morning meeting mentions budget cuts, "
                "I need to upload a report plan to the cloud before I leave office.'"
            ),
            "Only-After": (
                "Example: 'My manager requires me to send the finalized contract to the client "
                "for signing after receiving a confirmed-all-clear "
                "email from the legal department.'"
            ),
            "Mutual_Exclusion": (
                "[Non-numeric blocking] Example: 'Demo materials sent via "
                "work email to external vendors must never contain API private keys.'\n"
                "[Numeric blocking] Example: 'Reimbursement amount must never exceed X.'"
            ),
        },
    },
    {
        "domain": "Home/Domestic",
        "base_description": (
            "Home-life scenarios involving shopping, families, housekeeping, weather-based chore decisions, pet feeding "
            "with allergy awareness, special garment washing restrictions, and smart "
            "appliance permission settings, etc."
        ),
        "logic_inspirations": {
            "Forward_If-Then": (
                "Example: 'If the morning news forecasts heavy rain this "
                "afternoon, I have to bring the patio cushions inside before I leave home "
                "for the supermarket.'"
            ),
            "Only-After": (
                "Example: 'I need to activate the indoor infrared security alarm after all family members have left the "
                "house.'"
            ),
            "Mutual_Exclusion": (
                "[Non-numeric blocking] Example: 'When washing the "
                "expensive wool coat, the washing machine must never use a high-temperature wash.'\n"
                "[Numeric blocking] Example: 'Each online purchase must "
                "never exceed X in budget.'"
            ),
        },
    },
    {
        "domain": "Finance/Commerce",
        "base_description": (
            "Financial and commerce scenarios involving virtual credit card security, "
            "food-delivery budget tracking, overseas shopping customs declarations, "
            "investment stop-loss rules, and travel booking budgets, etc."
        ),
        "logic_inspirations": {
            "Forward_If-Then": (
                "Example: 'If the stock app alerts that a pharma stock "
                "has dropped below cost basis today, I should reassess the position before "
                "the market closes this afternoon.'"
            ),
            "Only-After": (
                "Example: 'I need to place a new order for that electronics product after the refund for a previously "
                "cancelled overseas order has arrived in my bank account.'"
            ),
            "Mutual_Exclusion": (
                "[Numeric blocking] Example: 'To save aggressively, "
                "the combined daily total for train tickets and hotel bookings must never "
                "exceed 500 dollars.'"
            ),
        },
    },
    {
        "domain": "Health/Dietary",
        "base_description": (
            "Health and dietary scenarios involving medication contraindications and "
            "antagonistic drug interactions, fasting requirements before blood draws, "
            "strict low-carb or special dietary restrictions, and fitness tracking, etc."
        ),
        "logic_inspirations": {
            "Forward_If-Then": (
                "Example: 'If my smartwatch syncs and shows I walked "
                "fewer than 5,000 steps today, I need to do 30 minutes of stretching "
                "before I go to bed tonight to maintain enough exercise.'"
            ),
            "Only-After": (
                "Example: 'I need to order some breakfast delivery after I have completed the fasting "
                "blood draw at the hospital tomorrow morning.'"
            ),
            "Mutual_Exclusion": (
                "[Non-numeric blocking] Example: 'I just started "
                "cephalosporin antibiotics by IV drip for my cold — any drink I order at "
                "the bar must contain absolutely zero alcohol.'"
            ),
        },
    },
    {
        "domain": "Parenting/Social",
        "base_description": (
            "Parenting and social scenarios involving rules for "
            "children education/entertainment, cultural taboos and gift-giving etiquette, customised menus for "
            "friend gatherings, and allergy precautions, etc."
        ),
        "logic_inspirations": {
            "Forward_If-Then": (
                "Example: 'If Person A messages on WeChat that they are "
                "bringing their vegetarian girlfriend to dinner this weekend, I need to"
                "buy extra leafy greens before I go downstairs to the market.'"
            ),
            "Only-After": (
                "Example: 'I should give my ten-year-old son the iPad child-lock passcode after he has "
                "finished his math homework.'"
            ),
            "Mutual_Exclusion": (
                "[Non-numeric blocking] Example: 'A friend just had a "
                "baby girl born in the Year of the Rabbit — when picking a Mid-Autumn gift, "
                "never choose anything with negative-themed prints like wolves.'"
            ),
        },
    },
    {
        "domain": "Transit/Travel",
        "base_description": (
            "Transit and travel scenarios, rental car "
            "fuel-type mismatches, international travel security precautions, customs "
            "prohibited-item awareness, and digital ticket management, etc."
        ),
        "logic_inspirations": {
            "Forward_If-Then": (
                "Example: 'If my flight is delayed by more than 3 hours, "
                "I will go to the airport counter and collect free meal vouchers for"
                " compensation before taking off.'"
            ),
            "Only-After": (
                "Example: 'I need to pick up my luggage after I have cleared customs at the airport.'"
            ),
            "Mutual_Exclusion": (
                "[Non-numeric blocking] Example: 'The rental Jeep "
                "Wrangler for this road trip is a diesel version — when refueling at any "
                "gas station along the route, it must never be filled with 92 or 95 octane "
                "gasoline.'"
            ),
        },
    },
    {
        "domain": "Study/Academic",
        "base_description": (
            "Academic and study scenarios involving exam preparation, thesis writing, "
            "course registration deadlines, library resource reservations, study group "
            "coordination, research paper submissions, academic advisor meetings, "
            "scholarship applications, and campus event planning, etc."
        ),
        "logic_inspirations": {
            "Forward_If-Then": (
                "Example: 'If my professor announces in today's lecture that "
                "Chapter 7 will be on the midterm, I need to review my Chapter 7 notes "
                "before going to the study group session tonight.'"
            ),
            "Only-After": (
                "Example: 'I should submit my final thesis draft to the "
                "department committee after my advisor sends the approval email.'"
            ),
            "Mutual_Exclusion": (
                "[Non-numeric blocking] Example: 'My thesis uses "
                "double-blind peer review data — any draft I share with classmates "
                "must never include the raw participant identifiers.'\n"
                "[Numeric blocking] Example: 'I am on a strict study schedule this "
                "finals week — I must not spend more than 20 minutes on social media "
                "per session.'"
            ),
        },
    },
]

LOGIC_TYPES = ["Forward_If-Then", "Mutual_Exclusion", "Only-After"]

# Per-combo item count: 7 domains × 3 logic types = 21 combos
# Default: 6 items per combo → 126 total
COMBO_ITEM_COUNT: dict[str, int] = {}
for _d in CONDITION_TRIGGER_DOMAINS:
    for _lt in LOGIC_TYPES:
        COMBO_ITEM_COUNT[f"{_d['domain']}__{_lt}"] = 6


MIN_TIME_GAP_MINUTES = 30      # >= 30 min
MAX_TIME_GAP_MINUTES = 2880    # <= 2 days


# ═══════════════════════════════════════════════════════════════════════════
#  Explicit Reminder Keywords — constraints must be IMPLICIT
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


def _normalize_overlap_text(text: str) -> str:
    """Expand contractions, strip punctuation, lowercase — used for overlap checks."""
    t = text.lower().strip()
    t = re.sub(r"what's", "what is", t)
    t = re.sub(r"n't", " not", t)
    t = re.sub(r"'re", " are", t)
    t = re.sub(r"'ve", " have", t)
    t = re.sub(r"'ll", " will", t)
    t = re.sub(r"'m", " am", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def check_trigger_distracting_overlap(trigger_text: str, distracting_task: str) -> bool:
    """Return True if trigger text already contains the distracting task (Stage 4 would fail)."""
    dt_norm = _normalize_overlap_text(distracting_task)
    trig_norm = _normalize_overlap_text(trigger_text)
    if not dt_norm or not trig_norm:
        return False
    if dt_norm in trig_norm:
        return True
    stopwords = {
        "a", "an", "the", "is", "to", "of", "in", "on", "at", "it", "i", "my",
        "me", "we", "do", "be", "so", "or", "if", "no", "up", "by", "as",
    }
    dt_words = set(dt_norm.split()) - stopwords
    trig_words = set(trig_norm.split()) - stopwords
    if len(dt_words) >= 3:
        overlap_ratio = len(dt_words & trig_words) / len(dt_words)
        if overlap_ratio >= 0.55:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Seed Selection  — per logic_type, include ALL seeds of that type
# ═══════════════════════════════════════════════════════════════════════════

def select_seed_examples_for_logic_type(logic_type: str) -> list[dict]:
    """Return all seed examples matching the given logic_type (should be 2)."""
    lt_map = {
        "Forward_If-Then": "Forward_If-Then",
        "Mutual_Exclusion": "Mutual_Exclusion",
        "Only-After": "Only-After",
    }
    target = lt_map.get(logic_type, logic_type)
    return [
        s for s in LOGICAL_ADHERENCE_SEEDS
        if s.get("meta", {}).get("logic_type") == target
    ]


def strip_seed_for_prompt(seed: dict) -> dict:
    """Remove task_id and meta from seed for prompt inclusion."""
    out = deepcopy(seed)
    out.pop("task_id", None)
    out.pop("meta", None)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  History Formatting  — compact summary for dedup
# ═══════════════════════════════════════════════════════════════════════════

def format_history_compact(items: list, logic_type: str) -> str:
    """Build a concise history string showing rule + trigger for dedup."""
    if not items:
        return "(none yet)"
    parts = []
    for i, item in enumerate(items, 1):
        story = item.get("story", {})
        rule = story.get("rule_constraint", "")

        if logic_type == "Forward_If-Then":
            cond_pos = story.get("condition_constraint", {}).get("positive", "")
            trigger = story.get("trigger", "")
            tg1 = story.get("time_gap_rule_to_condition", "?")
            tg2 = story.get("time_gap_condition_to_trigger", "?")
            parts.append(
                f"  --- #{i} ---\n"
                f"  Rule:           {rule}\n"
                f"  Condition(+):   {cond_pos}\n"
                f"  Trigger:        {trigger}\n"
                f"  TimeGaps:       rule→condition: {tg1},  condition→trigger: {tg2}"
            )
        else:
            trigger_pos = story.get("trigger", {}).get("positive", "")
            tg = story.get("time_gap_between_constraint_and_trigger", "?")
            parts.append(
                f"  --- #{i} ---\n"
                f"  Rule:           {rule}\n"
                f"  Trigger(+):     {trigger_pos}\n"
                f"  TimeGap:        {tg}"
            )
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt Construction
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert benchmark-data creator for **TriggerBench**, a benchmark that evaluates \
whether LLM agents can follow user-defined complex logic rules in long conversations and \
**proactively** intervene or stay silent at the right moment.

Your job: generate ONE new conditional-trigger "blueprint" (scenario skeleton) as a single \
JSON object.

────────────────────────────────────────────────────
CORE CONCEPT — Dynamic Rule Adherence
────────────────────────────────────────────────────
Unlike state-change triggers (tracking physical environment changes), conditional triggers \
test whether the agent can:
  1. Remember a user-defined RULE (policy, constraint, threshold) stated earlier
  2. Accurately determine whether that rule's condition has been met or violated
  3. Proactively intervene (in Positive cases) or correctly stay silent (in Negative cases)

This is about **logic faithfulness** — can the agent remember and correctly apply explicit \
user rules over a long conversation?

We generate PAIRED data: each blueprint contains both a Positive scenario (agent MUST act) \
and a Negative scenario (agent must stay SILENT). This enables precise control-variable \
evaluation.

────────────────────────────────────────────────────
IMPLICIT CONSTRAINT — NO EXPLICIT REMINDER LANGUAGE
────────────────────────────────────────────────────
The `rule_constraint` (and `condition_constraint` if applicable) must be expressed as the \
user's IMPLICIT intention or personal rule — the user states what they WILL do, NEED to do, \
SHOULD do, or MUST NOT do. The user NEVER explicitly asks the agent to remind them.

FORBIDDEN phrases in rule_constraint and condition_constraint:
  "remind me", "alert me", "warn me", "notify me", "don't let me forget", \
  "do not let me forget", "take a note", "set a reminder", "set an alarm", \
  "make sure I", "remember to", "don't forget", "do not forget", "keep in mind", \
  "heads up later", "ping me", "flag this", "note to self", "todo"

GOOD: "If the CEO mentions Q3 project, I should submit the report before leaving."
GOOD: "I must not place any takeout order over 50 dollars."
GOOD: "I need to pick up the pizza after the soccer game is over."

BAD:  "If the CEO mentions Q3 project, remind me to submit the report before leaving."

The constraint should read like a personal statement of intent or rule, NOT a command to the \
agent. The whole point of this benchmark is to test whether the agent can proactively recall \
and apply a passively stated user rule — if the user explicitly asks to be reminded, it \
becomes trivial and defeats the purpose.

────────────────────────────────────────────────────
OVERLOADED TRIGGER — Distracting Task (CRITICAL SEPARATION RULE)
────────────────────────────────────────────────────
Every blueprint includes a `trigger_distracting_task`: a context-free question that will \
be injected alongside the trigger IN A LATER PIPELINE STAGE.

*** ABSOLUTE RULE — TRIGGER AND DISTRACTING TASK MUST BE COMPLETELY SEPARATE ***

The trigger field(s) must contain ONLY the trigger event itself. They must NEVER include, \
embed, merge, or append any part of the `trigger_distracting_task` content. The distracting \
task is a SEPARATE field that will be combined with the trigger text in Stage 4.

FORBIDDEN patterns in trigger text:
  - "...event happened. Also, [distracting question]?"
  - "...event happened. By the way, [distracting question]?"
  - "...event happened. Separate question: [distracting question]?"
  - ANY form of appending a question/task from trigger_distracting_task into trigger text

CORRECT: trigger contains ONLY the event / request.
  Example: "Finance just sent the signed expense form back with their approval."
  trigger_distracting_task: "What's a concise way to explain depreciation?"
  (Two completely separate fields. The trigger does NOT contain the question.)

WRONG: trigger merges both.
  Example: "Finance just sent the signed expense form back. Also, what's a concise way \
to explain depreciation?"

Other rules for trigger_distracting_task:
  - It may relate to the Background or Trigger scene, but must NEVER mention anything \
about the Rule Constraint.
  - It must be answerable by the LLM using only parametric knowledge (e.g., recipe, \
currency conversion, history summary, translation). No external search or private data needed.

────────────────────────────────────────────────────
TIME GAP REQUIREMENTS
────────────────────────────────────────────────────
All time gaps must be >= 30 minutes and <= 2 days. Format: "N minutes", "N hours", or "N days" \
(e.g., "45 minutes", "2 hours", "1 day").

────────────────────────────────────────────────────
SLOT VALUE RULES (for positive_gold only)
────────────────────────────────────────────────────
  - 1-3 slots, each with 2-4 aliases in slot_values.
  - **GROUNDING RULE**: For each slot, at least one alias must literally appear (as a \
substring, case-insensitive) in the `rule_constraint` text or `condition_constraint` text \
(for If-Then type).
  - negative_gold must always have empty slots: `"required_slots":[], "slot_values": {}`

────────────────────────────────────────────────────
WITHIN-COMBO DIVERSITY
────────────────────────────────────────────────────
Each call targets a specific (domain × logic_type). Stay within the designated domain and \
logic type, but vary specific scenarios, objects, rules, and contexts. Do NOT reuse the \
same rule pattern from seed examples or from previously generated items in the history.

────────────────────────────────────────────────────
OUTPUT
────────────────────────────────────────────────────
Return ONLY a single, valid JSON object. No markdown fences, no comments, no extra text.\
"""


# ── Logic-type specific prompt blocks ──

IFTHEN_LOGIC_BLOCK = """\
═══ LOGIC TYPE: Forward_If-Then (Multi-turn Condition Update) ═══
Core logic: "If event [A] happens, I will/need to/should/am supposed to/must/have to/would/would like to do [C] when event/time [B] arrives."

IMPORTANT: The rule_constraint must be an IMPLICIT personal statement. The user states what \
they will/should/need to do — they NEVER ask the agent to "remind me" or "alert me". \
Forbidden: "remind me", "alert me", "warn me", "notify me", "don't let me forget", etc.

Structure:
  - `rule_constraint`: The user defines an If-Then rule as a personal intention/plan.
  - `condition_constraint`: Later in conversation, a SEPARATE turn confirms or denies \
whether the "If" condition was met. Contains TWO variants:
    * `positive`: The condition WAS met (→ positive gold intent must remind at trigger time)
    * `negative`: The condition was NOT met (→ negative gold intent should NOT remind at trigger time)
  - `trigger`: A SINGLE shared trigger event (related to [B]) that occurs later.
  - Two time gaps: `time_gap_rule_to_condition` and `time_gap_condition_to_trigger`.

Key design points:
  - The positive/negative condition_constraint should be NATURAL conversation — the user \
mentions the outcome as part of normal dialogue, not as a formal report.
  - The trigger is SHARED between positive and negative — only the condition differs.
  - The positive condition confirmation should be embedded naturally, not stated as \
"the condition was met." Make it sound like real speech.
  - The negative condition denial should also be natural — the user mentions what happened \
instead (e.g., "The meeting was mostly about team building").
  - The trigger dialogue MUST NOT mention the intended action (Action C)!
  - The trigger must contain ONLY the trigger event. Do NOT append trigger_distracting_task.

═══ REQUIRED OUTPUT JSON SCHEMA (Forward_If-Then) ═══
{
  "story": {
    "background": "<1-2 sentence scene setting>",
    "rule_constraint": "If [Condition] happens, I will/need to/should/am supposed to/must/have to/would/would like to do [Action] when [Trigger event/time happens].",
    "condition_constraint": {
      "positive": "<Natural dialogue confirming the If-condition was met>",
      "negative": "<Natural dialogue indicating the If-condition was NOT met>"
    },
    "time_gap_rule_to_condition": "<>= 30 min, <= 2 days. e.g., '2 hours'>",
    "time_gap_condition_to_trigger": "<>= 30 min, <= 2 days. e.g., '6 hours'>",
    "trigger": "<User initiates the trigger event related to [B]>",
    "trigger_distracting_task": "<Context-free LLM query — relates to trigger scene, NOT to rule>"
  },
  "gold_labels": {
    "positive_gold": {
      "intent": "<What the agent should proactively remind when condition IS met>",
      "slots": {
        "required_slots": ["<slot_name_1>", ...],
        "slot_values": {
          "<slot_name_1>": ["alias1", "alias2"]
        }
      }
    },
    "negative_gold": {
      "intent": "<Explain why agent should NOT remind — condition was not met>",
      "slots": { "required_slots": [], "slot_values": {} }
    }
  }
}\
"""


EXCLUSION_LOGIC_BLOCK = """\
═══ LOGIC TYPE: Mutual_Exclusion (Single-turn Red-line Blocking) ═══
Core logic: The user sets an absolute red line or threshold. If the user later tries to \
cross it, the agent MUST block; if within bounds, the agent should not remind.

IMPORTANT: The rule_constraint must be an IMPLICIT personal rule/policy. The user states \
what they MUST NOT do or what is absolutely forbidden — they NEVER ask the agent to \
"remind me", "warn me", "alert me", "don't let me forget", etc.

Structure:
  - `rule_constraint`: The user defines a clear, absolute rule/threshold.
  - `trigger`: Contains TWO variants (the request itself differs):
    * `positive`: User makes a request that VIOLATES the rule → positive gold intent must remind/block.
    * `negative`: User makes a request that OBEYS the rule → negative gold intent should NOT remind.
  - One time gap: `time_gap_between_constraint_and_trigger`.

Key design points:
  - Rules can be QUANTITATIVE (numeric thresholds: budget limits, calorie caps, distance \
limits) or QUALITATIVE (category-based: material restrictions, allergen bans, security \
protocols, cultural taboos, fuel-type incompatibilities).
  - Generate a MIX of both quantitative and qualitative rules. Be creative with values, categorical, material, safety-protocol, and \
cultural-taboo exclusions.
  - The positive trigger should be a natural user request that happens to violate the rule. \
The user is NOT trying to test the agent — they genuinely forgot the rule.
  - The negative trigger should be a similar category of request that is clearly safe.
  - *** trigger.positive and trigger.negative must contain ONLY the request. ***
    Do NOT append or embed the trigger_distracting_task into the trigger text.

═══ REQUIRED OUTPUT JSON SCHEMA (Mutual_Exclusion) ═══
{
  "story": {
    "background": "<1-2 sentence scene setting>",
    "rule_constraint": "<Absolute rule: never do X / threshold Y (NO 'remind me' / 'alert me' / 'notify me' etc.)>",
    "time_gap_between_constraint_and_trigger": "<>= 30 min, <= 2 days>",
    "trigger": {
      "positive": "<User request that VIOLATES the rule — agent must remind/block>",
      "negative": "<User request that OBEYS the rule — agent should NOT remind>"
    },
    "trigger_distracting_task": "<Context-free LLM query>"
  },
  "gold_labels": {
    "positive_gold": {
      "intent": "<Why the agent should block this request>",
      "slots": {
        "required_slots": ["<slot_name_1>", ...],
        "slot_values": { "<slot_name_1>": ["alias1", "alias2"] }
      }
    },
    "negative_gold": {
      "intent": "<Why the agent should NOT intervene — request is within bounds>",
      "slots": { "required_slots": [], "slot_values": {} }
    }
  }
}\
"""


ONLYAFTER_LOGIC_BLOCK = """\
═══ LOGIC TYPE: Only-After (As-Soon-As) ═══
Core logic: The user wants to execute a specific task after a specific future event happens/finishes. It tests the agent's ability to monitor conversational state and proactively awaken exactly when the condition hits, remaining silent beforehand.

IMPORTANT: The rule_constraint must be an IMPLICIT personal intention/plan. The user states \
what they NEED/WANT/PLAN to do after some event — they NEVER ask the agent to "remind me", \
"alert me", "notify me", "don't let me forget", etc.

CRITICAL DIFFERENCE FROM MUTUAL EXCLUSION:
This is NOT a permission gatekeeper (e.g., DO NOT WRITE: "Only after Legal approves may I send the file").
Instead, this is a deferred proactive to-do (e.g., MUST WRITE: "I need to pick up pizza after the soccer game is over and when I goes home.").

Structure:
  - `rule_constraint`: An explicitly stated delayed intention using phrasing like: "Need/want/prefer to [Action X], after / as soon as [Event Z] (happens/finishes)." 
  - `trigger`: Contains TWO variants based on user's progress updates. The `trigger` should NEVER mention action X.
    * `positive`: The user announces that Event Z has EXACTLY happened (e.g., "The soccer game is finally over, we're heading to the car.") -> positive gold intent MUST immediately awaken and remind the user of Action X.
    * `negative`: The user describes a pseudo-event or intermediate state prior to Event Z (e.g., "It's halftime", "We are stuck in the 3rd quarter.") -> negative gold intent MUST stay silent regarding Action X because the condition is not yet strictly met.
  - One time gap: `time_gap_between_constraint_and_trigger`.

Key design points:
  - The positive trigger should naturally convey that the prerequisite was achieved — \
through dialogue, not a formal check-in (e.g., "He just finished the last page" not \
"The reading condition has been satisfied").
  - Make the Negative trigger variant tricky! It shouldn't just be "I am sleeping". It should be an adjacent event that shows partial progress but explicitly misses the trigger condition (e.g., If rule says "after flight lands", negative should be "boarding the plane" or "taking off"). Make it sound like real conversation, not a test.
  - The trigger dialogue MUST NOT mention the intended action (Action X)!
  - *** trigger.positive and trigger.negative must contain ONLY the trigger event. ***
    Do NOT append or embed the trigger_distracting_task question into the trigger text.

═══ REQUIRED OUTPUT JSON SCHEMA (Only-After) ═══
{
  "story": {
    "background": "<1-2 sentence scene setting>",
    "rule_constraint": "<Need/want/prefer to/should/would like to do [Action X], after [Event Z] (happens). (NO 'remind me' / 'alert me' / 'notify me' etc.)",
    "time_gap_between_constraint_and_trigger": "<>= 30 min, <= 2 days>",
    "trigger": {
      "positive": "<Dialogue indicating prerequisite IS fulfilled ([Event Z] has EXACTLY happened) — agent must remind/provide>",
      "negative": "<Dialogue indicating prerequisite is NOT fulfilled, describing an intermediate state (e.g. halftime, halfway there) — agent must NOT remind>"
    },
    "trigger_distracting_task": "<Context-free LLM query>"
  },
  "gold_labels": {
    "positive_gold": {
      "intent": "<Why the agent should now provide/remind the locked action, noting the condition has been met>",
      "slots": {
        "required_slots": ["<slot_name_1>", ...],
        "slot_values": { "<slot_name_1>": ["alias1", "alias2"] }
      }
    },
    "negative_gold": {
      "intent": "<State that the condition has not yet fully arrived (explain the intermediate state), so do not remind.>",
      "slots": { "required_slots": [], "slot_values": {} }
    }
  }
}\
"""

LOGIC_BLOCK_MAP = {
    "Forward_If-Then": IFTHEN_LOGIC_BLOCK,
    "Mutual_Exclusion": EXCLUSION_LOGIC_BLOCK,
    "Only-After": ONLYAFTER_LOGIC_BLOCK,
}


def build_user_prompt(
    domain_info: dict,
    logic_type: str,
    seed_examples: list[dict],
    history: list,
) -> str:
    """Assemble the user prompt for one generation call."""

    domain = domain_info["domain"]
    base_desc = domain_info["base_description"]
    logic_inspiration = domain_info["logic_inspirations"].get(logic_type, "")
    logic_block = LOGIC_BLOCK_MAP[logic_type]

    # Assemble domain description: base + only the relevant logic inspiration
    description = (
        f"{base_desc}\n"
        f"Logic inspiration ({logic_type}): {logic_inspiration}\n"
        f"The above is just one example pattern — feel free to create other creative "
        f"patterns beyond this inspiration. Make the scenario diverse and distinct."
    )

    # Format seed examples
    seed_texts = []
    for i, seed in enumerate(seed_examples, 1):
        stripped = strip_seed_for_prompt(seed)
        seed_texts.append(
            f"--- Seed Example {i} ---\n"
            f"{json.dumps(stripped, indent=2, ensure_ascii=False)}"
        )
    seeds_block = "\n\n".join(seed_texts)

    history_text = format_history_compact(history, logic_type)

    prompt = f"""\
Generate ONE new conditional-trigger blueprint for:
  - domain = {domain}
  - logic_type = {logic_type}

{logic_block}

═══ DOMAIN DESCRIPTION (for context & diversity inspiration) ═══
{description}

═══ SEED EXAMPLES (for format & quality — do NOT duplicate these) ═══
{seeds_block}

═══ ALREADY GENERATED FOR THIS (domain × logic) COMBO (avoid repetition) ═══
{history_text}

Final reminders:
- The output must strictly follow the JSON schema for {logic_type} shown above.
- **IMPLICIT CONSTRAINT (CRITICAL)**: rule_constraint (and condition_constraint if applicable) \
must NEVER contain explicit reminder language such as "remind me", "alert me", "warn me", \
"notify me", "don't let me forget", "make sure I", "remember to", "set a reminder", etc. \
The user states what they WILL/SHOULD/NEED TO/MUST (NOT) do — NEVER asks the agent to remind. \
This is the core design of the benchmark: testing implicit prospective memory.
- Time gaps must be >= 30 minutes and <= 2 days. Use format: "N minutes", "N hours (0.5 allowed)", or "N days".
- positive_gold slots: 1-3 slots, each with 2-4 aliases. At least one alias per slot must \
appear as a substring in rule_constraint (or condition_constraint for If-Then).
- negative_gold slots MUST be empty: {{"required_slots":[], "slot_values": {{}}}}.
- negative_gold intent should explicitly state that the condition was not met / rule was not \
violated, and the agent should "Do not remind user" to avoid over-intervention.
- trigger_distracting_task must be context-free: related to background/trigger scene, NEVER \
about the rule_constraint. Must be answerable by LLM parametric knowledge only.
- *** CRITICAL: trigger text must contain ONLY the pure trigger event/request. Do NOT embed, \
append, or merge any part of trigger_distracting_task into it. ***
- Stay within domain = {domain} and logic_type = {logic_type}.
- For Mutual_Exclusion: mix both quantitative (numeric threshold) and qualitative (category, \
material, allergy, safety protocol) rules. \
- Be creative and diverse — different scenarios, objects, rules, and contexts from seeds & history.
- Make the constraint and trigger conversational, natural and diverse.
- Make the time gaps diverse, natural and realistic for the scenario. 
"""
    return prompt


# ═══════════════════════════════════════════════════════════════════════════
#  Validation — logic-type aware
# ═══════════════════════════════════════════════════════════════════════════

def validate_blueprint(item: dict, logic_type: str) -> tuple[bool, list[str]]:
    """Validate a generated conditional-trigger blueprint. Returns (is_valid, errors)."""
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

    # ── 2. Story fields depend on logic type ────────────────────────────────
    if "background" not in story:
        errors.append("Missing story.background")
    if "rule_constraint" not in story:
        errors.append("Missing story.rule_constraint")
    if "trigger_distracting_task" not in story:
        errors.append("Missing story.trigger_distracting_task")
    elif len(story.get("trigger_distracting_task", "").strip()) < 10:
        errors.append("trigger_distracting_task is too short")

    rule_text = story.get("rule_constraint", "").strip()
    if len(rule_text) < 15:
        errors.append(f"rule_constraint too short ({len(rule_text)} chars)")

    # ── 2b. Implicit constraint check — no explicit reminder language ──────
    rule_lower = rule_text.lower()
    for kw in EXPLICIT_KEYWORDS:
        if kw in rule_lower:
            errors.append(
                f"rule_constraint contains FORBIDDEN explicit keyword '{kw}'. "
                f"Constraints must be IMPLICIT (state what user will/should/must do, "
                f"never ask agent to remind)."
            )

    # ── 3. Logic-type specific structure ────────────────────────────────────
    grounding_texts = [rule_text.lower()]  # texts to check slot grounding against

    if logic_type == "Forward_If-Then":
        # Must have condition_constraint with positive/negative
        if "condition_constraint" not in story:
            errors.append("Missing story.condition_constraint (required for If-Then)")
        else:
            cc = story["condition_constraint"]
            if not isinstance(cc, dict):
                errors.append("story.condition_constraint must be a dict with 'positive' and 'negative'")
            else:
                if "positive" not in cc:
                    errors.append("Missing story.condition_constraint.positive")
                elif len(cc["positive"].strip()) < 10:
                    errors.append("condition_constraint.positive is too short")
                else:
                    grounding_texts.append(cc["positive"].strip().lower())
                    # Check for explicit keywords in condition_constraint.positive
                    cc_pos_lower = cc["positive"].strip().lower()
                    for kw in EXPLICIT_KEYWORDS:
                        if kw in cc_pos_lower:
                            errors.append(
                                f"condition_constraint.positive contains FORBIDDEN explicit keyword '{kw}'. "
                                f"Must be implicit."
                            )
                            break
                if "negative" not in cc:
                    errors.append("Missing story.condition_constraint.negative")
                elif len(cc["negative"].strip()) < 10:
                    errors.append("condition_constraint.negative is too short")
                else:
                    # Check for explicit keywords in condition_constraint.negative
                    cc_neg_lower = cc["negative"].strip().lower()
                    for kw in EXPLICIT_KEYWORDS:
                        if kw in cc_neg_lower:
                            errors.append(
                                f"condition_constraint.negative contains FORBIDDEN explicit keyword '{kw}'. "
                                f"Must be implicit."
                            )
                            break

        # trigger must be a single string
        if "trigger" not in story:
            errors.append("Missing story.trigger")
        elif isinstance(story["trigger"], dict):
            errors.append("For If-Then, story.trigger must be a single string, not a dict")
        elif len(story["trigger"].strip()) < 10:
            errors.append("story.trigger is too short")
        else:
            dt = story.get("trigger_distracting_task", "")
            if dt and check_trigger_distracting_overlap(story["trigger"], dt):
                errors.append(
                    "OVERLAP DETECTED: trigger contains trigger_distracting_task content. "
                    "These MUST be completely separate fields."
                )

        # Time gaps
        for tg_key in ["time_gap_rule_to_condition", "time_gap_condition_to_trigger"]:
            tg = story.get(tg_key, "")
            if not tg or len(tg.strip()) < 3:
                errors.append(f"Missing or too short: story.{tg_key}")
            else:
                parsed = parse_time_gap_minutes(tg)
                if parsed is None:
                    errors.append(f"{tg_key} '{tg}' could not be parsed — use 'N minutes/hours/days'")
                elif parsed < MIN_TIME_GAP_MINUTES:
                    errors.append(f"{tg_key} '{tg}' → {parsed:.0f} min < {MIN_TIME_GAP_MINUTES} min minimum")
                elif parsed > MAX_TIME_GAP_MINUTES:
                    errors.append(f"{tg_key} '{tg}' → {parsed:.0f} min > {MAX_TIME_GAP_MINUTES} min maximum (2 days)")

        # Should NOT have time_gap_between_constraint_and_trigger
        if "time_gap_between_constraint_and_trigger" in story:
            errors.append("If-Then should use time_gap_rule_to_condition + time_gap_condition_to_trigger, NOT time_gap_between_constraint_and_trigger")

    else:  # Mutual_Exclusion or Only-After
        # trigger must be a dict with positive/negative
        if "trigger" not in story:
            errors.append("Missing story.trigger")
        elif not isinstance(story["trigger"], dict):
            # Also accept trigger_turns as alias
            if "trigger_turns" in story and isinstance(story["trigger_turns"], dict):
                story["trigger"] = story.pop("trigger_turns")
            else:
                errors.append(f"For {logic_type}, story.trigger must be a dict with 'positive' and 'negative'")
        if isinstance(story.get("trigger"), dict):
            trg = story["trigger"]
            if "positive" not in trg:
                errors.append("Missing story.trigger.positive")
            elif len(trg["positive"].strip()) < 10:
                errors.append("story.trigger.positive is too short")
            if "negative" not in trg:
                errors.append("Missing story.trigger.negative")
            elif len(trg["negative"].strip()) < 10:
                errors.append("story.trigger.negative is too short")

            dt = story.get("trigger_distracting_task", "")
            if dt and "positive" in trg and trg["positive"]:
                if check_trigger_distracting_overlap(trg["positive"], dt):
                    errors.append(
                        "OVERLAP DETECTED: trigger.positive contains trigger_distracting_task "
                        "content. These MUST be completely separate fields."
                    )
            if dt and "negative" in trg and trg["negative"]:
                if check_trigger_distracting_overlap(trg["negative"], dt):
                    errors.append(
                        "OVERLAP DETECTED: trigger.negative contains trigger_distracting_task "
                        "content. These MUST be completely separate fields."
                    )

        # time_gap_between_constraint_and_trigger
        tg = story.get("time_gap_between_constraint_and_trigger", "")
        if not tg or len(tg.strip()) < 3:
            errors.append("Missing or too short: story.time_gap_between_constraint_and_trigger")
        else:
            parsed = parse_time_gap_minutes(tg)
            if parsed is None:
                errors.append(f"time_gap '{tg}' could not be parsed — use 'N minutes/hours/days'")
            elif parsed < MIN_TIME_GAP_MINUTES:
                errors.append(f"time_gap '{tg}' → {parsed:.0f} min < {MIN_TIME_GAP_MINUTES} min minimum")
            elif parsed > MAX_TIME_GAP_MINUTES:
                errors.append(f"time_gap '{tg}' → {parsed:.0f} min > {MAX_TIME_GAP_MINUTES} min maximum (2 days)")

        # Should NOT have condition_constraint
        if "condition_constraint" in story:
            errors.append(f"{logic_type} should NOT have condition_constraint (that's for If-Then only)")

        # Should NOT have If-Then time gaps
        for bad_key in ["time_gap_rule_to_condition", "time_gap_condition_to_trigger"]:
            if bad_key in story:
                errors.append(f"{logic_type} should NOT have {bad_key} (use time_gap_between_constraint_and_trigger)")

    # ── 4. gold_labels validation ───────────────────────────────────────────
    if "positive_gold" not in gl:
        errors.append("Missing gold_labels.positive_gold")
    if "negative_gold" not in gl:
        errors.append("Missing gold_labels.negative_gold")

    # Positive gold
    pg = gl.get("positive_gold", {})
    if "intent" not in pg:
        errors.append("Missing gold_labels.positive_gold.intent")
    elif len(pg["intent"].strip()) < 10:
        errors.append("positive_gold.intent is too short")

    if "slots" not in pg:
        errors.append("Missing gold_labels.positive_gold.slots")
    else:
        slots = pg["slots"]
        if "required_slots" not in slots:
            errors.append("Missing positive_gold.slots.required_slots")
        elif not isinstance(slots["required_slots"], list) or len(slots["required_slots"]) < 1:
            errors.append("positive_gold.slots.required_slots must be a non-empty list (1-3 slots)")
        if "slot_values" not in slots:
            errors.append("Missing positive_gold.slots.slot_values")
        elif not isinstance(slots["slot_values"], dict) or len(slots["slot_values"]) < 1:
            errors.append("positive_gold.slots.slot_values must be a non-empty dict")

        if (isinstance(slots.get("required_slots"), list)
                and isinstance(slots.get("slot_values"), dict)):
            if len(slots["required_slots"]) > 3:
                errors.append(
                    f"Too many slots: {len(slots['required_slots'])} (max 3)")
            for s in slots["required_slots"]:
                if s not in slots["slot_values"]:
                    errors.append(f"required_slot '{s}' missing from slot_values")
                elif not isinstance(slots["slot_values"][s], list):
                    errors.append(f"slot_values['{s}'] must be a list")
                elif len(slots["slot_values"][s]) < 2:
                    errors.append(f"slot_values['{s}'] needs >= 2 aliases")

            # Grounding check: each slot ≥1 alias in constraint(s)
            for s in slots["required_slots"]:
                aliases = slots.get("slot_values", {}).get(s, [])
                if isinstance(aliases, list) and len(aliases) >= 1:
                    if not any(
                        a.lower() in gt for a in aliases for gt in grounding_texts
                    ):
                        errors.append(
                            f"slot '{s}': none of {aliases} appears in "
                            f"rule_constraint / condition_constraint (grounding rule violated)"
                        )

    # Negative gold — must have empty slots
    ng = gl.get("negative_gold", {})
    if "intent" not in ng:
        errors.append("Missing gold_labels.negative_gold.intent")
    elif len(ng["intent"].strip()) < 10:
        errors.append("negative_gold.intent is too short")
    if "slots" not in ng:
        errors.append("Missing gold_labels.negative_gold.slots")
    else:
        ng_slots = ng["slots"]
        if ng_slots.get("required_slots") and len(ng_slots["required_slots"]) > 0:
            errors.append(
                f"negative_gold.slots.required_slots must be empty [], "
                f"got {ng_slots['required_slots']}"
            )
        if ng_slots.get("slot_values") and len(ng_slots["slot_values"]) > 0:
            errors.append(
                f"negative_gold.slots.slot_values must be empty {{}}, "
                f"got {ng_slots['slot_values']}"
            )

    return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Generate Logical Adherence blueprints")
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--k-per-combo", type=int, default=None,
                    help="Override item count per (domain × logic) combo (default from COMBO_ITEM_COUNT)")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to existing output JSON to resume from")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    random.Random(args.seed)

    if args.k_per_combo is not None:
        for key in COMBO_ITEM_COUNT:
            COMBO_ITEM_COUNT[key] = args.k_per_combo

    def call_llm(sys_p, usr_p):
        return complete(sys_p, usr_p, model=args.model, temperature=args.temperature)

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else DIM_DIR / "outputs" / "stage1_blueprints"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"blueprints_logical_adherence_{args.model}.json"

    all_items: list[dict] = []
    per_combo_history: dict[str, list] = {}
    for d in CONDITION_TRIGGER_DOMAINS:
        for lt in LOGIC_TYPES:
            per_combo_history[f"{d['domain']}__{lt}"] = []

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as f:
                all_items = json.load(f)
            for item in all_items:
                meta = item.get("meta", {})
                combo_key = f"{meta.get('domain', '')}__{meta.get('logic_type', '')}"
                if combo_key in per_combo_history:
                    per_combo_history[combo_key].append(item)
            log.info("Resumed %d items from %s", len(all_items), resume_path)

    global_idx = len(all_items)

    total_target = sum(COMBO_ITEM_COUNT.values())
    log.info("Target: %d blueprints across %d combos (7 domains × 3 logic types)",
             total_target, len(COMBO_ITEM_COUNT))
    for ck in sorted(COMBO_ITEM_COUNT):
        log.info("  %-50s  %d", ck, COMBO_ITEM_COUNT[ck])
    log.info("Model=%s, T=%.2f, max_retries=%d",
             args.model, args.temperature, args.max_retries)

    combo_counter = 0
    total_combos = len(CONDITION_TRIGGER_DOMAINS) * len(LOGIC_TYPES)

    for domain_info in CONDITION_TRIGGER_DOMAINS:
        domain = domain_info["domain"]

        for logic_type in LOGIC_TYPES:
            if args.sleep:
                time.sleep(args.sleep)
            combo_counter += 1
            combo_key = f"{domain}__{logic_type}"
            history = per_combo_history[combo_key]
            k_target = COMBO_ITEM_COUNT.get(combo_key, 0)

            if k_target == 0:
                log.info(
                    "L1×L2 [%d/%d]: %s — target=0, skipping",
                    combo_counter, total_combos, combo_key,
                )
                continue

            already_done = len(history)
            if already_done >= k_target:
                log.info(
                    "L1×L2 [%d/%d]: %s — already has %d/%d, skipping",
                    combo_counter, total_combos,
                    combo_key, already_done, k_target,
                )
                continue

            log.info("=" * 70)
            log.info(
                "L1×L2 [%d/%d]: %s  (history=%d, generating %d more)",
                combo_counter, total_combos,
                combo_key, already_done, k_target - already_done,
            )

            seed_examples = select_seed_examples_for_logic_type(logic_type)

            for k_idx in range(already_done, k_target):
                sample_tag = f"{combo_key}__#{k_idx + 1}"
                log.info("  Generating %s ...", sample_tag)

                user_prompt = build_user_prompt(domain_info, logic_type, seed_examples, history)
                success = False

                for retry in range(args.max_retries):
                    try:
                        raw = call_llm(SYSTEM_PROMPT, user_prompt)
                        item = extract_json_object(raw)

                        if "story" in item and "trigger_turns" in item["story"] and "trigger" not in item["story"]:
                            item["story"]["trigger"] = item["story"].pop("trigger_turns")

                        valid, errs = validate_blueprint(item, logic_type)

                        if valid:
                            global_idx += 1
                            item["task_id"] = f"logical_adherence_{global_idx:03d}"
                            item["meta"] = {
                                "domain": domain,
                                "logic_type": logic_type,
                                "trigger_type": "conditional",
                            }

                            all_items.append(item)
                            history.append(item)
                            log.info(
                                "  ✓ OK (attempt %d/%d)  task_id=%s",
                                retry + 1, args.max_retries, item["task_id"],
                            )
                            success = True
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
                                        domain_info, logic_type,
                                        seed_examples, history,
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
        key = f"{meta.get('domain', '?')}__{meta.get('logic_type', '?')}"
        stats[key] = stats.get(key, 0) + 1
    log.info("Per-combo counts:")
    for key in sorted(stats):
        log.info("  %-50s  %d", key, stats[key])


if __name__ == "__main__":
    main()

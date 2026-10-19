#!/usr/bin/env python3
"""Stage 1 — Blueprint design for the Attention Recovery dimension.

Generates "interruption" scenario blueprints where:
  - A user is in the middle of Task A (suspended)
  - An external cognitive hijack (Task B) interrupts via a bridge prefix
  - After Task B concludes, the user announces it's done (trigger)
  - A smart agent should proactively remind the user to resume Task A

Loop structure:
  L1: 4 domains (Work/Professional, Home/Domestic, Finance/Commerce, Study/Academic)
    L2: 2 payload types (math, code)
      → K items per (domain × payload_type) combo

Total = 4 domains × 2 payload_types × K_PER_COMBO
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
from seeds.blueprints import ATTENTION_RECOVERY_SEEDS

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
#  Domains
# ═══════════════════════════════════════════════════════════════════════════

DOMAINS = [
    {
        "name": "Work/Professional",
        "description": (
            "The user is performing a work-related task: filling in reports, "
            "preparing presentations, reviewing documents, organizing files, "
            "drafting emails, scheduling meetings, handling administrative "
            "paperwork, etc."
        ),
    },
    {
        "name": "Home/Domestic",
        "description": (
            "The user is performing a household task: cooking, cleaning, "
            "organizing closets, assembling furniture, doing laundry, "
            "gardening, packing for a trip, or other domestic chores."
        ),
    },
    {
        "name": "Finance/Commerce",
        "description": (
            "The user is managing personal finances or shopping: reconciling "
            "bank statements, comparing insurance policies, tracking expenses, "
            "filing taxes, researching purchases, applying for services, "
            "handling online orders, etc."
        ),
    },
    {
        "name": "Study/Academic",
        "description": (
            "The user is engaged in academic work: reading textbooks, writing "
            "essays, preparing for exams, doing research, taking notes, "
            "working on a thesis, planning a presentation, etc. NOTE: The "
            "academic task must NOT involve math, physics, chemistry, biology, "
            "coding, or logic puzzles (since those overlap with the payload "
            "domains)."
        ),
    },
]

# ═══════════════════════════════════════════════════════════════════════════
#  Bridge Prefix Variants (3 per payload type)
# ═══════════════════════════════════════════════════════════════════════════

# Bridge prefixes include a short pause-and-resume sentence (paper eval set:
# interrupt-explicit-blueprint). It does not name Task A.
BRIDGE_VARIANTS = {
    "math": [
        {
            "bridge": (
                "Wait a second — my cousin who's in high school just texted me "
                "for help. He says there's a math competition problem he's "
                "completely stuck on. He sent me the problem. "
                "Let me pause what I was doing — I'll pick it back up after we "
                "help him out. Can you help me solve this problem:\n"
            ),
            "resolution_cue": "explained the math problem to him",
        },
        {
            "bridge": (
                "Hold on — my sister just sent me a message asking for help "
                "with a difficult math problem she's been struggling with for "
                "hours. She really needs it solved tonight. "
                "I'll set aside what I was in the middle of — can finish it up "
                "after this. Please help me work through this:\n"
            ),
            "resolution_cue": "helped my sister figure out that math problem",
        },
        {
            "bridge": (
                "One moment — a student just emailed me urgently about a "
                "tricky math puzzle that they can't "
                "crack. They forwarded it to me. "
                "Let me step away from what I was doing — I'll circle back to it "
                "once this is sorted. Can you help me solve it:\n"
            ),
            "resolution_cue": "solved that math problem for my student",
        },
    ],
    "code": [
        {
            "bridge": (
                "Something urgent just came up! A friend ran into a coding "
                "problem and needs me to take a look right away. "
                "I'll shelve what I was working on for now — I'll get back to it "
                "after we handle this. Can you help me figure this out:\n"
            ),
            "resolution_cue": "solved the coding problem",
        },
        {
            "bridge": (
                "Wait — my coworker just pinged me with a critical code issue they "
                "can't solve. They need someone to look at it immediately before "
                "their deployment deadline. "
                "Let me drop what I was doing — I'll resume it once we've fixed "
                "this. Help me solve this:\n"
            ),
            "resolution_cue": "fixed that code issue for my coworker",
        },
        {
            "bridge": (
                "Hold on a second — a classmate just sent me a programming "
                "challenge they're stuck on. They have a submission deadline "
                "tonight and really need help. "
                "I need to pause what I was in the middle of — I'll get back to "
                "it once we're done here. Can you take a look at this:\n"
            ),
            "resolution_cue": "helped my classmate with that coding challenge",
        },
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
#  Time Gap Sampling
# ═══════════════════════════════════════════════════════════════════════════

def sample_time_gaps(rng: random.Random) -> tuple[str, str]:
    """Sample time gaps for a blueprint.

    - time_gap_between_task_a_and_interruption: uniform 2-6 minutes
    - time_gap_between_interruption_and_trigger:
        70% probability → uniform 60-90 minutes
        30% probability → uniform 40-60 minutes

    Returns:
        (gap_a_to_interrupt, gap_interrupt_to_trigger) as human-readable strings.
    """
    gap_a = rng.randint(2, 6)

    if rng.random() < 0.7:
        gap_trigger = rng.randint(60, 90)
    else:
        gap_trigger = rng.randint(40, 60)

    return f"{gap_a} minutes", f"{gap_trigger} minutes"


# ═══════════════════════════════════════════════════════════════════════════
#  Per-Combo Item Count
# ═══════════════════════════════════════════════════════════════════════════

K_PER_COMBO = 8  # 4 domains × 2 payload_types × 8 = 64 total
PAYLOAD_TYPES = ["math", "code"]

COMBO_ITEM_COUNT: dict[str, int] = {}
for d in DOMAINS:
    for pt in PAYLOAD_TYPES:
        key = f"{d['name']}__{pt}"
        COMBO_ITEM_COUNT[key] = K_PER_COMBO

# ═══════════════════════════════════════════════════════════════════════════
#  Forbidden keywords — Task A must NOT overlap with payload domains
# ═══════════════════════════════════════════════════════════════════════════

# If task_a_start mentions these, it overlaps with the payload datasets
PAYLOAD_DOMAIN_KEYWORDS = [
    # math
    "math", "equation", "algebra", "calculus", "geometry", "theorem",
    "integral", "derivative", "polynomial", "trigonometry", "proof",
    "mathematical", "arithmetic",
    # code
    "code", "coding", "programming", "debug", "algorithm", "function",
    "variable", "compile", "software", "python", "java", "javascript",
    "script", "API", "bug", "git", "repository",
    # physics
    "physics", "force", "velocity", "acceleration", "quantum",
    "thermodynamics", "momentum", "gravity", "electromagnetic",
    # chemistry
    "chemistry", "molecule", "atom", "reaction", "compound",
    "chemical", "element", "periodic table", "bond",
    # biology
    "biology", "cell", "DNA", "gene", "organism", "evolution",
    "protein", "enzyme", "biological",
    # puzzle/logic
    "puzzle", "riddle", "logic puzzle", "brain teaser",
]

# Explicit reminder language (must NOT appear in task_a_start)
EXPLICIT_KEYWORDS = [
    "remind me", "warn me", "alert me", "notify me",
    "don't let me forget", "do not let me forget",
    "take a note", "set a reminder", "set an alarm",
    "make sure i", "remember to", "todo",
    "don't forget", "do not forget", "keep in mind",
    "heads up later", "ping me", "flag this",
    "note to self",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Seed Example Selection
# ═══════════════════════════════════════════════════════════════════════════

def select_seed_examples(domain: str, payload_type: str) -> list[dict]:
    """Return seed examples matching the given domain × payload_type.

    Falls back to same-domain seeds, then to any 2 seeds.
    """
    # Exact match
    exact = [
        s for s in ATTENTION_RECOVERY_SEEDS
        if s.get("meta", {}).get("domain") == domain
        and s.get("meta", {}).get("payload_type") == payload_type
    ]
    if exact:
        return exact

    # Same domain
    domain_only = [
        s for s in ATTENTION_RECOVERY_SEEDS
        if s.get("meta", {}).get("domain") == domain
    ]
    if domain_only:
        return domain_only[:2]

    # Same payload_type
    pt_only = [
        s for s in ATTENTION_RECOVERY_SEEDS
        if s.get("meta", {}).get("payload_type") == payload_type
    ]
    if pt_only:
        return pt_only[:2]

    return ATTENTION_RECOVERY_SEEDS[:2]


def strip_seed_for_prompt(seed: dict) -> dict:
    """Remove task_id, meta, interruption_bridge_prefix, and time_gap fields from seed for prompt display."""
    out = deepcopy(seed)
    out.pop("task_id", None)
    out.pop("meta", None)
    story = out.get("story", {})
    story.pop("interruption_bridge_prefix", None)
    story.pop("time_gap_between_task_a_and_interruption", None)
    story.pop("time_gap_between_interruption_and_trigger", None)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  History Formatting
# ═══════════════════════════════════════════════════════════════════════════

def format_history_compact(items: list) -> str:
    """Build concise history string for dedup context."""
    if not items:
        return "(none yet)"
    parts = []
    for i, item in enumerate(items, 1):
        story = item.get("story", {})
        ta = story.get("task_a_start", "")
        trig = story.get("trigger", {})
        clean = trig.get("clean", "") if isinstance(trig, dict) else str(trig)
        overloaded = trig.get("overloaded", "") if isinstance(trig, dict) else ""
        parts.append(
            f"  --- #{i} ---\n"
            f"  Task A: {ta}\n"
            f"  Trigger (clean): {clean}\n"
            f"  Trigger (overloaded): {overloaded}"
        )
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt Construction
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert benchmark-data creator for **TriggerBench**, a benchmark that evaluates \
whether LLM agents can *proactively* remind users about suspended tasks after a long \
cognitive interruption.

This is the Attention Recovery dimension (also called Interrupt-Trigger).

Your job: generate ONE new "interrupt-trigger" blueprint as a single JSON object.

────────────────────────────────────────────────────
CORE CONCEPT — Dual-Task Interference & Implicit Prospective Memory
────────────────────────────────────────────────────
The Interrupt-Trigger paradigm tests whether an LLM agent can recall a **suspended Task A** \
after being cognitively "hijacked" by a very long external reasoning task (Task B). The flow is:

  Turn 1 (User): States they are in the middle of Task A, which gets suspended.
  Turn 2 (User): Introduces an urgent external interruption (Task B) — a math/code problem \
someone else needs help with. The bridge prefix introduces this naturally.
  Turn 2 (Asst): Solves the math/code problem (very long reasoning — thousands of tokens).
  Turn 3 (User): Announces Task B is resolved naturally and fluently (the "trigger").
  → The agent should PROACTIVELY remind the user to resume Task A.

The key challenge: Task B's extreme length (10,000–35,000 tokens of dense reasoning) should \
displace the agent's attention from Task A. A good agent "wakes up" and reminds the user.

────────────────────────────────────────────────────
WHAT YOU GENERATE
────────────────────────────────────────────────────
You generate ONLY:
  1. story.background — Brief scene setting (1-2 sentences)
  2. story.task_a_start — The user's in-progress task that gets suspended mid-way
  3. story.trigger.clean — User's statement that Task B is done (contains resolution_cue semantics)
  4. story.trigger.overloaded — Same as clean + user immediately starts a NEW unrelated task
  5. gold_labels — What the agent should remind about

You do NOT generate: task_id, meta, or interruption_bridge_prefix (those are added by code).

────────────────────────────────────────────────────
ABSOLUTE RULES (violations = rejection)
────────────────────────────────────────────────────
1. **TASK A MUST BE IN-PROGRESS AND SUSPENDED**
   task_a_start must describe a task the user is CURRENTLY doing, with a clear indication \
of progress (e.g., "I just finished X, and I'm about to do Y"). The task is interrupted \
mid-stream — the user hasn't completed the next step yet.

2. **TASK A MUST BE ORTHOGONAL TO OPENTHOUGHTS DOMAINS**
   task_a_start must NEVER involve: mathematics, coding/programming, physics, chemistry, \
biology, or logic puzzles. These are the payload domains of the interrupting Task B. \
Task A should be about completely different activities (e.g., cooking, writing essays, \
organizing files, financial tasks, cleaning, shopping, administrative work).

3. **TRIGGER MUST CONTAIN RESOLUTION CUE SEMANTICS**
   trigger.clean must signal that the external interruption (Task B) is finished. It should \
reference the resolution_cue I provide (e.g., "solved the math problem", "fixed the code"). \
It must NOT contain any keywords from task_a_start.

4. **TRIGGER MUST NOT LEAK TASK A KEYWORDS**
   trigger.clean and trigger.overloaded must have ZERO keyword overlap with task_a_start. \
The user does NOT mention Task A in the trigger — they've forgotten about it!

5. **OVERLOADED TRIGGER ADDS A NEW UNRELATED TASK**
   trigger.overloaded = trigger.clean + a brand new request or a new task the user plans to start, which is:
   - Unrelated to both Task A and Task B
   - Something the agent can help with (recommendation, search, drafting, etc.)
   - **Context-free**: Must be answerable or replied by LLM parametric knowledge only (e.g., recipe, \
currency conversion, trivia, translation, general knowledge, ask for suggestions, email/message drafting etc.). No external tools, private \
data, or real-time information needed. Make it diverse, interesting, and conversational. \
    For example, "By the way, I want to cook some lasagna. Can you recommend a good recipe?", \
        "Set an alarm to remind me to buy printer ink tomorrow morning. Can you recommend a good brand?", \
        "Help me draft a quick text to my landlord asking about the broken heater.", \
        "I plan to start writing my essay for the Economics class now. Can you give me a brief summary of the concept of opportunity cost that I can include in my introduction?"
   This makes recall harder because the agent now has a competing new request.

6. **SLOT GROUNDING RULE**
   - Define 1-2 slots in gold_labels.slots.required_slots
   - Each slot must have a list of 2-4 aliases in slot_values
   - For EACH slot, at least ONE alias must appear as an exact substring (case-insensitive) \
in task_a_start
   - Slot values capture what the agent should remind about (the suspended object/action)

7. **NO EXPLICIT REMINDER LANGUAGE IN task_a_start**
   task_a_start must NOT contain: "remind me", "don't forget", "make sure I", "remember to", \
"set a reminder", etc. The task is just naturally stated — the user expects to continue it.

8. **DIVERSITY WITHIN COMBO**
   Each call targets a specific domain. Vary the specific tasks, objects, and scenarios. \
Do NOT reuse patterns from the seed examples or previously generated items in the history.

────────────────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────────────────
Return ONLY a single, valid JSON object. No markdown fences, no comments, no extra text.\
"""


def build_user_prompt(
    domain_info: dict,
    payload_type: str,
    bridge_variant: dict,
    seed_examples: list[dict],
    history: list,
    time_gaps: tuple[str, str] | None = None,
) -> str:
    """Assemble the user prompt for one generation call."""

    domain = domain_info["name"]
    domain_desc = domain_info["description"]
    resolution_cue = bridge_variant["resolution_cue"]

    # Format seed examples (without meta/bridge)
    seed_texts = []
    for i, seed in enumerate(seed_examples, 1):
        stripped = strip_seed_for_prompt(seed)
        seed_texts.append(
            f"--- Example {i} ---\n"
            f"{json.dumps(stripped, indent=2, ensure_ascii=False)}"
        )
    seeds_block = "\n\n".join(seed_texts)

    history_text = format_history_compact(history)

    # Time gap context (sampled by code, provided as reference)
    time_gap_section = ""
    if time_gaps:
        gap_a, gap_trigger = time_gaps
        time_gap_section = f"""

═══ TIME CONTEXT (for reference — these are added by code, NOT generated by you) ═══
In this scenario:
- The interruption arrives ~{gap_a} after the user states Task A.
- The interruption (Task B) lasts ~{gap_trigger} before the user announces it's done.
Keep this temporal context in mind when crafting the scenario — the user has been away \
from Task A for {gap_trigger}, which is long enough to plausibly forget about it."""

    prompt = f"""\
Generate ONE new interrupt-trigger blueprint for:
  - domain = {domain}
  - payload_type = {payload_type}
  - resolution_cue = "{resolution_cue}"

═══ DOMAIN DESCRIPTION ═══
{domain_desc}

═══ WHAT IS THE INTERRUPTION? ═══
The user will be interrupted by an urgent {payload_type} problem that someone else needs help \
with. The bridge prefix (provided separately, NOT generated by you) introduces this interruption \
naturally. After the agent solves the {payload_type} problem (thousands of tokens of reasoning), \
the user announces it's done — that's the "trigger".

Your trigger.clean MUST semantically reference: "{resolution_cue}"
(Paraphrase it naturally — don't copy it verbatim, but convey the same meaning.)

═══ CRITICAL CONSTRAINT: TASK A vs PAYLOAD DOMAINS ═══
task_a_start must NEVER involve ANY of these topics (they are the interrupting payload domains):
- Mathematics, arithmetic, equations, proofs, geometry
- Coding, programming, debugging, algorithms, software development
- Physics, mechanics, thermodynamics, quantum
- Chemistry, molecules, reactions, compounds
- Biology, cells, genetics, organisms
- Logic puzzles, riddles, brain teasers

Good Task A examples for "{domain}": everyday activities like writing reports, organizing \
files, cooking, cleaning, shopping, scheduling, reading non-STEM material, etc.

═══ SEED EXAMPLES (for format & quality — do NOT duplicate) ═══
{seeds_block}

═══ ALREADY GENERATED FOR THIS COMBO (avoid repetition) ═══
{history_text}

═══ REQUIRED OUTPUT JSON SCHEMA ═══
{{
  "story": {{
    "background": "<1-2 sentence scene setting>",
    "task_a_start": "<User's in-progress task that gets suspended. Must show progress \
('I just finished X, and I'm about to Y'). Must NOT relate to math/code/physics/chemistry/\
biology/puzzles. Must NOT contain reminder language.>",
    "trigger": {{
      "clean": "<User announces Task B is done. Must semantically reference the resolution_cue. \
Must NOT contain any keywords from task_a_start.>",
      "overloaded": "<Same as clean + user immediately starts a new unrelated request. \
The new request must be context-free: answerable by LLM parametric knowledge only. No private data or real-time info needed.>"
    }}
  }},
  "gold_labels": {{
    "intent": "<What the agent should proactively remind the user about — resuming Task A.>",
    "slots": {{
      "required_slots": ["<slot_name_1>", "<slot_name_2>"],
      "slot_values": {{
        "<slot_name_1>": ["alias1", "alias2"],
        "<slot_name_2>": ["alias1", "alias2"]
      }}
    }}
  }}
}}

SLOT RULES:
- 1-2 slots, each with 2-4 aliases
- For EACH slot, at least ONE alias must literally appear in task_a_start (exact substring, case-insensitive)
- Slots should capture the key information the agent needs to remind about

Final reminders:
- task_a_start must be an in-progress task with clear next step
- task_a_start must be ORTHOGONAL to math/code/physics/chemistry/biology/puzzles
- trigger.clean references "{resolution_cue}" semantically
- trigger has ZERO keyword overlap with task_a_start
- Be creative — different tasks, objects, and scenarios from seeds & history
- trigger.overloaded adds a completely new, unrelated request or the user wants to start a new unrelated task after the clean trigger \
(must be context-free — answerable by LLM parametric knowledge only, no external tools needed). Make it diverse and distinct.\
"""
    # Append time context section if provided
    prompt += time_gap_section

    return prompt


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_blueprint(item: dict, payload_type: str, resolution_cue: str) -> tuple[bool, list[str]]:
    """Validate a generated interrupt-trigger blueprint. Returns (is_valid, errors)."""
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
    required_story_keys = ["background", "task_a_start", "trigger"]
    for key in required_story_keys:
        if key not in story:
            errors.append(f"Missing story.{key}")
    if errors:
        return False, errors

    # ── 3. task_a_start validation ──────────────────────────────────────────
    task_a = story.get("task_a_start", "").strip()
    if len(task_a) < 30:
        errors.append(f"task_a_start too short ({len(task_a)} chars)")

    # Check no payload domain keywords (word-boundary match to avoid
    # false positives like "evolution" inside "revolution")
    ta_lower = task_a.lower()
    for kw in PAYLOAD_DOMAIN_KEYWORDS:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, ta_lower):
            errors.append(
                f"task_a_start contains payload domain keyword: '{kw}' "
                f"(Task A must be orthogonal to math/code/physics/chemistry/biology/puzzle)"
            )
            break

    # Check no explicit reminder language
    for kw in EXPLICIT_KEYWORDS:
        if kw in ta_lower:
            errors.append(f"task_a_start contains forbidden explicit keyword: '{kw}'")
            break

    # ── 4. Trigger validation ───────────────────────────────────────────────
    trigger = story.get("trigger", {})
    if not isinstance(trigger, dict):
        errors.append("story.trigger must be a dict with 'clean' and 'overloaded' keys")
        return False, errors

    clean = trigger.get("clean", "").strip()
    overloaded = trigger.get("overloaded", "").strip()

    if len(clean) < 15:
        errors.append(f"trigger.clean too short ({len(clean)} chars)")
    if len(overloaded) < 20:
        errors.append(f"trigger.overloaded too short ({len(overloaded)} chars)")

    # trigger.clean must NOT contain task_a keywords (check significant words)
    if task_a and clean:
        # Extract significant words from task_a (>= 4 chars, not common stop words)
        stop_words = {
            "just", "about", "that", "this", "with", "from", "have", "been",
            "will", "would", "could", "should", "going", "getting", "doing",
            "being", "into", "onto", "they", "them", "their", "what", "when",
            "where", "which", "there", "here", "some", "other", "more", "most",
            "very", "really", "then", "than", "also", "only", "still", "already",
            "before", "after", "while", "during", "through", "between",
            "started", "finished", "ready", "need", "start",
        }
        task_a_words = set()
        for word in re.findall(r"[a-zA-Z]{4,}", task_a.lower()):
            if word not in stop_words:
                task_a_words.add(word)

        clean_lower = clean.lower()
        overlap = [w for w in task_a_words if w in clean_lower]
        if overlap:
            errors.append(
                f"trigger.clean shares keywords with task_a_start: {overlap[:5]}. "
                f"Trigger must have ZERO keyword overlap with Task A."
            )

    # ── 5. gold_labels validation ───────────────────────────────────────────
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
        elif len(slots["required_slots"]) > 2:
            errors.append(
                f"Too many slots: {len(slots['required_slots'])} (max 2)"
            )

        if "slot_values" not in slots:
            errors.append("Missing gold_labels.slots.slot_values")
        elif not isinstance(slots["slot_values"], dict) or len(slots["slot_values"]) < 1:
            errors.append("gold_labels.slots.slot_values must be a non-empty dict")

        # Cross-check: every required_slot must have aliases in slot_values
        if (isinstance(slots.get("required_slots"), list)
                and isinstance(slots.get("slot_values"), dict)):
            for s in slots["required_slots"]:
                if s not in slots["slot_values"]:
                    errors.append(f"required_slot '{s}' missing from slot_values")
                elif not isinstance(slots["slot_values"][s], list):
                    errors.append(f"slot_values['{s}'] must be a list")
                elif len(slots["slot_values"][s]) < 2:
                    errors.append(f"slot_values['{s}'] needs >= 2 aliases")

            # Grounding check: each slot must have ≥1 alias in task_a_start
            for s in slots["required_slots"]:
                aliases = slots.get("slot_values", {}).get(s, [])
                if isinstance(aliases, list) and len(aliases) >= 1:
                    if not any(a.lower() in ta_lower for a in aliases):
                        errors.append(
                            f"Slot '{s}': none of {aliases} appears as substring in "
                            f"task_a_start (grounding rule violated). At least one alias "
                            f"must be a case-insensitive substring of task_a_start."
                        )

    # ── 6. Background check ─────────────────────────────────────────────────
    bg = story.get("background", "").strip()
    if len(bg) < 10:
        errors.append("story.background is too short")

    return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Generate Attention Recovery blueprints"
    )
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--k-per-combo", type=int, default=None,
                    help="Override K_PER_COMBO (items per domain×payload combo)")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to existing output JSON to resume from")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Override K if provided
    k_per_combo = args.k_per_combo if args.k_per_combo else K_PER_COMBO
    combo_counts = {}
    for d in DOMAINS:
        for pt in PAYLOAD_TYPES:
            key = f"{d['name']}__{pt}"
            combo_counts[key] = k_per_combo

    def call_llm(sys_p, usr_p):
        return complete(sys_p, usr_p, model=args.model, temperature=args.temperature)

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else DIM_DIR / "outputs" / "stage1_blueprints"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"blueprints_attention_recovery_{args.model}.json"

    all_items: list[dict] = []
    per_combo_history: dict[str, list] = {k: [] for k in combo_counts}

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as f:
                all_items = json.load(f)
            for item in all_items:
                meta = item.get("meta", {})
                combo_key = f"{meta.get('domain', '')}__{meta.get('payload_type', '')}"
                if combo_key in per_combo_history:
                    per_combo_history[combo_key].append(item)
            log.info("Resumed %d items from %s", len(all_items), resume_path)

    global_idx = len(all_items)

    total_target = sum(combo_counts.values())
    log.info("Target: %d blueprints across %d combos (K=%d per combo)",
             total_target, len(combo_counts), k_per_combo)
    for ck, cv in sorted(combo_counts.items()):
        log.info("  %-40s  %d", ck, cv)
    log.info("Model=%s, T=%.2f, max_retries=%d",
             args.model, args.temperature, args.max_retries)

    # ═══════════════════════════════════════════════════════════════════════
    #  Generation loop:  L1 domain → L2 payload_type → K items
    # ═══════════════════════════════════════════════════════════════════════
    for domain_idx, domain_info in enumerate(DOMAINS):
        domain = domain_info["name"]

        for payload_type in PAYLOAD_TYPES:
            if args.sleep:
                time.sleep(args.sleep)
            combo_key = f"{domain}__{payload_type}"
            history = per_combo_history[combo_key]
            k_target = combo_counts[combo_key]

            # Skip combos already completed via resume
            already_done = len(history)
            if already_done >= k_target:
                log.info(
                    "L1 [%s × %s]: already has %d/%d, skipping",
                    domain, payload_type, already_done, k_target,
                )
                continue

            log.info("=" * 70)
            log.info(
                "L1 [%d/%d] domain=%s, payload=%s  (history=%d, generating %d more)",
                domain_idx + 1, len(DOMAINS),
                domain, payload_type, already_done, k_target - already_done,
            )

            for k_idx in range(already_done, k_target):
                sample_tag = f"{combo_key}__#{k_idx + 1}"
                log.info("  Generating %s ...", sample_tag)

                # Randomly select a bridge variant
                bridge_variant = rng.choice(BRIDGE_VARIANTS[payload_type])

                # Sample time gaps for this item
                time_gaps = sample_time_gaps(rng)

                # Pick matching seeds as ICL examples
                seed_examples = select_seed_examples(domain, payload_type)

                user_prompt = build_user_prompt(
                    domain_info, payload_type, bridge_variant,
                    seed_examples, history, time_gaps,
                )
                success = False

                for retry in range(args.max_retries):
                    try:
                        raw = call_llm(SYSTEM_PROMPT, user_prompt)
                        item = extract_json_object(raw)
                        valid, errs = validate_blueprint(
                            item, payload_type, bridge_variant["resolution_cue"]
                        )

                        if valid:
                            # Post-process: add task_id, meta, bridge prefix, time gaps
                            global_idx += 1
                            item["task_id"] = f"attention_recovery_{global_idx:03d}"
                            item["meta"] = {
                                "domain": domain,
                                "payload_type": payload_type,
                                "trigger_type": "interruption",
                            }
                            # Inject bridge prefix into story
                            item["story"]["interruption_bridge_prefix"] = bridge_variant["bridge"]
                            # Inject sampled time gaps into story
                            item["story"]["time_gap_between_task_a_and_interruption"] = time_gaps[0]
                            item["story"]["time_gap_between_interruption_and_trigger"] = time_gaps[1]

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
                                        domain_info, payload_type,
                                        bridge_variant, seed_examples, history,
                                        time_gaps,
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
        key = f"{meta.get('domain', '?')}__{meta.get('payload_type', '?')}"
        stats[key] = stats.get(key, 0) + 1
    log.info("Per-combo counts:")
    for key in sorted(stats):
        log.info("  %-40s  %d", key, stats[key])


if __name__ == "__main__":
    main()

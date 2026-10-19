#!/usr/bin/env python3
"""Stage 5 — Negative-Clean variant for State-Tracking.

Inserts a resolution user–assistant pair between the constraint and the trigger
so the latent risk is already resolved. The agent should remain silent.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from shared.json_utils import save_json
from shared.llm_client import chat
from shared.time_utils import add_minutes, parse_iso, time_diff_minutes
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
#  System Prompts
# ═══════════════════════════════════════════════════════════════════════════
# 1. It should sound like a natural impromptu interjection mid-conversation \
# 1. Keep it short, natural, fluent, and conversational \
# (e.g., "I met such a nice driver, he just..." or "Good news, I just..." \
# "Such a relief! I recalled ...", "Someone just told me ..."). 
# Make the pattern diverse. Avoid using "Oh, good catch -

SYSTEM_PROMPT_RESOLUTION_USER = """\
You are a rewriting assistant. Convert the following third-person description \
of a "crisis resolved" event into a first-person English sentence that a user \
would casually say to their AI assistant in a chat.

Requirements:
1. Keep it short, natural, fluent, and conversational.
2. Output plain text only — no quotes, no explanation.

CRITICAL TIME-LOGIC CONSTRAINT:
This utterance will be inserted in the MIDDLE of an ongoing conversation. \
The final key event (arriving at a destination, getting off, finishing an activity, \
etc.) is still some time away.
- If the resolution event implies imminent arrival/departure/ending, you MUST \
reframe it as a mid-journey incidental event or an early proactive precaution.
- Do NOT use phrases that imply the user is about to leave, arrive, or finish \
(e.g., "when I leave", "before I get out", "as I'm getting off", \
"before I head out").
- Instead, frame it as something that just happened naturally during the trip \
or wait (e.g., "the driver just mentioned don't forget...", "I just noticed...", \
"someone just told me don't forget...", "funny thing, I just...", "What a relief! I just recalled ...").
- Keep the key intent as it: the goal is to remind the user don't forget to do X, need to do X, or avoid doing X.

Examples:
- Event: "The conductor reminded him to take his luggage before getting off."
  Good: "The conductor just walked by and kindly reminded \
me not to forget my luggage later!"
  Bad: "The conductor reminded me to grab my luggage before I get off." \
(implies imminent departure)
- Event: "She smelled vinegar from the kitchen before leaving the house."
  Good: "I just caught a whiff of vinegar from the kitchen — let me go \
check the stove real quick."
  Bad: "Before I head out the door I noticed the vinegar smell." \
(implies imminent leaving)\
"""

SYSTEM_PROMPT_RESOLUTION_ASSISTANT = """\
You are a friendly, natural AI assistant.

The user just shared some good news or mentioned that they happened to \
resolve a minor concern (e.g., avoided losing an item, dodged a safety issue).

Based on the conversation context so far, respond with 1-2 sentences in a \
very natural way. Do NOT over-elaborate, and absolutely do NOT give unsolicited \
follow-up safety warnings or reminders.

CRITICAL persona requirement: You assume the user is a fully capable adult. \
Unless the user EXPLICITLY asks you for safety advice or EXPLICITLY requests \
you to set a reminder, you must NEVER proactively offer unsolicited advice, \
NEVER proactively warn about potential risks, and NEVER say things like \
"remember to do X". Maintain proper boundaries.

IMPORTANT TIME-AWARENESS: The conversation is still ongoing — the user is NOT \
about to leave, arrive, or finish their current activity anytime soon. Keep \
your response casual and mid-conversation in tone. Do NOT imply the user is \
about to depart, arrive, or wrap up (avoid phrases like "before you head out", \
"when you get there", "perfect timing" in a departure context).

Requirements:
1. Show natural relief, appreciation, or agreement (e.g., "That was so nice \
of them!", "Woo! It's nice! You won't forget that!", "That's great! Glad you resolved that!", "Good, glad you caught that!", "Ha, that's great!").
2. Keep it very brief (15-30 words). Do NOT summarize the entire conversation.
3. Smoothly follow the context, like a human friend wrapping up the topic.\
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Blacklist (reused from Stage 3 to check assistant reply)
# ═══════════════════════════════════════════════════════════════════════════

BLACKLIST_PATTERNS = [
    r"remind you",
    r"remind me",
    r"i('ll| will) remind",
    r"don'?t forget",
    r"do not forget",
    r"keep in mind",
    r"make sure (to|you)",
    r"just a heads[\s-]?up",
    r"by the way.{0,20}(earlier|before|mentioned|said)",
    r"heads[\s-]?up",
    r"be careful",
    r"be aware",
    r"warning",
    r"alert you",
    r"notify you",
    r"i('ll| will) let you know",
    r"set (a |an )?(reminder|alarm|timer)",
    r"before you (start|begin|go)",
    r"you (might|may) want to.{0,30}(first|before)",
    r"you still (need|have) to",
    r"you should.{0,20}(before|first)",
    r"earlier you (said|mentioned|told)",
    r"remember to",
]

_BLACKLIST_RE = [re.compile(p, re.IGNORECASE) for p in BLACKLIST_PATTERNS]


def check_blacklist(text: str) -> list:
    """Return list of matched blacklist patterns (empty = clean)."""
    return [p for p, rx in zip(BLACKLIST_PATTERNS, _BLACKLIST_RE) if rx.search(text)]


# Timestamp helpers live in shared.time_utils


# ═══════════════════════════════════════════════════════════════════════════
#  Sanity check: all turns in strictly non-decreasing time order
# ═══════════════════════════════════════════════════════════════════════════

def validate_time_order(turns: list) -> tuple:
    """Check all turns have non-decreasing timestamps.

    Returns (is_valid, error_msg).
    """
    for i in range(1, len(turns)):
        t_prev = parse_iso(turns[i - 1]["time"])
        t_curr = parse_iso(turns[i]["time"])
        if t_prev is None or t_curr is None or t_curr < t_prev:
            return False, (
                "Time regression at turn_id {}: {} < {} (turn_id {})"
                .format(turns[i]["turn_id"], turns[i]["time"],
                        turns[i - 1]["time"], turns[i - 1]["turn_id"])
            )
    return True, "OK"


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1: Find insertion point
# ═══════════════════════════════════════════════════════════════════════════

def find_insertion_point(turns: list, dialogue_id: str) -> dict:
    """Find insertion point using weighted random selection biased toward trigger.

    Strategy:
      1. Collect all valid thematic_connection assistant reply slots between
         constraint and trigger where the subsequent gap >= 3 min.
      2. If >= 3 valid slots, exclude the FIRST (adjacent to constraint) and
         LAST (adjacent to trigger) to avoid placing resolution right next
         to either boundary.
      3. Among remaining slots, assign linearly increasing weights toward
         trigger: [1, 2, ..., N]. This makes positions closer to trigger
         more likely but not dominant.
      4. Use dialogue_id as random seed for reproducibility (resume-safe).
      5. Fallback: any assistant→user gap >= 3 → Branch A.
      6. If no gap >= 3 exists → Branch B (force insert + time shift).

    Returns a dict with:
      - insert_after_idx: index in turns[] after which to insert
      - new_user_time: str
      - new_assistant_time: str
      - branch: "A" or "B"
      - shift_from_idx: index from which to shift timestamps (Branch B only)
    """
    # Find constraint assistant reply index and trigger index
    constraint_ast_idx = None
    trigger_idx = None

    for i, t in enumerate(turns):
        if t["function"] == "constraint" and t["role"] == "user":
            if i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
                constraint_ast_idx = i + 1
        if t["function"] == "trigger":
            trigger_idx = i

    if constraint_ast_idx is None or trigger_idx is None:
        return None

    # ── Collect all valid thematic slots with gap >= 3 ──────────────────
    valid_slots = []
    for i in range(constraint_ast_idx + 1, trigger_idx - 1):
        if (turns[i]["role"] == "user"
                and turns[i]["function"] == "thematic_connection"
                and i + 1 < trigger_idx
                and turns[i + 1]["role"] == "assistant"):
            ast_idx = i + 1
            if ast_idx + 1 < len(turns):
                gap = time_diff_minutes(
                    turns[ast_idx]["time"], turns[ast_idx + 1]["time"]
                )
                if gap >= 3:
                    valid_slots.append(ast_idx)

    if valid_slots:
        # Exclude first (near constraint) and last (near trigger) if >= 3
        candidates = valid_slots
        if len(valid_slots) >= 3:
            candidates = valid_slots[1:-1]

        # Weighted random: linearly increasing toward trigger
        n = len(candidates)
        weights = list(range(1, n + 1))

        # Deterministic seed from dialogue_id for reproducibility
        seed = int(hashlib.md5(dialogue_id.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        chosen = rng.choices(candidates, weights=weights, k=1)[0]

        return {
            "insert_after_idx": chosen,
            "new_user_time": add_minutes(turns[chosen]["time"], 1),
            "new_assistant_time": add_minutes(turns[chosen]["time"], 2),
            "branch": "A",
            "shift_from_idx": None,
        }

    # ── Fallback: any assistant→user gap >= 3 (Branch A) ───────────────
    best_gap = -1
    best_ast_idx = None

    for i in range(constraint_ast_idx, trigger_idx - 1):
        if turns[i]["role"] == "assistant" and turns[i + 1]["role"] == "user":
            gap = time_diff_minutes(turns[i]["time"], turns[i + 1]["time"])
            if gap > best_gap:
                best_gap = gap
                best_ast_idx = i

    if best_ast_idx is not None and best_gap >= 3:
        return {
            "insert_after_idx": best_ast_idx,
            "new_user_time": add_minutes(turns[best_ast_idx]["time"], 1),
            "new_assistant_time": add_minutes(turns[best_ast_idx]["time"], 2),
            "branch": "A",
            "shift_from_idx": None,
        }

    # ── Branch B: force insert + time shift ────────────────────────────
    first_thematic = None
    for i in range(constraint_ast_idx + 1, trigger_idx - 1):
        if (turns[i]["role"] == "user"
                and turns[i]["function"] == "thematic_connection"
                and i + 1 < trigger_idx
                and turns[i + 1]["role"] == "assistant"):
            first_thematic = i + 1
            break

    insert_idx = first_thematic if first_thematic is not None else constraint_ast_idx

    return {
        "insert_after_idx": insert_idx,
        "new_user_time": add_minutes(turns[insert_idx]["time"], 1),
        "new_assistant_time": add_minutes(turns[insert_idx]["time"], 2),
        "branch": "B",
        "shift_from_idx": insert_idx + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2: Generate User Resolution Utterance
# ═══════════════════════════════════════════════════════════════════════════

def generate_resolution_user(
    resolution_description: str,
    model: str,
    temperature: float,
    constraint_text: str = None,
    trigger_time: str = None,
    trigger_text: str = None,
    resolution_time: str = None,
    max_retries: int = 5,
) -> str:
    """Generate a first-person user utterance from the resolution description."""
    user_content = "Resolution Event: {}".format(resolution_description)

    if constraint_text:
        user_content += (
            "\n\nProblem context (the user previously said this — "
            "do NOT repeat it, but understand what problem is being resolved):\n"
            "\"{}\""
        ).format(constraint_text)

    if trigger_time and trigger_text and resolution_time:
        delta = int(time_diff_minutes(resolution_time, trigger_time))
        user_content += (
            "\n\nTiming context (CRITICAL — read carefully, but do NOT "
            "mention any of these details in your output):\n"
            "- Current chat time: {}.\n"
            "- The trigger event happens at {} ({} minutes from now): "
            "\"{}\"\n"
            "- Since the trigger is still {} minutes away, the resolution "
            "MUST sound like a mid-journey incidental event, NOT something "
            "happening right before arrival/departure.\n"
            "- Do NOT reference or reveal the trigger event content."
        ).format(resolution_time, trigger_time, delta, trigger_text, delta)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_RESOLUTION_USER},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            raw = chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=256,
            )
            text = raw.strip()
            # Strip wrapping quotes
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                text = text[1:-1].strip()

            if len(text) < 10:
                log.warning("  Resolution user attempt %d/%d: too short (%d chars)",
                            attempt, max_retries, len(text))
                continue
            if len(text) > 500:
                log.warning("  Resolution user attempt %d/%d: too long (%d chars)",
                            attempt, max_retries, len(text))
                continue
            return text
        except Exception as e:
            log.warning("  Resolution user attempt %d/%d API error: %s",
                        attempt, max_retries, e)
            time_module.sleep(2)

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3: Generate Assistant Resolution Reply (context-aware)
# ═══════════════════════════════════════════════════════════════════════════

_TIMESTAMP_PREFIX_RE = re.compile(r"^\s*\[?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}\]?\s*")


def strip_timestamp_prefix(text: str) -> str:
    return _TIMESTAMP_PREFIX_RE.sub("", text).strip()


def generate_resolution_assistant(
    constraint_turns: list,
    resolution_user_text: str,
    resolution_user_time: str,
    model: str,
    temperature: float,
    trigger_time: str = None,
    trigger_text: str = None,
    max_retries: int = 5,
) -> str:
    """Generate a context-aware assistant reply to the resolution.

    Uses only the constraint user-assistant pair (not the full history) to
    keep the LLM focused on the problem context without thematic distractions.
    """
    system_content = SYSTEM_PROMPT_RESOLUTION_ASSISTANT
    if trigger_time and trigger_text and resolution_user_time:
        delta = int(time_diff_minutes(resolution_user_time, trigger_time))
        system_content += (
            "\n\n[Hidden context — do NOT reveal to the user] "
            "The key event is still {} minutes away (at {}): \"{}\". "
            "Keep your tone mid-conversation, not end-of-journey."
        ).format(delta, trigger_time, trigger_text)

    messages = [{"role": "system", "content": system_content}]

    # Add only the constraint pair as context (problem background)
    for t in constraint_turns:
        messages.append({
            "role": t["role"],
            "content": "[{}] {}".format(t["time"], t["text"]),
        })

    # Add the new resolution user message
    messages.append({
        "role": "user",
        "content": "[{}] {}".format(resolution_user_time, resolution_user_text),
    })

    for attempt in range(1, max_retries + 1):
        try:
            raw = chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=256,
            )
            text = strip_timestamp_prefix(raw.strip())

            if len(text) < 5:
                log.warning("  Resolution assistant attempt %d/%d: too short",
                            attempt, max_retries)
                continue
            if len(text) > 500:
                log.warning("  Resolution assistant attempt %d/%d: too long (%d chars)",
                            attempt, max_retries, len(text))
                continue

            # Blacklist check
            hits = check_blacklist(text)
            if hits:
                log.warning("  Resolution assistant attempt %d/%d: blacklist hit: %s",
                            attempt, max_retries, hits)
                continue

            return text
        except Exception as e:
            log.warning("  Resolution assistant attempt %d/%d API error: %s",
                        attempt, max_retries, e)
            time_module.sleep(2)

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Blueprint Loader (reused from Stage 4)
# ═══════════════════════════════════════════════════════════════════════════

def load_blueprint_map(blueprint_file: str) -> dict:
    """Load blueprints from both seed and LLM-generated sources."""
    bp_map = {}

    for bp in STATE_TRACKING_SEEDS:
        bp_map[bp["task_id"]] = bp

    bp_path = Path(blueprint_file)
    if bp_path.exists():
        with open(bp_path, encoding="utf-8") as f:
            blueprints = json.load(f)
        for bp in blueprints:
            bp_map[bp["task_id"]] = bp
    else:
        log.warning("Blueprint file not found: %s", bp_path)

    log.info("Loaded %d blueprints (seed + LLM-generated)", len(bp_map))
    return bp_map


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
        description="Stage 5: Derive Negative-Clean variants for State-Tracking"
    )
    ap.add_argument("--dialogue-input", required=True,
                    help="Path to Stage 3 Positive-Clean dialogues JSON")
    ap.add_argument("--blueprint-input", required=True,
                    help="Path to Stage 1 LLM-generated blueprint JSON")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    # ── Load data ───────────────────────────────────────────────────────────
    log.info("Loading full dialogues from: %s", args.dialogue_input)
    with open(args.dialogue_input, encoding="utf-8") as f:
        dialogues = json.load(f)
    log.info("  -> %d dialogues", len(dialogues))

    bp_map = load_blueprint_map(args.blueprint_input)

    # ── Output setup ────────────────────────────────────────────────────────
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else DIM_DIR / "outputs" / "stage5_negative_clean"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dialogues_negative_clean_{}.json".format(args.model)

    # ── Resume support ──────────────────────────────────────────────────────
    completed = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as f:
                loaded = json.load(f)
            for item in loaded:
                completed[item["dialogue_id"]] = item
            log.info("Resumed %d dialogues from %s", len(completed), resume_path)

    # ── Generation loop ─────────────────────────────────────────────────────
    results = list(completed.values())
    success = len(results)
    fail = 0

    for dlg in tqdm(dialogues, desc="Stage 5 Negative-Clean"):
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
            log.info("  %s — already done, skipping", negative_id)
            continue

        # Skip partial/failed dialogues
        if dlg.get("status") != "complete":
            log.warning("  %s — status is '%s', skipping",
                        dialogue_id, dlg.get("status"))
            fail += 1
            continue

        # Look up blueprint
        bp = bp_map.get(task_id)
        if bp is None:
            log.warning("  %s — no blueprint found, skipping", task_id)
            fail += 1
            continue

        # Extract resolution_for_negative
        story = bp.get("story", {})
        variations = story.get("variations", story.get("variation", {}))
        resolution_desc = variations.get("resolution_for_negative")
        if not resolution_desc:
            log.warning("  %s — no resolution_for_negative in blueprint, skipping",
                        task_id)
            fail += 1
            continue

        turns = dlg["turns"]

        # ── Step 1: Find insertion point ────────────────────────────────────
        insertion = find_insertion_point(turns, negative_id)
        if insertion is None:
            log.warning("  %s — could not find insertion point, skipping", task_id)
            fail += 1
            continue

        insert_idx = insertion["insert_after_idx"]
        branch = insertion["branch"]

        log.info("  %s — Branch %s, inserting after idx %d (turn_id %d)",
                 task_id, branch, insert_idx, turns[insert_idx]["turn_id"])

        # ── Extract key turns ─────────────────────────────────────────
        trigger_turn = next(
            (t for t in turns if t["function"] == "trigger"), None
        )
        constraint_turn = next(
            (t for t in turns if t["function"] == "constraint" and t["role"] == "user"),
            None,
        )
        # Constraint pair: constraint user + its assistant reply
        constraint_turns = []
        if constraint_turn:
            ci = turns.index(constraint_turn)
            constraint_turns.append(constraint_turn)
            if ci + 1 < len(turns) and turns[ci + 1]["role"] == "assistant":
                constraint_turns.append(turns[ci + 1])

        # ── Step 2: Generate user resolution utterance ──────────────────────
        resolution_user_text = generate_resolution_user(
            resolution_description=resolution_desc,
            model=args.model,
            temperature=args.temperature,
            constraint_text=constraint_turn["text"] if constraint_turn else None,
            trigger_time=trigger_turn["time"] if trigger_turn else None,
            trigger_text=trigger_turn["text"] if trigger_turn else None,
            resolution_time=insertion["new_user_time"],
            max_retries=args.max_retries,
        )
        if resolution_user_text is None:
            log.error("  %s — FAILED to generate resolution user text", task_id)
            fail += 1
            continue

        # ── Step 3: Generate assistant resolution reply ─────────────────────
        resolution_ast_text = generate_resolution_assistant(
            constraint_turns=constraint_turns,
            resolution_user_text=resolution_user_text,
            resolution_user_time=insertion["new_user_time"],
            model=args.model,
            temperature=args.temperature,
            trigger_time=trigger_turn["time"] if trigger_turn else None,
            trigger_text=trigger_turn["text"] if trigger_turn else None,
            max_retries=args.max_retries,
        )
        if resolution_ast_text is None:
            log.error("  %s — FAILED to generate resolution assistant text", task_id)
            fail += 1
            continue

        # ── Step 4: Assemble new dialogue ───────────────────────────────────
        new_dlg = copy.deepcopy(dlg)
        new_turns = new_dlg["turns"]

        # Branch B: shift timestamps for turns after insertion point
        if branch == "B":
            shift_from = insertion["shift_from_idx"]
            for i in range(shift_from, len(new_turns)):
                new_turns[i]["time"] = add_minutes(new_turns[i]["time"], 3)

        # Create new turn nodes
        new_user_turn = {
            "turn_id": 0,  # will be renumbered
            "role": "user",
            "time": insertion["new_user_time"],
            "function": "negative_resolution",
            "text": resolution_user_text,
        }
        new_ast_turn = {
            "turn_id": 0,  # will be renumbered
            "role": "assistant",
            "time": insertion["new_assistant_time"],
            "function": "assistant_reply",
            "text": resolution_ast_text,
        }

        # Insert after insert_idx
        insert_pos = insert_idx + 1
        new_turns.insert(insert_pos, new_user_turn)
        new_turns.insert(insert_pos + 1, new_ast_turn)

        # Renumber turn_ids
        for i, t in enumerate(new_turns):
            t["turn_id"] = i + 1

        # ── Sanity check: time order ────────────────────────────────────────
        ok, err = validate_time_order(new_turns)
        if not ok:
            log.error("  %s — TIME ORDER VIOLATION: %s", task_id, err)
            fail += 1
            continue

        # Update metadata
        new_dlg["dialogue_id"] = negative_id
        new_dlg["_insertion_branch"] = branch

        results.append(new_dlg)
        completed[negative_id] = new_dlg
        success += 1

        # Incremental save
        save_output(results, out_file)
        log.info("  %s — done (Branch %s): user='%s' | ast='%s'",
                 task_id, branch,
                 resolution_user_text[:80],
                 resolution_ast_text[:80])

    # ── Final save ──────────────────────────────────────────────────────────
    save_output(results, out_file)

    log.info("=" * 70)
    log.info("Done!  %d / %d negative-clean dialogues generated  (%d failed).",
             success, len(dialogues), fail)
    log.info("Saved -> %s", out_file)


if __name__ == "__main__":
    main()

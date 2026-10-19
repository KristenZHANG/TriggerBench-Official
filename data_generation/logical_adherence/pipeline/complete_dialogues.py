#!/usr/bin/env python3
"""Stage 3 — Dialogue completion (Positive-Clean) for Logical Adherence.

Takes Stage 2 user-only dialogues and generates assistant responses for each
user turn EXCEPT the final trigger turn (N-1 completion rule).

Key differences from State-Tracking / Temporal Grounding Stage 3:
1. System Prompt emphasizes neutral acknowledgement of user rules/conditions —
   assistant must NOT proactively track, infer, or remind about conditional logic.
2. Two constraint-like functions: rule_constraint AND condition_constraint (Type A).
   Both receive the same blacklist treatment (constraint-subset).
3. Blacklist extended with conditional-logic leakage patterns (e.g., "condition is met",
   "since you said if", "that means you should").
"""

from __future__ import annotations

import argparse
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
from shared.time_utils import compute_assistant_time

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
#  System Prompt — Logical Adherence Enhanced
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
#  Blacklist — Full (for thematic_connection turns)
# ═══════════════════════════════════════════════════════════════════════════

BLACKLIST_PATTERNS_FULL = [
    # --- Core proactive reminder patterns (shared) ---
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
    r"set (a |an )?(reminder|alarm|timer|countdown)",
    r"before you (start|begin|go)",
    r"you (might|may) want to.{0,30}(first|before)",
    r"you still (need|have) to",
    r"you should.{0,20}(before|first)",
    r"earlier you (said|mentioned|told)",
    r"remember to",
    # --- Conditional-logic leakage patterns (new) ---
    r"since you (said|mentioned|told me).{0,40}(if|when|only after|unless)",
    r"(that|this) (means|implies|triggers|activates)",
    r"your (rule|condition|constraint|plan) .{0,20}(applies|triggered|met|satisfied)",
    r"condition.{0,10}(is |has been )?(met|fulfilled|satisfied)",
    r"now that .{0,30}(you (said|mentioned)|the condition)",
    r"based on what you (said|mentioned) earlier",
    r"you (previously|earlier) (said|stated|mentioned).{0,30}(if|when|rule|condition)",
    r"i('ll| will) (track|monitor|watch|keep an eye on)",
    r"want me to (track|monitor|remind|flag|alert)",
    r"shall i (remind|alert|notify|flag)",
    r"i('ll| will) (flag|note) (this|that) for (later|you|when)",
    r"let me know.{0,20}(condition|rule|when)",
]

# --- Constraint-only blacklist (for rule_constraint / condition_constraint turns) ---
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

_BLACKLIST_RE_FULL = [re.compile(p, re.IGNORECASE) for p in BLACKLIST_PATTERNS_FULL]
_BLACKLIST_RE_CONSTRAINT = [re.compile(p, re.IGNORECASE) for p in BLACKLIST_PATTERNS_CONSTRAINT]


def check_blacklist(text: str, mode: str = "full") -> list:
    """Return list of matched blacklist patterns (empty = clean).

    mode="full": all patterns (for thematic_connection turns)
    mode="constraint": conditional-logic subset (for rule_constraint / condition_constraint)
    """
    if mode == "constraint":
        return [p for p, rx in zip(BLACKLIST_PATTERNS_CONSTRAINT, _BLACKLIST_RE_CONSTRAINT)
                if rx.search(text)]
    return [p for p, rx in zip(BLACKLIST_PATTERNS_FULL, _BLACKLIST_RE_FULL)
            if rx.search(text)]


# ═══════════════════════════════════════════════════════════════════════════
#  Max token selection
# ═══════════════════════════════════════════════════════════════════════════

_LONG_RESPONSE_KEYWORDS = [
    "explain in detail", "write me", "write a", "draft a",
    "give me a detailed", "essay", "report", "comprehensive",
    "step by step", "step-by-step", "thorough", "in depth",
    "long", "elaborate", "full explanation", "full guide",
    "introduce", "summarize", "summary", "history of",
]

# Constraint-like functions that get short responses
_CONSTRAINT_FUNCTIONS = {"rule_constraint", "condition_constraint"}


def pick_max_tokens(user_text: str, user_function: str) -> int:
    if user_function in _CONSTRAINT_FUNCTIONS:
        return 512
    text_lower = user_text.lower()
    for kw in _LONG_RESPONSE_KEYWORDS:
        if kw in text_lower:
            return 2048
    return 1024


# ═══════════════════════════════════════════════════════════════════════════
#  Strip timestamp prefix from LLM output
# ═══════════════════════════════════════════════════════════════════════════

_TIMESTAMP_PREFIX_RE = re.compile(r"^\s*\[?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}\]?\s*")


def strip_timestamp_prefix(text: str) -> str:
    return _TIMESTAMP_PREFIX_RE.sub("", text).strip()


# ═══════════════════════════════════════════════════════════════════════════
#  Message Assembly
# ═══════════════════════════════════════════════════════════════════════════

def build_messages(
    user_turns: list,
    assistant_replies: list,
    current_turn_idx: int,
    assistant_time: str = None,
) -> list:
    """Build ChatML messages for generating the k-th assistant reply.

    If assistant_time is provided, a prefill assistant message with just the
    timestamp is appended so the LLM knows its current response time.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i in range(current_turn_idx):
        u = user_turns[i]
        messages.append({
            "role": "user",
            "content": "[{}] {}".format(u["time"], u["text"]),
        })
        a = assistant_replies[i]
        messages.append({
            "role": "assistant",
            "content": "[{}] {}".format(a["time"], a["text"]),
        })

    curr = user_turns[current_turn_idx]
    messages.append({
        "role": "user",
        "content": "[{}] {}".format(curr["time"], curr["text"]),
    })

    # Prefill: tell the LLM its response timestamp
    if assistant_time:
        messages.append({
            "role": "assistant",
            "content": "[{}]".format(assistant_time),
        })

    return messages


# ═══════════════════════════════════════════════════════════════════════════
#  Flat Turn Assembly
# ═══════════════════════════════════════════════════════════════════════════

def assemble_flat_turns(
    user_turns: list,
    assistant_replies: list,
    include_trigger: bool = True,
) -> list:
    """Build a flat interleaved turn list."""
    flat = []
    turn_id = 1
    for k in range(len(assistant_replies)):
        u = user_turns[k]
        flat.append({
            "turn_id": turn_id, "role": "user",
            "time": u["time"], "function": u["function"], "text": u["text"],
        })
        turn_id += 1
        a = assistant_replies[k]
        flat.append({
            "turn_id": turn_id, "role": "assistant",
            "time": a["time"], "function": "assistant_reply", "text": a["text"],
        })
        turn_id += 1

    if include_trigger:
        trigger = user_turns[-1]
        flat.append({
            "turn_id": turn_id, "role": "user",
            "time": trigger["time"], "function": trigger["function"],
            "text": trigger["text"],
        })
    return flat


# ═══════════════════════════════════════════════════════════════════════════
#  Single Dialogue Completion
# ═══════════════════════════════════════════════════════════════════════════

def complete_one_dialogue(
    task_id: str,
    user_turns: list,
    model: str,
    temperature: float,
    max_retries: int,
    blacklist_constraint: bool = True,
    existing_replies: list = None,
    on_progress=None,
) -> list:
    """Generate assistant replies for one dialogue (standard/unmerged).

    Args:
        blacklist_constraint: If True, apply constraint-subset blacklist on
            rule_constraint and condition_constraint turns. If False, fully
            exempt those turns from blacklist.

    Returns flat interleaved turn list, or None on failure.
    """
    n_user = len(user_turns)
    if n_user < 2:
        log.error("[%s] Dialogue has fewer than 2 user turns — skipping", task_id)
        return None

    assistant_replies = list(existing_replies or [])
    start_k = len(assistant_replies)

    if start_k > 0:
        log.info("[%s] Resuming from turn %d (have %d cached replies)",
                 task_id, start_k + 1, start_k)

    for k in range(start_k, n_user - 1):
        curr_user = user_turns[k]
        next_user = user_turns[k + 1]

        ast_time = compute_assistant_time(curr_user["time"], next_user["time"])
        max_tok = pick_max_tokens(curr_user["text"], curr_user["function"])

        # Determine blacklist mode based on turn function
        func = curr_user["function"]
        if func == "thematic_connection":
            bl_mode = "full"
        elif func in _CONSTRAINT_FUNCTIONS:
            bl_mode = "constraint" if blacklist_constraint else None
        else:
            bl_mode = None

        messages = build_messages(user_turns, assistant_replies, k,
                                  assistant_time=ast_time)

        reply_text = None
        for attempt in range(max_retries):
            try:
                raw = chat(
                    messages, model=model,
                    temperature=temperature, max_tokens=max_tok,
                )
                raw = strip_timestamp_prefix(raw)

                if bl_mode:
                    hits = check_blacklist(raw, mode=bl_mode)
                    if hits:
                        log.warning(
                            "[%s] turn %d attempt %d — blacklist hit (%s): %s",
                            task_id, k + 1, attempt + 1, bl_mode, hits,
                        )
                        if attempt < max_retries - 1:
                            continue
                        # Last attempt: accept with warning
                        log.warning("[%s] turn %d — accepting on final attempt",
                                    task_id, k + 1)

                reply_text = raw
                break

            except Exception as e:
                log.error("[%s] turn %d attempt %d — API error: %s",
                          task_id, k + 1, attempt + 1, e)
                time_module.sleep(2)

        if reply_text is None:
            log.error("[%s] Failed turn %d after %d retries", task_id, k + 1, max_retries)
            if on_progress and assistant_replies:
                partial = assemble_flat_turns(user_turns, assistant_replies,
                                             include_trigger=False)
                on_progress(task_id, partial, False)
            return None

        assistant_replies.append({"time": ast_time, "text": reply_text})

        is_complete = (k == n_user - 2)
        if on_progress:
            turns_so_far = assemble_flat_turns(
                user_turns, assistant_replies, include_trigger=is_complete,
            )
            on_progress(task_id, turns_so_far, is_complete)

    return assemble_flat_turns(user_turns, assistant_replies, include_trigger=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Save Helper
# ═══════════════════════════════════════════════════════════════════════════

def _save_output(completed: dict, status: dict, out_file: Path):
    output = []
    for did in sorted(completed):
        output.append({
            "dialogue_id": "{}_positive_clean".format(did),
            "status": status.get(did, "complete"),
            "turns": completed[did],
        })
    save_json(output, out_file)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage 3: Complete Logical Adherence dialogues (Positive-Clean)"
    )
    ap.add_argument("--input", required=True,
                    help="Path to Stage 2 user-dialogue JSON file")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--blacklist-constraint", dest="blacklist_constraint",
                    action="store_true", default=True,
                    help="Apply constraint-subset blacklist on rule_constraint and "
                         "condition_constraint turns (default: True).")
    ap.add_argument("--no-blacklist-constraint", dest="blacklist_constraint",
                    action="store_false",
                    help="Fully exempt rule_constraint and condition_constraint turns "
                         "from the blacklist.")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to existing output JSON to resume from")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    input_path = Path(args.input)
    with open(input_path, encoding="utf-8") as f:
        base_dialogues = json.load(f)
    log.info("Loaded %d base dialogues from %s", len(base_dialogues), input_path)

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else DIM_DIR / "outputs" / "stage3_positive_clean"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dialogues_positive_clean_{}.json".format(args.model)

    completed = {}
    completed_status = {}
    cached_replies = {}

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as f:
                loaded = json.load(f)
            for item in loaded:
                did = item.get("dialogue_id", "")
                tid = did
                for suffix in ("_positive_clean", "_standard"):
                    if tid.endswith(suffix):
                        tid = tid[: -len(suffix)]
                        break
                status = item.get("status", "complete")
                turns = item.get("turns", [])
                completed[tid] = turns
                completed_status[tid] = status
                if status == "partial":
                    cached_replies[tid] = [
                        {"time": t["time"], "text": t["text"]}
                        for t in turns if t["role"] == "assistant"
                    ]
            n_complete = sum(1 for s in completed_status.values() if s == "complete")
            n_partial = sum(1 for s in completed_status.values() if s == "partial")
            log.info("Resumed %d complete + %d partial from %s",
                     n_complete, n_partial, resume_path)

    def _on_progress(tid, flat_turns, is_complete):
        completed[tid] = flat_turns
        completed_status[tid] = "complete" if is_complete else "partial"
        _save_output(completed, completed_status, out_file)

    total = len(base_dialogues)
    completed_ids = {tid for tid, s in completed_status.items() if s == "complete"}
    success_count = len(completed_ids)
    fail_count = 0
    task_ids = list(base_dialogues.keys())

    log.info("Target: %d dialogues | Model=%s, T=%.2f, max_retries=%d, bl_constraint=%s",
             total, args.model, args.temperature, args.max_retries,
             args.blacklist_constraint)

    for bp_idx, task_id in enumerate(tqdm(task_ids, desc="Stage 3 Positive-Clean")):
        if args.sleep:
            time_module.sleep(args.sleep)

        if task_id in completed_ids:
            log.info("[%d/%d] %s — already done, skipping", bp_idx + 1, total, task_id)
            continue

        user_turns = base_dialogues[task_id]
        existing = cached_replies.pop(task_id, None)

        log.info("[%d/%d] %s — %d user turns, generating %d assistant replies",
                 bp_idx + 1, total, task_id, len(user_turns), len(user_turns) - 1)

        flat_turns = complete_one_dialogue(
            task_id=task_id,
            user_turns=user_turns,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            blacklist_constraint=args.blacklist_constraint,
            existing_replies=existing,
            on_progress=_on_progress,
        )

        if flat_turns is not None:
            completed_ids.add(task_id)
            success_count += 1
            log.info("[%s] completed — %d total turns", task_id, len(flat_turns))
        else:
            fail_count += 1
            log.error("[%s] FAILED (partial progress saved)", task_id)

    _save_output(completed, completed_status, out_file)

    log.info("=" * 70)
    log.info("Done!  %d / %d dialogues completed  (%d failed).",
             success_count, total, fail_count)
    log.info("Saved -> %s", out_file)


if __name__ == "__main__":
    main()

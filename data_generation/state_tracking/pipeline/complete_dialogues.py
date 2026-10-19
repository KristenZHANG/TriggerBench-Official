#!/usr/bin/env python3
"""Stage 3 — Dialogue completion (Positive-Clean) for State-Tracking.

Takes Stage 2 user-only dialogues and generates assistant responses for every
turn except the final trigger (the N-1 completion rule).
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
#  System Prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a professional, naturally conversational, and knowledgeable AI assistant.

CRITICAL persona requirement: You assume the user is a fully capable adult. \
Unless the user EXPLICITLY asks you for safety advice or EXPLICITLY requests \
you to set a reminder, you must NEVER proactively offer unsolicited advice, \
NEVER proactively warn about potential risks, and NEVER say things like \
"remember to do X". Maintain proper boundaries — do NOT be a preachy assistant.

Prioritize answering the user's explicitly stated question or task thoroughly \
and in detail. Do NOT over-intervene on the user's casually mentioned daily \
behaviors or physical states.

Length guideline: Keep responses professional and concise, typically under \
500 words. However, if the user requests a detailed explanation, summary, \
or report, provide a thorough and in-depth answer, up to 1000-2000 words.\
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Blacklist for Proactive Reminder Detection
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
#  Max token selection
# ═══════════════════════════════════════════════════════════════════════════

_LONG_RESPONSE_KEYWORDS = [
    "explain in detail", "write me", "write a", "draft a",
    "give me a detailed", "essay", "report", "comprehensive",
    "step by step", "step-by-step", "thorough", "in depth",
    "long", "elaborate", "full explanation", "full guide",
    "introduce", "summarize", "summary", "history of",
]


def pick_max_tokens(user_text: str, user_function: str) -> int:
    if user_function == "constraint":
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
) -> list:
    """Build ChatML messages for generating the k-th assistant reply."""
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
    existing_replies: list = None,
    on_progress=None,
) -> list:
    """Generate assistant replies for one dialogue (standard/unmerged).

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

        # Blacklist only for thematic_connection turns
        use_blacklist = curr_user["function"] == "thematic_connection"

        messages = build_messages(user_turns, assistant_replies, k)

        reply_text = None
        for attempt in range(max_retries):
            try:
                raw = chat(
                    messages, model=model,
                    temperature=temperature, max_tokens=max_tok,
                )
                raw = strip_timestamp_prefix(raw)

                if use_blacklist:
                    hits = check_blacklist(raw)
                    if hits:
                        log.warning(
                            "[%s] turn %d attempt %d — blacklist hit: %s",
                            task_id, k + 1, attempt + 1, hits,
                        )
                        if attempt < max_retries - 1:
                            continue
                        # Last attempt: accept with warning
                        log.warning("[%s] turn %d — accepting on final attempt", task_id, k + 1)

                reply_text = raw
                break

            except Exception as e:
                log.error("[%s] turn %d attempt %d — API error: %s",
                          task_id, k + 1, attempt + 1, e)
                time_module.sleep(2)

        if reply_text is None:
            log.error("[%s] Failed turn %d after %d retries", task_id, k + 1, max_retries)
            if on_progress and assistant_replies:
                partial = assemble_flat_turns(user_turns, assistant_replies, include_trigger=False)
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
        description="Stage 3: Complete State-Tracking dialogues (Positive-Clean)"
    )
    ap.add_argument("--input", required=True,
                    help="Path to Stage 2 user-dialogue JSON file")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to existing output JSON to resume from")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    # ── Load Stage 2 dialogues ──────────────────────────────────────────────
    input_path = Path(args.input)
    with open(input_path, encoding="utf-8") as f:
        base_dialogues = json.load(f)
    log.info("Loaded %d base dialogues from %s", len(base_dialogues), input_path)

    # ── Output setup ────────────────────────────────────────────────────────
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else DIM_DIR / "outputs" / "stage3_positive_clean"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dialogues_positive_clean_{}.json".format(args.model)

    # ── Resume support ──────────────────────────────────────────────────────
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
                tid = did.removesuffix("_positive_clean").removesuffix("_standard")
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

    # ── Progress callback ───────────────────────────────────────────────────
    def _on_progress(tid, flat_turns, is_complete):
        completed[tid] = flat_turns
        completed_status[tid] = "complete" if is_complete else "partial"
        _save_output(completed, completed_status, out_file)

    # ── Generation loop ─────────────────────────────────────────────────────
    total = len(base_dialogues)
    completed_ids = {tid for tid, s in completed_status.items() if s == "complete"}
    success_count = len(completed_ids)
    fail_count = 0
    task_ids = sorted(base_dialogues.keys())

    log.info("Target: %d dialogues | Model=%s, T=%.2f, max_retries=%d",
             total, args.model, args.temperature, args.max_retries)

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

    # ── Final save ──────────────────────────────────────────────────────────
    _save_output(completed, completed_status, out_file)

    log.info("=" * 70)
    log.info("Done!  %d / %d dialogues completed  (%d failed).",
             success_count, total, fail_count)
    log.info("Saved -> %s", out_file)


if __name__ == "__main__":
    main()

"""Letta-Sim PM evaluation: core + archival memory, simplified from Letta.

https://github.com/letta-ai/letta

Same system prompt as blended_close. Base split only (no 40K).
Flags: ../README.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from openai import RateLimitError
from longcontext_blended_close import get_system_prompt

# ── Path setup (relative to prospective_memory/) ──────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent                    # source/
PKG_ROOT = SCRIPT_DIR.parent                                    # prospective_memory/
BENCHMARK_ROOT = PKG_ROOT.parent                                # benchmark/
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT / "utils"))

from client import LLMClient, CONFIG, _get_client_kwargs, _parse_env_list
from pm_judge import PMJudge
from pm_scorer import (
    NEGATIVE_VARIANTS, PMScore,
    score_sample as pm_score_sample,
    score_to_dict,
    aggregate as pm_aggregate,
)

# ── Constants ────────────────────────────────────────────────────────────
DATA_DIR = PKG_ROOT / "data" / "base_context"
OUTPUT_ROOT = PKG_ROOT / "main_results" / "letta_sim"

CATEGORIES = [
    "state_tracking",
    "temporal_grounding",
    "logical_adherence",
    "attention_recovery",
    "safe_coding",
]
VARIANTS = ["positive_clean", "positive_overloaded", "negative_clean"]
BIN_LABELS = ["base"]

JUDGE_DEPLOYMENT = "gpt-4o"

# Turn function labels that mark constraint information
CONSTRAINT_FUNCTIONS = {"constraint", "rule_constraint", "condition_constraint"}

# Default core memory character limit (mirrors Letta's CORE_MEMORY_BLOCK_CHAR_LIMIT)
DEFAULT_CORE_LIMIT = 2000

# text-embedding-3-small uses cl100k_base and accepts up to 8191 tokens
_EMB_MAX_TOKENS = 8191
_EMB_ENCODING = tiktoken.get_encoding("cl100k_base")

# Max tokens per turn chunk for curation prompts.
# Long turns (e.g. interrupt reasoning traces) are split so
# the curation LLM sees manageable pieces.
_CURATION_MAX_TOKENS = 2_000
# Max tokens per chunk for embedding archival memories.
_CHUNK_MAX_TOKENS = 8_000


# ── Memory curation prompt ───────────────────────────────────────────────

MEMORY_CURATION_PROMPT = """\
You are the memory manager for a conversational AI assistant (similar to MemGPT/Letta).
You manage a two-tier memory system:

## CORE MEMORY (always visible to the assistant, character limit: {core_limit})
Critical facts about the user and context that should ALWAYS be accessible.
Currently contains:
---
{core_memory}
---
(Current length: {core_len}/{core_limit} chars)

## ARCHIVAL MEMORY (searchable via embeddings, unlimited)
Background details stored for later retrieval. Not always visible.

## NEW CONVERSATION TURN
{turn_text}

## YOUR TASK
Decide how to update memory based on this turn. Consider:
- Is this information critical enough to keep always visible (core)?
- Is it background info that might be useful later (archival)?
- Would core memory overflow? If so, should you replace less important content?
- Focus on user messages but also store assistant commitments, plans, or promises.

Output EXACTLY ONE JSON object (no extra text):
{{
  "action": "core_append" | "core_replace" | "archival_insert" | "both" | "no_action",
  "core_content": "text to append/replace in core (null if no core change)",
  "core_old": "text to replace in core (only for core_replace, null otherwise)",
  "archival_content": "text to store in archival (null if no archival change)",
  "reasoning": "brief reason for decision (1 sentence)"
}}

Rules:
- "both" means core_append/core_replace AND archival_insert together.
- For "core_replace": core_old must be an exact substring currently in core memory.
- Keep core entries concise (summarize, don't copy verbatim).
- Prioritize: user commitments, locations of objects, time-sensitive plans, safety-critical info.
- If core is near limit, replace less important info rather than skipping important new info.
"""


# ── Embedding helpers ────────────────────────────────────────────────────

def _embedding_client() -> tuple[Any, str]:
    """Build an Azure OpenAI embeddings client.

      AZURE_OPENAI_EMBEDDINGS_ENDPOINT   (falls back to AZURE_OPENAI_ENDPOINT)
      AZURE_OPENAI_EMBEDDINGS_DEPLOYMENTS (required)
      AZURE_OPENAI_EMBEDDINGS_API_VERSION
      AZURE_OPENAI_EMBEDDINGS_SCOPE
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    endpoint = (
        os.environ.get("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    deployments = _parse_env_list("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENTS")
    if not endpoint or not deployments:
        raise ValueError(
            "Set AZURE_OPENAI_EMBEDDINGS_DEPLOYMENTS (and optionally "
            "AZURE_OPENAI_EMBEDDINGS_ENDPOINT)"
        )
    api_version = os.environ.get(
        "AZURE_OPENAI_EMBEDDINGS_API_VERSION",
        os.environ.get("AZURE_OPENAI_API_VERSION",
                        CONFIG["defaults"]["api_version"]),
    )
    emb_scope = os.environ.get("AZURE_OPENAI_EMBEDDINGS_SCOPE")
    if emb_scope:
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, emb_scope)
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
    else:
        client = AzureOpenAI(**_get_client_kwargs(endpoint, api_version))
    return client, deployments[0]


def _tail_truncate(text: str, max_tokens: int = _EMB_MAX_TOKENS) -> str:
    """Keep the last *max_tokens* tokens of *text* for embedding."""
    tokens = _EMB_ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _EMB_ENCODING.decode(tokens[-max_tokens:])


def _embed_batch(emb_client: Any, emb_deployment: str,
                 texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returns L2-normalized matrix."""
    vecs: list[list[float]] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = [_tail_truncate(t) if len(t) > 0 else " "
                 for t in texts[i:i + batch_size]]
        for attempt in range(6):
            try:
                resp = emb_client.embeddings.create(
                    model=emb_deployment, input=batch)
                break
            except RateLimitError:
                if attempt == 5:
                    raise
                wait = min(2 ** attempt, 30)
                print(f"  Embedding rate-limited, retrying in {wait}s...")
                time.sleep(wait)
        for item in resp.data:
            vecs.append(item.embedding)
    m = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    return m / norms


# ── Helpers ──────────────────────────────────────────────────────────────

def _format_turn(turn: dict) -> str:
    """Format a single turn for display / embedding."""
    ts = f" [{turn['timestamp']}]" if turn.get("timestamp") else ""
    speaker = "USER" if turn["role"] == "user" else "ASSISTANT"
    return f"{speaker}{ts}: {turn['text']}"


def _count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_EMB_ENCODING.encode(text))


def _chunk_text(text: str, max_tokens: int) -> list[str]:
    """Split text into chunks of at most *max_tokens*, breaking at paragraph/sentence boundaries."""
    if _count_tokens(text) <= max_tokens:
        return [text]
    cpt = max(1, len(text) / max(1, _count_tokens(text)))
    chunks: list[str] = []
    while text:
        if _count_tokens(text) <= max_tokens:
            chunks.append(text)
            break
        est_chars = int(max_tokens * cpt * 0.95)
        est_chars = min(est_chars, len(text))
        cut = text.rfind("\n\n", 0, est_chars)
        if cut < est_chars * 0.3:
            cut = text.rfind(". ", 0, est_chars)
        if cut < est_chars * 0.3:
            cut = text.rfind("\n", 0, est_chars)
        if cut < est_chars * 0.3:
            cut = est_chars
        candidate = text[:cut + 1].rstrip()
        while candidate and _count_tokens(candidate) > max_tokens:
            cut = int(cut * 0.9)
            candidate = text[:cut + 1].rstrip()
        if not candidate:
            tokens = _EMB_ENCODING.encode(text)
            candidate = _EMB_ENCODING.decode(tokens[:max_tokens])
            cut = len(candidate) - 1
        chunks.append(candidate)
        text = text[cut + 1:].lstrip()
    return chunks if chunks else [text]


def _find_constraint_turns(turns: list[dict],
                           trigger_idx: int) -> dict[str, list[int]]:
    """Return {function_name: [turn_indices]} for constraint turns before trigger."""
    result: dict[str, list[int]] = {}
    for i, t in enumerate(turns[:trigger_idx]):
        func = t.get("function", "")
        if func in CONSTRAINT_FUNCTIONS:
            result.setdefault(func, []).append(i)
    return result


def _make_run_tag(model: str, temperature: float, concurrency: int,
                  top_k: int, core_limit: int, emb_deployment: str) -> str:
    """Build a filesystem-safe run tag.
    """
    tag = re.sub(r'[^a-zA-Z0-9]+', '_', model).strip('_')
    emb = re.sub(r'[^a-zA-Z0-9]+', '_', emb_deployment).strip('_')
    return (
        f"{tag}_temp{temperature}_con{concurrency}_k{top_k}"
        f"_core{core_limit}_emb_{emb}"
    )


def _parse_curation_response(response_text: str) -> dict:
    """Parse the JSON response from memory curation LLM call."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"action": "no_action", "reasoning": "Failed to parse response"}


def _apply_memory_action(
    action_dict: dict,
    core_memory: str,
    archival_memory: list[str],
    core_limit: int,
) -> tuple[str, list[str]]:
    """Apply a memory curation action and return updated (core, archival)."""
    action = action_dict.get("action", "no_action")
    core_content = action_dict.get("core_content")
    core_old = action_dict.get("core_old")
    archival_content = action_dict.get("archival_content")

    if action in ("core_append", "both"):
        if core_content:
            new_core = core_memory.rstrip()
            if new_core:
                new_core += "\n" + core_content.strip()
            else:
                new_core = core_content.strip()
            # Enforce limit: if over, truncate from the beginning
            if len(new_core) > core_limit:
                lines = new_core.split("\n")
                while len("\n".join(lines)) > core_limit and len(lines) > 1:
                    lines.pop(0)
                new_core = "\n".join(lines)
            core_memory = new_core

    elif action == "core_replace":
        if core_content and core_old:
            if core_old in core_memory:
                core_memory = core_memory.replace(core_old, core_content.strip(), 1)
            else:
                # Fallback to append if old text not found
                new_core = core_memory.rstrip()
                if new_core:
                    new_core += "\n" + core_content.strip()
                else:
                    new_core = core_content.strip()
                if len(new_core) > core_limit:
                    lines = new_core.split("\n")
                    while len("\n".join(lines)) > core_limit and len(lines) > 1:
                        lines.pop(0)
                    new_core = "\n".join(lines)
                core_memory = new_core

    # Handle archival for "both" or "archival_insert"
    if action in ("archival_insert", "both"):
        if archival_content:
            archival_memory.append(archival_content.strip())

    return core_memory, archival_memory


# ── Core evaluation ─────────────────────────────────────────────────────

def _run_one_letta_sim(
    entry: dict,
    client: LLMClient,
    temperature: float,
    top_k: int,
    core_limit: int,
    emb_client: Any | None = None,
    emb_deployment: str | None = None,
    category: str = "",
) -> dict:
    """Run simulated Letta PM evaluation for a single sample.

    1. Iterate turns before trigger: 1 LLM call per turn for memory curation.
    2. At trigger: retrieve from archival + use core memory in prompt.
    3. Generate response and score.
    """
    sample_dict = entry["sample"]
    trigger_idx = sample_dict["ground_truth"]["trigger"]["trigger_turn_idx"]
    turns = sample_dict["turns"]
    constraint_turns = _find_constraint_turns(turns, trigger_idx)

    # ── 1. Memory construction phase ────────────────────────────────
    core_memory = ""
    archival_memory: list[str] = []
    curation_log: list[dict] = []

    for turn_idx, t in enumerate(turns[:trigger_idx]):
        turn_text = _format_turn(t)

        # Chunk long turns so the curation LLM sees manageable pieces
        chunks = _chunk_text(turn_text, _CURATION_MAX_TOKENS)

        for chunk_i, chunk in enumerate(chunks):
            chunk_label = chunk if len(chunks) == 1 else f"[Part {chunk_i+1}/{len(chunks)}] {chunk}"

            # Build the curation prompt
            curation_prompt = MEMORY_CURATION_PROMPT.format(
                core_limit=core_limit,
                core_memory=core_memory if core_memory else "(empty)",
                core_len=len(core_memory),
                turn_text=chunk_label,
            )

            # LLM call for memory curation
            try:
                curation_response = client.complete_chat(
                    messages=[{"role": "user", "content": curation_prompt}],
                    max_tokens=300,
                    temperature=0.1,  # Low temp for consistent memory decisions
                )
            except Exception as e:
                curation_response = '{"action": "no_action", "reasoning": "LLM error"}'

            action_dict = _parse_curation_response(curation_response)

            # Apply the action
            core_memory, archival_memory = _apply_memory_action(
                action_dict, core_memory, archival_memory, core_limit
            )

        curation_log.append({
            "turn_idx": turn_idx,
            "n_chunks": len(chunks),
            "action": action_dict.get("action", "no_action"),
            "reasoning": action_dict.get("reasoning", ""),
            "core_len_after": len(core_memory),
            "archival_count_after": len(archival_memory),
        })

    # ── 2. Trigger: retrieve from archival ──────────────────────────
    trigger_text = _format_turn(turns[trigger_idx])
    archival_retrieved: list[str] = []
    archival_similarities: list[float] = []

    if archival_memory and emb_client and emb_deployment:
        # Vector search over archival memory
        memory_vecs = _embed_batch(emb_client, emb_deployment, archival_memory)
        query_vec = _embed_batch(emb_client, emb_deployment, [trigger_text])[0]
        sims = memory_vecs @ query_vec
        k = min(top_k, len(archival_memory))
        top_indices = np.argsort(-sims)[:k]
        archival_retrieved = [archival_memory[i] for i in top_indices]
        archival_similarities = [float(sims[i]) for i in top_indices]

    # ── 3. Generate response ────────────────────────────────────────
    system_prompt = get_system_prompt(category)
    core_block = core_memory if core_memory else "(no core memories stored)"
    archival_block = (
        "\n".join(f"- {m}" for m in archival_retrieved)
        if archival_retrieved
        else "(no archival memories retrieved)"
    )

    gen_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "=== CORE MEMORY (always visible) ===\n"
                f"{core_block}\n\n"
                "=== ARCHIVAL MEMORY (retrieved by relevance) ===\n"
                f"{archival_block}\n\n"
                "=== LATEST USER MESSAGE ===\n"
                f"{trigger_text}\n\n"
                "Respond to the latest user message. Use information from your "
                "memory (both core and archival) if relevant."
            ),
        },
    ]

    try:
        reply = client.complete_chat(
            messages=gen_messages, max_tokens=1024, temperature=temperature
        )
    except Exception as e:
        reply = f"[ERROR: {e}]"

    # ── 4. Score ────────────────────────────────────────────────────
    score = pm_score_sample(entry, reply)
    result = score_to_dict(score)
    result["pm_reply"] = reply
    result["core_memory_final"] = core_memory
    result["core_memory_len"] = len(core_memory)
    result["n_archival_total"] = len(archival_memory)
    result["n_archival_retrieved"] = len(archival_retrieved)
    result["archival_retrieved"] = archival_retrieved
    result["archival_similarities"] = archival_similarities
    result["constraint_turns"] = constraint_turns
    result["curation_log"] = curation_log

    # Check if constraint was captured in core memory
    result["constraint_in_core"] = False
    if constraint_turns:
        for func, idxs in constraint_turns.items():
            for idx in idxs:
                constraint_text = turns[idx]["text"].lower()
                words = set(re.findall(r'\b\w{4,}\b', constraint_text))
                core_lower = core_memory.lower()
                overlap = sum(1 for w in words if w in core_lower)
                if overlap >= 2:  # At least 2 key words from constraint in core
                    result["constraint_in_core"] = True
                    break

    return result


async def _run_batch_async(
    entries: list[dict],
    client: LLMClient,
    temperature: float,
    concurrency: int,
    top_k: int,
    core_limit: int,
    emb_client: Any | None = None,
    emb_deployment: str | None = None,
    category: str = "",
    out_path: Path | None = None,
) -> list[dict]:
    """Run simulated Letta PM evaluation concurrently with incremental saving."""
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    t0 = time.time()

    # Open file in append mode for incremental saving
    out_file = None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("a")

    async def _worker(entry):
        async with sem:
            return await asyncio.to_thread(
                _run_one_letta_sim, entry, client, temperature,
                top_k, core_limit, emb_client, emb_deployment,
                category,
            )

    tasks = [_worker(e) for e in entries]
    total = len(entries)
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        # Save immediately
        if out_file:
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
        done = i + 1
        if done % 5 == 0 or done == total:
            elapsed = time.time() - t0
            per_sample = elapsed / done
            remaining = per_sample * (total - done)
            mins_left = remaining / 60
            print(f"  Progress: {done}/{total} | "
                  f"~{per_sample:.0f}s/sample | "
                  f"ETA: {mins_left:.1f}min")

    if out_file:
        out_file.close()

    return results


# ── Judge ────────────────────────────────────────────────────────────────

def _run_judge_batch(
    results: list[dict],
    entries: list[dict],
    judge: PMJudge,
    judge_path: Path | None = None,
) -> list[dict]:
    """Run LLM-as-a-Judge on PM results, saving incrementally."""
    entry_by_id = {e["task_id"]: e for e in entries}
    judge_results = []

    judge_file = None
    if judge_path:
        judge_path.parent.mkdir(parents=True, exist_ok=True)
        judge_file = judge_path.open("a")

    for i, r in enumerate(results):
        entry = entry_by_id.get(r["task_id"])
        if entry is None:
            rec = {"task_id": r["task_id"], "verdict": "ERROR",
                   "reason": "Entry not found"}
            judge_results.append(rec)
            if judge_file:
                judge_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                judge_file.flush()
            continue

        sample = entry["sample"]
        intent_summary = sample["ground_truth"]["intent"]["summary"]
        constraint = sample["ground_truth"]["intent"].get("constraint", "")
        trigger_idx = sample["ground_truth"]["trigger"]["trigger_turn_idx"]
        trigger_text = sample["turns"][trigger_idx]["text"]

        # Conditional negatives: constraint/trigger are flipped, so use
        # intent-only evaluation to avoid logical mismatch.
        is_negative = r.get("is_negative", False)
        use_intent_only = (is_negative and r["category"] == "logical_adherence")

        verdict = judge.judge(
            intent_summary=intent_summary,
            trigger_text=trigger_text,
            assistant_reply=r["pm_reply"],
            constraint=constraint,
            category=r["category"],
            intent_only=use_intent_only,
        )

        if is_negative:
            judge_correct = (verdict["verdict"] == "NO")
        else:
            judge_correct = (verdict["verdict"] == "YES")

        rec = {
            "task_id": r["task_id"],
            "category": r["category"],
            "variant": r["variant"],
            "bin": r["bin"],
            "is_negative": is_negative,
            "intent_summary": intent_summary,
            "verdict": verdict["verdict"],
            "judge_correct": judge_correct,
            "reason": verdict["reason"],
            "rule_based_slot_match": r["pm_slot_match"],
            "pm_reply": r["pm_reply"],
            "core_memory_final": r.get("core_memory_final", ""),
            "constraint_in_core": r.get("constraint_in_core", False),
        }
        judge_results.append(rec)
        if judge_file:
            judge_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            judge_file.flush()

        if (i + 1) % 10 == 0 or (i + 1) == len(results):
            jc = sum(1 for j in judge_results if j["judge_correct"])
            print(f"  Judge progress: {i+1}/{len(results)} | "
                  f"Correct: {jc}/{len(judge_results)} = "
                  f"{jc/len(judge_results):.1%}")

    if judge_file:
        judge_file.close()

    return judge_results


# ── Run one (category, variant, bin) ─────────────────────────────────────

def run_pm(category: str, variant: str, bin_label: str,
           client: LLMClient, temperature: float, concurrency: int,
           results_dir: Path, judge_dir: Path,
           emb_client: Any, emb_deployment: str,
           top_k: int, core_limit: int,
           judge: PMJudge | None = None,
           judge_only: bool = False,
           max_samples: int = 0) -> tuple[list[dict], list[dict] | None]:
    """Run simulated Letta PM evaluation for a specific category/variant/bin."""
    data_path = DATA_DIR / variant / category / f"{category}_{bin_label}.jsonl"
    if not data_path.exists():
        print(f"  [{category}/{variant}/{bin_label}] SKIP — "
              f"data not found at {data_path}")
        return [], None

    entries = [json.loads(l) for l in data_path.read_text().splitlines()
               if l.strip()]
    if not entries:
        return [], None

    if max_samples > 0:
        entries = entries[:max_samples]

    # ── Judge-only mode ──────────────────────────────────────────────
    if judge_only:
        pm_path = (results_dir / variant / category
                   / f"pm_{category}_{bin_label}.jsonl")
        if not pm_path.exists():
            print(f"  [{category}/{variant}/{bin_label}] SKIP judge-only — "
                  f"no PM results at {pm_path}")
            return [], None
        results = [json.loads(l) for l in pm_path.read_text().splitlines()
                   if l.strip()]
        print(f"\n  [{category}/{variant}/{bin_label}] "
              f"Judge-only on {len(results)} existing PM results...")
        if judge is not None:
            j_path = (judge_dir / variant / category
                      / f"judge_{category}_{bin_label}.jsonl")
            judge_results = _run_judge_batch(
                results, entries, judge, judge_path=j_path,
            )
            jc = sum(1 for j in judge_results if j["judge_correct"])
            print(f"  [{category}/{variant}/{bin_label}] Judge done | "
                  f"Correct: {jc}/{len(judge_results)} = "
                  f"{jc/len(judge_results):.1%}")
            return results, judge_results
        return results, None

    # ── Resume support: load existing results ────────────────────────
    out_path = (results_dir / variant / category
                / f"pm_{category}_{bin_label}.jsonl")
    existing_results = []
    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                existing_results.append(rec)
                done_ids.add(rec["task_id"])

    remaining = [e for e in entries if e["task_id"] not in done_ids]

    if not remaining:
        print(f"\n  [{category}/{variant}/{bin_label}] "
              f"SKIP — all {len(entries)} already done")
        results = existing_results
    else:
        print(
            f"\n  [{category}/{variant}/{bin_label}] "
            f"Running Letta-sim (core={core_limit}, top-k={top_k}) "
            f"on {len(remaining)} samples..."
            + (f" [{len(done_ids)} already done]" if done_ids else "")
        )

        new_results = asyncio.run(_run_batch_async(
            remaining, client, temperature, concurrency,
            top_k, core_limit, emb_client, emb_deployment,
            category, out_path=out_path,
        ))

        results = existing_results + new_results

    # ── Print summary ───────────────────────────────────────────────
    scores = [PMScore(**{k: r[k] for k in PMScore.__dataclass_fields__})
              for r in results]
    agg = pm_aggregate(scores)
    is_neg = variant in NEGATIVE_VARIANTS
    neg_label = " [NEG]" if is_neg else ""

    print(f"  [{category}/{variant}/{bin_label}]{neg_label} "
          f"Done" +
          (f" | Avg slot: {agg.get('pos_avg_slot_match', 0):.3f}"
           if not is_neg else ""))

    # ── Judge ────────────────────────────────────────────────────────
    judge_results = None
    if judge is not None:
        j_path = (judge_dir / variant / category
                  / f"judge_{category}_{bin_label}.jsonl")

        # Resume support for judge
        existing_judge = []
        judge_done_ids = set()
        if j_path.exists():
            for line in j_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    existing_judge.append(rec)
                    judge_done_ids.add(rec["task_id"])

        remaining_for_judge = [r for r in results
                               if r["task_id"] not in judge_done_ids]
        if not remaining_for_judge:
            print(f"  [{category}/{variant}/{bin_label}] "
                  f"Judge SKIP — all done")
            judge_results = existing_judge
        else:
            if judge_done_ids:
                print(f"  [{category}/{variant}/{bin_label}] Judge resuming: "
                      f"{len(judge_done_ids)} done, "
                      f"{len(remaining_for_judge)} remaining...")
            else:
                print(f"  [{category}/{variant}/{bin_label}] "
                      f"Running LLM judge...")

            remaining_entries = [e for e in entries
                                if e["task_id"] not in judge_done_ids]
            new_judge = _run_judge_batch(
                remaining_for_judge, remaining_entries, judge,
                judge_path=j_path,
            )

            judge_results = existing_judge + new_judge
            jc = sum(1 for j in judge_results
                     if j.get("judge_correct", False))
            print(f"  [{category}/{variant}/{bin_label}] Judge done | "
                  f"Correct: {jc}/{len(judge_results)} = "
                  f"{jc/len(judge_results):.1%}")

    return results, judge_results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Simulated Letta (MemGPT) tiered-memory PM evaluation"
    )
    ap.add_argument("--category", choices=CATEGORIES)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--bin", choices=BIN_LABELS, default="base")
    ap.add_argument("--all", action="store_true",
                    help="Run all categories and variants")
    ap.add_argument("--judge", action="store_true",
                    help="Also run LLM-as-a-Judge semantic evaluation")
    ap.add_argument("--judge-only", action="store_true",
                    help="Re-run judge on existing PM results (skip memory construction)")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="Temperature for response generation")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Number of concurrent samples (keep low, many LLM "
                         "calls per sample). Default: 1")
    ap.add_argument("--top-k", type=int, default=6,
                    help="Number of archival memories to retrieve at trigger")
    ap.add_argument("--core-limit", type=int, default=DEFAULT_CORE_LIMIT,
                    help="Core memory character limit (default: 2000)")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Limit number of samples per category/variant (0 = all)")
    args = ap.parse_args()

    # ── Resolve model name and build output dirs ────────────────────────
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENTS", "unknown-model")
    emb_client, emb_deployment = _embedding_client()
    run_tag = _make_run_tag(model, args.temperature, args.concurrency,
                            args.top_k, args.core_limit, emb_deployment)
    results_dir = OUTPUT_ROOT / run_tag / "pm"
    judge_dir = OUTPUT_ROOT / run_tag / "pm_judge"

    print(f"  Model         : {model}")
    print(f"  Embedding     : {emb_deployment}")
    print(f"  Top-k         : {args.top_k}")
    print(f"  Core limit    : {args.core_limit} chars")
    print(f"  Run tag       : {run_tag}")
    print(f"  Results dir   : {results_dir}")

    client = LLMClient()
    judge_enabled = args.judge or args.judge_only
    judge_client = LLMClient(deployment=JUDGE_DEPLOYMENT) if judge_enabled else None
    judge = PMJudge(client=judge_client) if judge_enabled else None

    if args.all:
        categories = CATEGORIES
        variants = VARIANTS
    elif args.category and args.variant:
        categories = [args.category]
        variants = [args.variant]
    elif args.category:
        categories = [args.category]
        variants = VARIANTS
    elif args.variant:
        categories = CATEGORIES
        variants = [args.variant]
    else:
        ap.error("Provide --category and/or --variant, or use --all")

    bins = [args.bin]

    all_results: list[dict] = []
    all_judge: list[dict] = []
    for cat in categories:
        for variant in variants:
            for bl in bins:
                results, judge_results = run_pm(
                    cat, variant, bl, client,
                    args.temperature, args.concurrency,
                    results_dir, judge_dir,
                    emb_client, emb_deployment,
                    args.top_k, args.core_limit,
                    judge=judge,
                    judge_only=args.judge_only,
                    max_samples=args.max_samples,
                )
                if results:
                    all_results.extend(results)
                if judge_results:
                    all_judge.extend(judge_results)

    # ── Summary ─────────────────────────────────────────────────────────
    if not all_results:
        print("\nNo results produced.")
        return

    print(f"\n{'='*80}")
    print("  Letta-Sim PM Evaluation Summary")
    print(f"{'='*80}")

    def _to_scores(result_dicts: list[dict]) -> list[PMScore]:
        fields = set(PMScore.__dataclass_fields__)
        return [PMScore(**{k: r[k] for k in fields}) for r in result_dicts]

    by_key = defaultdict(list)
    for r in all_results:
        by_key[(r["category"], r["variant"])].append(r)
    judge_by_key = defaultdict(list)
    for j in all_judge:
        judge_by_key[(j["category"], j["variant"])].append(j)

    header = (f"  {'Category':<14} {'Variant':<22} {'n':>4}  "
              f"{'Slot':>6}")
    if all_judge:
        header += f"  {'Jdg Corr':>10}"
    print(header)
    print(f"  {'-'*70}")

    for cat in CATEGORIES:
        for variant in VARIANTS:
            key = (cat, variant)
            if key not in by_key:
                continue
            rs = by_key[key]
            agg = pm_aggregate(_to_scores(rs))
            is_neg = variant in NEGATIVE_VARIANTS
            slot_str = (f"{agg.get('pos_avg_slot_match', 0):>6.3f}"
                        if not is_neg else f"{'N/A':>6}")
            line = (f"  {cat:<14} {variant:<22} {agg['n']:>4}"
                    f"  {slot_str}")
            if key in judge_by_key:
                js = judge_by_key[key]
                jc = sum(1 for j in js if j["judge_correct"])
                line += f"  {jc/len(js):>9.1%}"
            print(line)

    # Overall summary by variant type
    print(f"  {'-'*70}")
    for variant in VARIANTS:
        rs = [r for r in all_results if r["variant"] == variant]
        if not rs:
            continue
        agg = pm_aggregate(_to_scores(rs))
        is_neg = variant in NEGATIVE_VARIANTS
        slot_str = (f"{agg.get('pos_avg_slot_match', 0):>6.3f}"
                    if not is_neg else f"{'N/A':>6}")
        line = (f"  {'OVERALL':<14} {variant:<22} {agg['n']:>4}"
                f"  {slot_str}")
        js = [j for j in all_judge if j["variant"] == variant]
        if js:
            jc = sum(1 for j in js if j["judge_correct"])
            line += f"  {jc/len(js):>9.1%}"
        print(line)

    print(f"{'='*80}")


if __name__ == "__main__":
    main()

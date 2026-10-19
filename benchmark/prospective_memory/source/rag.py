"""RAG PM evaluation: retrieve top-k prior turns instead of full context.

Same system prompt as blended_close. Supports base and 40K splits.
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
DATA_ROOT = PKG_ROOT / "data"
DATA_DIR_MAP = {
    "base": DATA_ROOT / "base_context",
    "40K":  DATA_ROOT / "40K_context",
}

OUTPUT_ROOT = PKG_ROOT / "main_results" / "rag"

CATEGORIES = [
    "state_tracking",
    "temporal_grounding",
    "logical_adherence",
    "attention_recovery",
    "safe_coding",
]
VARIANTS = ["positive_clean", "positive_overloaded", "negative_clean"]
BIN_LABELS = ["base", "40K"]

# Judge always uses gpt-4o regardless of the evaluation model.
JUDGE_DEPLOYMENT = "gpt-4o"

# Turn function labels that mark constraint information
CONSTRAINT_FUNCTIONS = {"constraint", "rule_constraint", "condition_constraint"}

# Retrieval cache — keyed by embedding model, independent of LLM / top-k
CACHE_ROOT = OUTPUT_ROOT / "retrieval_cache"

# Max tokens per chunk for interrupt turns.
# text-embedding-3-small accepts up to 8191 tokens; we leave a margin.
_CHUNK_MAX_TOKENS = 8_000


def _make_run_tag(model: str, temperature: float, concurrency: int,
                  top_k: int, emb_deployment: str) -> str:
    """Build a filesystem-safe run tag.
    """
    tag = re.sub(r'[^a-zA-Z0-9]+', '_', model).strip('_')
    emb = re.sub(r'[^a-zA-Z0-9]+', '_', emb_deployment).strip('_')
    return f"{tag}_temp{temperature}_con{concurrency}_k{top_k}_emb_{emb}"


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


# text-embedding-3-small uses cl100k_base and accepts up to 8191 tokens.
# We keep the *tail* of long texts so that the conclusion/answer
# (typically at the end) dominates the embedding rather than preamble
# or chain-of-thought reasoning.
_EMB_MAX_TOKENS = 8191
_EMB_ENCODING = tiktoken.get_encoding("cl100k_base")


def _tail_truncate(text: str, max_tokens: int = _EMB_MAX_TOKENS) -> str:
    """Keep the last *max_tokens* tokens of *text* for embedding."""
    tokens = _EMB_ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _EMB_ENCODING.decode(tokens[-max_tokens:])


def _embed_batch(emb_client: Any, emb_deployment: str,
                 texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returns L2-normalized matrix.

    Long texts are tail-truncated to ~8 191 tokens before embedding so
    that the answer/conclusion portion is captured rather than being
    lost to the API's front-truncation default.
    """
    vecs: list[list[float]] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = [_tail_truncate(t) if len(t) > 0 else " "
                 for t in texts[i:i + batch_size]]
        # Retry with exponential backoff on rate-limit errors
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


# ── Retrieval helpers ────────────────────────────────────────────────────

def _find_constraint_turns(turns: list[dict],
                           trigger_idx: int) -> dict[str, list[int]]:
    """Return {function_name: [turn_indices]} for constraint turns before trigger.

    Recognised functions: constraint, rule_constraint, condition_constraint.
    """
    result: dict[str, list[int]] = {}
    for i, t in enumerate(turns[:trigger_idx]):
        func = t.get("function", "")
        if func in CONSTRAINT_FUNCTIONS:
            result.setdefault(func, []).append(i)
    return result


def _compute_full_ranking(
    memory_texts: list[str],
    emb_client: Any,
    emb_deployment: str,
    trigger_text: str,
) -> tuple[list[int], list[float]]:
    """Embed memories + query, return (ranked_indices, similarities).

    ranked_indices is sorted by descending cosine similarity.
    """
    if not memory_texts:
        return [], []
    memory_vecs = _embed_batch(emb_client, emb_deployment, memory_texts)
    query_vec = _embed_batch(emb_client, emb_deployment, [trigger_text])[0]
    sims = memory_vecs @ query_vec
    ranked = np.argsort(-sims)
    return ranked.tolist(), [float(sims[i]) for i in ranked]


# ── Core evaluation ─────────────────────────────────────────────────────

def _format_turn(turn: dict) -> str:
    """Format a single turn for embedding / memory storage."""
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
    # Approximate chars-per-token ratio for efficient boundary search
    cpt = max(1, len(text) / max(1, _count_tokens(text)))
    chunks: list[str] = []
    while text:
        if _count_tokens(text) <= max_tokens:
            chunks.append(text)
            break
        # Estimate char budget from token limit (with small margin)
        est_chars = int(max_tokens * cpt * 0.95)
        est_chars = min(est_chars, len(text))
        # Try paragraph boundary first
        cut = text.rfind("\n\n", 0, est_chars)
        if cut < est_chars * 0.3:
            cut = text.rfind(". ", 0, est_chars)
        if cut < est_chars * 0.3:
            cut = text.rfind("\n", 0, est_chars)
        if cut < est_chars * 0.3:
            cut = est_chars
        # Refine: if chunk still exceeds max_tokens, shrink
        candidate = text[:cut + 1].rstrip()
        while candidate and _count_tokens(candidate) > max_tokens:
            cut = int(cut * 0.9)
            candidate = text[:cut + 1].rstrip()
        if not candidate:
            # Fallback: encode-decode exact token slice
            tokens = _EMB_ENCODING.encode(text)
            candidate = _EMB_ENCODING.decode(tokens[:max_tokens])
            cut = len(candidate) - 1
        chunks.append(candidate)
        text = text[cut + 1:].lstrip()
    return chunks if chunks else [text]


def _run_one_rag(
    entry: dict,
    client: LLMClient,
    temperature: float,
    top_k: int,
    emb_client: Any | None = None,
    emb_deployment: str | None = None,
    cached_retrieval: dict | None = None,
    category: str = "",
) -> tuple[dict, dict]:
    """Run RAG PM evaluation for a single sample.

    Returns (result_dict, retrieval_cache_entry).
    If *cached_retrieval* is provided the embedding step is skipped.
    """
    sample_dict = entry["sample"]
    trigger_idx = sample_dict["ground_truth"]["trigger"]["trigger_turn_idx"]
    turns = sample_dict["turns"]
    need_chunk = category.startswith("attention_recovery")

    # 1. Build memory texts — optionally chunk long turns
    memory_texts: list[str] = []
    chunk_to_turn: list[int] = []  # chunk_idx -> original turn idx
    n_chunked_turns = 0
    for turn_idx, t in enumerate(turns[:trigger_idx]):
        text = _format_turn(t)
        if need_chunk:
            chunks = _chunk_text(text, _CHUNK_MAX_TOKENS)
            if len(chunks) > 1:
                n_chunked_turns += 1
                # Prepend speaker/timestamp prefix to continuation chunks
                ts = f" [{t['timestamp']}]" if t.get("timestamp") else ""
                speaker = "USER" if t["role"] == "user" else "ASSISTANT"
                prefix = f"{speaker}{ts} (cont'd): "
                for ci, chunk in enumerate(chunks):
                    memory_texts.append(prefix + chunk if ci > 0 else chunk)
                    chunk_to_turn.append(turn_idx)
                print(
                    f"  Turn {t.get('idx', turn_idx)} "
                    f"({t['role']}): {_count_tokens(text)} tokens → {len(chunks)} chunks"
                )
            else:
                memory_texts.append(chunks[0])
                chunk_to_turn.append(turn_idx)
        else:
            memory_texts.append(text)
            chunk_to_turn.append(turn_idx)
    if n_chunked_turns > 0:
        print(
            f"  Chunking summary: {n_chunked_turns} turns split "
            f"(total memory items: {len(memory_texts)})"
        )

    # 2. Compute or load retrieval ranking
    constraint_turns = _find_constraint_turns(turns, trigger_idx)
    if cached_retrieval is not None:
        ranked_indices = cached_retrieval["ranked_indices"]
        similarities = cached_retrieval["similarities"]
        # Override constraint_turns from cache if available
        constraint_turns = cached_retrieval.get("constraint_turns", constraint_turns)
    else:
        trigger_text = _format_turn(turns[trigger_idx])
        ranked_indices, similarities = _compute_full_ranking(
            memory_texts, emb_client, emb_deployment, trigger_text,
        )

    # 3. Select top-k in chronological order
    k = min(top_k, len(memory_texts))
    selected_indices = sorted(ranked_indices[:k])
    retrieved = [memory_texts[i] for i in selected_indices]

    # 4. Check which constraint turns are in the top-k
    #    Map chunk indices back to original turn indices
    selected_turn_set = set(chunk_to_turn[i] for i in selected_indices)
    constraint_in_topk: dict[str, bool] = {
        func: any(i in selected_turn_set for i in idxs)
        for func, idxs in constraint_turns.items()
    }

    # 5. Generate with retrieved memories
    mem_block = ("\n".join(f"- {m}" for m in retrieved)
                 if retrieved else "(no memories yet)")
    current_msg = _format_turn(turns[trigger_idx])
    prompt = get_system_prompt(category)

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                "Relevant memories retrieved from our conversation history:\n"
                f"{mem_block}\n\n"
                f"Latest user message:\n{current_msg}\n\n"
                "Respond helpfully."
            ),
        },
    ]

    try:
        reply = client.complete_chat(
            messages=messages, max_tokens=1024, temperature=temperature
        )
    except Exception as e:
        reply = f"[ERROR: {e}]"

    # 6. Score
    score = pm_score_sample(entry, reply)
    result = score_to_dict(score)
    result["pm_reply"] = reply
    result["n_retrieved"] = len(retrieved)
    result["retrieved_turn_indices"] = sorted(set(chunk_to_turn[i] for i in selected_indices))
    result["constraint_turns"] = constraint_turns
    result["constraint_in_topk"] = constraint_in_topk

    # 7. Build retrieval cache entry (full ranking, reusable for any top-k)
    cache_entry = {
        "task_id": entry["task_id"],
        "n_memory_turns": len(memory_texts),
        "ranked_indices": ranked_indices,
        "similarities": similarities,
        "constraint_turns": constraint_turns,
    }

    return result, cache_entry


async def _run_batch_async(
    entries: list[dict],
    client: LLMClient,
    temperature: float,
    concurrency: int,
    top_k: int,
    emb_client: Any | None = None,
    emb_deployment: str | None = None,
    retrieval_cache: dict[str, dict] | None = None,
    category: str = "",
    out_path: Path | None = None,
    cache_path: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run RAG PM evaluation concurrently with incremental saving.

    Returns (results, cache_entries).
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    cache_entries: list[dict] = []

    # Open files in append mode for incremental saving
    out_file = None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("a")
    cache_file = None
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path.open("a")

    async def _worker(entry):
        cached = (retrieval_cache.get(entry["task_id"])
                  if retrieval_cache else None)
        async with sem:
            return await asyncio.to_thread(
                _run_one_rag, entry, client, temperature, top_k,
                emb_client, emb_deployment, cached, category,
            )

    tasks = [_worker(e) for e in entries]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result, cache_entry = await coro
        results.append(result)
        cache_entries.append(cache_entry)
        # Save immediately
        if out_file:
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
        if cache_file:
            cache_file.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")
            cache_file.flush()
        if (i + 1) % 10 == 0 or (i + 1) == len(entries):
            print(f"  Progress: {i+1}/{len(entries)}")

    if out_file:
        out_file.close()
    if cache_file:
        cache_file.close()

    return results, cache_entries


# ══════════════════════════════════════════════════════════════════════════
#  LLM-as-a-Judge
# ══════════════════════════════════════════════════════════════════════════

def _run_judge_batch(
    results: list[dict],
    entries: list[dict],
    judge: PMJudge,
    judge_path: Path | None = None,
) -> list[dict]:
    """Run LLM-as-a-Judge on PM results, saving incrementally."""
    entry_by_id = {e["task_id"]: e for e in entries}
    judge_results = []

    # Open file for incremental saving
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


# ══════════════════════════════════════════════════════════════════════════
#  Cache helpers
# ══════════════════════════════════════════════════════════════════════════

def _load_retrieval_cache(cache_path: Path) -> dict[str, dict] | None:
    """Load retrieval cache JSONL. Returns {task_id: cache_entry} or None."""
    if not cache_path.exists():
        return None
    cache: dict[str, dict] = {}
    for line in cache_path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            cache[entry["task_id"]] = entry
    return cache


def _save_retrieval_cache(cache_entries: list[dict], cache_path: Path) -> None:
    """Save retrieval cache JSONL."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        for entry in cache_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════
#  Run one (category, variant, bin)
# ══════════════════════════════════════════════════════════════════════════

def run_pm(
    category: str, variant: str, bin_label: str,
    client: LLMClient, temperature: float, concurrency: int,
    results_dir: Path, judge_dir: Path,
    emb_client: Any, emb_deployment: str,
    top_k: int,
    judge: PMJudge | None = None,
    max_samples: int | None = None,
) -> tuple[list[dict], list[dict] | None]:
    """Run RAG PM evaluation for a specific category, variant, and bin.

    Saves results and cache incrementally. Resumes from partial results.
    """
    data_dir = DATA_DIR_MAP.get(bin_label, DATA_DIR_MAP["base"])
    data_path = data_dir / variant / category / f"{category}_{bin_label}.jsonl"
    if not data_path.exists():
        return [], None

    entries = [json.loads(l) for l in data_path.read_text().splitlines()
               if l.strip()]
    if not entries:
        return [], None

    if max_samples is not None:
        entries = entries[:max_samples]

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

    # ── Try loading retrieval cache (partial or full) ────────────────
    cache_path = (CACHE_ROOT / emb_deployment / variant / category
                  / f"retrieval_{category}_{bin_label}.jsonl")
    retrieval_cache = _load_retrieval_cache(cache_path)

    # Determine which entries still need processing
    remaining = [e for e in entries if e["task_id"] not in done_ids]

    if not remaining:
        print(f"\n  [{category}/{variant}/{bin_label}] "
              f"SKIP — all {len(entries)} already done")
        results = existing_results
    else:
        # Check if cache covers remaining entries
        remaining_ids = {e["task_id"] for e in remaining}
        if retrieval_cache is not None and remaining_ids <= retrieval_cache.keys():
            print(f"\n  [{category}/{variant}/{bin_label}] "
                  f"Using cached retrieval | top-k={top_k} | "
                  f"{len(remaining)} remaining (of {len(entries)})...")
            use_cache = True
        else:
            print(f"\n  [{category}/{variant}/{bin_label}] "
                  f"Running RAG (top-k={top_k}) on {len(remaining)} samples "
                  f"(computing embeddings)..."
                  + (f" [{len(done_ids)} already done]" if done_ids else ""))
            use_cache = False
            if retrieval_cache is None:
                retrieval_cache = None

        start = time.time()

        new_results, cache_entries = asyncio.run(_run_batch_async(
            remaining, client, temperature, concurrency, top_k,
            emb_client=None if use_cache else emb_client,
            emb_deployment=None if use_cache else emb_deployment,
            retrieval_cache=retrieval_cache,
            category=category,
            out_path=out_path,
            cache_path=None if use_cache else cache_path,
        ))

        elapsed = time.time() - start

        results = existing_results + new_results
        is_neg = variant in NEGATIVE_VARIANTS
        neg_label = " [NEG]" if is_neg else ""
        scores = [PMScore(**{k: r[k] for k in PMScore.__dataclass_fields__})
                  for r in results]
        agg = pm_aggregate(scores)
        print(f"  [{category}/{variant}/{bin_label}]{neg_label} "
              f"Done in {elapsed:.1f}s" +
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

        remaining_for_judge = [r for r in results if r["task_id"] not in judge_done_ids]
        if not remaining_for_judge:
            print(f"  [{category}/{variant}/{bin_label}] Judge SKIP — all done")
            judge_results = existing_judge
        else:
            if judge_done_ids:
                print(f"  [{category}/{variant}/{bin_label}] Judge resuming: "
                      f"{len(judge_done_ids)} done, {len(remaining_for_judge)} remaining...")
            else:
                print(f"  [{category}/{variant}/{bin_label}] Running LLM judge...")

            remaining_entries = [e for e in entries if e["task_id"] not in judge_done_ids]
            new_judge = _run_judge_batch(
                remaining_for_judge, remaining_entries, judge, judge_path=j_path
            )

            judge_results = existing_judge + new_judge
            jc = sum(1 for j in judge_results if j.get("judge_correct", False))
            print(f"  [{category}/{variant}/{bin_label}] Judge done | "
                  f"Correct: {jc}/{len(judge_results)} = "
                  f"{jc/len(judge_results):.1%}")

    return results, judge_results


# ══════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════

def _print_summary(
    all_results: list[dict],
    all_judge: list[dict],
    model_label: str,
    top_k: int,
):
    from collections import defaultdict

    print(f"\n{'='*80}")
    print(f"  RAG PM Evaluation Summary (top-k={top_k}, {model_label})")
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

    header = f"  {'Category':<14} {'Variant':<22} {'n':>4}  {'Slot':>6}"
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

    # Per variant
    print(f"  {'-'*70}")
    for variant in VARIANTS:
        rs = [r for r in all_results if r["variant"] == variant]
        if not rs:
            continue
        agg = pm_aggregate(_to_scores(rs))
        is_neg = variant in NEGATIVE_VARIANTS
        slot_str = (f"{agg.get('pos_avg_slot_match', 0):>6.3f}"
                    if not is_neg else f"{'N/A':>6}")
        line = (f"  {'ALL':<14} {variant:<22} {agg['n']:>4}"
                f"  {slot_str}")
        js = [j for j in all_judge if j["variant"] == variant]
        if js:
            jc = sum(1 for j in js if j["judge_correct"])
            line += f"  {jc/len(js):>9.1%}"
        print(line)

    print(f"{'='*80}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="RAG-based PM evaluation")
    ap.add_argument("--category", choices=CATEGORIES)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--bin", default="base", choices=BIN_LABELS,
                    help="Bin label: 'base' (short) or '40K' (40K-token context)")
    ap.add_argument("--all", action="store_true",
                    help="Run all categories and variants")
    ap.add_argument("--judge", action="store_true",
                    help="Also run LLM-as-a-Judge semantic evaluation")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=6,
                    help="Number of memory snippets to retrieve")
    ap.add_argument("--recompute-cache", action="store_true",
                    help="Force recompute embeddings even if cache exists")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Max samples per (category, variant, bin). For quick testing.")
    args = ap.parse_args()

    # ── Resolve model name and build output dirs ────────────────────
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENTS", "unknown-model")
    emb_client, emb_deployment = _embedding_client()
    run_tag = _make_run_tag(model, args.temperature, args.concurrency,
                            args.top_k, emb_deployment)
    results_dir = OUTPUT_ROOT / run_tag / "pm"
    judge_dir = OUTPUT_ROOT / run_tag / "pm_judge"

    # If --recompute-cache, delete existing cache for this embedding model
    if args.recompute_cache:
        import shutil
        cache_dir = CACHE_ROOT / emb_deployment
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"  Cleared cache: {cache_dir}")

    print(f"  Model       : {model}")
    print(f"  Embedding   : {emb_deployment}")
    print(f"  Top-k       : {args.top_k}")
    print(f"  Bin         : {args.bin}")
    print(f"  Run tag     : {run_tag}")
    print(f"  Results dir : {results_dir}")
    print(f"  Cache dir   : {CACHE_ROOT / emb_deployment}")

    client = LLMClient()
    judge_client = LLMClient(deployment=JUDGE_DEPLOYMENT) if args.judge else None
    judge = PMJudge(client=judge_client) if args.judge else None

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
                    emb_client, emb_deployment, args.top_k,
                    judge=judge,
                    max_samples=args.max_samples,
                )
                if results:
                    all_results.extend(results)
                if judge_results:
                    all_judge.extend(judge_results)

    # ── Summary ─────────────────────────────────────────────────────
    _print_summary(all_results, all_judge, model, args.top_k)


if __name__ == "__main__":
    main()

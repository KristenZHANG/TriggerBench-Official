"""Mem0 PM evaluation: LLM fact extraction + vector retrieval.

Same system prompt as blended_close. Base split only (no 40K).
Flags: ../README.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import tiktoken

# Give each process its own MEM0_DIR to avoid the global
# ~/.mem0/migrations_qdrant/ Qdrant file-lock conflict when running
# multiple mem0_eval.py instances in parallel (e.g. different categories).
# Must be set BEFORE `from mem0 import Memory` (even lazy imports).
_MEM0_PROCESS_DIR = tempfile.mkdtemp(prefix="mem0_home_")
os.environ["MEM0_DIR"] = _MEM0_PROCESS_DIR
# Newer mem0 also opens a second local Qdrant at {MEM0_DIR}/migrations_qdrant
# for telemetry. Local Qdrant takes an exclusive flock, so a second
# Memory.from_config() in the same process crashes unless that client is
# closed. Disable telemetry entirely — it is unused for scoring.
os.environ.setdefault("MEM0_TELEMETRY", "False")
import atexit as _atexit
_atexit.register(shutil.rmtree, _MEM0_PROCESS_DIR, True)

# ── Path setup (relative to prospective_memory/) ──────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent                    # source/
PKG_ROOT = SCRIPT_DIR.parent                                    # prospective_memory/
BENCHMARK_ROOT = PKG_ROOT.parent                                # benchmark/
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT / "utils"))

from client import CONFIG, LLMClient, _base_url, _get_api_key, _parse_env_list
from longcontext_blended_close import get_system_prompt
from pm_judge import PMJudge
from pm_scorer import (
    NEGATIVE_VARIANTS, PMScore,
    score_sample as pm_score_sample,
    score_to_dict,
    aggregate as pm_aggregate,
)

# ── Constants ────────────────────────────────────────────────────────────
DATA_ROOT = PKG_ROOT / "data"
DATA_DIR = DATA_ROOT / "base_context"                            # base bin only

OUTPUT_ROOT = PKG_ROOT / "main_results" / "mem0"

CATEGORIES = [
    "state_tracking",
    "temporal_grounding",
    "logical_adherence",
    "attention_recovery",
    "safe_coding",
]
VARIANTS = ["positive_clean", "positive_overloaded", "negative_clean"]

JUDGE_DEPLOYMENT = "gpt-4o"

# Turn function labels that mark constraint information
CONSTRAINT_FUNCTIONS = {"constraint", "rule_constraint", "condition_constraint"}

# text-embedding-3-small uses cl100k_base, max 8191 tokens.
_ENC = tiktoken.get_encoding("cl100k_base")

# Max tokens per turn chunk for Mem0 ingestion.  Long turns
# (e.g. interrupt reasoning traces) are split into multiple
# chunk-messages so all content is ingested without exceeding the
# embedding limit and LLM fact-extraction stays high-quality.
MAX_TURN_TOKENS = 2_000   # per chunk
MAX_BATCH_TOKENS = 8_000  # per Mem0 add() call (embedding dedup concat)


# ── Mem0 setup ───────────────────────────────────────────────────────────

# Serialize Memory() construction across threads (Mem0 locks ~/.mem0/migrations_qdrant)
_MEM0_INIT_LOCK = threading.Lock()

_EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small_1": 1536,
}


def _chat_llm_config() -> dict:
    """Mem0 fact-extraction LLM: same OpenAI-compatible env as LLMClient."""
    deployments = _parse_env_list("AZURE_OPENAI_DEPLOYMENTS")
    if not deployments:
        model = os.environ.get("OPENAI_MODEL", "").strip()
        deployments = [model] if model else []
    if not deployments:
        raise ValueError("Set AZURE_OPENAI_DEPLOYMENTS (or OPENAI_MODEL)")
    chat_deployment = deployments[0]
    api_key = _get_api_key()
    if not api_key:
        raise ValueError(
            "export AZURE_OPENAI_API_KEY=...  (OPENAI_API_KEY is also accepted)"
        )
    llm_cfg: dict = {
        "model": chat_deployment,
        "temperature": 0.2,
        "max_tokens": 4096,
        "api_key": api_key,
    }
    base_url = _base_url()
    if base_url:
        llm_cfg["openai_base_url"] = base_url
    return {"provider": "openai", "config": llm_cfg}


def _embedder_config() -> tuple[dict, int]:
    """Build Mem0 Azure OpenAI embedder config.

      AZURE_OPENAI_EMBEDDINGS_ENDPOINT   (falls back to AZURE_OPENAI_ENDPOINT)
      AZURE_OPENAI_EMBEDDINGS_DEPLOYMENTS (required)
      AZURE_OPENAI_EMBEDDINGS_API_VERSION
      AZURE_OPENAI_EMBEDDINGS_SCOPE
    """
    api_version = os.environ.get(
        "AZURE_OPENAI_API_VERSION", CONFIG["defaults"]["api_version"]
    )
    emb_endpoint = (
        os.environ.get("AZURE_OPENAI_EMBEDDINGS_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    deployments = _parse_env_list("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENTS")
    if not emb_endpoint or not deployments:
        raise ValueError(
            "Set AZURE_OPENAI_EMBEDDINGS_DEPLOYMENTS (and optionally "
            "AZURE_OPENAI_EMBEDDINGS_ENDPOINT)"
        )
    emb_deployment = deployments[0]
    emb_api_version = os.environ.get(
        "AZURE_OPENAI_EMBEDDINGS_API_VERSION", api_version
    )

    emb_scope = os.environ.get("AZURE_OPENAI_EMBEDDINGS_SCOPE")
    if emb_scope:
        import mem0.embeddings.azure_openai as _mem0_emb_mod
        _mem0_emb_mod.SCOPE = emb_scope

    return {
        "provider": "azure_openai",
        "config": {
            "model": emb_deployment,
            "azure_kwargs": {
                "azure_endpoint": emb_endpoint,
                "api_version": emb_api_version,
                "azure_deployment": emb_deployment,
            },
        },
    }, _EMBED_DIMS.get(emb_deployment, 1536)


def _build_mem0_config(storage_path: str, collection_name: str) -> dict:
    """Build Mem0 config: OpenAI-compatible chat + Azure embedder."""
    embedder, dims = _embedder_config()
    return {
        "llm": _chat_llm_config(),
        "embedder": embedder,
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": dims,
                "path": storage_path,
                "on_disk": True,
            },
        },
    }


def _close_mem0(mem) -> None:
    """Drop exclusive local-Qdrant flocks (sample store + telemetry + entity)."""
    for attr in ("vector_store", "_telemetry_vector_store", "_entity_store"):
        store = getattr(mem, attr, None)
        client = getattr(store, "client", None) if store is not None else None
        if client is None:
            continue
        try:
            client.close()
        except Exception:
            pass
    try:
        mem.close()
    except Exception:
        pass


def _make_run_tag(model: str, temperature: float, concurrency: int,
                  top_k: int, batch_size: int) -> str:
    """Build a filesystem-safe run tag."""
    tag = re.sub(r'[^a-zA-Z0-9]+', '_', model).strip('_')
    return f"{tag}_temp{temperature}_con{concurrency}_k{top_k}_b{batch_size}"


# ── Helpers ──────────────────────────────────────────────────────────────

def _find_constraint_turns(turns: list[dict],
                           trigger_idx: int) -> dict[str, list[int]]:
    """Return {function_name: [turn_indices]} for constraint turns before trigger."""
    result: dict[str, list[int]] = {}
    for i, t in enumerate(turns[:trigger_idx]):
        func = t.get("function", "")
        if func in CONSTRAINT_FUNCTIONS:
            result.setdefault(func, []).append(i)
    return result


def _format_turn(turn: dict) -> str:
    """Format a single turn for memory ingestion."""
    ts = f" [{turn['timestamp']}]" if turn.get("timestamp") else ""
    speaker = "USER" if turn["role"] == "user" else "ASSISTANT"
    return f"{speaker}{ts}: {turn['text']}"


def _count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_ENC.encode(text))


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
            tokens = _ENC.encode(text)
            candidate = _ENC.decode(tokens[:max_tokens])
            cut = len(candidate) - 1
        chunks.append(candidate)
        text = text[cut + 1:].lstrip()
    return chunks if chunks else [text]


# ── Core evaluation ─────────────────────────────────────────────────────

def _run_one_mem0(
    entry: dict,
    client: LLMClient,
    temperature: float,
    top_k: int,
    batch_size: int,
    category: str = "",
) -> dict:
    """Run Mem0 PM evaluation for a single sample.

    1. Create isolated Mem0 Memory instance (per-sample Qdrant).
    2. Ingest all turns before trigger in batches → LLM fact extraction.
    3. Search with trigger message → retrieve top-k memory facts.
    4. Generate response with system prompt + memories + trigger.
    5. Score.
    """
    from mem0 import Memory

    sample_dict = entry["sample"]
    trigger_idx = sample_dict["ground_truth"]["trigger"]["trigger_turn_idx"]
    turns = sample_dict["turns"]
    constraint_turns = _find_constraint_turns(turns, trigger_idx)

    # ── 1. Create isolated Mem0 instance ────────────────────────────
    tag = uuid.uuid4().hex[:12]
    storage = tempfile.mkdtemp(prefix=f"pm_mem0_{tag}_", dir="/tmp")
    config = _build_mem0_config(
        storage_path=storage, collection_name=f"pm_{tag}"
    )

    user_id = f"pm_{uuid.uuid4().hex[:10]}"

    try:
        # Mem0 uses a GLOBAL ~/.mem0/migrations_qdrant/ Qdrant instance
        # (for telemetry) plus per-sample instances.  Qdrant local mode
        # acquires NON_BLOCKING exclusive file locks during _load(), so
        # concurrent Memory construction / first-use will crash.
        # We serialise ALL mem0 I/O (init + add + search) under one lock;
        # the LLM generation call afterwards runs outside the lock.
        with _MEM0_INIT_LOCK:
            mem = Memory.from_config(config)

            # ── 2. Ingest all turns before trigger ──────────────────
            # Chunk any turn exceeding MAX_TURN_TOKENS for better
            # fact-extraction quality and to stay within embedding limits.
            messages = []
            n_chunked_turns = 0
            for t in turns[:trigger_idx]:
                role = "user" if t["role"] == "user" else "assistant"
                content = _format_turn(t)
                tok_count = _count_tokens(content)
                if tok_count > MAX_TURN_TOKENS:
                    chunks = _chunk_text(content, MAX_TURN_TOKENS)
                    if len(chunks) > 1:
                        n_chunked_turns += 1
                        ts = f" [{t['timestamp']}]" if t.get("timestamp") else ""
                        speaker = "USER" if t["role"] == "user" else "ASSISTANT"
                        prefix = f"{speaker}{ts} (cont'd): "
                        for ci, chunk in enumerate(chunks):
                            messages.append({"role": role, "content": prefix + chunk if ci > 0 else chunk})
                    else:
                        messages.append({"role": role, "content": chunks[0]})
                else:
                    messages.append({"role": role, "content": content})
            if n_chunked_turns > 0:
                print(
                    f"  Chunked {n_chunked_turns} long turns "
                    f"→ {len(messages)} total messages"
                )

            n_ingested = 0
            if messages:
                # Build batches respecting both message count AND total
                # token budget.  Mem0 concatenates the whole batch before
                # embedding for dedup, so the total must stay under the
                # embedding model's 8191-token limit.
                batches: list[list[dict]] = []
                cur_batch: list[dict] = []
                cur_tokens = 0
                for msg in messages:
                    msg_tokens = _count_tokens(msg["content"])
                    if cur_batch and (
                        len(cur_batch) >= batch_size
                        or cur_tokens + msg_tokens > MAX_BATCH_TOKENS
                    ):
                        batches.append(cur_batch)
                        cur_batch = []
                        cur_tokens = 0
                    cur_batch.append(msg)
                    cur_tokens += msg_tokens
                if cur_batch:
                    batches.append(cur_batch)

                for batch in batches:
                    try:
                        mem.add(batch, user_id=user_id)
                        n_ingested += len(batch)
                    except Exception as e:
                        print(
                            f"  Mem0 add batch ({len(batch)} msgs, "
                            f"{sum(len(m['content']) for m in batch)} chars) "
                            f"failed: {type(e).__name__}: {str(e)[:150]}"
                        )

            # ── 3. Search with trigger message ──────────────────────
            trigger_text = _format_turn(turns[trigger_idx])
            try:
                results = mem.search(
                    trigger_text,
                    filters={"user_id": user_id},
                    top_k=top_k,
                )
                if isinstance(results, dict):
                    memories = results.get("results", [])
                else:
                    memories = results
                mem_texts = [m.get("memory", "") for m in memories if m.get("memory")]
            except Exception as e:
                mem_texts = []
                print(f"  Mem0 search error: {type(e).__name__}: {str(e)[:150]}")

            # Close all local Qdrant clients before the next sample can
            # Memory.from_config() (local mode uses exclusive file locks).
            _close_mem0(mem)

        # ── 4. Generate response (outside lock) ────────────────────
        trigger_text = _format_turn(turns[trigger_idx])
        prompt = get_system_prompt(category)
        mem_block = (
            "\n".join(f"- {m}" for m in mem_texts)
            if mem_texts
            else "(no memories retrieved)"
        )

        gen_messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "Relevant memories extracted from our conversation history:\n"
                    f"{mem_block}\n\n"
                    f"Latest user message:\n{trigger_text}\n\n"
                    "Respond helpfully."
                ),
            },
        ]

        try:
            reply = client.complete_chat(
                messages=gen_messages, max_tokens=1024, temperature=temperature
            )
        except Exception as e:
            reply = f"[ERROR: {e}]"

    finally:
        # ── Cleanup ─────────────────────────────────────────────────
        shutil.rmtree(storage, ignore_errors=True)

    # ── 5. Score ────────────────────────────────────────────────────
    score = pm_score_sample(entry, reply)
    result = score_to_dict(score)
    result["pm_reply"] = reply
    result["n_memories_retrieved"] = len(mem_texts)
    result["memories_retrieved"] = mem_texts
    result["n_turns_ingested"] = n_ingested
    result["constraint_turns"] = constraint_turns

    return result


async def _run_batch_async(
    entries: list[dict],
    client: LLMClient,
    temperature: float,
    concurrency: int,
    top_k: int,
    batch_size: int,
    category: str = "",
) -> list[dict]:
    """Run Mem0 PM evaluation concurrently.

    Note: concurrency should be kept low (2-4) because each sample
    creates its own Qdrant instance and makes multiple LLM calls
    for fact extraction during ingest.
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    t0 = time.time()

    async def _worker(entry):
        async with sem:
            return await asyncio.to_thread(
                _run_one_mem0, entry, client, temperature, top_k, batch_size,
                category,
            )

    tasks = [_worker(e) for e in entries]
    total = len(entries)
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        done = i + 1
        if done % 5 == 0 or done == total:
            elapsed = time.time() - t0
            per_sample = elapsed / done
            remaining = per_sample * (total - done)
            mins_left = remaining / 60
            print(
                f"  Progress: {done}/{total} | "
                f"~{per_sample:.0f}s/sample | "
                f"ETA: {mins_left:.1f}min"
            )

    return results


# ══════════════════════════════════════════════════════════════════════════
#  LLM-as-a-Judge
# ══════════════════════════════════════════════════════════════════════════

def _run_judge_batch(
    results: list[dict],
    entries: list[dict],
    judge: PMJudge,
) -> list[dict]:
    """Run LLM-as-a-Judge on PM results."""
    entry_by_id = {e["task_id"]: e for e in entries}
    judge_results = []

    for i, r in enumerate(results):
        entry = entry_by_id.get(r["task_id"])
        if entry is None:
            judge_results.append({
                "task_id": r["task_id"], "verdict": "ERROR",
                "reason": "Entry not found",
            })
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

        judge_results.append({
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
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(results):
            jc = sum(1 for j in judge_results if j["judge_correct"])
            print(
                f"  Judge progress: {i+1}/{len(results)} | "
                f"Correct: {jc}/{len(judge_results)} = "
                f"{jc/len(judge_results):.1%}"
            )

    return judge_results


# ══════════════════════════════════════════════════════════════════════════
#  Run one (category, variant)
# ══════════════════════════════════════════════════════════════════════════

def run_pm(
    category: str,
    variant: str,
    client: LLMClient,
    temperature: float,
    concurrency: int,
    results_dir: Path,
    judge_dir: Path,
    top_k: int,
    batch_size: int,
    judge: PMJudge | None = None,
    max_samples: int | None = None,
) -> tuple[list[dict], list[dict] | None]:
    """Run Mem0 PM evaluation for a specific category and variant (base bin only)."""
    data_path = DATA_DIR / variant / category / f"{category}_base.jsonl"
    if not data_path.exists():
        return [], None

    entries = [
        json.loads(l) for l in data_path.read_text().splitlines() if l.strip()
    ]
    if not entries:
        return [], None

    if max_samples is not None:
        entries = entries[:max_samples]

    print(
        f"\n  [{category}/{variant}/base] "
        f"Running Mem0 (top-k={top_k}, batch={batch_size}) on "
        f"{len(entries)} samples..."
    )
    start = time.time()

    results = asyncio.run(
        _run_batch_async(
            entries, client, temperature, concurrency, top_k, batch_size,
            category,
        )
    )

    elapsed = time.time() - start

    # ── Print summary ───────────────────────────────────────────────
    scores = [
        PMScore(**{k: r[k] for k in PMScore.__dataclass_fields__})
        for r in results
    ]
    agg = pm_aggregate(scores)
    is_neg = variant in NEGATIVE_VARIANTS
    neg_label = " [NEG]" if is_neg else ""
    print(
        f"  [{category}/{variant}/base]{neg_label} "
        f"Done in {elapsed:.1f}s"
        + (
            f" | Avg slot: {agg.get('pos_avg_slot_match', 0):.3f}"
            if not is_neg
            else ""
        )
    )

    # Save PM results
    out_path = (
        results_dir / variant / category / f"pm_{category}_base.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Run judge if requested
    judge_results = None
    if judge is not None:
        print(f"  [{category}/{variant}/base] Running LLM judge...")
        judge_results = _run_judge_batch(results, entries, judge)

        jc = sum(1 for j in judge_results if j["judge_correct"])
        print(
            f"  [{category}/{variant}/base] Judge done | "
            f"Correct: {jc}/{len(judge_results)} = "
            f"{jc/len(judge_results):.1%}"
        )

        j_path = (
            judge_dir / variant / category / f"judge_{category}_base.jsonl"
        )
        j_path.parent.mkdir(parents=True, exist_ok=True)
        with j_path.open("w") as f:
            for j in judge_results:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")

    return results, judge_results


# ══════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════

def _print_summary(
    all_results: list[dict],
    all_judge: list[dict],
    model_label: str,
    top_k: int,
    batch_size: int,
):
    from collections import defaultdict

    print(f"\n{'='*76}")
    print(
        f"  Mem0 PM Evaluation Summary "
        f"(top-k={top_k}, batch={batch_size}, {model_label})"
    )
    print(f"{'='*76}")

    def _to_scores(result_dicts: list[dict]) -> list[PMScore]:
        fields = set(PMScore.__dataclass_fields__)
        return [PMScore(**{k: r[k] for k in fields}) for r in result_dicts]

    by_key = defaultdict(list)
    for r in all_results:
        by_key[(r["category"], r["variant"])].append(r)
    judge_by_key = defaultdict(list)
    for j in all_judge:
        judge_by_key[(j["category"], j["variant"])].append(j)

    header = (
        f"  {'Category':<14} {'Variant':<22} {'n':>4}"
        f"  {'Slot':>6}"
    )
    if all_judge:
        header += f"  {'Judge':>8}"
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
            slot_str = (
                f"{agg.get('pos_avg_slot_match', 0):>6.3f}"
                if not is_neg
                else f"{'N/A':>6}"
            )
            line = (
                f"  {cat:<14} {variant:<22} {agg['n']:>4}"
                f"  {slot_str}"
            )
            if key in judge_by_key:
                js = judge_by_key[key]
                jc = sum(1 for j in js if j["judge_correct"])
                line += f"  {jc/len(js):>7.1%}"
            print(line)

    # Per variant totals
    print(f"  {'-'*70}")
    for variant in VARIANTS:
        rs = [r for r in all_results if r["variant"] == variant]
        if not rs:
            continue
        agg = pm_aggregate(_to_scores(rs))
        is_neg = variant in NEGATIVE_VARIANTS
        slot_str = (
            f"{agg.get('pos_avg_slot_match', 0):>6.3f}"
            if not is_neg
            else f"{'N/A':>6}"
        )
        line = (
            f"  {'ALL':<14} {variant:<22} {agg['n']:>4}"
            f"  {slot_str}"
        )
        js = [j for j in all_judge if j["variant"] == variant]
        if js:
            jc = sum(1 for j in js if j["judge_correct"])
            line += f"  {jc/len(js):>7.1%}"
        print(line)

    print(f"{'='*76}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Mem0-based PM evaluation")
    ap.add_argument("--category", choices=CATEGORIES)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument(
        "--all", action="store_true", help="Run all categories and variants"
    )
    ap.add_argument(
        "--judge",
        action="store_true",
        help="Also run LLM-as-a-Judge semantic evaluation",
    )
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max samples in parallel (keep low — each sample does LLM "
        "extraction). Default: 1",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of memory facts to retrieve",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Number of turns per Mem0 add() call (smaller = better "
        "extraction but more LLM calls). Default: 40",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per (category, variant). For quick testing.",
    )
    args = ap.parse_args()

    # ── Resolve model name and build output dirs ────────────────────
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENTS", "unknown-model")
    run_tag = _make_run_tag(
        model, args.temperature, args.concurrency, args.top_k, args.batch_size
    )
    results_dir = OUTPUT_ROOT / run_tag / "pm"
    judge_dir = OUTPUT_ROOT / run_tag / "pm_judge"

    print(f"  Model       : {model}")
    print(f"  Top-k       : {args.top_k}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Concurrency : {args.concurrency}")
    print(f"  Run tag     : {run_tag}")
    print(f"  Results dir : {results_dir}")

    client = LLMClient()
    judge_client = (
        LLMClient(deployment=JUDGE_DEPLOYMENT) if args.judge else None
    )
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

    all_results: list[dict] = []
    all_judge: list[dict] = []
    for cat in categories:
        for variant in variants:
            results, judge_results = run_pm(
                cat, variant, client,
                args.temperature, args.concurrency,
                results_dir, judge_dir,
                args.top_k, args.batch_size,
                judge=judge,
                max_samples=args.max_samples,
            )
            if results:
                all_results.extend(results)
            if judge_results:
                all_judge.extend(judge_results)

    # ── Summary ─────────────────────────────────────────────────────
    _print_summary(all_results, all_judge, model, args.top_k, args.batch_size)


if __name__ == "__main__":
    main()

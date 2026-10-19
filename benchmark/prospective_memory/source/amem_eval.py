"""A-MEM PM evaluation, using Memory-in-the-LLM-Era's memory_layer.py.

https://github.com/Yanchen398/Memory-in-the-LLM-Era

Same system prompt as blended_close. Base split only (no 40K).
Flags: ../README.md
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from typing import Any
from openai import RateLimitError

# ── Path setup (relative to prospective_memory/) ──────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent                    # source/
PKG_ROOT = SCRIPT_DIR.parent                                    # prospective_memory/
BENCHMARK_ROOT = PKG_ROOT.parent                                # benchmark/
AMEM_CODE_ROOT = BENCHMARK_ROOT / "Memory-in-the-LLM-Era" / "code"

sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT / "utils"))
sys.path.insert(0, str(AMEM_CODE_ROOT))

from client import CONFIG, LLMClient, _parse_env_list, _get_client_kwargs, get_openai_client
from longcontext_blended_close import get_system_prompt
from pm_judge import PMJudge
from pm_scorer import (
    NEGATIVE_VARIANTS, PMScore,
    score_sample as pm_score_sample,
    score_to_dict,
    aggregate as pm_aggregate,
)

from Method.amem.memory_layer import (
    AgenticMemorySystem,
    BaseLLMController,
    LLMController,
    MemoryNote,
    SimpleEmbeddingRetriever,  # monkey-patched to skip local model loading
)

# ── Constants ────────────────────────────────────────────────────────────
DATA_ROOT = PKG_ROOT / "data"
DATA_DIR = DATA_ROOT / "base_context"                            # base bin only

OUTPUT_ROOT = PKG_ROOT / "main_results" / "amem"
LOG_ROOT = PKG_ROOT / "main_results" / "amem" / "_logs"
EXTRACT_ROOT = PKG_ROOT / "main_results" / "amem" / "_extractions"

CATEGORIES = [
    "state_tracking",
    "temporal_grounding",
    "logical_adherence",
    "attention_recovery",
    "safe_coding",
]
VARIANTS = ["positive_clean", "positive_overloaded", "negative_clean"]

JUDGE_DEPLOYMENT = "gpt-4o"

CONSTRAINT_FUNCTIONS = {"constraint", "rule_constraint", "condition_constraint"}


# ── Embedding retriever (drop-in for SimpleEmbeddingRetriever) ───────────

class AzureEmbeddingRetriever:
    """Retriever using Azure OpenAI text-embedding-3-small, same API as RAG.

    Drop-in replacement for A-MEM's SimpleEmbeddingRetriever so the
    comparison with RAG uses identical embeddings.
    """

    def __init__(self, emb_client: Any, emb_deployment: str):
        self.emb_client = emb_client
        self.emb_deployment = emb_deployment
        self.corpus: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.document_ids: dict[str, int] = {}

    # ── Embedding helper (mirrors rag.py _embed_batch) ──────────────
    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = [t if len(t) > 0 else " " for t in texts[i:i + batch_size]]
            for attempt in range(6):
                try:
                    resp = self.emb_client.embeddings.create(
                        model=self.emb_deployment, input=batch)
                    break
                except RateLimitError:
                    if attempt == 5:
                        raise
                    wait = min(2 ** attempt, 30)
                    time.sleep(wait)
            for item in resp.data:
                vecs.append(item.embedding)
        m = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        return m / norms

    def add_documents(self, documents: list[str]):
        """Add documents — same interface as SimpleEmbeddingRetriever."""
        if not self.corpus:
            self.corpus = documents
            self.embeddings = self._embed(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embs = self._embed(documents)
            if self.embeddings is None:
                self.embeddings = new_embs
            else:
                self.embeddings = np.vstack([self.embeddings, new_embs])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

    def search(self, query: str, k: int = 5):
        """Return top-k indices by cosine similarity."""
        if not self.corpus:
            return []
        query_vec = self._embed([query])[0]
        sims = self.embeddings @ query_vec
        k = min(k, len(self.corpus))
        return np.argsort(sims)[-k:][::-1].tolist()


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


# ── Azure-compatible LLM controller for A-MEM ───────────────────────────

class AzureOpenAIController(BaseLLMController):
    """Drop-in replacement for A-MEM's OpenAIController.

    Uses the same OpenAI-compatible chat env as LLMClient
    (AZURE_OPENAI_API_KEY + OPENAI_BASE_URL).
    """

    def __init__(self, deployment: str):
        self.deployment = deployment
        self.call_count = 0  # Track real LLM calls
        self.client = get_openai_client()

    def get_completion(self, prompt: str, response_format: dict = None,
                       temperature: float = 0.7) -> str:
        kwargs = dict(
            model=self.deployment,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=2000,
        )
        if response_format:
            kwargs["response_format"] = response_format
        response = self.client.chat.completions.create(**kwargs)
        self.call_count += 1
        return response.choices[0].message.content


def _build_amem_system(deployment: str) -> AgenticMemorySystem:
    """Create an AgenticMemorySystem with OpenAI-compatible chat + embeddings.

    Skips local SentenceTransformer loading (no GPU needed) and uses
    AzureEmbeddingRetriever (text-embedding-3-small) for fair comparison
    with RAG.
    """
    # Skip loading the local SentenceTransformer model entirely —
    # we replace the retriever with Azure embeddings immediately.
    _orig_retriever_init = SimpleEmbeddingRetriever.__init__

    def _dummy_init(self, *args, **kwargs):
        self.model = None
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}

    SimpleEmbeddingRetriever.__init__ = _dummy_init
    try:
        system = AgenticMemorySystem(
            model_name="dummy",
            llm_backend="openai",
            llm_model="dummy",
            api_key="dummy",
            base_url="http://dummy",
        )
    finally:
        SimpleEmbeddingRetriever.__init__ = _orig_retriever_init

    system.llm_controller.llm = AzureOpenAIController(deployment)

    emb_client, emb_deployment = _embedding_client()
    system.retriever = AzureEmbeddingRetriever(emb_client, emb_deployment)

    return system


# ── Helpers ──────────────────────────────────────────────────────────────

def _format_turn(turn: dict) -> str:
    """Format a single turn for memory ingestion."""
    ts = f" [{turn['timestamp']}]" if turn.get("timestamp") else ""
    speaker = "USER" if turn["role"] == "user" else "ASSISTANT"
    return f"{speaker}{ts}: {turn['text']}"


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
                  top_k: int, emb_deployment: str) -> str:
    """Build a filesystem-safe run tag.
    """
    tag = re.sub(r'[^a-zA-Z0-9]+', '_', model).strip('_')
    emb = re.sub(r'[^a-zA-Z0-9]+', '_', emb_deployment).strip('_')
    return f"{tag}_temp{temperature}_con{concurrency}_k{top_k}_emb_{emb}"


# ── Core evaluation ─────────────────────────────────────────────────────

def _run_one_amem(
    entry: dict,
    client: LLMClient,
    temperature: float,
    top_k: int,
    category: str = "",
    no_evolution: bool = False,
    cached_notes: list[dict] | None = None,
) -> dict:
    """Run A-MEM PM evaluation for a single sample.

    1. Create fresh AgenticMemorySystem (per-sample, in-memory).
    2. Ingest all turns before trigger via add_note() — each turn triggers:
       - LLM call: analyze_content (extract keywords/context/tags)
       - LLM call: process_memory (evolution decision with neighbors)
    3. Retrieve related memories using trigger text (embedding-based).
    4. Generate response with system prompt + retrieved memories + trigger.
    5. Score.
    """
    sample_dict = entry["sample"]
    trigger_idx = sample_dict["ground_truth"]["trigger"]["trigger_turn_idx"]
    turns = sample_dict["turns"]
    constraint_turns = _find_constraint_turns(turns, trigger_idx)

    deployments = _parse_env_list("AZURE_OPENAI_DEPLOYMENTS")
    if not deployments:
        model = os.environ.get("OPENAI_MODEL", "").strip()
        deployments = [model] if model else []
    if not deployments:
        raise ValueError("Set AZURE_OPENAI_DEPLOYMENTS (or OPENAI_MODEL)")
    deployment = deployments[0]

    # ── 1. Create fresh A-MEM system ────────────────────────────────
    memory_system = _build_amem_system(deployment=deployment)

    # ── 2. Ingest all turns before trigger ──────────────────────────
    n_ingested = 0
    raw_notes_list: list[dict] = []
    _log_buf = io.StringIO()  # capture A-MEM's verbose prints for log file

    if cached_notes is not None and no_evolution:
        # No-evolution with cache: skip both analyze_content and evolution
        print(f"  [{entry['task_id']}] Loading {len(cached_notes)} notes "
              f"from extraction cache (no-evolution, 0 LLM calls)")
        for nd in cached_notes:
            note = MemoryNote(
                content=nd["content"],
                context=nd["context"],
                keywords=nd["keywords"],
                tags=nd["tags"],
                timestamp=nd["timestamp"],
                llm_controller=None,  # skip analyze_content
            )
            memory_system.memories[note.id] = note
            try:
                memory_system.retriever.add_documents(
                    [note.context + " keywords: " + ", ".join(note.keywords)]
                )
            except Exception as e:
                print(f"  Warning: add_documents failed: {e}")
            n_ingested += 1
    elif cached_notes is not None and not no_evolution:
        # Cached extraction + evolution: skip analyze_content, run process_memory
        # Same order as add_note(): process_memory BEFORE storing in memories/retriever
        print(
            f"  [{entry['task_id']}] Loading {len(cached_notes)} notes from "
            f"extraction cache + running evolution (~{len(cached_notes)} LLM calls)"
        )
        for nd in cached_notes:
            note = MemoryNote(
                content=nd["content"],
                context=nd["context"],
                keywords=nd["keywords"],
                tags=nd["tags"],
                timestamp=nd["timestamp"],
                llm_controller=None,  # skip analyze_content
            )
            # Run evolution with existing memories as neighbors
            try:
                with contextlib.redirect_stdout(_log_buf):
                    memory_system.process_memory(note)
            except Exception as e:
                print(f"  Warning: process_memory failed for note {n_ingested}: "
                      f"{type(e).__name__}: {str(e)[:150]}")
            memory_system.memories[note.id] = note
            try:
                memory_system.retriever.add_documents(
                    [note.context + " keywords: " + ", ".join(note.keywords)]
                )
            except Exception as e:
                print(f"  Warning: add_documents failed: {e}")
            n_ingested += 1
    else:
        # Full ingestion from scratch
        # Monkey-patch process_memory to capture pre-evolution state
        _orig_pm = memory_system.process_memory

        def _patched_pm(note):
            raw_notes_list.append({
                "content": note.content,
                "context": note.context,
                "keywords": list(note.keywords),
                "tags": list(note.tags),
                "timestamp": note.timestamp,
            })
            if no_evolution:
                return False, note  # skip evolution
            return _orig_pm(note)

        memory_system.process_memory = _patched_pm

        evo_label = "no-evolution" if no_evolution else "full evolution"
        print(f"  [{entry['task_id']}] Ingesting {trigger_idx} turns "
              f"from scratch ({evo_label})")
        for t in turns[:trigger_idx]:
            content = _format_turn(t)
            ts = t.get("timestamp", "")
            try:
                with contextlib.redirect_stdout(_log_buf):
                    memory_system.add_note(content, time=ts)
                n_ingested += 1
            except Exception as e:
                print(f"  Warning: add_note failed for turn {n_ingested}: "
                      f"{type(e).__name__}: {str(e)[:150]}")

    # ── 3. Retrieve with trigger message ────────────────────────────
    # Directly use retriever.search() + memories dict to get exactly
    # top-k structured memories.  Avoids find_related_memories_raw()
    # which returns a flat string that breaks on multi-line content.
    trigger_text = _format_turn(turns[trigger_idx])
    mem_texts: list[str] = []
    try:
        if memory_system.memories:
            indices = memory_system.retriever.search(trigger_text, k=top_k)
            all_memories = list(memory_system.memories.values())
            seen = set()
            for i in indices:
                if i >= len(all_memories):
                    continue
                m = all_memories[i]
                # Format each memory as a single structured entry
                entry_str = (
                    f"[{m.timestamp}] {m.content} "
                    f"(context: {m.context}; "
                    f"keywords: {', '.join(m.keywords)}; "
                    f"tags: {', '.join(m.tags)})"
                )
                mem_texts.append(entry_str)
                seen.add(i)
                # Also expand linked neighbors (up to top_k total)
                for neighbor_idx in m.links:
                    if len(mem_texts) >= top_k:
                        break
                    if (isinstance(neighbor_idx, int)
                            and 0 <= neighbor_idx < len(all_memories)
                            and neighbor_idx not in seen):
                        nb = all_memories[neighbor_idx]
                        nb_str = (
                            f"[{nb.timestamp}] {nb.content} "
                            f"(context: {nb.context}; "
                            f"keywords: {', '.join(nb.keywords)}; "
                            f"tags: {', '.join(nb.tags)})"
                        )
                        mem_texts.append(nb_str)
                        seen.add(neighbor_idx)
                if len(mem_texts) >= top_k:
                    break
    except Exception as e:
        mem_texts = []
        print(f"  A-MEM retrieval error: {type(e).__name__}: {str(e)[:150]}")

    # ── 4. Generate response ────────────────────────────────────────
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
            messages=gen_messages, max_tokens=1024, temperature=temperature,
        )
    except Exception as e:
        reply = f"[ERROR: {e}]"

    # ── 5. Score ────────────────────────────────────────────────────
    score = pm_score_sample(entry, reply)
    result = score_to_dict(score)
    result["pm_reply"] = reply
    result["n_memories_retrieved"] = len(mem_texts)
    result["memories_retrieved"] = mem_texts
    result["n_turns_ingested"] = n_ingested
    result["constraint_turns"] = constraint_turns
    # Real LLM call count: amem ingestion calls + 1 generation call
    amem_llm_calls = memory_system.llm_controller.llm.call_count
    result["llm_calls_amem"] = amem_llm_calls
    result["llm_calls_total"] = amem_llm_calls + 1  # +1 for generation
    result["_amem_log"] = _log_buf.getvalue()  # captured A-MEM stdout
    result["_raw_notes"] = raw_notes_list if raw_notes_list else None

    return result


async def _run_batch_async(
    entries: list[dict],
    client: LLMClient,
    temperature: float,
    concurrency: int,
    top_k: int,
    category: str = "",
    out_path: Path | None = None,
    log_path: Path | None = None,
    no_evolution: bool = False,
    extract_path: Path | None = None,
) -> list[dict]:
    """Run A-MEM PM evaluation one sample at a time.

    ``main`` rejects concurrency other than 1. ``_build_amem_system`` swaps
    ``SimpleEmbeddingRetriever.__init__`` on the global class, so overlapping
    samples can restore another thread's method. That path is not supported.

    Results are appended to *out_path* as each sample completes so that
    interrupted runs can be resumed without re-processing.
    """
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    t0 = time.time()

    # Load extraction cache (shared across runs with different temp/top-k)
    extract_cache: dict[str, list[dict]] = {}
    if extract_path and extract_path.exists():
        for line in extract_path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                extract_cache[obj["task_id"]] = obj["notes"]
        if extract_cache:
            print(f"  Loaded {len(extract_cache)} extraction caches")

    async def _worker(entry):
        async with sem:
            cached = extract_cache.get(entry["task_id"]) if extract_cache else None
            return await asyncio.to_thread(
                _run_one_amem, entry, client, temperature, top_k,
                category,
                no_evolution=no_evolution,
                cached_notes=cached,
            )

    tasks = [_worker(e) for e in entries]
    total = len(entries)
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro

        # ── Incremental save: write to disk as each sample completes ──
        if out_path is not None:
            amem_log = result.pop("_amem_log", "")
            raw_notes = result.pop("_raw_notes", None)
            with out_path.open("a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if log_path is not None and amem_log:
                with log_path.open("a") as lf:
                    lf.write(f"===== task_id={result['task_id']} =====\n")
                    lf.write(amem_log)
                    lf.write("\n")
            # Save extraction cache (new extractions only)
            if extract_path is not None and raw_notes:
                extract_path.parent.mkdir(parents=True, exist_ok=True)
                with extract_path.open("a") as ef:
                    ef.write(json.dumps({
                        "task_id": result["task_id"],
                        "notes": raw_notes,
                    }, ensure_ascii=False) + "\n")
        else:
            result.pop("_amem_log", None)
            result.pop("_raw_notes", None)

        results.append(result)
        done = i + 1
        elapsed = time.time() - t0
        per_sample = elapsed / done
        remaining = per_sample * (total - done)
        mins_left = remaining / 60
        if done % 5 == 0 or done == total:
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
                "reason": "Entry not found", "judge_correct": False,
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
    log_dir: Path,
    top_k: int,
    max_samples: int | None = None,
    judge: PMJudge | None = None,
    no_evolution: bool = False,
) -> tuple[list[dict], list[dict] | None]:
    """Run A-MEM PM evaluation for a specific category and variant (base bin only)."""
    data_path = DATA_DIR / variant / category / f"{category}_base.jsonl"
    if not data_path.exists():
        return [], None

    entries = [
        json.loads(l) for l in data_path.read_text().splitlines() if l.strip()
    ]
    if not entries:
        return [], None

    if max_samples is not None and max_samples > 0:
        entries = entries[:max_samples]

    # ── Compute output paths for incremental save & resume ──────────
    out_path = (
        results_dir / variant / category / f"pm_{category}_base.jsonl"
    )
    log_path = (
        log_dir / variant / category / f"amem_{category}_base.log"
    )

    # ── Resume: skip already-completed samples ──────────────────────
    completed_ids: set[str] = set()
    existing_results: list[dict] = []
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                completed_ids.add(r["task_id"])
                existing_results.append(r)

    remaining = [e for e in entries if e["task_id"] not in completed_ids]

    if not remaining:
        print(
            f"\n  [{category}/{variant}/base] "
            f"All {len(entries)} samples already completed, skipping."
        )
        results = existing_results
        elapsed = 0.0
    else:
        evo_str = " no-evolution" if no_evolution else ""
        if completed_ids:
            print(
                f"\n  [{category}/{variant}/base] "
                f"Resuming{evo_str}: {len(completed_ids)} done, "
                f"{len(remaining)} remaining "
                f"(top-k={top_k})..."
            )
        else:
            print(
                f"\n  [{category}/{variant}/base] "
                f"Running A-MEM{evo_str} (top-k={top_k}) on "
                f"{len(remaining)} samples..."
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        start = time.time()
        # Compute extraction cache path (shared across temp/top-k runs).
        # positive_clean and positive_overloaded share identical pre-trigger
        # turns, so they use the same extraction cache under "positive/".
        _model_tag = re.sub(
            r'[^a-zA-Z0-9]+', '_',
            os.environ.get("AZURE_OPENAI_DEPLOYMENTS", "unknown"),
        ).strip('_')
        if variant.startswith("positive"):
            _extract_variant = "positive"
        else:
            _extract_variant = variant
        extract_path = (
            EXTRACT_ROOT / _model_tag / _extract_variant / category
            / f"notes_{category}_base.jsonl"
        )
        if extract_path.exists():
            print(f"  Extraction cache: {extract_path} (will reuse)")
        else:
            print(f"  Extraction cache: {extract_path} (not found, will extract fresh)")

        new_results = asyncio.run(
            _run_batch_async(
                remaining, client, temperature, concurrency, top_k,
                category,
                out_path=out_path, log_path=log_path,
                no_evolution=no_evolution,
                extract_path=extract_path,
            )
        )
        elapsed = time.time() - start
        results = existing_results + new_results

    # ── Print summary (all results, including resumed) ──────────────
    if not results:
        return [], None

    scores = [
        PMScore(**{k: r[k] for k in PMScore.__dataclass_fields__})
        for r in results
    ]
    agg = pm_aggregate(scores)
    is_neg = variant in NEGATIVE_VARIANTS
    neg_label = " [NEG]" if is_neg else ""
    avg_mems = (
        sum(r["n_memories_retrieved"] for r in results) / len(results)
        if results
        else 0
    )
    time_str = f"in {elapsed:.1f}s" if elapsed > 0 else "(cached)"
    print(
        f"  [{category}/{variant}/base]{neg_label} "
        f"Done {time_str}"
        + (
            f" | Avg slot: {agg.get('pos_avg_slot_match', 0):.3f}"
            if not is_neg
            else ""
        )
        + f" | Avg mems: {avg_mems:.1f}"
    )
    avg_llm = (
        sum(r.get("llm_calls_total", 0) for r in results) / len(results)
        if results
        else 0
    )
    total_llm = sum(r.get("llm_calls_total", 0) for r in results)
    print(
        f"  [{category}/{variant}/base] "
        f"LLM calls: {total_llm} total | {avg_llm:.1f} avg/sample"
    )

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
    top_k: int,
    no_evolution: bool,
):
    print(f"\n{'='*76}")
    evo_label = " no-evolution" if no_evolution else ""
    print(f"  A-MEM PM Evaluation Summary (top-k={top_k}{evo_label})")
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
        f"  {'Slot':>6}  {'AvgMem':>6}"
    )
    if all_judge:
        header += f"  {'Judge':>8}"
    print(header)
    print(f"  {'-'*76}")

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
            avg_mem = sum(r["n_memories_retrieved"] for r in rs) / len(rs)
            line = (
                f"  {cat:<14} {variant:<22} {agg['n']:>4}"
                f"  {slot_str}  {avg_mem:>6.1f}"
            )
            if key in judge_by_key:
                js = judge_by_key[key]
                jc = sum(1 for j in js if j["judge_correct"])
                line += f"  {jc/len(js):>7.1%}"
            print(line)

    # Per variant totals
    print(f"  {'-'*76}")
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
        avg_mem = sum(r["n_memories_retrieved"] for r in rs) / len(rs)
        line = (
            f"  {'ALL':<14} {variant:<22} {agg['n']:>4}"
            f"  {slot_str}  {avg_mem:>6.1f}"
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
    ap = argparse.ArgumentParser(description="A-MEM based PM evaluation")
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
        help="Only 1 is supported. Parallel samples are not safe: A-MEM "
        "temporarily replaces a global retriever constructor. Default: 1",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of memories to retrieve",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per (category, variant) for testing",
    )
    ap.add_argument(
        "--no-evolution",
        action="store_true",
        help="Skip A-MEM memory evolution (process_memory). Uses cached "
        "extractions if available, otherwise runs analyze_content only. "
        "Cuts LLM calls from ~2N to ~N per sample (or 0 if cached).",
    )
    args = ap.parse_args()
    # Official runs fix concurrency at 1 (README). Do not overlap
    # _build_amem_system: it patches SimpleEmbeddingRetriever.__init__ globally.
    if args.concurrency != 1:
        ap.error(
            "A-MEM only supports --concurrency 1 "
            "(the global retriever patch is not safe to overlap)"
        )

    # ── Resolve model name and build output dirs ────────────────────
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENTS", "unknown-model")
    emb_deployment = _parse_env_list("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENTS")
    if not emb_deployment:
        raise ValueError(
            "Set AZURE_OPENAI_EMBEDDINGS_DEPLOYMENTS (and optionally "
            "AZURE_OPENAI_EMBEDDINGS_ENDPOINT)"
        )
    emb_deployment = emb_deployment[0]
    run_tag = _make_run_tag(
        model, args.temperature, args.concurrency, args.top_k, emb_deployment,
    )
    if args.no_evolution:
        run_tag += "_noevo"
    results_dir = OUTPUT_ROOT / run_tag / "pm"
    judge_dir = OUTPUT_ROOT / run_tag / "pm_judge"
    log_dir = LOG_ROOT / run_tag

    print(f"  Model         : {model}")
    print(f"  Top-k         : {args.top_k}")
    print(f"  Embedding     : {emb_deployment}")
    print(f"  No evolution  : {args.no_evolution}")
    print(f"  Concurrency   : {args.concurrency}")
    print(f"  Max samples   : {args.max_samples or 'ALL'}")
    print(f"  Run tag       : {run_tag}")
    print(f"  Results dir   : {results_dir}")

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
                results_dir, judge_dir, log_dir,
                args.top_k,
                max_samples=args.max_samples,
                judge=judge,
                no_evolution=args.no_evolution,
            )
            if results:
                all_results.extend(results)
            if judge_results:
                all_judge.extend(judge_results)

    # ── Summary ─────────────────────────────────────────────────────
    _print_summary(all_results, all_judge, args.top_k, args.no_evolution)


if __name__ == "__main__":
    main()

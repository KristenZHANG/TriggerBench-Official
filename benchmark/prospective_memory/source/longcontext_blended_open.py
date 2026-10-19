"""Full-context PM evaluation against a local vLLM server.

Judge still uses Azure OpenAI gpt-4o. Supports base and 40K splits.
Start the server with run_vllm_server.sh. Flags: ../README.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from openai import OpenAI
from longcontext_blended_close import get_system_prompt

# ── Path setup (relative to prospective_memory/) ──────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent                    # source/
PKG_ROOT = SCRIPT_DIR.parent                                    # prospective_memory/
BENCHMARK_ROOT = PKG_ROOT.parent                                # benchmark/
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT / "utils"))

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

OUTPUT_ROOT = PKG_ROOT / "main_results" / "blended_open"

CATEGORIES = [
    "state_tracking",
    "temporal_grounding",
    "logical_adherence",
    "attention_recovery",
    "safe_coding",
]
VARIANTS = ["positive_clean", "positive_overloaded", "negative_clean"]
BIN_LABELS = ["base", "40K"]

# Judge always uses Azure OpenAI gpt-4o.
JUDGE_DEPLOYMENT = "gpt-4o"


def _make_run_tag(model: str, temperature: float, concurrency: int) -> str:
    """Build a filesystem-safe run tag, e.g. 'Qwen3_32B_temp0.6_con8'."""
    tag = re.sub(r'[^a-zA-Z0-9]+', '_', model).strip('_')
    return f"{tag}_temp{temperature}_con{concurrency}"


# ── Evaluation helpers ───────────────────────────────────────────────────

def _format_history(sample_dict: dict) -> str:
    """Format conversation up to and including the trigger turn."""
    trigger_idx = sample_dict["ground_truth"]["trigger"]["trigger_turn_idx"]
    lines = []
    for t in sample_dict["turns"][:trigger_idx + 1]:
        ts = f" [{t['timestamp']}]" if t.get("timestamp") else ""
        speaker = "USER" if t["role"] == "user" else "ASSISTANT"
        lines.append(f"{speaker}{ts}: {t['text']}")
    return "\n".join(lines)


# ── Local vLLM client ───────────────────────────────────────────────────

_vllm_client: OpenAI | None = None
_vllm_lock = threading.Lock()


def _get_vllm_client(base_url: str) -> OpenAI:
    """Global singleton vLLM client."""
    global _vllm_client
    if _vllm_client is None:
        with _vllm_lock:
            if _vllm_client is None:
                _vllm_client = OpenAI(
                    api_key="EMPTY",
                    base_url=base_url,
                )
    return _vllm_client


def _call_vllm(
    messages: list[dict],
    model: str,
    max_tokens: int = 8192,
    temperature: float = 0.6,
) -> tuple[str, str | None]:
    """Call locally deployed vLLM model.

    Returns (answer, reasoning_content). With --reasoning-parser qwen3,
    the thinking part is separated into reasoning_content.
    """
    client = _vllm_client
    if client is None:
        raise RuntimeError("vLLM client not initialized. Call _get_vllm_client first.")

    if temperature > 0:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            extra_body={"top_k": 20},
        )
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body={"top_k": 1},
        )
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    return msg.content, reasoning


# ── Core evaluation ─────────────────────────────────────────────────────

def _run_one_pm(entry: dict, model: str, temperature: float) -> dict:
    """Run PM evaluation for a single sample."""
    sample_dict = entry["sample"]
    transcript = _format_history(sample_dict)
    prompt = get_system_prompt(entry.get("category", ""))

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"Here is the conversation so far:\n\n{transcript}\n\n"
                "Respond to the latest user message."
            ),
        },
    ]

    try:
        reply, reasoning = _call_vllm(messages, model=model, temperature=temperature)
    except Exception as e:
        reply = f"[ERROR: {e}]"
        reasoning = None

    score = pm_score_sample(entry, reply)
    result = score_to_dict(score)
    result["pm_reply"] = reply
    if reasoning:
        result["reasoning"] = reasoning
    return result


async def _run_batch_async(
    entries: list[dict],
    model: str,
    temperature: float,
    concurrency: int,
) -> list[dict]:
    """Run PM evaluation concurrently."""
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    t0 = time.time()

    async def _worker(entry):
        async with sem:
            return await asyncio.to_thread(_run_one_pm, entry, model, temperature)

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
            print(
                f"  Progress: {done}/{total} | "
                f"~{per_sample:.1f}s/sample | "
                f"ETA: {remaining/60:.1f}min"
            )

    return results


# ── LLM-as-a-Judge ──────────────────────────────────────────────────────

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
            print(f"  Judge progress: {i+1}/{len(results)} | "
                  f"Correct: {jc}/{len(judge_results)} = {jc/len(judge_results):.1%}")

    return judge_results


# ── Run one (category, variant, bin) ────────────────────────────────────

def run_pm(
    category: str, variant: str, bin_label: str,
    model: str, temperature: float, concurrency: int,
    results_dir: Path, judge_dir: Path,
    judge: PMJudge | None = None,
    max_samples: int | None = None,
) -> tuple[list[dict], list[dict] | None]:
    """Run PM evaluation for a specific category, variant, and bin."""
    data_dir = DATA_DIR_MAP.get(bin_label, DATA_DIR_MAP["base"])
    data_path = data_dir / variant / category / f"{category}_{bin_label}.jsonl"
    if not data_path.exists():
        return [], None

    entries = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    if not entries:
        return [], None

    if max_samples is not None:
        entries = entries[:max_samples]

    print(f"\n  [{category}/{variant}/{bin_label}] Running PM on {len(entries)} samples...")
    start = time.time()

    results = asyncio.run(_run_batch_async(entries, model, temperature, concurrency))

    elapsed = time.time() - start
    scores = [PMScore(**{k: r[k] for k in PMScore.__dataclass_fields__}) for r in results]
    agg = pm_aggregate(scores)
    is_neg = variant in NEGATIVE_VARIANTS
    neg_label = " [NEG]" if is_neg else ""
    print(
        f"  [{category}/{variant}/{bin_label}]{neg_label} Done in {elapsed:.1f}s"
        + (f" | Avg slot: {agg.get('pos_avg_slot_match', 0):.3f}" if not is_neg else "")
    )

    # Save PM results
    out_path = results_dir / variant / category / f"pm_{category}_{bin_label}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Run judge if requested
    judge_results = None
    if judge is not None:
        print(f"  [{category}/{variant}/{bin_label}] Running LLM judge...")
        judge_results = _run_judge_batch(results, entries, judge)

        jc = sum(1 for j in judge_results if j["judge_correct"])
        print(
            f"  [{category}/{variant}/{bin_label}] Judge done | "
            f"Correct: {jc}/{len(judge_results)} = {jc/len(judge_results):.1%}"
        )

        j_path = judge_dir / variant / category / f"judge_{category}_{bin_label}.jsonl"
        j_path.parent.mkdir(parents=True, exist_ok=True)
        with j_path.open("w") as f:
            for j in judge_results:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")

    return results, judge_results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Long-context PM evaluation (local vLLM open-source models)"
    )
    ap.add_argument("--category", choices=CATEGORIES)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--bin", default="base", choices=BIN_LABELS,
                    help="Bin label: 'base' (short) or '40K' (40K-token context)")
    ap.add_argument("--all", action="store_true",
                    help="Run all categories and variants")
    ap.add_argument("--judge", action="store_true",
                    help="Also run LLM-as-a-Judge (uses Azure gpt-4o)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--model", type=str,
                    default="Qwen3-32B",
                    help="Model name/path served by vLLM (default: Qwen3-32B)")
    ap.add_argument("--port", type=int, default=8000,
                    help="vLLM server port (default: 8000)")
    ap.add_argument("--host", type=str, default="localhost",
                    help="vLLM server host (default: localhost)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Max samples per (category, variant, bin). For quick testing.")
    args = ap.parse_args()

    # ── Initialize vLLM client ──────────────────────────────────────
    base_url = f"http://{args.host}:{args.port}/v1"
    _get_vllm_client(base_url)

    # ── Build output dirs ───────────────────────────────────────────
    model_short = Path(args.model).name  # e.g. "Qwen3-32B"
    run_tag = _make_run_tag(model_short, args.temperature, args.concurrency)
    results_dir = OUTPUT_ROOT / run_tag / "pm"
    judge_dir = OUTPUT_ROOT / run_tag / "pm_judge"

    print(f"  Model       : {args.model}")
    print(f"  vLLM server : {base_url}")
    print(f"  Concurrency : {args.concurrency}")
    print(f"  Temperature : {args.temperature}")
    print(f"  Run tag     : {run_tag}")
    print(f"  Results dir : {results_dir}")

    # Judge still uses Azure OpenAI gpt-4o
    judge = None
    if args.judge:
        from client import LLMClient
        judge_client = LLMClient(deployment=JUDGE_DEPLOYMENT)
        judge = PMJudge(client=judge_client)
        print(f"  Judge       : {JUDGE_DEPLOYMENT} (Azure OpenAI)")

    # ── Build category / variant lists ──────────────────────────────
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
                    cat, variant, bl, args.model,
                    args.temperature, args.concurrency,
                    results_dir, judge_dir, judge=judge,
                    max_samples=args.max_samples,
                )
                if results:
                    all_results.extend(results)
                if judge_results:
                    all_judge.extend(judge_results)

    # ── Summary ─────────────────────────────────────────────────────
    from collections import defaultdict

    print(f"\n{'='*80}")
    print(f"  Long-context PM Evaluation Summary (local: {model_short})")
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

    has_judge = bool(all_judge)
    header = (f"  {'Category':<28} {'Variant':<22} {'n':>4}"
              f"  {'Slot':>6}")
    if has_judge:
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
            slot_str = (f"{agg.get('pos_avg_slot_match', 0):>6.3f}"
                        if not is_neg else f"{'N/A':>6}")
            line = (f"  {cat:<28} {variant:<22} {agg['n']:>4}"
                    f"  {slot_str}")
            if key in judge_by_key:
                js = judge_by_key[key]
                jc = sum(1 for j in js if j["judge_correct"])
                line += f"  {jc/len(js):>7.1%}"
            print(line)

    print(f"  {'-'*76}")
    for variant in VARIANTS:
        rs = [r for r in all_results if r["variant"] == variant]
        if not rs:
            continue
        agg = pm_aggregate(_to_scores(rs))
        is_neg = variant in NEGATIVE_VARIANTS
        slot_str = (f"{agg.get('pos_avg_slot_match', 0):>6.3f}"
                    if not is_neg else f"{'N/A':>6}")
        line = (f"  {'ALL':<28} {variant:<22} {agg['n']:>4}"
                f"  {slot_str}")
        js = [j for j in all_judge if j["variant"] == variant]
        if js:
            jc = sum(1 for j in js if j["judge_correct"])
            line += f"  {jc/len(js):>7.1%}"
        print(line)

    print(f"{'='*80}")


if __name__ == "__main__":
    main()

"""Full-context PM evaluation via an OpenAI-compatible chat API.

Standard models use temperature (paper default 0.6). Reasoning models
(--reasoning / --reasoning-effort) use max_completion_tokens instead.
Supports base and 40K splits. Flags and examples: ../README.md
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

# ── Path setup (relative to prospective_memory/) ──────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent                    # source/
PKG_ROOT = SCRIPT_DIR.parent                                    # prospective_memory/
BENCHMARK_ROOT = PKG_ROOT.parent                                # benchmark/
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT / "utils"))

from client import LLMClient, get_openai_client
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

OUTPUT_ROOT = PKG_ROOT / "main_results" / "blended_close"

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


def _make_run_tag(model: str, temperature: float, concurrency: int) -> str:
    """Build a filesystem-safe run tag, e.g. 'gpt_4o_temp0.6_con8'."""
    tag = re.sub(r'[^a-zA-Z0-9]+', '_', model).strip('_')
    return f"{tag}_temp{temperature}_con{concurrency}"


# ── System Prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful conversational assistant. Respond naturally to the user's message.

Each turn is prefixed with a timestamp (e.g., [2026-04-19T10:00]). \
You MUST treat this as the exact, real-world current time for that turn.

A few things to keep in mind as you help:
1. Answer the user's actual question directly and concisely.
2. If the user asks a question about your past conversation, answer based \
on what was discussed earlier. Prefer using \
the original words or phrases from the conversation.
3. Pay attention to things or rules the user mentioned earlier that could become \
relevant now. For example, 
    - an object left somewhere they might forget, \
    - a time commitment that is approaching that the user might forget, \
    - a requirement or rule that is about to be violated, or \
    - an unfinished task that the user might forget resuming after sudden interruptions. 
If the current situation calls for it, mention it in your reply — a heads-up \
remind or intervention is better than letting the user forget. Do not repeat the same reminder \
if you have already brought it up.
4. If you notice something or action that seems unsafe or likely to cause a problem,
  mention it.
5. Match the user's tone (casual if they are casual, terse if they are terse).
"""

SYSTEM_PROMPT_CODING = """\
You are a coding assistant helping a developer through a multi-step \
software engineering session. Help with the user's current task. Match \
their tone. If something looks like it could cause a problem, bring it \
up at the right moment."""

# Category → prompt mapping.  safe_coding uses the concise coding prompt;
# all other categories use the detailed general prompt.
_CATEGORY_PROMPTS: dict[str, str] = {
    "safe_coding": SYSTEM_PROMPT_CODING,
}


def get_system_prompt(category: str) -> str:
    """Return the appropriate system prompt for *category*."""
    return _CATEGORY_PROMPTS.get(category, SYSTEM_PROMPT)


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


# ══════════════════════════════════════════════════════════════════════════
#  Standard model support (gpt-4o, gpt-4.1, etc.)
# ══════════════════════════════════════════════════════════════════════════

def _run_one_pm_standard(entry: dict, client: LLMClient, temperature: float) -> dict:
    """Run PM evaluation for a single sample using a standard chat model."""
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
        reply = client.complete_chat(
            messages=messages, max_tokens=1024, temperature=temperature
        )
    except Exception as e:
        reply = f"[ERROR: {e}]"

    score = pm_score_sample(entry, reply)
    result = score_to_dict(score)
    result["pm_reply"] = reply
    return result


async def _run_batch_standard(
    entries: list[dict],
    client: LLMClient,
    temperature: float,
    concurrency: int,
) -> list[dict]:
    """Run PM evaluation concurrently with a standard model."""
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def _worker(entry):
        async with sem:
            return await asyncio.to_thread(
                _run_one_pm_standard, entry, client, temperature
            )

    tasks = [_worker(e) for e in entries]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        if (i + 1) % 10 == 0 or (i + 1) == len(entries):
            print(f"  Progress: {i+1}/{len(entries)}")

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Reasoning model support (gpt-5.x, o3, etc.)
# ══════════════════════════════════════════════════════════════════════════

def _get_chat_client():
    """Shared OpenAI-compatible client (same env as LLMClient)."""
    return get_openai_client()


def _call_reasoning(
    messages: list[dict],
    deployment: str,
    max_completion_tokens: int = 16384,
    reasoning_effort: str | None = None,
) -> tuple[str, str | None, dict | None]:
    """Call a reasoning model via the OpenAI-compatible chat API.

    Reasoning models do NOT support temperature, top_p, max_tokens.
    Must use max_completion_tokens instead (includes reasoning + answer tokens).

    Returns (answer, reasoning_content_or_None, usage_dict_or_None).
    """
    client = _get_chat_client()

    kwargs: dict = {
        "model": deployment,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    answer = msg.content or ""

    # Some thinking models put everything in reasoning and leave content empty
    if not answer and reasoning:
        answer = reasoning

    # Capture token usage (reasoning_tokens are in completion_tokens_details)
    usage = None
    if resp.usage:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        if (hasattr(resp.usage, "completion_tokens_details")
                and resp.usage.completion_tokens_details):
            details = resp.usage.completion_tokens_details
            usage["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)

    return answer, reasoning, usage


def _run_one_pm_reasoning(
    entry: dict,
    deployment: str,
    reasoning_effort: str | None,
) -> dict:
    """Run PM evaluation for a single sample using a reasoning model."""
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
        reply, reasoning, usage = _call_reasoning(
            messages, deployment=deployment,
            reasoning_effort=reasoning_effort,
        )
    except Exception as e:
        reply = f"[ERROR: {e}]"
        reasoning = None
        usage = None

    score = pm_score_sample(entry, reply)
    result = score_to_dict(score)
    result["pm_reply"] = reply
    if reasoning:
        result["reasoning"] = reasoning
    if usage:
        result["usage"] = usage
    return result


async def _run_batch_reasoning(
    entries: list[dict],
    deployment: str,
    concurrency: int,
    reasoning_effort: str | None,
) -> list[dict]:
    """Run PM evaluation concurrently with a reasoning model."""
    sem = asyncio.Semaphore(concurrency)
    results = []
    t0 = time.time()

    async def _worker(entry):
        async with sem:
            return await asyncio.to_thread(
                _run_one_pm_reasoning, entry, deployment, reasoning_effort,
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
            print(
                f"  Progress: {done}/{total} | "
                f"~{per_sample:.1f}s/sample | "
                f"ETA: {remaining/60:.1f}min"
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
            print(f"  Judge progress: {i+1}/{len(results)} | "
                  f"Correct: {jc}/{len(judge_results)} = {jc/len(judge_results):.1%}")

    return judge_results


# ══════════════════════════════════════════════════════════════════════════
#  Run one (category, variant, bin)
# ══════════════════════════════════════════════════════════════════════════

def run_pm(
    category: str, variant: str, bin_label: str,
    *,
    # Standard mode args
    client: LLMClient | None = None,
    temperature: float = 0.3,
    # Reasoning mode args
    reasoning: bool = False,
    deployment: str | None = None,
    reasoning_effort: str | None = None,
    # Common args
    concurrency: int = 8,
    results_dir: Path,
    judge_dir: Path,
    judge: PMJudge | None = None,
    judge_only: bool = False,
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

    # ── Judge-only mode: load existing PM results ───────────────────
    if judge_only:
        pm_path = results_dir / variant / category / f"pm_{category}_{bin_label}.jsonl"
        if not pm_path.exists():
            print(f"  [{category}/{variant}/{bin_label}] SKIP — no PM results at {pm_path}")
            return [], None
        results = [json.loads(l) for l in pm_path.read_text().splitlines() if l.strip()]
        print(f"\n  [{category}/{variant}/{bin_label}] Loaded {len(results)} existing PM results (judge-only)")
    else:
        print(f"\n  [{category}/{variant}/{bin_label}] Running PM on {len(entries)} samples...")
        start = time.time()

        if reasoning:
            results = asyncio.run(_run_batch_reasoning(
                entries, deployment, concurrency, reasoning_effort,
            ))
        else:
            results = asyncio.run(_run_batch_standard(
                entries, client, temperature, concurrency,
            ))

        elapsed = time.time() - start
        scores = [PMScore(**{k: r[k] for k in PMScore.__dataclass_fields__}) for r in results]
        agg = pm_aggregate(scores)
        is_neg = variant in NEGATIVE_VARIANTS
        neg_label = " [NEG]" if is_neg else ""

        status_parts = []
        if reasoning:
            n_reasoning = sum(1 for r in results if r.get("reasoning"))
            status_parts.append(f"Reasoning: {n_reasoning}/{len(results)}")
        if not is_neg:
            status_parts.append(f"Avg slot: {agg.get('pos_avg_slot_match', 0):.3f}")

        print(f"  [{category}/{variant}/{bin_label}]{neg_label} Done in {elapsed:.1f}s"
              + ((" | " + " | ".join(status_parts)) if status_parts else ""))

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
        print(f"  [{category}/{variant}/{bin_label}] Judge done | "
              f"Correct: {jc}/{len(judge_results)} = {jc/len(judge_results):.1%}")

        j_path = judge_dir / variant / category / f"judge_{category}_{bin_label}.jsonl"
        j_path.parent.mkdir(parents=True, exist_ok=True)
        with j_path.open("w") as f:
            for j in judge_results:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")

    return results, judge_results


# ══════════════════════════════════════════════════════════════════════════
#  Summary printer
# ══════════════════════════════════════════════════════════════════════════

def _print_summary(
    all_results: list[dict],
    all_judge: list[dict],
    model_label: str,
    reasoning: bool = False,
):
    from collections import defaultdict

    print(f"\n{'='*80}")
    if reasoning:
        print(f"  PM Evaluation Summary (Reasoning: {model_label})")
    else:
        print(f"  Long-context PM Evaluation Summary ({model_label})")
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

    # Reasoning-specific stats
    if reasoning:
        n_with_reasoning = sum(1 for r in all_results if r.get("reasoning"))
        if n_with_reasoning:
            avg_reasoning_len = sum(
                len(r["reasoning"]) for r in all_results if r.get("reasoning")
            ) / n_with_reasoning
            print(f"\n  Reasoning content: {n_with_reasoning}/{len(all_results)} samples")
            print(f"  Avg reasoning length: {avg_reasoning_len:.0f} chars")

        usages = [r["usage"] for r in all_results if r.get("usage")]
        if usages:
            avg_prompt = sum(u["prompt_tokens"] for u in usages) / len(usages)
            avg_completion = sum(u["completion_tokens"] for u in usages) / len(usages)
            reasoning_tokens = [
                u.get("reasoning_tokens") for u in usages
                if u.get("reasoning_tokens")
            ]
            print(f"\n  Token usage ({len(usages)} samples):")
            print(f"    Avg prompt tokens    : {avg_prompt:.0f}")
            print(f"    Avg completion tokens : {avg_completion:.0f}")
            if reasoning_tokens:
                avg_rt = sum(reasoning_tokens) / len(reasoning_tokens)
                print(f"    Avg reasoning tokens  : {avg_rt:.0f} "
                      f"({len(reasoning_tokens)}/{len(usages)} reported)")

    print(f"{'='*80}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Long-context PM evaluation (OpenAI-compatible closed-source models)"
    )
    ap.add_argument("--category", choices=CATEGORIES)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--bin", default="base", choices=BIN_LABELS,
                    help="Bin label: 'base' (short) or '40K' (40K-token context)")
    ap.add_argument("--all", action="store_true",
                    help="Run all categories and variants")
    ap.add_argument("--judge", action="store_true",
                    help="Also run LLM-as-a-Judge semantic evaluation")
    ap.add_argument("--judge-only", action="store_true",
                    help="Skip PM prediction; load existing results and run judge only")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="Temperature (standard models only; default: 0.6)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--deployment", type=str, default=None,
                    help="Model / deployment name (default: AZURE_OPENAI_DEPLOYMENTS env)")

    # Reasoning model options
    ap.add_argument("--reasoning", action="store_true",
                    help="Use reasoning model API (no temperature, uses max_completion_tokens)")
    ap.add_argument("--reasoning-effort", type=str, default=None,
                    choices=["none", "low", "medium", "high", "xhigh"],
                    help="Reasoning effort level (implies --reasoning)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Max samples per (category, variant, bin). For quick testing.")

    args = ap.parse_args()

    # --reasoning-effort implies --reasoning
    reasoning = args.reasoning or (args.reasoning_effort is not None)

    deployment = (
        args.deployment
        or os.environ.get("AZURE_OPENAI_DEPLOYMENTS")
        or os.environ.get("OPENAI_MODEL", "")
    )
    if not deployment:
        ap.error("Provide --deployment or set AZURE_OPENAI_DEPLOYMENTS env var")

    # ── Initialize client ───────────────────────────────────────────
    if reasoning:
        _get_chat_client()
        client = None
    else:
        client = LLMClient(deployment=deployment) if args.deployment else LLMClient()

    # ── Build output dirs ───────────────────────────────────────────
    if reasoning:
        tag_name = deployment
        if args.reasoning_effort:
            tag_name = f"{deployment}-{args.reasoning_effort}"
        run_tag = _make_run_tag(tag_name, 0.0, args.concurrency)
    else:
        run_tag = _make_run_tag(deployment, args.temperature, args.concurrency)

    results_dir = OUTPUT_ROOT / run_tag / "pm"
    judge_dir = OUTPUT_ROOT / run_tag / "pm_judge"

    print(f"  Deployment      : {deployment}")
    if reasoning:
        print(f"  Mode            : reasoning")
        print(f"  Reasoning effort: {args.reasoning_effort or 'default (model decides)'}")
        print(f"  Temperature     : N/A (not supported for reasoning models)")
    else:
        print(f"  Mode            : standard")
        print(f"  Temperature     : {args.temperature}")
    print(f"  Concurrency     : {args.concurrency}")
    print(f"  Run tag         : {run_tag}")
    print(f"  Results dir     : {results_dir}")

    # Judge always uses gpt-4o
    judge = None
    if args.judge or args.judge_only:
        judge_client = LLMClient(deployment=JUDGE_DEPLOYMENT)
        judge = PMJudge(client=judge_client)
        print(f"  Judge           : {JUDGE_DEPLOYMENT}")

    judge_only = args.judge_only

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

    all_results = []
    all_judge = []
    for cat in categories:
        for variant in variants:
            for bl in bins:
                results, judge_results = run_pm(
                    cat, variant, bl,
                    client=client,
                    temperature=args.temperature,
                    reasoning=reasoning,
                    deployment=deployment,
                    reasoning_effort=args.reasoning_effort,
                    concurrency=args.concurrency,
                    results_dir=results_dir,
                    judge_dir=judge_dir,
                    judge=judge,
                    judge_only=judge_only,
                    max_samples=args.max_samples,
                )
                if results:
                    all_results.extend(results)
                if judge_results:
                    all_judge.extend(judge_results)

    # ── Summary ─────────────────────────────────────────────────────
    _print_summary(all_results, all_judge, deployment, reasoning=reasoning)


if __name__ == "__main__":
    main()

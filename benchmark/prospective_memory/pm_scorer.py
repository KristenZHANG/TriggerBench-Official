"""Prospective Memory rule-based scorer.

Implements rule-based metrics for PM evaluation:

- pm_slot_match: fraction of required_slots matched (positive only; 0 for negative)

Shared across PM evaluation pipelines (longcontext, agentic, etc.).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

# ── Negative variant detection ───────────────────────────────────────────
NEGATIVE_VARIANTS = {"negative_clean"}


# ── Dataclass ────────────────────────────────────────────────────────────

@dataclass
class PMScore:
    task_id: str
    category: str
    variant: str
    bin: str
    is_negative: bool = False
    pm_slot_match: float = 0.0  # fraction of required_slots matched (0 for negatives)


def slot_match(reply: str, sample_dict: dict) -> float:
    """Fraction of required_slots whose accepted surface forms appear in reply."""
    action = sample_dict["ground_truth"]["action"]
    required_slots = action.get("required_slots", [])
    if not required_slots:
        return 1.0
    reply_lc = (reply or "").lower()
    hits = 0
    for slot in required_slots:
        accepted = action.get("slot_values", {}).get(slot, [])
        if any(a.lower() in reply_lc for a in accepted):
            hits += 1
    return hits / len(required_slots)


# ── Sample-level scoring ────────────────────────────────────────────────

def score_sample(entry: dict, reply: str) -> PMScore:
    """Score a single PM sample given the LLM reply.

    Args:
        entry: data entry dict with keys: task_id, category, variant, bin,
               tokens, sample.
        reply: the LLM's reply text.

    Returns:
        PMScore with rule-based metrics.
    """
    sample_dict = entry["sample"]
    # Normalise category: data files may use sub-types like
    # "interrupt-explicit-blueprint" — map to canonical "interrupt".
    raw_cat = entry["category"]
    category = raw_cat.split("-")[0] if "-" in raw_cat else raw_cat
    is_negative = entry["variant"] in NEGATIVE_VARIANTS

    if is_negative:
        pm_slot = 0.0       # slot_match not applicable for negatives
    else:
        pm_slot = slot_match(reply, sample_dict)

    return PMScore(
        task_id=entry["task_id"],
        category=category,
        variant=entry["variant"],
        bin=entry["bin"],
        is_negative=is_negative,
        pm_slot_match=pm_slot,
    )


def score_to_dict(score: PMScore) -> dict:
    """Convert PMScore to a plain dict (for JSON serialization)."""
    return asdict(score)


# ── Aggregation ──────────────────────────────────────────────────────────

def aggregate(scores: Iterable[PMScore]) -> dict[str, float | int]:
    """Aggregate PM scores with positive/negative awareness.

    Returns dict with:
        n, n_pos, n_neg,
        pos_avg_slot_match (positive only)
    """
    scores = list(scores)
    if not scores:
        return {"n": 0}

    pos = [s for s in scores if not s.is_negative]
    neg = [s for s in scores if s.is_negative]

    result: dict[str, float | int] = {
        "n": len(scores),
        "n_pos": len(pos),
        "n_neg": len(neg),
    }

    if pos:
        avg_slot = sum(s.pm_slot_match for s in pos) / len(pos)
        result["pos_avg_slot_match"] = avg_slot

    return result

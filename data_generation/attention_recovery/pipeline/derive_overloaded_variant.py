#!/usr/bin/env python3
"""Stage 4 — Positive-Overloaded variant for Attention Recovery.

Pure local JSON transformation — NO LLM API calls needed.

Because the Attention Recovery Stage 1 blueprint already contains the complete
`story.trigger.overloaded` string (which starts with `trigger.clean`), this
stage simply:
  1. Matches each Stage 3 dialogue to its blueprint via task_id.
  2. Deep-copies the dialogue.
  3. Replaces the last turn's text with `trigger.overloaded`.
  4. Annotates with _clean_trigger and _distracting_suffix for traceability.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
DIM_DIR = PIPELINE_DIR.parent
GEN_ROOT = DIM_DIR.parent
sys.path.insert(0, str(GEN_ROOT))
sys.path.insert(0, str(DIM_DIR))

from shared.json_utils import save_json
from seeds.blueprints import ATTENTION_RECOVERY_SEEDS

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
#  Blueprint Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_blueprint_map(blueprint_file: str) -> dict:
    """Load blueprints from both seed and LLM-generated sources.

    Returns dict: task_id -> blueprint.
    """
    bp_map = {}

    # 1. Seed blueprints
    for bp in ATTENTION_RECOVERY_SEEDS:
        bp_map[bp["task_id"]] = bp

    # 2. LLM-generated blueprints
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
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage 4: Attention Recovery overloaded trigger variant (no LLM)"
    )
    ap.add_argument("--dialogue-input", required=True,
                    help="Path to Stage 3 Positive-Clean dialogues JSON")
    ap.add_argument("--blueprint-input", required=True,
                    help="Path to Stage 1 LLM-generated blueprint JSON")
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    log.info("Loading full dialogues from: %s", args.dialogue_input)
    with open(args.dialogue_input, encoding="utf-8") as f:
        dialogues = json.load(f)
    log.info("  -> %d dialogues", len(dialogues))

    bp_map = load_blueprint_map(args.blueprint_input)

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else DIM_DIR / "outputs" / "stage4_positive_overloaded"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dialogues_positive_overloaded.json"

    results = []
    success = 0
    skipped = 0

    for dlg in dialogues:
        dialogue_id = dlg["dialogue_id"]
        task_id = dialogue_id
        for suffix in ("_positive_clean", "_standard"):
            if task_id.endswith(suffix):
                task_id = task_id[: -len(suffix)]
                break
        overloaded_id = "{}_positive_overloaded".format(task_id)

        # Skip incomplete dialogues
        if dlg.get("status") != "complete":
            log.warning("  %s — status '%s', skipping", dialogue_id, dlg.get("status"))
            skipped += 1
            continue

        # Look up blueprint
        bp = bp_map.get(task_id)
        if bp is None:
            log.warning("  %s — no blueprint found, skipping", task_id)
            skipped += 1
            continue

        # Extract trigger texts from blueprint
        trigger_obj = bp["story"]["trigger"]
        if isinstance(trigger_obj, dict):
            clean_text = trigger_obj.get("clean", "")
            overloaded_text = trigger_obj.get("overloaded", "")
        else:
            log.warning("  %s — trigger is not a dict, skipping", task_id)
            skipped += 1
            continue

        if not overloaded_text:
            log.warning("  %s — empty overloaded trigger in blueprint, skipping", task_id)
            skipped += 1
            continue

        # Validate last turn is user trigger
        turns = dlg["turns"]
        last_turn = turns[-1]
        if last_turn["role"] != "user" or last_turn["function"] != "trigger":
            log.warning("  %s — last turn is not user/trigger (role=%s, func=%s), skipping",
                        task_id, last_turn.get("role"), last_turn.get("function"))
            skipped += 1
            continue

        # Compute distracting suffix
        if overloaded_text.startswith(clean_text):
            distracting_suffix = overloaded_text[len(clean_text):].strip()
        else:
            # Minor LLM variation — use the full overloaded as-is
            distracting_suffix = overloaded_text
            log.info("  %s — overloaded does not start with clean (minor variation)", task_id)

        # Deep copy and modify
        new_dlg = copy.deepcopy(dlg)
        new_dlg["dialogue_id"] = overloaded_id
        new_dlg["turns"][-1]["function"] = "trigger_overloaded"
        new_dlg["turns"][-1]["text"] = overloaded_text
        new_dlg["turns"][-1]["_clean_trigger"] = clean_text
        new_dlg["turns"][-1]["_distracting_suffix"] = distracting_suffix

        results.append(new_dlg)
        success += 1

    save_json(results, out_file)

    log.info("=" * 70)
    log.info("Done!  %d / %d overloaded dialogues generated  (%d skipped).",
             success, len(dialogues), skipped)
    log.info("Saved -> %s", out_file)


if __name__ == "__main__":
    main()

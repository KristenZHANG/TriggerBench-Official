#!/usr/bin/env python3
"""Stage 4 — Positive-Overloaded variant for Temporal Grounding.

Takes Stage 3 Positive-Clean dialogues and appends a distracting task to the
trigger turn while preserving the clean trigger as an exact prefix.

Few-shot examples are Temporal Grounding specific (Time_Query + Time_Blind_Action).
"""

from __future__ import annotations

import argparse
import copy
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
from seeds.blueprints import TEMPORAL_GROUNDING_SEEDS

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
#  System Prompt for overloaded trigger generation
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert dialogue writer. Your task is to write a COMPLETE user message \
that combines two parts into one natural, continuous utterance:

1. The user's original trigger sentence (MUST be preserved exactly as given).
2. A follow-up where the user casually asks about a different question or task \
(the "distracting task").

You will output the FULL combined message: trigger sentence + distracting task request.

═══ STRICT RULES ═══

1. **Preserve the trigger sentence EXACTLY**: Your output MUST begin with the \
trigger sentence word-for-word. Do NOT modify, rephrase, or omit any part of it.

2. **First-person voice for distracting task**: Write as the user ("I", "me", "my").

3. **Transition phrase — use when needed**: By default, directly follow the \
trigger sentence with the distracting task request (no transition phrase) if it is fluent and natural. \
Add a transition phrase if the direct combination reads unnaturally and disrupts the flow. \
When you do use one, choose from phrases like:
   - "By the way, "
   - "Oh, while I'm at it, "
   - "Also, "
   - "Actually, quick question — "
   - "Oh, that reminds me, "
   - "Anyway, "
   - "And, "
   - "On a different note, "
   - "So, "

4. **Convert task description to speech**: The distracting task is given as a \
third-person description. Convert it into a natural first-person request or \
question. Make it fluent and natural.

5. **Keep the distracting part concise**: 1-3 sentences for the distracting task portion.

6. **Plain text only**: No markdown, no preamble, no explanation. \
Output ONLY the full combined user message.\
"""
# - "Also, "


# ═══════════════════════════════════════════════════════════════════════════
#  Few-shot examples — Temporal Grounding specific
# ═══════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = [
    # Time_Query + knowledge question (matches seed_02 pattern)
    {
        "clean_trigger": "What time is it right now? I got completely lost in reading this history paper.",
        "distracting_task": "The user asks the agent to summarize the main causes of the French Revolution.",
        "output": "What time is it right now? I got completely lost in reading this history paper. By the way, before I keep reading, can you summarize the main causes of the French Revolution for me?"
    },
    # Time_Blind_Action + task request (matches seed_01 pattern)
    {
        "clean_trigger": "I'm feeling unexpectedly exhausted today. I'm going to put on my sleep mask, and crash for the night.",
        "distracting_task": "The user asks the agent to recommend a 10-minute deep sleep meditation script.",
        "output": "I'm feeling unexpectedly exhausted today. I'm going to put on my sleep mask, and crash for the night. Could you quickly recommend a 10-minute deep sleep meditation script I can read before closing my eyes?"
    },
    # Time_Query + casual ask (matches seed_03 pattern — series marathon)
    {
        "clean_trigger": "I've been so lost in this marathon of my favorite series... Hey, could you tell me what time it is right now?",
        "distracting_task": "The user asks the agent to explain the concept of a 'MacGuffin' in cinematic storytelling and provide three famous examples from classic films.",
        "output": "I've been so lost in this marathon of my favorite series... Hey, could you tell me what time it is right now? Another question, I keep hearing about this thing called a 'MacGuffin' in film discussions — can you explain what it is and give me three famous examples from classic films?"
    },
    # Time_Blind_Action + knowledge question (matches seed_04 pattern — leaving lab)
    {
        "clean_trigger": "I finally finished the writing of my homework. I'm going to head down to the cafeteria to grab a quick late lunch and catch up with some colleagues.",
        "distracting_task": "The user asks the agent to provide a list of the top five most common high-protein plant-based foods.",
        "output": "I finally finished the writing of my homework. I'm going to head down to the cafeteria to grab a quick late lunch and catch up with some colleagues. Oh, while I'm at it, can you list the top five most common high-protein plant-based foods? I'm trying to eat healthier."
    },
    # Time_Query + casual research (generic time-trigger)
    {
        "clean_trigger": "Wait, what time is it now? I've been deep in this debugging session and totally lost track.",
        "distracting_task": "The user asks the agent to explain the difference between TCP and UDP protocols in simple terms.",
        "output": "Wait, what time is it now? I've been deep in this debugging session and totally lost track. Actually, quick question — can you explain the difference between TCP and UDP in simple terms? I keep mixing them up."
    },
]


def build_user_prompt(clean_trigger: str, distracting_task: str) -> str:
    """Build the user prompt with few-shot examples."""
    examples_text = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += (
            "\n── Example {} ──\n"
            "Trigger sentence: \"{}\"\n"
            "Distracting task description: \"{}\"\n"
            "Your output: {}\n"
        ).format(i, ex["clean_trigger"], ex["distracting_task"], ex["output"])

    return (
        "Here are some examples of the expected behavior:\n"
        "{examples}\n\n"
        "IMPORTANT REMINDERS:\n"
        "- Your output MUST start with the trigger sentence exactly as given.\n"
        "- By default, go directly from trigger to distracting task (no transition phrase).\n"
        "- Only add a transition phrase (By the way, Oh that reminds me, Actually, etc.) "
        "if the direct combination reads awkwardly without one.\n\n"
        "── Your Turn ──\n"
        "Trigger sentence: \"{trigger}\"\n"
        "Distracting task description: \"{task}\"\n"
        "Your output:"
    ).format(
        examples=examples_text,
        trigger=clean_trigger,
        task=distracting_task,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Validation & Cleaning
# ═══════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def validate_output(output: str, clean_trigger: str) -> tuple:
    """Validate the LLM-generated full output. Returns (is_valid, reason).

    Sanity checks:
    1. Output is non-empty and not too long.
    2. Clean trigger is preserved as a prefix (with normalization).
    3. Output contains more than just the trigger (distracting part exists).
    """
    output = output.strip()

    if not output:
        return False, "Empty output"

    if len(output) > 1500:
        return False, "Output too long ({} chars)".format(len(output))

    # Sanity check: clean_trigger must appear as prefix (normalized)
    norm_output = normalize_text(output)
    norm_trigger = normalize_text(clean_trigger)

    if not norm_output.startswith(norm_trigger):
        # Also check if trigger is contained anywhere (more lenient)
        if norm_trigger not in norm_output:
            return False, "Clean trigger not found in output"
        else:
            return False, "Clean trigger found but not as prefix"

    # Check that something was added after the trigger
    suffix_part = output[len(clean_trigger):].strip()
    if len(suffix_part) < 10:
        return False, "Distracting part too short or missing ({} chars)".format(
            len(suffix_part))

    return True, "OK"


def clean_output(output: str) -> str:
    """Clean up LLM-generated output (strip quotes, whitespace)."""
    output = output.strip()
    if (output.startswith('"') and output.endswith('"')) or \
       (output.startswith("'") and output.endswith("'")):
        output = output[1:-1].strip()
    return output


# ═══════════════════════════════════════════════════════════════════════════
#  Core generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_overloaded_trigger(
    clean_trigger: str,
    distracting_task: str,
    model: str = "gpt-4.1",
    temperature: float = 0.8,
    max_retries: int = 5,
) -> str:
    """Generate the full overloaded trigger via LLM, with retry logic.

    The LLM outputs the complete text: trigger + (optional transition) + distracting task.
    We validate that the clean_trigger is preserved as a prefix.

    Returns the full overloaded text, or None if all attempts fail.
    """
    user_prompt = build_user_prompt(clean_trigger, distracting_task)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            raw = chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=512,
            )
            output = clean_output(raw)
            ok, reason = validate_output(output, clean_trigger)
            if ok:
                return output
            else:
                log.warning("  Attempt %d/%d validation failed: %s",
                            attempt, max_retries, reason)
        except Exception as e:
            log.warning("  Attempt %d/%d API error: %s", attempt, max_retries, e)
            time_module.sleep(2)

    log.error("  All %d attempts failed", max_retries)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Blueprint Loader (seed + LLM-generated)
# ═══════════════════════════════════════════════════════════════════════════

def load_blueprint_map(blueprint_file: str) -> dict:
    """Load blueprints from both seed and LLM-generated sources.

    Returns a dict mapping task_id -> blueprint.
    """
    bp_map = {}

    # 1. Seed blueprints
    for bp in TEMPORAL_GROUNDING_SEEDS:
        tid = bp["task_id"]
        bp_map[tid] = bp

    # 2. LLM-generated blueprints
    bp_path = Path(blueprint_file)
    if bp_path.exists():
        with open(bp_path, encoding="utf-8") as f:
            blueprints = json.load(f)
        for bp in blueprints:
            tid = bp["task_id"]
            bp_map[tid] = bp
    else:
        log.warning("Blueprint file not found: %s", bp_path)

    log.info("Loaded %d blueprints (seed + LLM-generated)", len(bp_map))
    return bp_map


# ═══════════════════════════════════════════════════════════════════════════
#  Save helper
# ═══════════════════════════════════════════════════════════════════════════

def save_output(results: list, path: Path):
    save_json(results, path)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage 4: Derive Positive-Overloaded trigger variants"
    )
    ap.add_argument("--dialogue-input", required=True,
                    help="Path to Stage 3 Positive-Clean dialogues JSON")
    ap.add_argument("--blueprint-input", required=True,
                    help="Path to Stage 1 LLM-generated blueprint JSON")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to existing output JSON to resume from")
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    # ── Load data ───────────────────────────────────────────────────────────
    log.info("Loading full dialogues from: %s", args.dialogue_input)
    with open(args.dialogue_input, encoding="utf-8") as f:
        dialogues = json.load(f)
    log.info("  -> %d dialogues", len(dialogues))

    bp_map = load_blueprint_map(args.blueprint_input)

    # ── Output setup ────────────────────────────────────────────────────────
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else DIM_DIR / "outputs" / "stage4_positive_overloaded"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dialogues_positive_overloaded_{}.json".format(args.model)

    # ── Resume support ──────────────────────────────────────────────────────
    completed = {}  # dialogue_id -> dialogue dict
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8") as f:
                loaded = json.load(f)
            for item in loaded:
                completed[item["dialogue_id"]] = item
            log.info("Resumed %d dialogues from %s", len(completed), resume_path)

    # ── Generation loop ─────────────────────────────────────────────────────
    results = list(completed.values())
    success = len(results)
    fail = 0

    for dlg in tqdm(dialogues, desc="Stage 4 Positive-Overloaded"):
        if args.sleep:
            time_module.sleep(args.sleep)

        dialogue_id = dlg["dialogue_id"]
        task_id = dialogue_id
        for suffix in ("_positive_clean", "_standard"):
            if task_id.endswith(suffix):
                task_id = task_id[: -len(suffix)]
                break
        overloaded_id = "{}_positive_overloaded".format(task_id)

        # Skip if already done
        if overloaded_id in completed:
            log.info("  %s — already done, skipping", overloaded_id)
            continue

        # Skip partial/failed dialogues
        if dlg.get("status") != "complete":
            log.warning("  %s — status is '%s', skipping", dialogue_id, dlg.get("status"))
            fail += 1
            continue

        # Look up blueprint
        bp = bp_map.get(task_id)
        if bp is None:
            log.warning("  %s — no blueprint found, skipping", task_id)
            fail += 1
            continue

        # Extract distracting_task from blueprint
        story = bp.get("story", {})
        variations = story.get("variations", {})
        distracting_task = variations.get("distracting_task")
        if not distracting_task:
            log.warning("  %s — no distracting_task in blueprint, skipping", task_id)
            fail += 1
            continue

        # Extract clean trigger (last user turn)
        turns = dlg["turns"]
        last_turn = turns[-1]
        if last_turn["role"] != "user" or last_turn["function"] != "trigger":
            log.warning("  %s — last turn is not user trigger (role=%s, func=%s), skipping",
                        task_id, last_turn.get("role"), last_turn.get("function"))
            fail += 1
            continue

        clean_trigger = last_turn["text"]

        log.info("  %s — generating overloaded trigger...", task_id)

        overloaded_text = generate_overloaded_trigger(
            clean_trigger=clean_trigger,
            distracting_task=distracting_task,
            model=args.model,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )

        if overloaded_text is None:
            log.error("  %s — FAILED to generate overloaded trigger", task_id)
            fail += 1
            continue

        # Build new dialogue (deep copy, modify last turn only)
        new_dlg = copy.deepcopy(dlg)
        new_dlg["dialogue_id"] = overloaded_id
        new_dlg["turns"][-1]["text"] = overloaded_text
        new_dlg["turns"][-1]["function"] = "trigger_overloaded"
        # Store originals for traceability
        new_dlg["turns"][-1]["_clean_trigger"] = clean_trigger
        new_dlg["turns"][-1]["_distracting_suffix"] = overloaded_text[len(clean_trigger):].strip()

        results.append(new_dlg)
        completed[overloaded_id] = new_dlg
        success += 1

        # Incremental save
        save_output(results, out_file)
        log.info("  %s — done: ...%s", task_id, overloaded_text[len(clean_trigger):][:120])

    # ── Final save ──────────────────────────────────────────────────────────
    save_output(results, out_file)

    log.info("=" * 70)
    log.info("Done!  %d / %d overloaded dialogues generated  (%d failed).",
             success, len(dialogues), fail)
    log.info("Saved -> %s", out_file)


if __name__ == "__main__":
    main()

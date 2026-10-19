#!/usr/bin/env bash
# Stage 3 — Dialogue completion / Positive-Clean (Logical Adherence)
set -euo pipefail

MODEL="${MODEL:-gpt-4.1}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_RETRY="${MAX_RETRY:-3}"
STAGE2_MODEL="${STAGE2_MODEL:-gpt-5.4}"
# false = fully exempt constraint turns (matches the original production run)
BLACKLIST_CONSTRAINT="${BLACKLIST_CONSTRAINT:-false}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE="${DIM_DIR}/pipeline/complete_dialogues.py"
STAGE2_FILE="${STAGE2_FILE:-${DIM_DIR}/outputs/stage2_user_dialogues/${STAGE2_MODEL}_temp1.0/user_dialogues_${STAGE2_MODEL}.json}"
OUTPUT_DIR="${DIM_DIR}/outputs/stage3_positive_clean/${MODEL}_temp${TEMPERATURE}"

EXTRA_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FILE="${OUTPUT_DIR}/dialogues_positive_clean_${MODEL}.json"
    if [[ -f "$RESUME_FILE" ]]; then
        EXTRA_ARGS+=(--resume "$RESUME_FILE")
        echo "Resuming from: $RESUME_FILE"
    fi
fi

if [[ "$BLACKLIST_CONSTRAINT" == "true" ]]; then
    BL_FLAG="--blacklist-constraint"
else
    BL_FLAG="--no-blacklist-constraint"
fi

echo "Stage 2 input: $STAGE2_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Blacklist constraint: $BLACKLIST_CONSTRAINT"

python3 "$PIPELINE" \
    --input "$STAGE2_FILE" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --max-retries "$MAX_RETRY" \
    --output-dir "$OUTPUT_DIR" \
    "$BL_FLAG" \
    "${EXTRA_ARGS[@]}"

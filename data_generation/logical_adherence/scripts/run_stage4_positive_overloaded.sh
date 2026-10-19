#!/usr/bin/env bash
# Stage 4 — Positive-Overloaded variant (Logical Adherence)
set -euo pipefail

MODEL="${MODEL:-gpt-4.1}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_RETRY="${MAX_RETRY:-5}"
STAGE1_MODEL="${STAGE1_MODEL:-gpt-5.4}"
STAGE3_MODEL="${STAGE3_MODEL:-gpt-4.1}"
K_PER_COMBO="${K_PER_COMBO:-7}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE="${DIM_DIR}/pipeline/derive_overloaded_variant.py"
STAGE3_FILE="${STAGE3_FILE:-${DIM_DIR}/outputs/stage3_positive_clean/${STAGE3_MODEL}_temp0.7/dialogues_positive_clean_${STAGE3_MODEL}.json}"
BLUEPRINT_FILE="${BLUEPRINT_FILE:-${DIM_DIR}/outputs/stage1_blueprints/${STAGE1_MODEL}_num${K_PER_COMBO}_temp1.0_max5/blueprints_logical_adherence_${STAGE1_MODEL}.json}"
OUTPUT_DIR="${DIM_DIR}/outputs/stage4_positive_overloaded"

EXTRA_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FILE="${OUTPUT_DIR}/dialogues_positive_overloaded_${MODEL}.json"
    if [[ -f "$RESUME_FILE" ]]; then
        EXTRA_ARGS+=(--resume "$RESUME_FILE")
        echo "Resuming from: $RESUME_FILE"
    fi
fi

echo "Stage 3 input: $STAGE3_FILE"
echo "Blueprint input: $BLUEPRINT_FILE"

python3 "$PIPELINE" \
    --dialogue-input "$STAGE3_FILE" \
    --blueprint-input "$BLUEPRINT_FILE" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --max-retries "$MAX_RETRY" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"

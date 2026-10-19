#!/usr/bin/env bash
# Stage 2 — User-dialogue stretching (Logical Adherence)
set -euo pipefail

MODEL="${MODEL:-gpt-5.4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_RETRY="${MAX_RETRY:-5}"
STAGE1_MODEL="${STAGE1_MODEL:-gpt-5.4}"
K_PER_COMBO="${K_PER_COMBO:-7}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE="${DIM_DIR}/pipeline/generate_user_dialogues.py"
BLUEPRINT_FILE="${BLUEPRINT_FILE:-${DIM_DIR}/outputs/stage1_blueprints/${STAGE1_MODEL}_num${K_PER_COMBO}_temp1.0_max5/blueprints_logical_adherence_${STAGE1_MODEL}.json}"
OUTPUT_DIR="${DIM_DIR}/outputs/stage2_user_dialogues/${MODEL}_temp${TEMPERATURE}"

EXTRA_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FILE="${OUTPUT_DIR}/user_dialogues_${MODEL}.json"
    if [[ -f "$RESUME_FILE" ]]; then
        EXTRA_ARGS+=(--resume "$RESUME_FILE")
        echo "Resuming from: $RESUME_FILE"
    fi
fi

echo "Blueprint file: $BLUEPRINT_FILE"
echo "Output dir: $OUTPUT_DIR"

python3 "$PIPELINE" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --max-retries "$MAX_RETRY" \
    --blueprint-file "$BLUEPRINT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --include-seeds \
    "${EXTRA_ARGS[@]}"

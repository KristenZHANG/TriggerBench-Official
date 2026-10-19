#!/usr/bin/env bash
# Stage 1 — Blueprint design (Logical Adherence)
set -euo pipefail

MODEL="${MODEL:-gpt-5.4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_RETRY="${MAX_RETRY:-5}"
K_PER_COMBO="${K_PER_COMBO:-7}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE="${DIM_DIR}/pipeline/generate_blueprints.py"
OUTPUT_DIR="${DIM_DIR}/outputs/stage1_blueprints/${MODEL}_num${K_PER_COMBO}_temp${TEMPERATURE}_max${MAX_RETRY}"

EXTRA_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FILE="${OUTPUT_DIR}/blueprints_logical_adherence_${MODEL}.json"
    if [[ -f "$RESUME_FILE" ]]; then
        EXTRA_ARGS+=(--resume "$RESUME_FILE")
        echo "Resuming from: $RESUME_FILE"
    else
        echo "No existing file to resume from, starting fresh."
    fi
fi

python3 "$PIPELINE" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --max-retries "$MAX_RETRY" \
    --k-per-combo "$K_PER_COMBO" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"

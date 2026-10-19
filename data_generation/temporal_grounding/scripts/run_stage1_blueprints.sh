#!/usr/bin/env bash
# Stage 1 — Blueprint design (Temporal Grounding)
set -euo pipefail

MODEL="${MODEL:-gpt-5.4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_RETRY="${MAX_RETRY:-5}"
N_PER_CELL="${N_PER_CELL:-5}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE="${DIM_DIR}/pipeline/generate_blueprints.py"
OUTPUT_DIR="${DIM_DIR}/outputs/stage1_blueprints/${MODEL}_temp${TEMPERATURE}_n${N_PER_CELL}_max${MAX_RETRY}"

EXTRA_ARGS=()
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FILE="${OUTPUT_DIR}/blueprints_temporal_grounding_${MODEL}.json"
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
    --n-per-cell "$N_PER_CELL" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"

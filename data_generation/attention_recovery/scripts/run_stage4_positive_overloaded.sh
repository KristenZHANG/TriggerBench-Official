#!/usr/bin/env bash
# Stage 4 — Positive-Overloaded variant (Attention Recovery)
# Local string swap — no LLM.
set -euo pipefail

STAGE1_MODEL="${STAGE1_MODEL:-gpt-5.4}"
STAGE3_MODEL="${STAGE3_MODEL:-gpt-4.1}"
K_PER_COMBO="${K_PER_COMBO:-6}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIPELINE="${DIM_DIR}/pipeline/derive_overloaded_variant.py"
STAGE3_FILE="${STAGE3_FILE:-${DIM_DIR}/outputs/stage3_positive_clean/${STAGE3_MODEL}_temp0.7/dialogues_positive_clean_${STAGE3_MODEL}.json}"
BLUEPRINT_FILE="${BLUEPRINT_FILE:-${DIM_DIR}/outputs/stage1_blueprints/${STAGE1_MODEL}_num${K_PER_COMBO}_temp1.0_max5/blueprints_attention_recovery_${STAGE1_MODEL}.json}"
OUTPUT_DIR="${DIM_DIR}/outputs/stage4_positive_overloaded"

echo "Stage 3 input:  $STAGE3_FILE"
echo "Blueprint input: $BLUEPRINT_FILE"
echo "Output dir:      $OUTPUT_DIR"

python3 "$PIPELINE" \
    --dialogue-input "$STAGE3_FILE" \
    --blueprint-input "$BLUEPRINT_FILE" \
    --output-dir "$OUTPUT_DIR"

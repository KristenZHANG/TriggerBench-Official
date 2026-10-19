#!/bin/bash
# Mem0 PM evaluation. Flags: ../README.md

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$SCRIPT_DIR"

ENVF="${SCRIPT_DIR}/.env"
if [[ ! -f "$ENVF" ]]; then
    echo "ERROR: missing $ENVF"
    echo "       cp .env.example .env  # then fill in"
    exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENVF"
set +a

export AZURE_OPENAI_DEPLOYMENTS="${AZURE_OPENAI_DEPLOYMENTS:-gpt-4.1}"

CONCURRENCY=1
TEMPERATURE=0.6
TOP_K=6
BATCH_SIZE=40
USE_JUDGE=true
MAX_SAMPLES=-1

CATEGORY_FLAG=""
VARIANT_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --category) CATEGORY_FLAG="--category $2"; shift 2;;
        --variant) VARIANT_FLAG="--variant $2"; shift 2;;
        --deployment) export AZURE_OPENAI_DEPLOYMENTS="$2"; shift 2;;
        --concurrency) CONCURRENCY=$2; shift 2;;
        --temperature) TEMPERATURE=$2; shift 2;;
        --top-k) TOP_K=$2; shift 2;;
        --batch-size) BATCH_SIZE=$2; shift 2;;
        --no-judge) USE_JUDGE=false; shift;;
        --max-samples) MAX_SAMPLES=$2; shift 2;;
        *) echo "Unknown flag: $1"; exit 1;;
    esac
done

if [[ -z "$CATEGORY_FLAG" && -z "$VARIANT_FLAG" ]]; then
    RUN_FLAG="--all"
else
    RUN_FLAG="$CATEGORY_FLAG $VARIANT_FLAG"
fi

JUDGE_FLAG=""
if [[ "$USE_JUDGE" == true ]]; then
    JUDGE_FLAG="--judge"
fi

DATA_DIR="$SCRIPT_DIR/../data/base_context"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Data not found at $DATA_DIR"
    exit 1
fi

if ! $PYTHON -c "from mem0 import Memory" 2>/dev/null; then
    echo "ERROR: mem0 not found in $PYTHON"
    echo "       Run: $PYTHON -m pip install mem0ai qdrant-client"
    exit 1
fi

MAX_SAMPLES_FLAG=""
if [[ $MAX_SAMPLES -gt 0 ]]; then
    MAX_SAMPLES_FLAG="--max-samples $MAX_SAMPLES"
fi

echo "Mem0 PM evaluation → $SCRIPT_DIR/../main_results/mem0/"

$PYTHON mem0_eval.py $RUN_FLAG \
    --concurrency $CONCURRENCY \
    --temperature $TEMPERATURE \
    --top-k $TOP_K \
    --batch-size $BATCH_SIZE \
    $JUDGE_FLAG \
    $MAX_SAMPLES_FLAG

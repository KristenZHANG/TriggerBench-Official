#!/bin/bash
# Local vLLM PM evaluation. Start the server with run_vllm_server.sh first.
# Flags: ../README.md

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

MODEL="Qwen3-32B"
PORT=8000
HOST="localhost"
CONCURRENCY=1
TEMPERATURE=0.6
USE_JUDGE=true
BIN="base"
MAX_SAMPLES=-1

export AZURE_OPENAI_DEPLOYMENTS="${AZURE_OPENAI_DEPLOYMENTS:-gpt-4o}"

CATEGORY_FLAG=""
VARIANT_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --category) CATEGORY_FLAG="--category $2"; shift 2;;
        --variant) VARIANT_FLAG="--variant $2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --port) PORT=$2; shift 2;;
        --host) HOST="$2"; shift 2;;
        --concurrency) CONCURRENCY=$2; shift 2;;
        --temperature) TEMPERATURE=$2; shift 2;;
        --no-judge) USE_JUDGE=false; shift;;
        --bin) BIN="$2"; shift 2;;
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

if ! curl -s "http://${HOST}:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM server not reachable at http://${HOST}:${PORT}"
    echo "       Start it first: bash run_vllm_server.sh qwen3-32b"
    exit 1
fi

if [[ "$BIN" == "40K" ]]; then
    DATA_DIR="$SCRIPT_DIR/../data/40K_context"
else
    DATA_DIR="$SCRIPT_DIR/../data/base_context"
fi
if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Data not found at $DATA_DIR"
    exit 1
fi

MAX_SAMPLES_FLAG=""
if [[ $MAX_SAMPLES -gt 0 ]]; then
    MAX_SAMPLES_FLAG="--max-samples $MAX_SAMPLES"
fi

echo "Blended (open / vLLM) PM evaluation → $SCRIPT_DIR/../main_results/blended_open/"

$PYTHON longcontext_blended_open.py $RUN_FLAG \
    --bin $BIN \
    --model "$MODEL" \
    --port $PORT \
    --host "$HOST" \
    --concurrency $CONCURRENCY \
    --temperature $TEMPERATURE \
    $JUDGE_FLAG \
    $MAX_SAMPLES_FLAG

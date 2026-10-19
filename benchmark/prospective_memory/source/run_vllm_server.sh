#!/bin/bash
# vLLM launcher for blended_open. See ../README.md
#   export LLM_DIR=/path/to/weights
#   bash run_vllm_server.sh [qwen3-32b|instruct|thinking|gemma3-27b]
#
# --served-model-name is the id you pass to run_blended_open.sh --model.

set -e

MODE="${1:-qwen3-32b}"
LLM_DIR="${LLM_DIR:?export LLM_DIR=/path/to/model/weights}"
VLLM="${VLLM:-vllm}"

case "$MODE" in
  instruct|235b-instruct)
    NAME="Qwen3-235B-A22B-Instruct-2507-FP8"
    echo "Starting ${NAME} (4 GPUs, no reasoning parser)"
    echo "Client: bash run_blended_open.sh --model ${NAME}   # add --bin 40K if needed"
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$VLLM" serve "${LLM_DIR}/${NAME}" \
      --served-model-name "$NAME" \
      --port 8000 \
      --max-model-len 131072 \
      --tensor-parallel-size 4 \
      --gpu-memory-utilization 0.85
    ;;
  thinking|235b-thinking)
    NAME="Qwen3-235B-A22B-Thinking-2507-FP8"
    echo "Starting ${NAME} (4 GPUs, deepseek_r1 parser)"
    echo "Client: bash run_blended_open.sh --model ${NAME}   # add --bin 40K if needed"
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$VLLM" serve "${LLM_DIR}/${NAME}" \
      --served-model-name "$NAME" \
      --port 8000 \
      --max-model-len 131072 \
      --tensor-parallel-size 4 \
      --gpu-memory-utilization 0.9 \
      --reasoning-parser deepseek_r1
    ;;
  qwen3-32b|32b|"")
    NAME="Qwen3-32B"
    echo "Starting ${NAME} (4 GPUs, qwen3 parser)"
    echo "Client: bash run_blended_open.sh --model ${NAME}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$VLLM" serve "${LLM_DIR}/${NAME}" \
      --served-model-name "$NAME" \
      --port 8000 \
      --max-model-len 32768 \
      --tensor-parallel-size 4 \
      --gpu-memory-utilization 0.7 \
      --reasoning-parser qwen3
    ;;
  gemma3-27b|gemma3|gemma)
    NAME="gemma-3-27b-it"
    echo "Starting ${NAME} (4 GPUs, 27B dense)"
    echo "Client: bash run_blended_open.sh --model ${NAME}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$VLLM" serve "${LLM_DIR}/${NAME}" \
      --served-model-name "$NAME" \
      --port 8000 \
      --max-model-len 32768 \
      --tensor-parallel-size 4 \
      --gpu-memory-utilization 0.85
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash run_vllm_server.sh [qwen3-32b|instruct|thinking|gemma3-27b]"
    exit 1
    ;;
esac

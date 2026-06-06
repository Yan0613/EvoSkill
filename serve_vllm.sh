#!/bin/bash
# Start vLLM server for EvoSkill benchmarks
# Usage: bash serve_vllm.sh
# Press Ctrl+C to stop the server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

HF_MODEL="Qwen/Qwen2.5-7B-Instruct"
VLLM_PORT=8765
TP_SIZE=4
LOG_DIR="$SCRIPT_DIR/results/logs"
mkdir -p "$LOG_DIR"

echo "Starting vLLM server..."
echo "  Model : $HF_MODEL"
echo "  Port  : $VLLM_PORT"
echo "  TP    : $TP_SIZE"
echo "  GPUs  : $CUDA_VISIBLE_DEVICES"
echo "  Log   : $LOG_DIR/vllm_server.log"
echo ""

vllm serve "$HF_MODEL" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --tokenizer-mode slow \
    2>&1 | tee "$LOG_DIR/vllm_server.log"

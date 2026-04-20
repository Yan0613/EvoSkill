#!/bin/bash
# Start vLLM server for Gemma 4 31B-IT
# Usage: bash serve_gemma4.sh
# Press Ctrl+C to stop the server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_PRELOAD=/hkfs/work/workspace/scratch/lmu_eqm3765-misc/conda_envs/evoskill/lib/libstdc++.so.6

MODEL_PATH="/hkfs/work/workspace/scratch/lmu_eqm3765-misc/hf_cache/models--google--gemma-4-31B-it"
VLLM_PORT=8765
TP_SIZE=4
LOG_DIR="$SCRIPT_DIR/results/logs"
mkdir -p "$LOG_DIR"

# Redirect all subsequent output (echo + vllm) to log file only
exec >> "$LOG_DIR/vllm_gemma4_server.log" 2>&1

echo "Starting vLLM server for Gemma 4..."
echo "  Model : $MODEL_PATH"
echo "  Port  : $VLLM_PORT"
echo "  TP    : $TP_SIZE"
echo "  GPUs  : $CUDA_VISIBLE_DEVICES"
echo ""

conda run -p /hkfs/work/workspace/scratch/lmu_eqm3765-misc/conda_envs/evoskill \
    vllm serve "$MODEL_PATH" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    2>&1 | tee "$LOG_DIR/vllm_gemma4_server.log"

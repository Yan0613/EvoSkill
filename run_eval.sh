#!/bin/bash
# Run all EvoSkill benchmarks (requires vLLM server already running)
# Usage: bash run_eval.sh
# Start vLLM first: bash serve_vllm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR"


HF_MODEL="Qwen/Qwen2.5-32B-Instruct"
MAX_TOKENS=4096
CONTEXT_LENGTH=32768
MAX_CONCURRENT=64
VLLM_PORT=8765
VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
LOG_DIR="$SCRIPT_DIR/results/logs"
mkdir -p "$LOG_DIR" results

echo "============================================================"
echo "  EvoSkill Benchmark Runner - Qwen2.5-72B-Instruct (vLLM)"
echo "  Model   : $HF_MODEL"
echo "  Backend : $VLLM_BASE_URL"
echo "  Concurrent: $MAX_CONCURRENT"
echo "  Time    : $(date)"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────
# 1. DABStep
# ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/4] DABStep benchmark"

rm -f results/dabstep_qwen25_72b.pkl

python3 -W ignore scripts/run_eval_dabstep.py \
    --sdk vllm \
    --model "$HF_MODEL" \
    --vllm-base-url "$VLLM_BASE_URL" \
    --vllm-max-tokens "$MAX_TOKENS" \
    --vllm-context-length "$CONTEXT_LENGTH" \
    --max-concurrent "$MAX_CONCURRENT" \
    --dataset .dataset/dabstep_dev.csv \
    --data-dir .dataset/DABstep-data/data/context \
    --output results/dabstep_qwen25_72b.pkl \
    --no-resume \
    2>&1 | grep -v "pynvml\|FutureWarning\|torch_dtype\|Loading checkpoint\|generation flags" \
    | tee "$LOG_DIR/dabstep.log"

echo ">>> DABStep done."

# ─────────────────────────────────────────────────────────────────
# 2. LiveCodeBench
# ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/4] LiveCodeBench benchmark"

rm -f results/livecodebench_qwen25_72b.pkl

python3 -W ignore scripts/run_eval_livecodebench.py \
    --sdk vllm \
    --model "$HF_MODEL" \
    --vllm-base-url "$VLLM_BASE_URL" \
    --vllm-max-tokens "$MAX_TOKENS" \
    --vllm-context-length "$CONTEXT_LENGTH" \
    --max-concurrent "$MAX_CONCURRENT" \
    --output results/livecodebench_qwen25_72b.pkl \
    --no-resume \
    2>&1 | grep -v "pynvml\|FutureWarning\|torch_dtype\|Loading checkpoint\|generation flags" \
    | tee "$LOG_DIR/livecodebench.log"

echo ">>> LiveCodeBench done."

# ─────────────────────────────────────────────────────────────────
# 3. SEAL-QA
# ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> [3/4] SEAL-QA benchmark"

if [ ! -f ".dataset/seal-0.csv" ]; then
    echo "    Downloading SEAL-QA dataset..."
    python3 -W ignore -c "
import os
from datasets import load_dataset
import os as _os
#set proxy and hf token there
_os.makedirs('.dataset', exist_ok=True)
ds = load_dataset('seal-research/seal-qa', split='test')
ds.to_csv('.dataset/seal-0.csv', index=False)
print('SEAL-QA downloaded:', len(ds), 'samples')
" 2>&1 | grep -v "pynvml\|FutureWarning\|Downloading\|Generating"
fi

if [ -f ".dataset/seal-0.csv" ]; then
    rm -f results/sealqa_qwen25_72b.pkl
    python3 -W ignore scripts/run_eval_sealqa.py \
        --sdk vllm \
        --model "$HF_MODEL" \
        --vllm-base-url "$VLLM_BASE_URL" \
        --vllm-max-tokens "$MAX_TOKENS" \
        --vllm-context-length "$CONTEXT_LENGTH" \
        --max-concurrent "$MAX_CONCURRENT" \
        --output results/sealqa_qwen25_72b.pkl \
        --no-resume \
        2>&1 | grep -v "pynvml\|FutureWarning\|torch_dtype\|Loading checkpoint\|generation flags" \
        | tee "$LOG_DIR/sealqa.log"
    echo ">>> SEAL-QA done."
else
    echo "    [SKIP] SEAL-QA dataset not available"
fi

# ─────────────────────────────────────────────────────────────────
# 4. OfficeQA
# ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> [4/4] OfficeQA benchmark"

OFFICEQA_DATASET=".dataset/officeqa/officeqa_pro.csv"

if [ ! -f "$OFFICEQA_DATASET" ]; then
    echo "    [SKIP] OfficeQA dataset not found at $OFFICEQA_DATASET"
else
    rm -f results/officeqa_qwen25_72b.pkl

    python3 -W ignore scripts/run_eval.py \
        --sdk vllm \
        --model "$HF_MODEL" \
        --vllm_base_url "$VLLM_BASE_URL" \
        --vllm_max_tokens "$MAX_TOKENS" \
        --vllm_context_length "$CONTEXT_LENGTH" \
        --max_concurrent "$MAX_CONCURRENT" \
        --dataset_path "$OFFICEQA_DATASET" \
        --output results/officeqa_qwen25_72b.pkl \
        --resume False \
        2>&1 | grep -v "pynvml\|FutureWarning\|torch_dtype\|Loading checkpoint\|generation flags" \
        | tee "$LOG_DIR/officeqa.log"

    echo ">>> OfficeQA done."
fi 

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  All benchmarks completed!  $(date)"
echo "  Results: $SCRIPT_DIR/results/"
echo "  Logs   : $LOG_DIR/"
echo "============================================================"
echo ""
echo "=== FINAL SCORES ==="
for log in "$LOG_DIR"/*.log; do
    [ "$(basename $log)" = "vllm_server.log" ] && continue
    name=$(basename "$log" .log)
    echo "--- $name ---"
    grep -E "Accuracy|Pass@1|Successful|Failed|Total completed" "$log" 2>/dev/null | tail -5
done
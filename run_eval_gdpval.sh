#!/usr/bin/env bash
# Run GDPVal evaluation with EvoSkill (vLLM backend).
#
# Usage:
#   bash run_eval_gdpval.sh                         # defaults (all 220 tasks)
#   LIMIT=20 bash run_eval_gdpval.sh                # quick test with 20 tasks
#   MAX_CONCURRENT=2 bash run_eval_gdpval.sh        # reduce concurrency
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config ───────────────────────────────────────────────────────────────────
export VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:8765/v1}"
VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-8192}"
VLLM_CONTEXT_LENGTH="${VLLM_CONTEXT_LENGTH:-131072}"
MODEL_PATH="${MODEL_PATH:-/hkfs/work/workspace/scratch/lmu_eqm3765-misc/hf_cache/models--google--gemma-4-31B-it}"

HF_CACHE="${HF_CACHE:-/hkfs/work/workspace/scratch/lmu_eqm3765-skill/SkillEvolve/data/benchmarks/gdpval}"
WORKSPACE_DIR="${WORKSPACE_DIR:-outputs/gdpval_workspaces}"
OUTPUT="${OUTPUT:-results/gdpval_gemma4_31b.pkl}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
LIMIT="${LIMIT:-}"

PYTHON="${PYTHON:-/hkfs/work/workspace/scratch/lmu_eqm3765-misc/conda_envs/skill/bin/python}"

export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

# ── Set vLLM backend ─────────────────────────────────────────────────────────

echo "================================================================"
echo "GDPVal evaluation — EvoSkill (vLLM)"
echo "API:          $VLLM_API_BASE"
echo "Model:        $MODEL_PATH"
echo "HF cache:     $HF_CACHE"
echo "Workspace:    $WORKSPACE_DIR"
echo "Output:       $OUTPUT"
echo "Concurrent:   $MAX_CONCURRENT"
echo "Limit:        ${LIMIT:-none}"
echo "Max tokens:   $VLLM_MAX_TOKENS"
echo "Context len:  $VLLM_CONTEXT_LENGTH"
echo "================================================================"

LIMIT_ARG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARG="--limit $LIMIT"
fi

"$PYTHON" src/evaluation/gdpval_runner.py \
    --cache-dir "$HF_CACHE" \
    --workspace-dir "$WORKSPACE_DIR" \
    --output "$OUTPUT" \
    --model "$MODEL_PATH" \
    --max-concurrent "$MAX_CONCURRENT" \
    --sdk vllm \
    --vllm-base-url "$VLLM_API_BASE" \
    --vllm-max-tokens "$VLLM_MAX_TOKENS" \
    --vllm-context-length "$VLLM_CONTEXT_LENGTH" \
    $LIMIT_ARG

echo "================================================================"
echo "Done. Results saved to: $OUTPUT"
echo "================================================================"

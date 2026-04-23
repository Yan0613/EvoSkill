#!/usr/bin/env bash
# Two-phase EvoSkill OfficeQA evaluation:
#   Phase 2 — loop:       run self-improving loop
#   Phase 3 — post-loop:  eval with evolved skill/prompt
#
# Usage:
#   bash scripts/eval_officeqa.sh              # both phases
#   PHASES=2 bash scripts/eval_officeqa.sh     # loop only
#   PHASES=3 bash scripts/eval_officeqa.sh     # eval only
#   LIMIT=20 bash scripts/eval_officeqa.sh     # cap eval at 20 samples
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# ── Environment ──────────────────────────────────────────────────────────────
export TERM="${TERM:-xterm-256color}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:8765/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export LITELLM_LOG="${LITELLM_LOG:-ERROR}"

PYTHON="${PYTHON:-/hkfs/work/workspace/scratch/lmu_eqm3765-misc/conda_envs/skill/bin/python}"
MODEL="${MODEL:-/hkfs/work/workspace/scratch/lmu_eqm3765-misc/hf_cache/models--google--gemma-4-31B-it}"
DATASET_CSV="${DATASET_CSV:-.dataset/officeqa/officeqa_pro.csv}"
LOOP_CSV="${LOOP_CSV:-.dataset/officeqa/officeqa_loop.csv}"
LIMIT="${LIMIT:-}"
PHASES="${PHASES:-2,3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
CONCURRENCY="${CONCURRENCY:-4}"
TRAIN_RATIO="${TRAIN_RATIO:-0.40}"
VAL_RATIO="${VAL_RATIO:-0.25}"
MODE="${MODE:-skill_only}"

TS=$(date +%Y%m%dT%H%M%S)
BASE_DIR="results/eval_officeqa/${TS}"
mkdir -p "${BASE_DIR}"

LOG="${BASE_DIR}/master.log"

log() { echo "$*" | tee -a "${LOG}"; }

maybe_limit() {
    if [[ -n "${LIMIT}" ]]; then echo "--num_samples ${LIMIT}"; fi
}

log "================================================================"
log "EvoSkill OfficeQA evaluation"
log "Model:      ${MODEL}"
log "vLLM API:   ${VLLM_API_BASE}"
log "Dataset:    ${DATASET_CSV}"
log "Phases:     ${PHASES}"
log "Mode:       ${MODE}"
log "Limit:      ${LIMIT:-none}"
log "Started:    $(date)"
log "Output:     ${BASE_DIR}"
log "================================================================"

# ── Prepare loop dataset (add category column from difficulty) ────────────────
if [[ ",${PHASES}," == *",2,"* ]]; then
    if [[ ! -f "${LOOP_CSV}" ]]; then
        log ""
        log "── Preparing loop dataset (difficulty → category) ──"
        "$PYTHON" -c "
import pandas as pd
df = pd.read_csv('${DATASET_CSV}')
df['category'] = df['difficulty']
df.to_csv('${LOOP_CSV}', index=False)
print('Categories:', dict(df['category'].value_counts()))
print('Saved to: ${LOOP_CSV}')
" | tee -a "${LOG}"
    else
        log "Loop dataset already exists: ${LOOP_CSV}"
    fi
fi

# ── Phase 2: Self-improving loop ──────────────────────────────────────────────
if [[ ",${PHASES}," == *",2,"* ]]; then
    log ""
    log "── [phase2_loop] starting at $(date) ──"
    if "$PYTHON" scripts/run_loop.py \
        --sdk vllm \
        --vllm_base_url "${VLLM_API_BASE}" \
        --model "${MODEL}" \
        --dataset "${LOOP_CSV}" \
        --mode "${MODE}" \
        --max_iterations "${MAX_ITERATIONS}" \
        --train_ratio "${TRAIN_RATIO}" \
        --val_ratio "${VAL_RATIO}" \
        --concurrency "${CONCURRENCY}" \
        2>&1 | tee "${BASE_DIR}/phase2_loop.log"; then
        log "── [phase2_loop] DONE at $(date) ──"
    else
        log "── [phase2_loop] FAILED at $(date) (exit $?) ──"
    fi
fi

# ── Phase 3: Post-loop eval (with evolved skill/prompt) ───────────────────────
if [[ ",${PHASES}," == *",3,"* ]]; then
    log ""
    log "── [phase3_post_loop] starting at $(date) ──"
    if "$PYTHON" scripts/run_eval.py \
        --sdk vllm \
        --vllm_base_url "${VLLM_API_BASE}" \
        --model "${MODEL}" \
        --dataset_path "${DATASET_CSV}" \
        --output "${BASE_DIR}/phase3_post_loop.pkl" \
        --max_concurrent "${CONCURRENCY}" \
        --resume false \
        $(maybe_limit) \
        2>&1 | tee "${BASE_DIR}/phase3_post_loop.log"; then
        log "── [phase3_post_loop] DONE at $(date) ──"
    else
        log "── [phase3_post_loop] FAILED at $(date) (exit $?) ──"
    fi
fi

log ""
log "================================================================"
log "All requested phases complete at $(date)"
log "Output: ${BASE_DIR}"
log "================================================================"

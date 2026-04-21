#!/usr/bin/env bash
# Three-phase EvoSkill SealQA evaluation:
#   Phase 1 — baseline:      no loop, eval only
#   Phase 2 — loop:          run self-improving loop
#   Phase 3 — post-loop:     eval with evolved skill/prompt
#
# Usage:
#   bash scripts/eval_sealqa.sh              # all 3 phases
#   PHASES=1 bash scripts/eval_sealqa.sh     # eval only (no loop)
#   PHASES=2,3 bash scripts/eval_sealqa.sh   # loop + post-loop eval
#   PHASES=2,3 LIMIT=20 bash ...             # cap dataset at 20 samples
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
DATASET="${DATASET:-.dataset/seal-0.csv}"
LIMIT="${LIMIT:-}"
PHASES="${PHASES:-1,2,3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
CONCURRENCY="${CONCURRENCY:-4}"
TRAIN_RATIO="${TRAIN_RATIO:-0.40}"
VAL_RATIO="${VAL_RATIO:-0.25}"
MODE="${MODE:-skill_only}"

TS=$(date +%Y%m%dT%H%M%S)
BASE_DIR="results/eval_sealqa/${TS}"
mkdir -p "${BASE_DIR}"

LOG="${BASE_DIR}/master.log"

log() { echo "$*" | tee -a "${LOG}"; }

maybe_limit() {
    if [[ -n "${LIMIT}" ]]; then echo "--num-samples ${LIMIT}"; fi
}

log "================================================================"
log "EvoSkill SealQA evaluation"
log "Model:      ${MODEL}"
log "vLLM API:   ${VLLM_API_BASE}"
log "Dataset:    ${DATASET}"
log "Phases:     ${PHASES}"
log "Mode:       ${MODE}"
log "Limit:      ${LIMIT:-none}"
log "Started:    $(date)"
log "Output:     ${BASE_DIR}"
log "================================================================"

# ── Phase 1: Baseline eval (no loop) ─────────────────────────────────────────
if [[ ",${PHASES}," == *",1,"* ]]; then
    log ""
    log "── [phase1_baseline] starting at $(date) ──"
    if "$PYTHON" scripts/run_eval_sealqa.py \
        --sdk vllm \
        --vllm-base-url "${VLLM_API_BASE}" \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --output "${BASE_DIR}/phase1_baseline.pkl" \
        --max-concurrent "${CONCURRENCY}" \
        --no-resume \
        $(maybe_limit) \
        2>&1 | tee "${BASE_DIR}/phase1_baseline.log"; then
        log "── [phase1_baseline] DONE at $(date) ──"
    else
        log "── [phase1_baseline] FAILED at $(date) (exit $?) ──"
    fi
fi

# ── Phase 2: Self-improving loop ──────────────────────────────────────────────
if [[ ",${PHASES}," == *",2,"* ]]; then
    log ""
    log "── [phase2_loop] starting at $(date) ──"
    if "$PYTHON" scripts/run_loop_sealqa.py \
        --vllm-base-url "${VLLM_API_BASE}" \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --mode "${MODE}" \
        --max-iterations "${MAX_ITERATIONS}" \
        --train-ratio "${TRAIN_RATIO}" \
        --val-ratio "${VAL_RATIO}" \
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
    if "$PYTHON" scripts/run_eval_sealqa.py \
        --sdk vllm \
        --vllm-base-url "${VLLM_API_BASE}" \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --output "${BASE_DIR}/phase3_post_loop.pkl" \
        --max-concurrent "${CONCURRENCY}" \
        --no-resume \
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

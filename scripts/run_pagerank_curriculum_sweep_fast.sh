#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${1:-0}"
PAGERANK_MASK_RATIOS="${PAGERANK_MASK_RATIOS:-0.10 0.15 0.20}"
PAGERANK_MASK_STEPS="${PAGERANK_MASK_STEPS:-5}"
PAGERANK_ITERS="${PAGERANK_ITERS:-20}"
PAGERANK_DAMPING="${PAGERANK_DAMPING:-0.85}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${ROOT_DIR}/logs/pagerank_curriculum_sweep_fast_$(date +%Y%m%d_%H%M%S)}"
SWEEP_SUMMARY="${SWEEP_LOG_DIR}/summary.log"

mkdir -p "${SWEEP_LOG_DIR}"
cd "${ROOT_DIR}" || exit 1

echo "PageRank curriculum fast sweep started: $(date)" > "${SWEEP_SUMMARY}"
echo "Device: ${DEVICE}" >> "${SWEEP_SUMMARY}"
echo "Python: ${PYTHON_BIN}" >> "${SWEEP_SUMMARY}"
echo "Ratios: ${PAGERANK_MASK_RATIOS}" >> "${SWEEP_SUMMARY}"
echo "Steps: ${PAGERANK_MASK_STEPS}" >> "${SWEEP_SUMMARY}"
echo "Iters: ${PAGERANK_ITERS}" >> "${SWEEP_SUMMARY}"
echo "Damping: ${PAGERANK_DAMPING}" >> "${SWEEP_SUMMARY}"
echo "Logs: ${SWEEP_LOG_DIR}" >> "${SWEEP_SUMMARY}"
echo "" >> "${SWEEP_SUMMARY}"

for ratio in ${PAGERANK_MASK_RATIOS}; do
  ratio_tag="${ratio/./p}"
  run_log_dir="${SWEEP_LOG_DIR}/ratio_${ratio_tag}"

  echo "RUN ratio=${ratio}"
  echo "LOG ${run_log_dir}"
  echo "RUN ratio=${ratio}" >> "${SWEEP_SUMMARY}"
  echo "LOG ${run_log_dir}" >> "${SWEEP_SUMMARY}"

  PAGERANK_MASK_RATIO="${ratio}" \
  PAGERANK_MASK_STEPS="${PAGERANK_MASK_STEPS}" \
  PAGERANK_ITERS="${PAGERANK_ITERS}" \
  PAGERANK_DAMPING="${PAGERANK_DAMPING}" \
  LOG_DIR="${run_log_dir}" \
    bash scripts/run_pagerank_curriculum_fast.sh "${DEVICE}"

  status=$?
  echo "STATUS ratio=${ratio}: ${status}" >> "${SWEEP_SUMMARY}"
  if [ -f "${run_log_dir}/summary.log" ]; then
    grep -E "OK   |FAIL |# final_|# early-stopping_|#Test_|real |user |sys " "${run_log_dir}/summary.log" >> "${SWEEP_SUMMARY}" || true
  fi
  echo "" >> "${SWEEP_SUMMARY}"
done

echo "PageRank curriculum fast sweep finished: $(date)" >> "${SWEEP_SUMMARY}"
echo "Summary: ${SWEEP_SUMMARY}"

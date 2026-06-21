#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${1:-0}"
PAGERANK_MASK_RATIO="${PAGERANK_MASK_RATIO:-0.20}"
PAGERANK_MASK_STEPS="${PAGERANK_MASK_STEPS:-5}"
PAGERANK_ITERS="${PAGERANK_ITERS:-20}"
PAGERANK_DAMPING="${PAGERANK_DAMPING:-0.85}"
PAGERANK_PRESERVE_POWER="${PAGERANK_PRESERVE_POWER:-0.5}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/pagerank_preserve_curriculum_large_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="${LOG_DIR}/summary.log"
TIME_BIN="$(command -v time || true)"

read -r -a LARGE_SEED_ARGS <<< "${LARGE_SEEDS:-0 1 2}"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}" || exit 1

run_task() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  local status=0

  echo "RUN  ${name}"
  echo "CMD  $*"
  echo "LOG  ${log_file}"

  {
    echo "START ${name}: $(date)"
    echo "COMMAND: $*"
    if [ -n "${TIME_BIN}" ]; then
      "${TIME_BIN}" -p "$@"
    else
      "$@"
    fi
    status=$?
    echo "END ${name}: $(date)"
    echo "STATUS ${status}"
  } > "${log_file}" 2>&1

  if [ "${status}" -eq 0 ]; then
    echo "OK   ${name}  ${log_file}" >> "${SUMMARY_FILE}"
  else
    echo "FAIL ${name}  ${log_file}" >> "${SUMMARY_FILE}"
  fi

  grep -E "# final_|# early-stopping_|#Test_|real |user |sys " "${log_file}" >> "${SUMMARY_FILE}" || true
  echo "" >> "${SUMMARY_FILE}"
}

common_args() {
  printf '%s\n' \
    --device "${DEVICE}" \
    --use_cfg \
    --mask_strategy pagerank_preserve_curriculum \
    --pagerank_mask_ratio "${PAGERANK_MASK_RATIO}" \
    --pagerank_mask_steps "${PAGERANK_MASK_STEPS}" \
    --pagerank_iters "${PAGERANK_ITERS}" \
    --pagerank_damping "${PAGERANK_DAMPING}" \
    --pagerank_preserve_power "${PAGERANK_PRESERVE_POWER}"
}

echo "PageRank preserve curriculum large run started: $(date)" > "${SUMMARY_FILE}"
echo "Device: ${DEVICE}" >> "${SUMMARY_FILE}"
echo "Python: ${PYTHON_BIN}" >> "${SUMMARY_FILE}"
echo "Large seeds: ${LARGE_SEED_ARGS[*]}" >> "${SUMMARY_FILE}"
echo "PageRank preserve mask ratio: ${PAGERANK_MASK_RATIO}" >> "${SUMMARY_FILE}"
echo "PageRank preserve mask steps: ${PAGERANK_MASK_STEPS}" >> "${SUMMARY_FILE}"
echo "PageRank preserve power: ${PAGERANK_PRESERVE_POWER}" >> "${SUMMARY_FILE}"
echo "PageRank iters: ${PAGERANK_ITERS}" >> "${SUMMARY_FILE}"
echo "PageRank damping: ${PAGERANK_DAMPING}" >> "${SUMMARY_FILE}"
echo "Datasets: NCI1 COLLAB REDDIT-BINARY" >> "${SUMMARY_FILE}"
echo "Logs: ${LOG_DIR}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

COMMON_ARGS=()
while IFS= read -r arg; do
  COMMON_ARGS+=("${arg}")
done < <(common_args)

run_task graph_NCI1_pagerank_preserve_curriculum \
  "${PYTHON_BIN}" main_graph.py --dataset NCI1 "${COMMON_ARGS[@]}" --seeds "${LARGE_SEED_ARGS[@]}"

run_task graph_COLLAB_pagerank_preserve_curriculum \
  "${PYTHON_BIN}" main_graph.py --dataset COLLAB "${COMMON_ARGS[@]}" --seeds "${LARGE_SEED_ARGS[@]}"

run_task graph_REDDIT_BINARY_pagerank_preserve_curriculum \
  "${PYTHON_BIN}" main_graph.py --dataset REDDIT-BINARY "${COMMON_ARGS[@]}" --seeds "${LARGE_SEED_ARGS[@]}"

echo "PageRank preserve curriculum large run finished: $(date)" >> "${SUMMARY_FILE}"
echo "Summary: ${SUMMARY_FILE}"

#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${1:-0}"
DEGREE_MASK_RATIO="${DEGREE_MASK_RATIO:-0.3}"
DEGREE_MASK_POWER="${DEGREE_MASK_POWER:-1.0}"
RUN_PPI="${RUN_PPI:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/degree_mixed_medium_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="${LOG_DIR}/summary.log"
TIME_BIN="$(command -v time || true)"

# Deadline-friendly default: one seed. Override with, for example:
#   SEEDS="0 1 2" bash scripts/run_degree_mixed_mask.sh 0
read -r -a SEED_ARGS <<< "${SEEDS:-0}"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}" || exit 1

run_task() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  local status=0

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
    --mask_strategy degree_mixed \
    --degree_mask_ratio "${DEGREE_MASK_RATIO}" \
    --degree_mask_power "${DEGREE_MASK_POWER}"
}

echo "Degree mixed medium run started: $(date)" > "${SUMMARY_FILE}"
echo "Device: ${DEVICE}" >> "${SUMMARY_FILE}"
echo "Python: ${PYTHON_BIN}" >> "${SUMMARY_FILE}"
echo "Seeds: ${SEED_ARGS[*]}" >> "${SUMMARY_FILE}"
echo "Degree mask ratio: ${DEGREE_MASK_RATIO}" >> "${SUMMARY_FILE}"
echo "Degree mask power: ${DEGREE_MASK_POWER}" >> "${SUMMARY_FILE}"
echo "Run PPI: ${RUN_PPI}" >> "${SUMMARY_FILE}"
echo "Logs: ${LOG_DIR}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

COMMON_ARGS=()
while IFS= read -r arg; do
  COMMON_ARGS+=("${arg}")
done < <(common_args)

run_task node_pubmed_degree_mixed \
  "${PYTHON_BIN}" main_transductive.py --dataset pubmed "${COMMON_ARGS[@]}" --seeds "${SEED_ARGS[@]}"

run_task graph_NCI1_degree_mixed \
  "${PYTHON_BIN}" main_graph.py --dataset NCI1 "${COMMON_ARGS[@]}" --pooling sum --seeds "${SEED_ARGS[@]}"

run_task graph_COLLAB_degree_mixed \
  "${PYTHON_BIN}" main_graph.py --dataset COLLAB "${COMMON_ARGS[@]}" --seeds "${SEED_ARGS[@]}"

run_task graph_REDDIT_BINARY_degree_mixed \
  "${PYTHON_BIN}" main_graph.py --dataset REDDIT-BINARY "${COMMON_ARGS[@]}" --seeds "${SEED_ARGS[@]}"

if [ "${RUN_PPI}" = "1" ]; then
  run_task node_ppi_degree_mixed \
    "${PYTHON_BIN}" main_inductive.py --dataset ppi "${COMMON_ARGS[@]}" --seeds "${SEED_ARGS[@]}"
fi

echo "Degree mixed medium run finished: $(date)" >> "${SUMMARY_FILE}"
echo "Summary: ${SUMMARY_FILE}"

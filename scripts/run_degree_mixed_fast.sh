#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${1:-0}"
DEGREE_MASK_RATIO="${DEGREE_MASK_RATIO:-0.3}"
DEGREE_MASK_POWER="${DEGREE_MASK_POWER:-1.0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/degree_mixed_fast_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="${LOG_DIR}/summary.log"
TIME_BIN="$(command -v time || true)"

# Fast set excludes NCI1, COLLAB, and REDDIT-BINARY.
read -r -a SMALL_SEED_ARGS <<< "${SMALL_SEEDS:-0 1 2 3 4 5}"
read -r -a MEDIUM_SEED_ARGS <<< "${MEDIUM_SEEDS:-0 1 2}"

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

echo "Degree mixed fast run started: $(date)" > "${SUMMARY_FILE}"
echo "Device: ${DEVICE}" >> "${SUMMARY_FILE}"
echo "Python: ${PYTHON_BIN}" >> "${SUMMARY_FILE}"
echo "Small seeds: ${SMALL_SEED_ARGS[*]}" >> "${SUMMARY_FILE}"
echo "Medium seeds: ${MEDIUM_SEED_ARGS[*]}" >> "${SUMMARY_FILE}"
echo "Degree mask ratio: ${DEGREE_MASK_RATIO}" >> "${SUMMARY_FILE}"
echo "Degree mask power: ${DEGREE_MASK_POWER}" >> "${SUMMARY_FILE}"
echo "Skipped: NCI1 COLLAB REDDIT-BINARY" >> "${SUMMARY_FILE}"
echo "Logs: ${LOG_DIR}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

COMMON_ARGS=()
while IFS= read -r arg; do
  COMMON_ARGS+=("${arg}")
done < <(common_args)

run_task node_citeseer_degree_mixed \
  "${PYTHON_BIN}" main_transductive.py --dataset citeseer "${COMMON_ARGS[@]}" --seeds "${SMALL_SEED_ARGS[@]}"

run_task node_cora_degree_mixed \
  "${PYTHON_BIN}" main_transductive.py --dataset cora "${COMMON_ARGS[@]}" --seeds "${SMALL_SEED_ARGS[@]}"

run_task node_pubmed_degree_mixed \
  "${PYTHON_BIN}" main_transductive.py --dataset pubmed "${COMMON_ARGS[@]}" --seeds "${MEDIUM_SEED_ARGS[@]}"

run_task graph_MUTAG_degree_mixed \
  "${PYTHON_BIN}" main_graph.py --dataset MUTAG "${COMMON_ARGS[@]}" --seeds "${SMALL_SEED_ARGS[@]}"

run_task graph_IMDB_BINARY_degree_mixed \
  "${PYTHON_BIN}" main_graph.py --dataset IMDB-BINARY "${COMMON_ARGS[@]}" --seeds "${SMALL_SEED_ARGS[@]}"

run_task graph_IMDB_MULTI_degree_mixed \
  "${PYTHON_BIN}" main_graph.py --dataset IMDB-MULTI "${COMMON_ARGS[@]}" --seeds "${SMALL_SEED_ARGS[@]}"

run_task graph_PROTEINS_degree_mixed \
  "${PYTHON_BIN}" main_graph.py --dataset PROTEINS "${COMMON_ARGS[@]}" --seeds "${SMALL_SEED_ARGS[@]}"

echo "Degree mixed fast run finished: $(date)" >> "${SUMMARY_FILE}"
echo "Summary: ${SUMMARY_FILE}"

#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/xsj/miniconda3/envs/graphmae/bin/python"
DEVICE="${1:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/overnight_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="${LOG_DIR}/summary.log"

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
    /usr/bin/time -p "$@"
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

echo "Overnight run started: $(date)" > "${SUMMARY_FILE}"
echo "Device: ${DEVICE}" >> "${SUMMARY_FILE}"
echo "Logs: ${LOG_DIR}" >> "${SUMMARY_FILE}"

run_task graph_MUTAG \
  "${PYTHON_BIN}" main_graph.py --dataset MUTAG --device "${DEVICE}" --use_cfg --seeds 0 1 2 3 4

run_task graph_IMDB_BINARY \
  "${PYTHON_BIN}" main_graph.py --dataset IMDB-BINARY --device "${DEVICE}" --use_cfg --seeds 0 1 2 3 4

run_task graph_IMDB_MULTI \
  "${PYTHON_BIN}" main_graph.py --dataset IMDB-MULTI --device "${DEVICE}" --use_cfg --seeds 0 1 2 3 4

run_task graph_PROTEINS \
  "${PYTHON_BIN}" main_graph.py --dataset PROTEINS --device "${DEVICE}" --use_cfg --seeds 0 1 2 3 4

run_task node_citeseer \
  "${PYTHON_BIN}" main_transductive.py --dataset citeseer --device "${DEVICE}" --use_cfg --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

run_task node_cora \
  "${PYTHON_BIN}" main_transductive.py --dataset cora --device "${DEVICE}" --use_cfg --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

echo "Overnight run finished: $(date)" >> "${SUMMARY_FILE}"
echo "Summary: ${SUMMARY_FILE}"

#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${1:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/remaining_3090_$(date +%Y%m%d_%H%M%S)}"
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

SEEDS_20=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
SEEDS_5=(0 1 2 3 4)

echo "Remaining 3090 run started: $(date)" > "${SUMMARY_FILE}"
echo "Device: ${DEVICE}" >> "${SUMMARY_FILE}"
echo "Python: ${PYTHON_BIN}" >> "${SUMMARY_FILE}"
echo "Logs: ${LOG_DIR}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

run_task node_pubmed \
  "${PYTHON_BIN}" main_transductive.py --dataset pubmed --device "${DEVICE}" --use_cfg --seeds "${SEEDS_20[@]}"

run_task node_ogbn_arxiv \
  "${PYTHON_BIN}" main_transductive.py --dataset ogbn-arxiv --device "${DEVICE}" --use_cfg --seeds "${SEEDS_20[@]}"

run_task node_ppi \
  "${PYTHON_BIN}" main_inductive.py --dataset ppi --device "${DEVICE}" --use_cfg --seeds "${SEEDS_5[@]}"

run_task node_reddit \
  "${PYTHON_BIN}" main_inductive.py --dataset reddit --device "${DEVICE}" --use_cfg --seeds "${SEEDS_5[@]}"

run_task graph_COLLAB \
  "${PYTHON_BIN}" main_graph.py --dataset COLLAB --device "${DEVICE}" --use_cfg --seeds "${SEEDS_5[@]}"

run_task graph_REDDIT_BINARY \
  "${PYTHON_BIN}" main_graph.py --dataset REDDIT-BINARY --device "${DEVICE}" --use_cfg --seeds "${SEEDS_5[@]}"

run_task graph_NCI1 \
  "${PYTHON_BIN}" main_graph.py --dataset NCI1 --device "${DEVICE}" --use_cfg --pooling sum --seeds "${SEEDS_5[@]}"

echo "Remaining 3090 run finished: $(date)" >> "${SUMMARY_FILE}"
echo "Summary: ${SUMMARY_FILE}"

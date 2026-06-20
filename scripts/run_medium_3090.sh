#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${1:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/medium_3090_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="${LOG_DIR}/summary.log"
TIME_BIN="$(command -v time || true)"

# Deadline-friendly default: one seed. Override with, for example:
#   SEEDS="0 1 2" bash scripts/run_medium_3090.sh 0
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

echo "Medium 3090 run started: $(date)" > "${SUMMARY_FILE}"
echo "Device: ${DEVICE}" >> "${SUMMARY_FILE}"
echo "Python: ${PYTHON_BIN}" >> "${SUMMARY_FILE}"
echo "Seeds: ${SEED_ARGS[*]}" >> "${SUMMARY_FILE}"
echo "Logs: ${LOG_DIR}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

run_task node_pubmed \
  "${PYTHON_BIN}" main_transductive.py --dataset pubmed --device "${DEVICE}" --use_cfg --seeds "${SEED_ARGS[@]}"

run_task graph_NCI1 \
  "${PYTHON_BIN}" main_graph.py --dataset NCI1 --device "${DEVICE}" --use_cfg --pooling sum --seeds "${SEED_ARGS[@]}"

run_task graph_COLLAB \
  "${PYTHON_BIN}" main_graph.py --dataset COLLAB --device "${DEVICE}" --use_cfg --seeds "${SEED_ARGS[@]}"

run_task graph_REDDIT_BINARY \
  "${PYTHON_BIN}" main_graph.py --dataset REDDIT-BINARY --device "${DEVICE}" --use_cfg --seeds "${SEED_ARGS[@]}"

echo "Medium 3090 run finished: $(date)" >> "${SUMMARY_FILE}"
echo "Summary: ${SUMMARY_FILE}"

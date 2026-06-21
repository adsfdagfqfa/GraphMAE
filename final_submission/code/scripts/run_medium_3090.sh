#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${1:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/medium_3090_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="${LOG_DIR}/summary.log"
TIME_BIN="$(command -v time || true)"
RUN_PPI="${RUN_PPI:-0}"

# Default: small datasets use 6 seeds, medium datasets use 3 seeds.
# Override with, for example:
#   SMALL_SEEDS="0 1 2" MEDIUM_SEEDS="0" bash scripts/run_medium_3090.sh 0
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

echo "Small/medium 3090 run started: $(date)" > "${SUMMARY_FILE}"
echo "Device: ${DEVICE}" >> "${SUMMARY_FILE}"
echo "Python: ${PYTHON_BIN}" >> "${SUMMARY_FILE}"
echo "Small seeds: ${SMALL_SEED_ARGS[*]}" >> "${SUMMARY_FILE}"
echo "Medium seeds: ${MEDIUM_SEED_ARGS[*]}" >> "${SUMMARY_FILE}"
echo "Run PPI: ${RUN_PPI}" >> "${SUMMARY_FILE}"
echo "Logs: ${LOG_DIR}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

run_task node_citeseer \
  "${PYTHON_BIN}" main_transductive.py --dataset citeseer --device "${DEVICE}" --use_cfg --seeds "${SMALL_SEED_ARGS[@]}"

run_task node_cora \
  "${PYTHON_BIN}" main_transductive.py --dataset cora --device "${DEVICE}" --use_cfg --seeds "${SMALL_SEED_ARGS[@]}"

run_task node_pubmed \
  "${PYTHON_BIN}" main_transductive.py --dataset pubmed --device "${DEVICE}" --use_cfg --seeds "${MEDIUM_SEED_ARGS[@]}"

run_task graph_MUTAG \
  "${PYTHON_BIN}" main_graph.py --dataset MUTAG --device "${DEVICE}" --use_cfg --seeds "${SMALL_SEED_ARGS[@]}"

run_task graph_IMDB_BINARY \
  "${PYTHON_BIN}" main_graph.py --dataset IMDB-BINARY --device "${DEVICE}" --use_cfg --seeds "${SMALL_SEED_ARGS[@]}"

run_task graph_IMDB_MULTI \
  "${PYTHON_BIN}" main_graph.py --dataset IMDB-MULTI --device "${DEVICE}" --use_cfg --seeds "${SMALL_SEED_ARGS[@]}"

run_task graph_PROTEINS \
  "${PYTHON_BIN}" main_graph.py --dataset PROTEINS --device "${DEVICE}" --use_cfg --seeds "${SMALL_SEED_ARGS[@]}"

run_task graph_NCI1 \
  "${PYTHON_BIN}" main_graph.py --dataset NCI1 --device "${DEVICE}" --use_cfg --pooling sum --seeds "${MEDIUM_SEED_ARGS[@]}"

run_task graph_COLLAB \
  "${PYTHON_BIN}" main_graph.py --dataset COLLAB --device "${DEVICE}" --use_cfg --seeds "${MEDIUM_SEED_ARGS[@]}"

run_task graph_REDDIT_BINARY \
  "${PYTHON_BIN}" main_graph.py --dataset REDDIT-BINARY --device "${DEVICE}" --use_cfg --seeds "${MEDIUM_SEED_ARGS[@]}"

if [ "${RUN_PPI}" = "1" ]; then
  run_task node_ppi \
    "${PYTHON_BIN}" main_inductive.py --dataset ppi --device "${DEVICE}" --use_cfg --seeds "${MEDIUM_SEED_ARGS[@]}"
fi

echo "Small/medium 3090 run finished: $(date)" >> "${SUMMARY_FILE}"
echo "Summary: ${SUMMARY_FILE}"

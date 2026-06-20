#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${1:-0}"
STRUCTMAE_BETA="${STRUCTMAE_BETA:-0.25}"
STRUCTMAE_SCORE="${STRUCTMAE_SCORE:-pagerank}"
STRUCTMAE_ITERS="${STRUCTMAE_ITERS:-20}"
STRUCTMAE_DAMPING="${STRUCTMAE_DAMPING:-0.85}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/structmae_fast_$(date +%Y%m%d_%H%M%S)}"
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
    --mask_strategy structmae \
    --structmae_beta "${STRUCTMAE_BETA}" \
    --structmae_score "${STRUCTMAE_SCORE}" \
    --structmae_iters "${STRUCTMAE_ITERS}" \
    --structmae_damping "${STRUCTMAE_DAMPING}"
}

echo "StructMAE-style fast run started: $(date)" > "${SUMMARY_FILE}"
echo "Device: ${DEVICE}" >> "${SUMMARY_FILE}"
echo "Python: ${PYTHON_BIN}" >> "${SUMMARY_FILE}"
echo "Small seeds: ${SMALL_SEED_ARGS[*]}" >> "${SUMMARY_FILE}"
echo "Medium seeds: ${MEDIUM_SEED_ARGS[*]}" >> "${SUMMARY_FILE}"
echo "StructMAE beta: ${STRUCTMAE_BETA}" >> "${SUMMARY_FILE}"
echo "StructMAE score: ${STRUCTMAE_SCORE}" >> "${SUMMARY_FILE}"
echo "StructMAE iters: ${STRUCTMAE_ITERS}" >> "${SUMMARY_FILE}"
echo "StructMAE damping: ${STRUCTMAE_DAMPING}" >> "${SUMMARY_FILE}"
echo "Skipped: NCI1 COLLAB REDDIT-BINARY" >> "${SUMMARY_FILE}"
echo "Logs: ${LOG_DIR}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

COMMON_ARGS=()
while IFS= read -r arg; do
  COMMON_ARGS+=("${arg}")
done < <(common_args)

run_task node_citeseer_structmae \
  "${PYTHON_BIN}" main_transductive.py --dataset citeseer "${COMMON_ARGS[@]}" --seeds "${SMALL_SEED_ARGS[@]}"

run_task node_cora_structmae \
  "${PYTHON_BIN}" main_transductive.py --dataset cora "${COMMON_ARGS[@]}" --seeds "${SMALL_SEED_ARGS[@]}"

run_task node_pubmed_structmae \
  "${PYTHON_BIN}" main_transductive.py --dataset pubmed "${COMMON_ARGS[@]}" --seeds "${MEDIUM_SEED_ARGS[@]}"

run_task graph_MUTAG_structmae \
  "${PYTHON_BIN}" main_graph.py --dataset MUTAG "${COMMON_ARGS[@]}" --seeds "${SMALL_SEED_ARGS[@]}"

run_task graph_IMDB_BINARY_structmae \
  "${PYTHON_BIN}" main_graph.py --dataset IMDB-BINARY "${COMMON_ARGS[@]}" --seeds "${SMALL_SEED_ARGS[@]}"

run_task graph_IMDB_MULTI_structmae \
  "${PYTHON_BIN}" main_graph.py --dataset IMDB-MULTI "${COMMON_ARGS[@]}" --seeds "${SMALL_SEED_ARGS[@]}"

run_task graph_PROTEINS_structmae \
  "${PYTHON_BIN}" main_graph.py --dataset PROTEINS "${COMMON_ARGS[@]}" --seeds "${SMALL_SEED_ARGS[@]}"

echo "StructMAE-style fast run finished: $(date)" >> "${SUMMARY_FILE}"
echo "Summary: ${SUMMARY_FILE}"

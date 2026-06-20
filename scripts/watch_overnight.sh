#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${1:-}"
PID_FILE="${2:-${ROOT_DIR}/logs/overnight_current.pid}"
INTERVAL="${INTERVAL:-60}"

if [ -z "${LOG_DIR}" ]; then
  LOG_DIR="$(ls -td "${ROOT_DIR}"/logs/overnight_* 2>/dev/null | head -1)"
fi

WATCH_LOG="${LOG_DIR}/watchdog.log"
ALERT_LOG="${LOG_DIR}/alerts.log"

mkdir -p "${LOG_DIR}"

echo "Watchdog started: $(date)" >> "${WATCH_LOG}"
echo "Log dir: ${LOG_DIR}" >> "${WATCH_LOG}"
echo "PID file: ${PID_FILE}" >> "${WATCH_LOG}"

while true; do
  now="$(date)"
  pid=""
  if [ -f "${PID_FILE}" ]; then
    pid="$(cat "${PID_FILE}")"
  fi

  if [ -n "${pid}" ] && ps -p "${pid}" >/dev/null 2>&1; then
    status="running"
  else
    status="stopped"
  fi

  current_task="$(pgrep -af 'main_graph.py|main_transductive.py|main_inductive.py' | grep -v pgrep | tail -1 || true)"
  summary_tail="$(tail -20 "${LOG_DIR}/summary.log" 2>/dev/null || true)"

  {
    echo "[$now] runner=${status} pid=${pid}"
    if [ -n "${current_task}" ]; then
      echo "current=${current_task}"
    fi
    echo "${summary_tail}" | tail -8
    echo ""
  } >> "${WATCH_LOG}"

  if grep -R -E "Traceback|CUDA out of memory|RuntimeError|DGLError|STATUS [1-9][0-9]*" "${LOG_DIR}"/*.log >/tmp/graphmae_watch_hits 2>/dev/null; then
    {
      echo "[$now] ALERT"
      cat /tmp/graphmae_watch_hits
      echo ""
    } >> "${ALERT_LOG}"
  fi

  if [ "${status}" = "stopped" ]; then
    echo "Watchdog stopped: $(date)" >> "${WATCH_LOG}"
    break
  fi

  sleep "${INTERVAL}"
done

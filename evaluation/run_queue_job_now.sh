#!/usr/bin/env bash
# Start one job of state/queue/jobs.txt now on a chosen GPU, outside the queue's own GPU choice, for instance
# next to a queue job that leaves half of a GPU unused. The job's state file is written as the queue writes it
# ("running ... pid=", then "done"/"failed"), so the queue does not start the job again and adopts it if the
# queue itself is restarted. VAR=value arguments replace the job's own settings of that variable (simple values).
#
#   setsid nohup run_queue_job_now.sh <job name> <gpu> [VAR=value ...] > /dev/null 2>&1 &
set -euo pipefail
ROOT="${HOME}/roboelo"
QDIR="${ROOT}/state/queue"
name="$1"; gpu="$2"; shift 2
if [[ -e "${QDIR}/${name}.state" ]]; then echo "${name} already has a state: $(cat "${QDIR}/${name}.state")"; exit 1; fi
cmd="$(awk -v n="${name}" -F'|' '{ k = $1; gsub(/ /, "", k) } k == n { sub(/^[^|]*\|[^|]*\|[^|]*\|/, ""); print; exit }' "${QDIR}/jobs.txt")"
if [[ -z "${cmd}" ]]; then echo "no job ${name} in ${QDIR}/jobs.txt"; exit 1; fi
for kv in "$@"; do
  var="${kv%%=*}"
  if [[ "${cmd}" =~ (^|[[:space:]])${var}= ]]; then
    cmd="$(sed -E "s#(^|[[:space:]])${var}=[^[:space:]]*#\\1${kv}#" <<<"${cmd}")"
  else
    cmd="${kv} ${cmd}"
  fi
done
log="${ROOT}/logs/queue/${name}_$(date +%Y%m%d_%H%M%S)_manual.log"
echo "running gpu=${gpu} pid=$$ since=$(date -Is) log=${log} started_by=run_queue_job_now.sh" > "${QDIR}/${name}.state"
echo "[$(date -Is)] ${name}: started on GPU ${gpu} outside the queue's choice (run_queue_job_now.sh $*), log ${log}" \
  >> "${ROOT}/logs/queue/queue.log"
set +e
( export GPU_FIXED="${gpu}"; exec bash -c "${cmd}" ) > "${log}" 2>&1 < /dev/null
rc=$?
if (( rc == 0 )); then
  echo "done at=$(date -Is) gpu=${gpu} started_by=run_queue_job_now.sh" > "${QDIR}/${name}.state"
else
  echo "failed rc=${rc} at=$(date -Is) gpu=${gpu} started_by=run_queue_job_now.sh" > "${QDIR}/${name}.state"
fi
echo "[$(date -Is)] ${name}: ended with rc=${rc} (run_queue_job_now.sh)" >> "${ROOT}/logs/queue/queue.log"
exit "${rc}"

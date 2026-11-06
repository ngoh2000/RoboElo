#!/usr/bin/env bash
# Pipeline test of one job of state/queue/jobs.txt on a few episodes, without touching the queue's state: the job's
# command runs with EPISODES=<subset>, OUT=tmp/smoke_<job>, STATUS=state/smoke_<job>.status (plus any VAR=value given).
# Afterwards, look at tmp/smoke_<job>/episodes.csv and create state/<job>_tested.done to let the queue run the job.
#
#   setsid nohup smoke_queue_job.sh <job name> <gpu> <episodes.jsonl> [VAR=value ...] > /dev/null 2>&1 &
set -euo pipefail
ROOT="${HOME}/roboelo"
name="$1"; gpu="$2"; episodes="$(realpath "$3")"; shift 3
cmd="$(awk -v n="${name}" -F'|' '{ k = $1; gsub(/ /, "", k) } k == n { sub(/^[^|]*\|[^|]*\|[^|]*\|/, ""); print; exit }' "${ROOT}/state/queue/jobs.txt")"
if [[ -z "${cmd}" ]]; then echo "no job ${name} in jobs.txt"; exit 1; fi
out="${ROOT}/tmp/smoke_${name}"
for kv in "$@"; do [[ "${kv}" == OUT=* ]] && out="${kv#OUT=}"; done
rm -rf "${out}"
for kv in "EPISODES=${episodes}" "OUT=${ROOT}/tmp/smoke_${name}" "STATUS=${ROOT}/state/smoke_${name}.status" "$@"; do
  var="${kv%%=*}"
  if [[ "${cmd}" =~ (^|[[:space:]])${var}= ]]; then
    cmd="$(sed -E "s#(^|[[:space:]])${var}=[^[:space:]]*#\\1${kv}#" <<<"${cmd}")"
  else
    cmd="${kv} ${cmd}"
  fi
done
log="${ROOT}/logs/$(basename "${out}").log"
echo "[$(date -Is)] ${name}: pipeline test on GPU ${gpu}, $(wc -l < "${episodes}") episodes" > "${log}"
set +e
( export GPU_FIXED="${gpu}"; exec bash -c "${cmd}" ) >> "${log}" 2>&1 < /dev/null
rc=$?
echo "[$(date -Is)] ${name}: pipeline test ended with rc=${rc}" >> "${log}"
echo "SMOKE_DONE rc=${rc}" >> "${log}"
exit "${rc}"

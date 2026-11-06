#!/usr/bin/env bash
# Runs the jobs of a list one after another on whichever GPU is free: one job per GPU at a time.
#
#   setsid nohup ~/roboelo/scripts/roboelo_gpu_queue.sh > ~/roboelo/logs/queue/queue.log 2>&1 &
#
# Job list: ~/roboelo/state/queue/jobs.txt, read again at every poll, so jobs can be added while the
# queue runs. One job per line, four fields separated by '|' (lines starting with # are ignored):
#
#   name | GPU memory needed (MiB) | files that must exist before it may start (space separated) | command
#
# The command is started with GPU_FIXED=<gpu index> in its environment and should use that GPU only.
#
# A GPU is free for a job when
#   (a) no job of this queue is running on it,
#   (b) no other process of this user is computing on it (a run that was started outside the queue),
#   (c) no process of another user that started less than OTHER_SETTLE_S seconds ago is computing on it:
#       people run series of short benchmarks whose memory use changes from one to the next, and a job
#       started in between would make theirs run out of memory. Next to a process that has been there
#       for a while, the job may run if the memory is there,
#   (d) it has the job's memory free,
# and all of that holds for STABLE_POLLS polls in a row. Jobs start in list order; a job whose files are
# not there yet is skipped until they are, so a model that is still downloading does not block the rest.
#
# state/queue/<name>.state says "running ...", "done ..." or "failed ...". A job with a state file is not
# started again: delete the file to run it again (the runners skip rollouts that are already recorded).
# If the queue itself is restarted, "running" jobs whose process is gone are put back in line.
# Create state/queue/STOP to make the queue exit once the jobs that are running have ended.
set -uo pipefail

ROOT="${HOME}/roboelo"
QDIR="${ROOT}/state/queue"
JOBS="${QDIR}/jobs.txt"
LOGS="${ROOT}/logs/queue"
POLL_S="${POLL_S:-60}"
STABLE_POLLS="${STABLE_POLLS:-6}"
OTHER_SETTLE_S="${OTHER_SETTLE_S:-1800}"
SMALL_OWN_MIB="${SMALL_OWN_MIB:-5000}"
mkdir -p "${QDIR}" "${LOGS}"
exec 8> "${QDIR}/queue.lock"
if ! flock -n 8; then echo "another queue is already running"; exit 1; fi

say() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
trim() { local v="$1"; v="${v#"${v%%[![:space:]]*}"}"; v="${v%"${v##*[![:space:]]}"}"; printf '%s' "${v}"; }
blockers_on() {  # compute processes on GPU $1 that keep the queue off it: this user's, and other users' recent ones.
  # This user's processes below SMALL_OWN_MIB do not count: the SAPIEN renderers of SimplerEnv runs whose policy
  # runs on the CPU (RT-1-X, Octo) hold a few hundred MiB at first and ~3 GB later.
  local uuid pid u mem owner age n=0
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1" | tr -d ' ')"
  while IFS=, read -r pid u mem; do
    pid="${pid// /}"; u="${u// /}"; mem="${mem//[!0-9]/}"
    if [[ "${u}" != "${uuid}" ]]; then continue; fi
    read -r owner age < <(ps -o uid=,etimes= -p "${pid}" 2>/dev/null) || continue
    if [[ "${owner}" == "$(id -u)" ]]; then
      if (( ${mem:-0} >= SMALL_OWN_MIB )); then n=$(( n + 1 )); fi
    elif (( age < OTHER_SETTLE_S )); then
      n=$(( n + 1 ))
    fi
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits)
  echo "${n}"
}

declare -A job_pid=() job_gpu=() stable=() adopted=()

# Jobs left "running" by an earlier queue: keep watching the ones that are alive, re-queue the others.
for f in "${QDIR}"/*.state; do
  [[ -e "${f}" ]] || continue
  name="$(basename "${f}" .state)"
  read -r word rest < "${f}" || true
  if [[ "${word}" != "running" ]]; then continue; fi
  pid="$(sed -n 's/.*pid=\([0-9]*\).*/\1/p' "${f}")"; gpu="$(sed -n 's/.*gpu=\([0-9]*\).*/\1/p' "${f}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    job_pid[${name}]="${pid}"; job_gpu[${name}]="${gpu}"; adopted[${name}]=1
    say "${name}: still running on GPU ${gpu} (pid ${pid}); watching it"
  else
    rm -f "${f}"; say "${name}: was marked running but its process is gone; back in line"
  fi
done

say "queue started; job list ${JOBS}"
polls=0
while :; do
  notes=""
  # ---- 1. jobs that have ended ----
  for name in "${!job_pid[@]}"; do
    pid="${job_pid[${name}]}"
    if kill -0 "${pid}" 2>/dev/null; then continue; fi
    if [[ -n "${adopted[${name}]:-}" ]]; then
      # not a child of this process, so its exit status is unknown: run it once more, which only
      # checks that every rollout is recorded
      rm -f "${QDIR}/${name}.state"; say "${name}: ended (started by an earlier queue); back in line for a check"
    else
      wait "${pid}"; rc=$?
      if (( rc == 0 )); then
        echo "done at=$(date -Is) gpu=${job_gpu[${name}]}" > "${QDIR}/${name}.state"; say "${name}: done"
      else
        echo "failed rc=${rc} at=$(date -Is) gpu=${job_gpu[${name}]}" > "${QDIR}/${name}.state"
        say "${name}: FAILED with exit status ${rc} (see ${LOGS}/${name}_*.log); not started again"
      fi
    fi
    unset "job_pid[${name}]" "job_gpu[${name}]" "adopted[${name}]"
  done

  # ---- 2. a job for every free GPU ----
  if [[ ! -e "${QDIR}/STOP" && -f "${JOBS}" ]]; then
    for gpu in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
      busy=0
      for name in "${!job_gpu[@]}"; do if [[ "${job_gpu[${name}]}" == "${gpu}" ]]; then busy=1; fi; done
      if (( busy )); then stable[${gpu}]=0; notes+=" GPU ${gpu}: a job of the queue;"; continue; fi
      if (( $(blockers_on "${gpu}") > 0 )); then
        stable[${gpu}]=0; notes+=" GPU ${gpu}: in use (own run outside the queue, or another user's recent process);"; continue
      fi
      free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
      pick="" pick_cmd=""
      while IFS='|' read -r name need files cmd; do
        name="$(trim "${name}")"; need="$(trim "${need}")"; cmd="$(trim "${cmd}")"
        if [[ -z "${name}" || "${name}" == \#* || -z "${cmd}" ]]; then continue; fi
        if [[ -e "${QDIR}/${name}.state" ]]; then continue; fi          # running, done or failed
        ready=1
        for file in ${files}; do if [[ ! -e "${file/#\~/${HOME}}" ]]; then ready=0; fi; done
        if (( ! ready )) || (( free < need )); then continue; fi
        pick="${name}"; pick_cmd="${cmd}"; break
      done < "${JOBS}"
      if [[ -z "${pick}" ]]; then stable[${gpu}]=0; notes+=" GPU ${gpu}: ${free} MiB free, no job that is ready fits;"; continue; fi
      stable[${gpu}]=$(( ${stable[${gpu}]:-0} + 1 ))
      if (( stable[${gpu}] < STABLE_POLLS )); then
        notes+=" GPU ${gpu}: free for ${pick}, poll ${stable[${gpu}]} of ${STABLE_POLLS};"; continue
      fi
      log="${LOGS}/${pick}_$(date +%Y%m%d_%H%M%S).log"
      ( export GPU_FIXED="${gpu}"; exec bash -c "${pick_cmd}" ) > "${log}" 2>&1 8>&- < /dev/null &
      job_pid[${pick}]=$!; job_gpu[${pick}]="${gpu}"; stable[${gpu}]=0
      echo "running gpu=${gpu} pid=${job_pid[${pick}]} since=$(date -Is) log=${log}" > "${QDIR}/${pick}.state"
      say "${pick}: started on GPU ${gpu} (${free} MiB free), pid ${job_pid[${pick}]}, log ${log}"
    done
  fi

  polls=$(( polls + 1 ))
  if (( polls % 10 == 1 )) && [[ -n "${notes}" ]]; then say "status:${notes}"; fi
  if [[ -e "${QDIR}/STOP" && ${#job_pid[@]} -eq 0 ]]; then say "STOP file found and nothing is running; exiting"; exit 0; fi
  sleep "${POLL_S}"
done

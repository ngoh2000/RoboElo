#!/usr/bin/env bash
# A policy on SimplerEnv's official episodes (state/simpler/episodes_official.jsonl), one CSV row per episode.
#
# Runs NUM_WORKERS processes of run_simpler_env.py on one GPU, worker i taking the episodes whose index % N == i,
# all appending to <OUT>/episodes.csv. A worker that dies is started again and continues where it stopped; the run
# stops after 3 restarts in a row without a new episode. Safe to re-run: recorded episodes are skipped.
# Started by roboelo_gpu_queue.sh with GPU_FIXED=<index>.
#
#   MODEL=rt1x CKPT_LABEL=rt_1_x_tf_trained_for_002272480_step PY=~/roboelo/envs/simpler-tf/bin/python \
#   FACTORY=rt1 POLICY_KWARGS='{"ckpt": "..."}' FAMILIES=google_robot,widowx NUM_WORKERS=4 \
#     ~/roboelo/scripts/run_simpler_supervisor.sh
set -Euo pipefail

ROOT="${HOME}/roboelo"
MODEL="${MODEL:?MODEL is not set}"
CKPT_LABEL="${CKPT_LABEL:?CKPT_LABEL is not set}"
PY="${PY:?PY (python of the model environment) is not set}"
FACTORY="${FACTORY:?FACTORY is not set (a function of scripts/simpler_policies.py, or <file>:<function>)}"
POLICY_KWARGS="${POLICY_KWARGS:-{\}}"
FAMILIES="${FAMILIES:-google_robot,widowx}"
PROTOCOLS="${PROTOCOLS:-visual_matching,variant_aggregation}"
NUM_WORKERS="${NUM_WORKERS:-2}"
# One model's episodes split over several jobs (one per GPU): this job's workers take shards SHARD_OFFSET ..
# SHARD_OFFSET+NUM_WORKERS-1 of SHARD_TOTAL (episode_index % SHARD_TOTAL); counts and the completion check cover only those.
SHARD_TOTAL="${SHARD_TOTAL:-${NUM_WORKERS}}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
GPU_FIXED="${GPU_FIXED:-0}"
SIMPLER_ROOT="${SIMPLER_ROOT:-${ROOT}/third_party/SimplerEnv}"   # cwd of the workers (relative asset paths)
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"                          # e.g. the fork's policies and model code
EPISODES="${EPISODES:-${ROOT}/state/simpler/episodes_official.jsonl}"
WORKER_ENV="${WORKER_ENV:-}"                                      # extra VAR=value pairs for the workers
MAX_BARREN_RESTARTS="${MAX_BARREN_RESTARTS:-3}"
PAUSE_S="${PAUSE_S:-600}"        # a worker that keeps dying without a new episode is paused, the others go on
MAX_PAUSES="${MAX_PAUSES:-12}"
WATCH_S="${WATCH_S:-30}"
RUN_NOTE="${RUN_NOTE:-}"
# Optional policy server on the same GPU, for models that cannot run inside the SimplerEnv Python (GR00T, LeRobot
# ports): started before the workers, restarted if it dies, stopped at the end. The workers' clients wait for it.
SERVER_CMD="${SERVER_CMD:-}"                                      # shell command, run with exec in SERVER_CWD
SERVER_CWD="${SERVER_CWD:-${ROOT}}"
SERVER_ENV="${SERVER_ENV:-}"                                      # extra VAR=value pairs for the server
SERVER_PORT="${SERVER_PORT:-}"
SERVER_WAIT_S="${SERVER_WAIT_S:-1800}"
OUT="${OUT:-${ROOT}/results/${MODEL}/simplerenv}"
STATUS="${STATUS:-${ROOT}/state/${MODEL}_simplerenv.status}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/logs/${MODEL}_simplerenv_${RUN_ID}"
[[ "${FACTORY}" == *:* ]] || FACTORY="${ROOT}/scripts/simpler_policies.py:${FACTORY}"
export HF_HOME="${ROOT}/hf_cache" HF_HUB_OFFLINE=1

mkdir -p "${LOGDIR}" "${OUT}"
say() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOGDIR}/supervisor.log"; }
status() { printf 'time=%s\nrun_id=%s\nstate=%s\nlog_dir=%s\nout=%s\n' "$(date -Is)" "${RUN_ID}" "$*" "${LOGDIR}" "${OUT}" > "${STATUS}"; }
json_str() { sed 's/["\\]/\\&/g' <<<"$1"; }
SHARDS="$(seq -s, "${SHARD_OFFSET}" $(( SHARD_OFFSET + NUM_WORKERS - 1 )))"
expected="$(python3 - "${EPISODES}" "${FAMILIES}" "${PROTOCOLS}" "${SHARD_TOTAL}" "${SHARDS}" <<'PY'
import json, sys
fam, prot = sys.argv[2].split(","), sys.argv[3].split(",")
n, shards = int(sys.argv[4]), {int(x) for x in sys.argv[5].split(",")}
print(sum(1 for l in open(sys.argv[1]) if (e := json.loads(l))["robot_family"] in fam and e["protocol"] in prot
          and e["episode_index"] % n in shards))
PY
)"
# Counts only this run's robot families, protocols and shards: several jobs (one per robot checkpoint, or one per GPU)
# may share OUT.
count() {
  if [[ -f "${OUT}/episodes.csv" ]]; then
    python3 - "${OUT}/episodes.csv" "${FAMILIES}" "${PROTOCOLS}" "$1" "${SHARD_TOTAL}" "${SHARDS}" <<'PY'
import csv, sys
fam, prot = sys.argv[2].split(","), sys.argv[3].split(",")
n, shards = int(sys.argv[5]), {int(x) for x in sys.argv[6].split(",")}
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["robot_family"] in fam and r["protocol"] in prot
        and int(r["episode_index"]) % n in shards]
print(len({r["episode_index"] for r in rows}) if sys.argv[4] == "rows" else sum(int(r["success"]) for r in rows))
PY
  else echo 0; fi
}
rows() { count rows; }
oks() { count oks; }

read -r -a worker_env <<<"${WORKER_ENV}"
worker_env+=(CUDA_VISIBLE_DEVICES="${GPU_FIXED}" PYTHONPATH="${EXTRA_PYTHONPATH}${EXTRA_PYTHONPATH:+:}${SIMPLER_ROOT}:${SIMPLER_ROOT}/ManiSkill2_real2sim"
             PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${WORKER_THREADS:-2}" DISPLAY=)
cat > "${OUT}/run_config_${RUN_ID}.json" <<JSON
{"run_id": "${RUN_ID}", "model": "${MODEL}", "checkpoint_label": "$(json_str "${CKPT_LABEL}")",
 "runner": "run_simpler_env.py: the loop of SimplerEnv's run_maniskill2_eval_single_episode over the official episode list",
 "episodes": "${EPISODES}", "families": "${FAMILIES}", "protocols": "${PROTOCOLS}", "expected_episodes": ${expected},
 "factory": "${FACTORY}", "policy_kwargs": "$(json_str "${POLICY_KWARGS}")", "workers": ${NUM_WORKERS}, "gpu": "${GPU_FIXED}",
 "shards": "${SHARDS} of ${SHARD_TOTAL}",
 "python": "${PY}", "simpler_root": "${SIMPLER_ROOT}", "simpler_commit": "$(git -C "${SIMPLER_ROOT}" rev-parse HEAD 2>/dev/null)",
 "extra_pythonpath": "${EXTRA_PYTHONPATH}", "policy_server": "$(json_str "${SERVER_CMD}")",
 "policy_server_cwd": "${SERVER_CWD}", "note": "$(json_str "${RUN_NOTE}")"}
JSON

server_pid=""
start_server() {
  [[ -n "${SERVER_CMD}" ]] || return 0
  local senv
  read -r -a senv <<<"${SERVER_ENV}"
  printf '\n===== server start %s =====\n' "$(date -Is)" >> "${LOGDIR}/server.log"
  ( cd "${SERVER_CWD}" && exec env "${senv[@]}" CUDA_VISIBLE_DEVICES="${GPU_FIXED}" PYTHONFAULTHANDLER=1 \
      TOKENIZERS_PARALLELISM=false bash -c "exec ${SERVER_CMD}" ) >> "${LOGDIR}/server.log" 2>&1 &
  server_pid=$!
}
server_up() { [[ -z "${SERVER_CMD}" ]] || { kill -0 "${server_pid}" 2>/dev/null && (echo > "/dev/tcp/127.0.0.1/${SERVER_PORT}") 2>/dev/null; }; }
wait_for_server() {
  [[ -n "${SERVER_CMD}" ]] || return 0
  local t=0
  while (( t < SERVER_WAIT_S )); do
    kill -0 "${server_pid}" 2>/dev/null || return 1
    server_up && return 0
    sleep 5; t=$(( t + 5 ))
  done
  return 1
}
stop_server() { if [[ -n "${server_pid}" ]]; then kill "${server_pid}" 2>/dev/null || true; fi; }

pids=() restarts=() barren=() rows_at_start=() finished=() paused_until=() pauses=()
start_worker() {
  local i="$1"
  printf '\n===== worker %s start %s =====\n' "${i}" "$(date -Is)" >> "${LOGDIR}/worker_${i}.log"
  rows_at_start[i]="$(rows)"
  ( cd "${SIMPLER_ROOT}" && exec env "${worker_env[@]}" "${PY}" "${ROOT}/scripts/run_simpler_env.py" \
      --episodes "${EPISODES}" --factory "${FACTORY}" --policy-kwargs "${POLICY_KWARGS}" --families "${FAMILIES}" \
      --protocols "${PROTOCOLS}" --out "${OUT}" --model "${MODEL}" --checkpoint "${CKPT_LABEL}" \
      --shard "$(( SHARD_OFFSET + i ))/${SHARD_TOTAL}" ) >> "${LOGDIR}/worker_${i}.log" 2>&1 &
  pids[i]=$!
}
stop_all() { local pid; for pid in "${pids[@]}"; do kill "${pid}" $(pgrep -P "${pid}" || true) 2>/dev/null || true; done; }
trap 'stop_all; stop_server' EXIT
trap 'exit 143' TERM INT HUP

say "GPU ${GPU_FIXED}; ${NUM_WORKERS} worker(s); ${expected} episodes expected"
status "starting gpu=${GPU_FIXED} workers=${NUM_WORKERS}"
if [[ -n "${SERVER_CMD}" ]]; then
  start_server
  say "policy server pid ${server_pid} (port ${SERVER_PORT}); waiting until it accepts connections"
  if ! wait_for_server; then status "server_not_ready"; say "the policy server did not come up; see ${LOGDIR}/server.log"; exit 5; fi
  say "policy server is up"
fi
for ((i = 0; i < NUM_WORKERS; i++)); do restarts[i]=0; barren[i]=0; finished[i]=0; start_worker "${i}"; done
gave_up=0 server_restarts=0 rows_at_server_start="$(rows)"
while :; do
  if [[ -n "${SERVER_CMD}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
    if [[ "$(rows)" == "${rows_at_server_start}" ]]; then server_restarts=$(( server_restarts + 1 )); else server_restarts=1; fi
    if (( server_restarts > MAX_BARREN_RESTARTS )); then say "policy server died ${server_restarts} times in a row without a new episode; giving up"; gave_up=1; break; fi
    say "policy server died (see ${LOGDIR}/server.log); restarting"
    rows_at_server_start="$(rows)"
    start_server
    wait_for_server || say "policy server not ready yet"
  fi
  n_finished=0
  for ((i = 0; i < NUM_WORKERS; i++)); do
    if (( finished[i] )); then n_finished=$(( n_finished + 1 )); continue; fi
    if (( ${paused_until[i]:-0} > 0 )); then  # waiting out a run of crashes (e.g. Vulkan device lost when the GPU is full)
      if (( $(date +%s) < paused_until[i] )); then continue; fi
      paused_until[i]=0; say "worker ${i}: pause over, starting again"; start_worker "${i}"; continue
    fi
    if kill -0 "${pids[i]}" 2>/dev/null; then continue; fi
    if wait "${pids[i]}"; then finished[i]=1; n_finished=$(( n_finished + 1 )); say "worker ${i} is complete"; continue; fi
    restarts[i]=$(( restarts[i] + 1 ))
    if [[ "$(rows)" == "${rows_at_start[i]}" ]]; then barren[i]=$(( barren[i] + 1 )); else barren[i]=0; fi
    if (( barren[i] >= MAX_BARREN_RESTARTS )); then
      pauses[i]=$(( ${pauses[i]:-0} + 1 )); barren[i]=0
      if (( pauses[i] > MAX_PAUSES )); then say "worker ${i} kept dying without a new episode after ${MAX_PAUSES} pauses; giving up"; gave_up=1; break; fi
      paused_until[i]=$(( $(date +%s) + PAUSE_S ))
      say "worker ${i} died ${MAX_BARREN_RESTARTS} times in a row without a new episode; pausing it ${PAUSE_S} s (pause ${pauses[i]} of ${MAX_PAUSES})"
      continue
    fi
    say "worker ${i} died (see ${LOGDIR}/worker_${i}.log); restart ${restarts[i]}"
    start_worker "${i}"
  done
  if (( gave_up )) || (( n_finished == NUM_WORKERS )); then break; fi
  status "running gpu=${GPU_FIXED} workers=${NUM_WORKERS} episodes_recorded=$(rows) successes=$(oks) expected=${expected}"
  sleep "${WATCH_S}"
done
stop_all
n="$(rows)"
if (( gave_up )) || (( n < expected )); then
  status "finished_with_missing_episodes episodes_recorded=${n} expected=${expected}"; say "stopped with episodes missing"; exit 1
fi
status "finished_ok episodes_recorded=${n} successes=$(oks)"
say "complete: ${n} episodes, $(oks) successes"

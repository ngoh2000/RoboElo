#!/usr/bin/env bash
# GR00T N1.7 on LIBERO: 4 suites x 10 tasks x 50 fixed initial states = 2,000 rollouts, one CSV row each.
#
# Waits until one GPU has enough free memory. Then, suite by suite (each suite has its own checkpoint),
# starts GR00T's policy server on that GPU and runs several clients against it, each on its own tasks.
# A server that dies is started again; the clients wait for it and redo the interrupted rollouts.
# Safe to re-run: finished rollouts are skipped.
#
#   setsid nohup ~/roboelo/scripts/run_groot_libero_supervisor.sh > ~/roboelo/logs/groot_libero_nohup.log 2>&1 &
#
# Pipeline test without a GPU (slow CPU inference, two policy calls):
#   DEVICE=cpu SUITES=libero_spatial NUM_TRIALS=1 TASK_IDS=0 NUM_CLIENTS=1 MAX_EPISODE_STEPS=16 PORT=18566 \
#     OUT=~/roboelo/tmp/groot_smoke STATUS=~/roboelo/tmp/groot_smoke.status ~/roboelo/scripts/run_groot_libero_supervisor.sh
set -Euo pipefail

ROOT="${HOME}/roboelo"
PORT="${PORT:-18565}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"       # free memory required on one GPU before starting
POLL_S="${POLL_S:-60}"
STABLE_POLLS="${STABLE_POLLS:-3}"           # the GPU must look free this many polls in a row
MAX_WAIT_H="${MAX_WAIT_H:-336}"
MAX_SERVER_RESTARTS="${MAX_SERVER_RESTARTS:-500}"  # per suite
MAX_BARREN_RESTARTS="${MAX_BARREN_RESTARTS:-3}"    # stop if the server dies this often in a row without a new rollout
WATCH_S="${WATCH_S:-30}"
NUM_TRIALS="${NUM_TRIALS:-50}"
NUM_CLIENTS="${NUM_CLIENTS:-5}"             # GR00T's LIBERO instructions use 5 parallel envs
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"      # as in GR00T's LIBERO instructions
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"              # as in GR00T's LIBERO instructions
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
DEVICE="${DEVICE:-cuda}"                    # "cpu" = pipeline test without a GPU
SERVER_GDB="${SERVER_GDB:-1}"               # 1 = run the server under gdb: a crash leaves a native backtrace in server.log
MIN_SUCCESS_AFTER="${MIN_SUCCESS_AFTER:-20}"   # sanity gate: after this many rollouts ...
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-50}"     # ... stop if fewer than this percent succeeded
OUT="${OUT:-${ROOT}/results/groot_n17/libero_50}"
STATUS="${STATUS:-${ROOT}/state/groot_libero.status}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/logs/groot_libero_${RUN_ID}"
GROOT="${ROOT}/third_party/Isaac-GR00T"
SERVER_PY="${GROOT}/.venv/bin/python"
CLIENT_PY="${GROOT}/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
CKPT="${ROOT}/checkpoints/GR00T-N1.7-LIBERO"
export HF_HOME="${ROOT}/hf_cache"

mkdir -p "${LOGDIR}" "${OUT}" "${ROOT}/state"
say() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOGDIR}/supervisor.log"; }
status() { printf 'time=%s\nrun_id=%s\nstate=%s\nlog_dir=%s\nout=%s\n' "$(date -Is)" "${RUN_ID}" "$*" "${LOGDIR}" "${OUT}" > "${STATUS}"; }
rows_in() {  # recorded rollouts of the given suites
  local s n=0 c
  for s in "$@"; do c="$(grep -vc '^model,' "${OUT}/${s}/episodes.csv" 2>/dev/null || true)"; n=$(( n + ${c:-0} )); done
  echo "${n}"
}
ok_in() {  # successful rollouts of the given suites (column 10)
  local s n=0 c
  for s in "$@"; do
    c="$(awk -F, 'NR>1 && $1!="model" {s+=$10} END {print s+0}' "${OUT}/${s}/episodes.csv" 2>/dev/null || true)"
    n=$(( n + ${c:-0} ))
  done
  echo "${n}"
}
test -x "${CLIENT_PY}" || { status "client_env_missing"; say "run scripts/setup_groot_libero_env.sh first"; exit 2; }

# ---- 0. the gated backbone nvidia/Cosmos-Reason2-2B must be in the local cache ------------------------
BACKBONE_DONE="${ROOT}/state/model_cosmos_reason2_2b.done"
if [[ ! -f "${BACKBONE_DONE}" ]]; then
  status "waiting_for_backbone"
  say "waiting for the backbone nvidia/Cosmos-Reason2-2B (${BACKBONE_DONE})"
  until [[ -f "${BACKBONE_DONE}" ]]; do sleep "${POLL_S}"; done
fi
# The checkpoints name the backbone as "nvidia/Cosmos-Reason2-2B". transformers asks the Hub about that
# name even when the files are cached, which fails offline. A folder of that name under the directory
# the server is started from makes transformers treat it as a local path and never go online.
snapshot="$(ls -d "${HF_HOME}"/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/* | head -1)"
mkdir -p "${GROOT}/nvidia"
ln -sfn "${snapshot}" "${GROOT}/nvidia/Cosmos-Reason2-2B"

# ---- 1. wait for a GPU (one supervisor at a time picks a GPU and loads its model) -------------------
GPU=0
exec 9> "${ROOT}/state/gpu_select.lock"
if [[ "${DEVICE}" != "cpu" ]]; then
  status "waiting_for_gpu min_free_mib=${MIN_FREE_MIB}"
  say "waiting for a GPU with >= ${MIN_FREE_MIB} MiB free (${STABLE_POLLS} polls of ${POLL_S}s in a row)"
  stable=0
  deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
  while :; do
    flock 9
    best="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)"
    gpu_now="$(cut -d, -f1 <<<"${best}" | tr -d ' ')"
    free_now="$(cut -d, -f2 <<<"${best}" | tr -d ' ')"
    if (( free_now >= MIN_FREE_MIB )); then
      stable=$(( stable + 1 ))
      if (( stable >= STABLE_POLLS )); then GPU="${gpu_now}"; break; fi   # keeps the lock
    else
      stable=0
    fi
    flock -u 9
    if (( $(date +%s) > deadline )); then status "gave_up_waiting_for_gpu"; say "no free GPU after ${MAX_WAIT_H} h"; exit 3; fi
    sleep "${POLL_S}"
  done
  say "GPU ${GPU} has ${free_now} MiB free; starting"
fi

# ---- 2. server and clients -------------------------------------------------------------------------
cd "${GROOT}"
if [[ "${DEVICE}" == "cpu" ]]; then
  # FlashAttention cannot run on a CPU: hide it for the pipeline test so the model falls back to sdpa attention.
  mkdir -p "${ROOT}/pydeps/no_flash_attn/flash_attn"
  echo 'raise ImportError("flash_attn is hidden for the CPU pipeline test")' > "${ROOT}/pydeps/no_flash_attn/flash_attn/__init__.py"
  server_env=(CUDA_VISIBLE_DEVICES= PYTHONPATH="${ROOT}/pydeps/no_flash_attn")
else
  server_env=(CUDA_VISIBLE_DEVICES="${GPU}")
fi
server_env+=(PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1)   # everything is local: no token needed
client_env=(MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="${EGL_GPU:-${GPU}}"
            LIBERO_CONFIG_PATH="${ROOT}/libero_config_groot" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=
            OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false)
SERVER_PID=""
pids=()
start_server() {
  local suite="$1"
  local cmd=("${SERVER_PY}" gr00t/eval/run_gr00t_server.py --model-path "${CKPT}/${suite}" --embodiment-tag LIBERO_PANDA
             --use-sim-policy-wrapper --device "${DEVICE}" --host 127.0.0.1 --port "${PORT}")
  if [[ "${SERVER_GDB}" == "1" ]]; then
    # Symbols are read only after a crash, so start-up stays fast.
    cmd=(gdb -q -batch -ex "set pagination off" -ex "set print thread-events off" -ex "set auto-solib-add off"
         -ex run -ex "echo ===== NATIVE BACKTRACE =====" -ex 'x/6i $pc' -ex "info registers"
         -ex sharedlibrary -ex "bt 80" -ex "info sharedlibrary" --args "${cmd[@]}")
  fi
  printf '\n===== server start %s (%s) =====\n' "$(date -Is)" "${suite}" >> "${LOGDIR}/server.log"
  env "${server_env[@]}" "${cmd[@]}" >> "${LOGDIR}/server.log" 2>&1 9>&- &
  SERVER_PID=$!
}
stop_server() {  # under gdb the Python process is a child of SERVER_PID
  local kids
  if [[ -z "${SERVER_PID}" ]]; then return; fi
  kids="$(pgrep -P "${SERVER_PID}" || true)"
  # shellcheck disable=SC2086
  kill "${SERVER_PID}" ${kids} 2>/dev/null || true
  for _ in $(seq 1 30); do if ! kill -0 "${SERVER_PID}" 2>/dev/null; then break; fi; sleep 1; done
  # shellcheck disable=SC2086
  kill -9 "${SERVER_PID}" ${kids} 2>/dev/null || true
  SERVER_PID=""
}
stop_clients() {
  local pid
  for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
  pids=()
}
wait_for_port() {  # up to 30 minutes
  for _ in $(seq 1 360); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then return 1; fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then return 0; fi
    sleep 5
  done
  return 1
}
trap 'stop_clients; stop_server' EXIT
trap 'exit 143' TERM INT HUP

read -r -a task_ids <<<"${TASK_IDS}"
expected_per_suite=$(( ${#task_ids[@]} * NUM_TRIALS ))
cat > "${OUT}/run_config_${RUN_ID}.json" <<JSON
{"run_id": "${RUN_ID}", "model": "GR00T N1.7", "checkpoint": "${CKPT}/<suite>", "suites": "${SUITES}",
 "num_trials_per_task": ${NUM_TRIALS}, "initial_states": "LIBERO fixed initial states 0..N-1",
 "num_steps_wait": ${NUM_STEPS_WAIT}, "env_seed": 7, "policy_seed": "none (as in GR00T's instructions)",
 "max_episode_steps": ${MAX_EPISODE_STEPS}, "n_action_steps": ${N_ACTION_STEPS}, "clients_per_suite": ${NUM_CLIENTS},
 "device": "${DEVICE}", "gpu": "${GPU}", "server_under_gdb": "${SERVER_GDB}",
 "gr00t_commit": "$(git -C "${GROOT}" rev-parse HEAD 2>/dev/null)",
 "libero_commit": "$(git -C "${GROOT}/external_dependencies/LIBERO" rev-parse HEAD 2>/dev/null)"}
JSON

gave_up=0
total_restarts=0
for suite in ${SUITES}; do
  if (( $(rows_in "${suite}") >= expected_per_suite )); then say "${suite}: already complete"; flock -u 9; continue; fi
  status "starting_server gpu=${GPU} suite=${suite}"
  start_server "${suite}"
  say "${suite}: policy server pid ${SERVER_PID}, port ${PORT}; waiting until it accepts connections"
  if ! wait_for_port; then
    flock -u 9
    status "server_not_ready suite=${suite}"; say "${suite}: the server did not come up; see ${LOGDIR}/server.log"; exit 5
  fi
  flock -u 9   # the model is in memory: other supervisors may now look at the GPUs
  say "${suite}: server is up"

  restarts=0 barren=0 n_at_restart=-1
  for pass in 1 2 3; do
    # each client takes every NUM_CLIENTS-th task
    pids=()
    for c in $(seq 0 $(( NUM_CLIENTS - 1 ))); do
      mine=()
      for j in "${!task_ids[@]}"; do if (( j % NUM_CLIENTS == c )); then mine+=("${task_ids[j]}"); fi; done
      if (( ${#mine[@]} == 0 )); then continue; fi
      env "${client_env[@]}" "${CLIENT_PY}" "${ROOT}/scripts/run_groot_libero_client.py" --suite "${suite}" \
        --out "${OUT}" --port "${PORT}" --num-trials "${NUM_TRIALS}" --task-ids "${mine[@]}" \
        --max-episode-steps "${MAX_EPISODE_STEPS}" --n-action-steps "${N_ACTION_STEPS}" \
        --num-steps-wait "${NUM_STEPS_WAIT}" >> "${LOGDIR}/client_${suite}_${c}.log" 2>&1 9>&- &
      pids+=("$!")
    done

    while :; do
      alive=0
      for pid in "${pids[@]}"; do if kill -0 "${pid}" 2>/dev/null; then alive=1; fi; done
      if (( ! alive )); then break; fi
      n_done="$(rows_in "${suite}")"
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        restarts=$(( restarts + 1 )); total_restarts=$(( total_restarts + 1 ))
        if (( n_done == n_at_restart )); then barren=$(( barren + 1 )); else barren=0; fi
        n_at_restart="${n_done}"
        if (( restarts > MAX_SERVER_RESTARTS || barren >= MAX_BARREN_RESTARTS )); then
          say "${suite}: policy server died ${restarts} times (${barren} in a row without a new rollout); giving up, re-run to resume"
          gave_up=1; break
        fi
        say "${suite}: policy server died with ${n_done} rollouts recorded (see ${LOGDIR}/server.log); restart ${restarts}"
        start_server "${suite}"
      fi
      # shellcheck disable=SC2086
      n_all="$(rows_in ${SUITES})"; ok_all="$(ok_in ${SUITES})"
      if (( n_all >= MIN_SUCCESS_AFTER && ok_all * 100 < n_all * MIN_SUCCESS_RATE )); then
        say "only ${ok_all} of ${n_all} rollouts succeeded: something is wrong with the set-up; stopping"
        status "stopped_low_success rollouts_recorded=${n_all} successes=${ok_all}"
        exit 6
      fi
      status "running gpu=${GPU} suite=${suite} server_restarts=${total_restarts} rollouts_recorded=${n_all} successes=${ok_all}"
      sleep "${WATCH_S}"
    done
    stop_clients
    if (( gave_up )) || (( $(rows_in "${suite}") >= expected_per_suite )); then break; fi
    say "${suite}: the clients ended with rollouts missing (pass ${pass}); see ${LOGDIR}/client_${suite}_*.log"
  done
  stop_server
  n_done="$(rows_in "${suite}")"
  say "${suite}: ${n_done} of ${expected_per_suite} rollouts recorded"
  if (( gave_up )); then break; fi
done

# ---- 3. wrap up ------------------------------------------------------------------------------------
"${CLIENT_PY}" "${ROOT}/scripts/roboelo_libero_common.py" "${OUT}" --num-trials "${NUM_TRIALS}" | tee -a "${LOGDIR}/supervisor.log"
# shellcheck disable=SC2086
n_all="$(rows_in ${SUITES})"
n_suites="$(wc -w <<<"${SUITES}")"
if (( gave_up )) || (( n_all < n_suites * expected_per_suite )); then
  status "finished_with_missing_rollouts rollouts_recorded=${n_all}"; say "stopped with rollouts missing; re-run to resume"; exit 1
fi
status "finished_ok rollouts_recorded=${n_all}"
say "all suites complete"

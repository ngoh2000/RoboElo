#!/usr/bin/env bash
# A LeRobot policy checkpoint on LIBERO: 10 tasks x 50 fixed initial states per suite, one CSV row per rollout.
#
# Runs NUM_WORKERS processes of run_lerobot_libero.py on one GPU, each with its own copy of the model
# and its own share of the suites. A worker that dies is started again and continues where it stopped.
# Safe to re-run: finished rollouts are skipped.
#
#   MODEL=smolvla CKPT=~/roboelo/checkpoints/smolvla-libero NUM_WORKERS=4 MIN_FREE_MIB=3000 \
#     setsid nohup ~/roboelo/scripts/run_lerobot_libero_supervisor.sh > ~/roboelo/logs/smolvla_libero_nohup.log 2>&1 &
#
# Started by roboelo_gpu_queue.sh with GPU_FIXED=<index>; without it the script waits for a GPU itself.
#
# Pipeline test without a GPU (slow CPU inference, a few policy calls):
#   MODEL=x CKPT=... DEVICE=cpu SUITE_GROUPS=libero_spatial NUM_TRIALS=1 TASK_IDS=0 EPISODE_LENGTH=10 WORKER_GDB=0 \
#     OUT=~/roboelo/tmp/x_smoke STATUS=~/roboelo/tmp/x_smoke.status ~/roboelo/scripts/run_lerobot_libero_supervisor.sh
set -Euo pipefail

ROOT="${HOME}/roboelo"
MODEL="${MODEL:?MODEL is not set (name of the results folder and label in the CSV)}"
CKPT="${CKPT:?CKPT is not set (checkpoint folder)}"
CKPT_LABEL="${CKPT_LABEL:-$(basename "${CKPT}")}"
POLICY_ARGS="${POLICY_ARGS:-}"              # extra lerobot-eval flags, space separated (no spaces inside one flag)
READY_FILES="${READY_FILES:-}"              # files that must exist before starting (download markers)
MIN_FREE_MIB="${MIN_FREE_MIB:-9000}"        # GPU memory one worker needs
NUM_WORKERS="${NUM_WORKERS:-2}"
SUITE_GROUPS="${SUITE_GROUPS:-}"            # one group of suites per worker, groups separated by ';'
GPU_FIXED="${GPU_FIXED:-}"                  # set by the GPU queue: use this GPU and do not wait for one
POLL_S="${POLL_S:-60}"
STABLE_POLLS="${STABLE_POLLS:-3}"           # the GPU must look free this many polls in a row
MAX_WAIT_H="${MAX_WAIT_H:-336}"
MAX_RESTARTS="${MAX_RESTARTS:-500}"         # restart a dead worker at most this many times
MAX_BARREN_RESTARTS="${MAX_BARREN_RESTARTS:-3}"   # ... and stop if it dies this often in a row without a new rollout
WATCH_S="${WATCH_S:-30}"
NUM_TRIALS="${NUM_TRIALS:-50}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"      # LeRobot's LIBERO default
EPISODE_LENGTHS="${EPISODE_LENGTHS:-}"      # per-suite horizon, e.g. "libero_spatial=300,libero_10=600"
RUN_NOTE="${RUN_NOTE:-}"                    # free text for run_config: provenance, differences from the official set-up
# Another runner with the same ROBOELO_* interface (e.g. run_openvla_libero.py) gets only POLICY_ARGS as its
# command line instead of the lerobot-eval flags.
WORKER_SCRIPT="${WORKER_SCRIPT:-${HOME}/roboelo/scripts/run_lerobot_libero.py}"
RUNNER_DESC="${RUNNER_DESC:-run_lerobot_libero.py (LeRobot policy, processors, LIBERO env and rollout loop)}"
CODE_REPO="${CODE_REPO:-${HOME}/roboelo/third_party/lerobot}"   # its commit goes into run_config
LIBERO_CONFIG="${LIBERO_CONFIG:-${HOME}/roboelo/libero_config_lerobot}"
ENV_SEED_DESC="${ENV_SEED_DESC:-seed + initial state id}"
EXTRA_WORKER_ENV="${EXTRA_WORKER_ENV:-}"     # more VAR=value pairs for the workers, e.g. ROBOELO_LOAD_WEIGHTS_ON_CPU=1
CKPT_CONFIG="${CKPT_CONFIG:-config.json}"   # file in CKPT that must exist and is copied next to the results
# Optional policy server shared by the workers (models whose own LIBERO client talks to a server):
SERVER_CMD="${SERVER_CMD:-}"                # shell command, run with exec in SERVER_CWD on the job's GPU
SERVER_CWD="${SERVER_CWD:-${ROOT:-${HOME}/roboelo}}"
SERVER_ENV="${SERVER_ENV:-}"                # extra VAR=value pairs for the server
SERVER_PORT="${SERVER_PORT:-}"
SERVER_WAIT_S="${SERVER_WAIT_S:-1800}"
SEED="${SEED:-1000}"                        # lerobot-eval's default
TASK_IDS="${TASK_IDS:-}"                    # debug only: these tasks of every suite
EPISODE_LENGTH="${EPISODE_LENGTH:-}"        # debug only: shorter horizon
DEVICE="${DEVICE:-cuda}"                    # "cpu" = pipeline test without a GPU
WORKER_GDB="${WORKER_GDB:-1}"               # 1 = run workers under gdb: a crash leaves a native backtrace in the log
MIN_SUCCESS_AFTER="${MIN_SUCCESS_AFTER:-40}"   # sanity gate: after this many rollouts ...
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-30}"     # ... stop if fewer than this percent succeeded
OUT="${OUT:-${ROOT}/results/${MODEL}/libero_50}"
STATUS="${STATUS:-${ROOT}/state/${MODEL}_libero.status}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/logs/${MODEL}_libero_${RUN_ID}"
# Python environment. envs/lerobot-mujoco332 is roboelo-lerobot with MuJoCo 3.3.2 instead of 3.8.1: with
# MuJoCo 3.4 and later some of LIBERO's stored initial states settle into a different scene (libero_spatial
# task 5: the black bowl no longer rests on the ramekin).
PY="${PY:-${ROOT}/envs/lerobot-mujoco332/bin/python}"
export HF_HOME="${ROOT}/hf_cache"

mkdir -p "${LOGDIR}" "${OUT}" "${ROOT}/state"
say() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOGDIR}/supervisor.log"; }
status() { printf 'time=%s\nrun_id=%s\nstate=%s\nlog_dir=%s\nout=%s\n' "$(date -Is)" "${RUN_ID}" "$*" "${LOGDIR}" "${OUT}" > "${STATUS}"; }
count_spec() {  # $1 = rows | ok | expected, $2 = suite spec ("libero_goal" or "libero_10:0-4")
  local suite="${2%%:*}" ids="" file
  if [[ "$2" == *:* ]]; then ids="${2#*:}"; fi
  file="${OUT}/${suite}/episodes.csv"
  if [[ "$1" == "expected" ]]; then file=/dev/null; elif [[ ! -f "${file}" ]]; then echo 0; return; fi
  awk -F, -v ids="${ids}" -v what="$1" -v all_tasks="${n_tasks_default}" -v trials="${NUM_TRIALS}" '
    BEGIN {
      n = split(ids, parts, "+")
      for (i = 1; i <= n; i++) {
        m = split(parts[i], ab, "-"); hi = (m > 1 ? ab[2] : ab[1])
        for (t = ab[1] + 0; t <= hi + 0; t++) { want[t] = 1; n_want++ }
      }
      if (what == "expected") { print (ids == "" ? all_tasks : n_want) * trials; exit }
    }
    FNR > 1 && $1 != "model" && (ids == "" || ($5 in want)) { rows++; ok += $10 }
    END { if (what == "rows") print rows + 0; if (what == "ok") print ok + 0 }' "${file}"
}
count_group() {  # $1 = rows | ok | expected, then suite specs
  local what="$1" spec n=0 c; shift
  for spec in "$@"; do c="$(count_spec "${what}" "${spec}")"; n=$(( n + ${c:-0} )); done
  echo "${n}"
}

# ---- 0. the checkpoint must be there ---------------------------------------------------------------
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
missing_files() { local f; for f in ${READY_FILES} "${CKPT}/${CKPT_CONFIG}"; do if [[ ! -e "${f}" ]]; then echo "${f}"; return; fi; done; }
if [[ -n "$(missing_files)" ]]; then
  status "waiting_for_files"
  say "waiting for $(missing_files)"
  until [[ -z "$(missing_files)" ]]; do
    if (( $(date +%s) > deadline )); then status "gave_up_waiting_for_files"; exit 3; fi
    sleep "${POLL_S}"
  done
fi
export HF_HUB_OFFLINE=1   # everything is local from here on

# ---- 1. GPU (assigned by the queue, or wait for one) -------------------------------------------------
GPU=0
exec 9> "${ROOT}/state/gpu_select.lock"
if [[ "${DEVICE}" != "cpu" && -n "${GPU_FIXED}" ]]; then
  GPU="${GPU_FIXED}"
  say "GPU ${GPU} was assigned by the queue; ${NUM_WORKERS} worker(s)"
elif [[ "${DEVICE}" != "cpu" ]]; then
  need=$(( MIN_FREE_MIB * NUM_WORKERS ))
  status "waiting_for_gpu min_free_mib=${need}"
  say "waiting for a GPU with >= ${need} MiB free (${STABLE_POLLS} polls of ${POLL_S}s in a row)"
  stable=0
  while :; do
    flock 9
    best="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)"
    gpu_now="$(cut -d, -f1 <<<"${best}" | tr -d ' ')"
    free_now="$(cut -d, -f2 <<<"${best}" | tr -d ' ')"
    if (( free_now >= need )); then
      stable=$(( stable + 1 ))
      if (( stable >= STABLE_POLLS )); then GPU="${gpu_now}"; break; fi   # keeps the lock
    else
      stable=0
    fi
    flock -u 9
    if (( $(date +%s) > deadline )); then status "gave_up_waiting_for_gpu"; say "no free GPU after ${MAX_WAIT_H} h"; exit 3; fi
    sleep "${POLL_S}"
  done
  say "GPU ${GPU} has ${free_now} MiB free; starting ${NUM_WORKERS} worker(s)"
fi
if [[ -z "${SUITE_GROUPS}" ]]; then
  # libero_10 rollouts are about 2.5 times as long as the others, so its tasks are spread over the workers
  case "${NUM_WORKERS}" in
    1) SUITE_GROUPS="libero_spatial libero_object libero_goal libero_10" ;;
    2) SUITE_GROUPS="libero_spatial libero_object libero_10:0-1;libero_goal libero_10:2-9" ;;
    3) SUITE_GROUPS="libero_spatial libero_object;libero_goal libero_10:0-3;libero_10:4-9" ;;
    4) SUITE_GROUPS="libero_spatial libero_10:0-1;libero_object libero_10:2;libero_goal libero_10:3-4;libero_10:5-9" ;;
    *) echo "set SUITE_GROUPS for NUM_WORKERS=${NUM_WORKERS}"; exit 2 ;;
  esac
fi
IFS=';' read -r -a groups <<<"${SUITE_GROUPS}"
all_specs="$(tr ';' ' ' <<<"${SUITE_GROUPS}")"
n_tasks_default=10; if [[ -n "${TASK_IDS}" ]]; then n_tasks_default="$(wc -w <<<"${TASK_IDS//,/ }")"; fi

# ---- 2. workers ------------------------------------------------------------------------------------
if [[ "$(basename "${WORKER_SCRIPT}")" == "run_lerobot_libero.py" ]]; then
  worker_cmd=("${PY}" "${WORKER_SCRIPT}" --policy.path="${CKPT}" --env.type=libero
              --env.task=libero_spatial --eval.batch_size=1 --eval.n_episodes="${NUM_TRIALS}" --seed="${SEED}"
              --policy.device="${DEVICE}")
else
  worker_cmd=("${PY}" "${WORKER_SCRIPT}")
fi
read -r -a extra_args <<<"${POLICY_ARGS}"
worker_cmd+=("${extra_args[@]}")
if [[ "${DEVICE}" == "cpu" ]]; then
  worker_env=(CUDA_VISIBLE_DEVICES= MUJOCO_EGL_DEVICE_ID="${EGL_GPU:-0}"); THREADS=16
else
  worker_env=(CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}"); THREADS="${WORKER_THREADS:-1}"
fi
if [[ -n "${EPISODE_LENGTH}" ]]; then worker_cmd+=(--env.episode_length="${EPISODE_LENGTH}"); fi
if [[ "${WORKER_GDB}" == "1" ]]; then
  # Symbols are read only after a crash, so start-up stays fast.
  worker_cmd=(gdb -q -batch -ex "set pagination off" -ex "set print thread-events off" -ex "set auto-solib-add off"
              -ex run -ex "echo ===== NATIVE BACKTRACE =====" -ex 'x/6i $pc' -ex "info registers"
              -ex sharedlibrary -ex "bt 80" -ex "info sharedlibrary" --args "${worker_cmd[@]}")
fi
worker_env+=(LIBERO_CONFIG_PATH="${LIBERO_CONFIG}"
             MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" TOKENIZERS_PARALLELISM=false
             TQDM_DISABLE=1 PYTHONFAULTHANDLER=1 ROBOELO_OUT="${OUT}" ROBOELO_NUM_TRIALS="${NUM_TRIALS}"
             ROBOELO_NUM_STEPS_WAIT="${NUM_STEPS_WAIT}" ROBOELO_EPISODE_LENGTHS="${EPISODE_LENGTHS}"
             ROBOELO_TASK_IDS="${TASK_IDS}" ROBOELO_MODEL="${MODEL}"
             ROBOELO_CHECKPOINT="${CKPT_LABEL}")
read -r -a extra_worker_env <<<"${EXTRA_WORKER_ENV}"
worker_env+=("${extra_worker_env[@]}")

pids=() restarts=() barren=() rows_at_start=() finished=()
start_worker() {
  local i="$1"
  printf '\n===== worker %s start %s =====\n' "${i}" "$(date -Is)" >> "${LOGDIR}/worker_${i}.log"
  # shellcheck disable=SC2086
  rows_at_start[i]="$(count_group rows ${groups[i]})"
  ( cd "${LOGDIR}" && exec env "${worker_env[@]}" ROBOELO_SUITES="${groups[i]}" "${worker_cmd[@]}" ) \
    >> "${LOGDIR}/worker_${i}.log" 2>&1 9>&- &
  pids[i]=$!
}
stop_all() {
  local pid kids
  for pid in "${pids[@]}"; do
    kids="$(pgrep -P "${pid}" || true)"
    # shellcheck disable=SC2086
    kill "${pid}" ${kids} 2>/dev/null || true
  done
}
server_pid=""
start_server() {
  [[ -n "${SERVER_CMD}" ]] || return 0
  local senv
  read -r -a senv <<<"${SERVER_ENV}"
  printf '\n===== server start %s =====\n' "$(date -Is)" >> "${LOGDIR}/server.log"
  ( cd "${SERVER_CWD}" && exec env "${senv[@]}" CUDA_VISIBLE_DEVICES="${GPU}" PYTHONFAULTHANDLER=1 \
      TOKENIZERS_PARALLELISM=false bash -c "exec ${SERVER_CMD}" ) >> "${LOGDIR}/server.log" 2>&1 9>&- &
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
trap 'stop_all; stop_server' EXIT
trap 'exit 143' TERM INT HUP

cp -f "${CKPT}/${CKPT_CONFIG}" "${OUT}/policy_${CKPT_CONFIG}"
cat > "${OUT}/run_config_${RUN_ID}.json" <<JSON
{"run_id": "${RUN_ID}", "model": "${MODEL}", "checkpoint": "${CKPT}", "checkpoint_label": "${CKPT_LABEL}",
 "runner": "$(sed 's/["\\]/\\&/g' <<<"${RUNNER_DESC}")", "worker_script": "${WORKER_SCRIPT}",
 "extra_lerobot_eval_flags": "$(sed 's/["\\]/\\&/g' <<<"${POLICY_ARGS}")", "extra_worker_env": "${EXTRA_WORKER_ENV}", "suite_groups": "${SUITE_GROUPS}", "num_trials_per_task": ${NUM_TRIALS},
 "initial_states": "LIBERO fixed initial states 0..N-1", "seed": ${SEED}, "env_seed": "${ENV_SEED_DESC}",
 "num_steps_wait": ${NUM_STEPS_WAIT}, "episode_lengths": "${EPISODE_LENGTHS}",
 "max_steps": "episode_lengths per suite if set, else --env.episode_length in the extra flags if set, else LeRobot defaults (spatial 280, object 280, goal 300, libero_10 520)",
 "note": "$(sed 's/["\\]/\\&/g' <<<"${RUN_NOTE}")",
 "policy_settings": "as stored in the checkpoint (policy_config.json) unless overridden by the extra flags",
 "device": "${DEVICE}", "gpu": "${GPU}", "workers": ${#groups[@]}, "under_gdb": "${WORKER_GDB}",
 "python": "${PY}", "mujoco": "$("${PY}" -c 'import mujoco; print(mujoco.__version__)' 2>/dev/null)",
 "libero_config": "${LIBERO_CONFIG}", "code_repo": "${CODE_REPO}",
 "policy_server": "$(sed 's/["\\]/\\&/g' <<<"${SERVER_CMD}")", "policy_server_cwd": "${SERVER_CWD}",
 "code_commit": "$(git -C "${CODE_REPO}" rev-parse HEAD 2>/dev/null)",
 "lerobot_commit": "$(git -C "${ROOT}/third_party/lerobot" rev-parse HEAD 2>/dev/null)"}
JSON

status "starting gpu=${GPU} workers=${#groups[@]}"
if [[ -n "${SERVER_CMD}" ]]; then
  start_server
  say "policy server pid ${server_pid} (port ${SERVER_PORT}); waiting until it accepts connections"
  if ! wait_for_server; then status "server_not_ready"; say "the policy server did not come up; see ${LOGDIR}/server.log"; exit 5; fi
  say "policy server is up"
fi
for i in "${!groups[@]}"; do restarts[i]=0; barren[i]=0; finished[i]=0; start_worker "${i}"; done
server_restarts=0 rows_at_server_start="$(count_group rows ${all_specs})"
if [[ "${DEVICE}" != "cpu" && -z "${GPU_FIXED}" ]]; then
  # keep the GPU-selection lock until the models are in memory (or 10 minutes at most)
  for _ in $(seq 1 60); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
    if (( used >= MIN_FREE_MIB * ${#groups[@]} * 6 / 10 )); then break; fi
    sleep 10
  done
fi
flock -u 9

gave_up=0
while :; do
  if [[ -n "${SERVER_CMD}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
    # shellcheck disable=SC2086
    n_now="$(count_group rows ${all_specs})"
    if [[ "${n_now}" == "${rows_at_server_start}" ]]; then server_restarts=$(( server_restarts + 1 )); else server_restarts=1; fi
    if (( server_restarts > MAX_BARREN_RESTARTS )); then say "policy server died ${server_restarts} times in a row without a new rollout; giving up"; gave_up=1; break; fi
    say "policy server died (see ${LOGDIR}/server.log); restarting"
    rows_at_server_start="${n_now}"
    start_server
    wait_for_server || say "policy server not ready yet"
  fi
  n_finished=0
  for i in "${!groups[@]}"; do
    if (( finished[i] )); then n_finished=$(( n_finished + 1 )); continue; fi
    if kill -0 "${pids[i]}" 2>/dev/null; then continue; fi
    # shellcheck disable=SC2086
    n_now="$(count_group rows ${groups[i]})"; n_expected="$(count_group expected ${groups[i]})"
    if (( n_now >= n_expected )); then
      finished[i]=1; n_finished=$(( n_finished + 1 )); say "worker ${i} (${groups[i]}) is complete"; continue
    fi
    restarts[i]=$(( restarts[i] + 1 ))
    if (( n_now == rows_at_start[i] )); then barren[i]=$(( barren[i] + 1 )); else barren[i]=0; fi
    if (( restarts[i] > MAX_RESTARTS || barren[i] >= MAX_BARREN_RESTARTS )); then
      say "worker ${i} died ${restarts[i]} times (${barren[i]} in a row without a new rollout); giving up, re-run to resume"
      gave_up=1; break
    fi
    say "worker ${i} died with ${n_now} of ${n_expected} rollouts recorded (see ${LOGDIR}/worker_${i}.log); restart ${restarts[i]}"
    start_worker "${i}"
  done
  if (( gave_up )) || (( n_finished == ${#groups[@]} )); then break; fi
  # shellcheck disable=SC2086
  n_all="$(count_group rows ${all_specs})"; ok_all="$(count_group ok ${all_specs})"
  if (( n_all >= MIN_SUCCESS_AFTER && ok_all * 100 < n_all * MIN_SUCCESS_RATE )); then
    say "only ${ok_all} of ${n_all} rollouts succeeded: something is wrong with the set-up; stopping"
    status "stopped_low_success rollouts_recorded=${n_all} successes=${ok_all}"
    stop_all; exit 6
  fi
  status "running gpu=${GPU} workers=${#groups[@]} restarts=$(IFS=+; echo "${restarts[*]}") rollouts_recorded=${n_all} successes=${ok_all}"
  sleep "${WATCH_S}"
done

# ---- 3. wrap up ------------------------------------------------------------------------------------
stop_all
"${PY}" "${ROOT}/scripts/roboelo_libero_common.py" "${OUT}" --num-trials "${NUM_TRIALS}" | tee -a "${LOGDIR}/supervisor.log"
# shellcheck disable=SC2086
n_all="$(count_group rows ${all_specs})"; n_expected="$(count_group expected ${all_specs})"
if (( gave_up )) || (( n_all < n_expected )); then
  status "finished_with_missing_rollouts rollouts_recorded=${n_all} expected=${n_expected}"
  say "stopped with rollouts missing; re-run to resume"; exit 1
fi
status "finished_ok rollouts_recorded=${n_all}"
say "all suites complete (${n_all} rollouts)"

#!/usr/bin/env bash
# MolmoAct2 (LeRobot checkpoint) on LIBERO: 4 suites x 10 tasks x 50 fixed initial states = 2,000 rollouts,
# one CSV row each.
#
# Waits until the base checkpoint is downloaded and one GPU has enough free memory, then runs one or two
# worker processes on that GPU (each holds its own copy of the model, about 12 GiB). A worker that dies
# is started again and continues where it stopped. Safe to re-run: finished rollouts are skipped.
#
#   setsid nohup ~/roboelo/scripts/run_molmoact2_libero_supervisor.sh > ~/roboelo/logs/molmoact2_libero_nohup.log 2>&1 &
#
# Pipeline test without a GPU (very slow CPU inference, one policy call):
#   DEVICE=cpu SUITE_GROUPS="libero_spatial" NUM_TRIALS=1 TASK_IDS=0 EPISODE_LENGTH=10 OUT=~/roboelo/tmp/molmoact2_smoke \
#     STATUS=~/roboelo/tmp/molmoact2_smoke.status ~/roboelo/scripts/run_molmoact2_libero_supervisor.sh
set -Euo pipefail

ROOT="${HOME}/roboelo"
MIN_FREE_MIB="${MIN_FREE_MIB:-15000}"       # free GPU memory needed per worker
NUM_WORKERS="${NUM_WORKERS:-}"              # empty = 2 if the GPU has room for two, else 1
POLL_S="${POLL_S:-60}"
STABLE_POLLS="${STABLE_POLLS:-3}"           # the GPU must look free this many polls in a row
MAX_WAIT_H="${MAX_WAIT_H:-336}"
MAX_RESTARTS="${MAX_RESTARTS:-500}"         # restart a dead worker at most this many times
MAX_BARREN_RESTARTS="${MAX_BARREN_RESTARTS:-3}"   # ... and stop if it dies this often in a row without a new rollout
WATCH_S="${WATCH_S:-30}"
NUM_TRIALS="${NUM_TRIALS:-50}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-50}"      # the MolmoAct2 authors report all LIBERO results with 50
SEED="${SEED:-1000}"                        # as in the MolmoAct2 LIBERO instructions
SUITE_GROUPS="${SUITE_GROUPS:-}"            # one group of suites per worker, groups separated by ';'
TASK_IDS="${TASK_IDS:-}"
EPISODE_LENGTH="${EPISODE_LENGTH:-}"        # debug only: shorter horizon
DEVICE="${DEVICE:-cuda}"                    # "cpu" = pipeline test without a GPU
WORKER_GDB="${WORKER_GDB:-1}"               # 1 = run workers under gdb: a crash leaves a native backtrace in the log
MIN_SUCCESS_AFTER="${MIN_SUCCESS_AFTER:-20}"   # sanity gate: after this many rollouts ...
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-50}"     # ... stop if fewer than this percent succeeded
OUT="${OUT:-${ROOT}/results/molmoact2/libero_50}"
STATUS="${STATUS:-${ROOT}/state/molmoact2_libero.status}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/logs/molmoact2_libero_${RUN_ID}"
PY="${ROOT}/micromamba/envs/roboelo-lerobot/bin/python"
CKPT_SRC="${ROOT}/checkpoints/MolmoAct2-LIBERO-LeRobot"
CKPT="${ROOT}/checkpoints/MolmoAct2-LIBERO-LeRobot-cfgfix"
BASE_DONE="${ROOT}/state/model_molmoact2_libero_base.done"
export HF_HOME="${ROOT}/hf_cache"

mkdir -p "${LOGDIR}" "${OUT}" "${ROOT}/state"

# The released config.json stores null for scheduler_decay_steps (learning-rate schedule, training only),
# which the pinned LeRobot commit declares as an int, so the config does not load. Use a copy of the
# checkpoint folder (links to the same files) whose config.json has that commit's default instead.
mkdir -p "${CKPT}"
for f in "${CKPT_SRC}"/*; do
  if [[ "$(basename "${f}")" != "config.json" ]]; then ln -sfn "${f}" "${CKPT}/$(basename "${f}")"; fi
done
python3 - "${CKPT_SRC}/config.json" "${CKPT}/config.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
assert cfg["scheduler_decay_steps"] is None, cfg["scheduler_decay_steps"]
cfg["scheduler_decay_steps"] = 100000
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
PY
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

# ---- 0. the base checkpoint must be there ----------------------------------------------------------
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
if [[ ! -f "${BASE_DONE}" ]]; then
  status "waiting_for_base_checkpoint"
  say "waiting for the base checkpoint allenai/MolmoAct2-LIBERO (${BASE_DONE})"
  until [[ -f "${BASE_DONE}" ]]; do
    if (( $(date +%s) > deadline )); then status "gave_up_waiting_for_base_checkpoint"; exit 3; fi
    sleep "${POLL_S}"
  done
fi
export HF_HUB_OFFLINE=1   # everything is local from here on

# ---- 1. wait for a GPU (one supervisor at a time picks a GPU and loads its model) -------------------
GPU=0
exec 9> "${ROOT}/state/gpu_select.lock"
if [[ "${DEVICE}" != "cpu" ]]; then
  status "waiting_for_gpu min_free_mib=${MIN_FREE_MIB}"
  say "waiting for a GPU with >= ${MIN_FREE_MIB} MiB free (${STABLE_POLLS} polls of ${POLL_S}s in a row)"
  stable=0
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
  if [[ -z "${NUM_WORKERS}" ]]; then
    if (( free_now >= 2 * MIN_FREE_MIB )); then NUM_WORKERS=2; else NUM_WORKERS=1; fi
  fi
  say "GPU ${GPU} has ${free_now} MiB free; starting ${NUM_WORKERS} worker(s)"
fi
NUM_WORKERS="${NUM_WORKERS:-1}"
if [[ -z "${SUITE_GROUPS}" ]]; then
  # libero_10 rollouts are about 2.5 times as long as the others
  if (( NUM_WORKERS >= 2 )); then SUITE_GROUPS="libero_spatial libero_object libero_goal;libero_10"
  else SUITE_GROUPS="libero_spatial libero_object libero_goal libero_10"; fi
fi
IFS=';' read -r -a groups <<<"${SUITE_GROUPS}"
all_suites="$(tr ';' ' ' <<<"${SUITE_GROUPS}")"
n_tasks=10; if [[ -n "${TASK_IDS}" ]]; then n_tasks="$(wc -w <<<"${TASK_IDS//,/ }")"; fi
expected_per_suite=$(( n_tasks * NUM_TRIALS ))

# ---- 2. workers ------------------------------------------------------------------------------------
worker_cmd=("${PY}" "${ROOT}/scripts/run_lerobot_libero.py"
  --policy.path="${CKPT}" --policy.inference_action_mode=continuous --policy.per_episode_seed=true
  --policy.eval_seed="${SEED}" --env.type=libero --env.task=libero_spatial
  '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'
  --eval.batch_size=1 --eval.n_episodes="${NUM_TRIALS}" --seed="${SEED}")
if [[ "${DEVICE}" == "cpu" ]]; then
  worker_cmd+=(--policy.device=cpu --policy.model_dtype=float32 --policy.use_amp=false
               --policy.enable_inference_cuda_graph=false)
  worker_env=(CUDA_VISIBLE_DEVICES= MUJOCO_EGL_DEVICE_ID="${EGL_GPU:-0}")
else
  worker_cmd+=(--policy.device=cuda --policy.model_dtype=bfloat16 --policy.use_amp=true
               --policy.enable_inference_cuda_graph=true)
  worker_env=(CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}")
fi
THREADS=1; if [[ "${DEVICE}" == "cpu" ]]; then THREADS=16; fi   # one thread per process, as in the MolmoAct2 instructions
if [[ -n "${EPISODE_LENGTH}" ]]; then worker_cmd+=(--env.episode_length="${EPISODE_LENGTH}"); fi
if [[ "${WORKER_GDB}" == "1" ]]; then
  # Symbols are read only after a crash, so start-up stays fast.
  worker_cmd=(gdb -q -batch -ex "set pagination off" -ex "set print thread-events off" -ex "set auto-solib-add off"
              -ex run -ex "echo ===== NATIVE BACKTRACE =====" -ex 'x/6i $pc' -ex "info registers"
              -ex sharedlibrary -ex "bt 80" -ex "info sharedlibrary" --args "${worker_cmd[@]}")
fi
worker_env+=(LIBERO_CONFIG_PATH="${ROOT}/libero_config_lerobot"
             MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" TOKENIZERS_PARALLELISM=false
             TQDM_DISABLE=1 PYTHONFAULTHANDLER=1 ROBOELO_OUT="${OUT}" ROBOELO_NUM_TRIALS="${NUM_TRIALS}"
             ROBOELO_NUM_STEPS_WAIT="${NUM_STEPS_WAIT}" ROBOELO_TASK_IDS="${TASK_IDS}" ROBOELO_MODEL=molmoact2
             ROBOELO_CHECKPOINT=MolmoAct2-LIBERO-LeRobot)

pids=() restarts=() barren=() rows_at_start=() finished=()
start_worker() {
  local i="$1"
  printf '\n===== worker %s start %s =====\n' "${i}" "$(date -Is)" >> "${LOGDIR}/worker_${i}.log"
  # shellcheck disable=SC2086
  rows_at_start[i]="$(rows_in ${groups[i]})"
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
trap 'stop_all' EXIT
trap 'exit 143' TERM INT HUP

cat > "${OUT}/run_config_${RUN_ID}.json" <<JSON
{"run_id": "${RUN_ID}", "model": "MolmoAct2 (LeRobot policy)", "checkpoint": "${CKPT}",
 "base_checkpoint": "allenai/MolmoAct2-LIBERO", "suite_groups": "${SUITE_GROUPS}", "num_trials_per_task": ${NUM_TRIALS},
 "initial_states": "LIBERO fixed initial states 0..N-1", "seed": ${SEED}, "env_seed": "seed + initial state id",
 "num_steps_wait": ${NUM_STEPS_WAIT}, "max_steps": "LeRobot defaults (spatial 280, object 280, goal 300, libero_10 520)",
 "inference": "continuous, bfloat16, amp, cuda graph, per-episode seed", "device": "${DEVICE}", "gpu": "${GPU}",
 "workers": ${NUM_WORKERS}, "under_gdb": "${WORKER_GDB}",
 "lerobot_commit": "$(git -C "${ROOT}/third_party/lerobot" rev-parse HEAD 2>/dev/null)"}
JSON

status "starting gpu=${GPU} workers=${NUM_WORKERS}"
for i in "${!groups[@]}"; do restarts[i]=0; barren[i]=0; finished[i]=0; start_worker "${i}"; done
if [[ "${DEVICE}" != "cpu" ]]; then
  # keep the GPU-selection lock until the models are in memory (or 10 minutes at most)
  for _ in $(seq 1 60); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
    if (( used >= 9000 * ${#groups[@]} )); then break; fi
    sleep 10
  done
fi
flock -u 9

gave_up=0
while :; do
  n_finished=0
  for i in "${!groups[@]}"; do
    if (( finished[i] )); then n_finished=$(( n_finished + 1 )); continue; fi
    if kill -0 "${pids[i]}" 2>/dev/null; then continue; fi
    # shellcheck disable=SC2086
    n_now="$(rows_in ${groups[i]})"
    n_suites="$(wc -w <<<"${groups[i]}")"
    if (( n_now >= n_suites * expected_per_suite )); then
      finished[i]=1; n_finished=$(( n_finished + 1 )); say "worker ${i} (${groups[i]}) is complete"; continue
    fi
    restarts[i]=$(( restarts[i] + 1 ))
    if (( n_now == rows_at_start[i] )); then barren[i]=$(( barren[i] + 1 )); else barren[i]=0; fi
    if (( restarts[i] > MAX_RESTARTS || barren[i] >= MAX_BARREN_RESTARTS )); then
      say "worker ${i} died ${restarts[i]} times (${barren[i]} in a row without a new rollout); giving up, re-run to resume"
      gave_up=1; break
    fi
    say "worker ${i} died with ${n_now} rollouts recorded (see ${LOGDIR}/worker_${i}.log); restart ${restarts[i]}"
    start_worker "${i}"
  done
  if (( gave_up )) || (( n_finished == ${#groups[@]} )); then break; fi
  # shellcheck disable=SC2086
  n_all="$(rows_in ${all_suites})"; ok_all="$(ok_in ${all_suites})"
  if (( n_all >= MIN_SUCCESS_AFTER && ok_all * 100 < n_all * MIN_SUCCESS_RATE )); then
    say "only ${ok_all} of ${n_all} rollouts succeeded: something is wrong with the set-up; stopping"
    status "stopped_low_success rollouts_recorded=${n_all} successes=${ok_all}"
    stop_all; exit 6
  fi
  status "running gpu=${GPU} workers=${NUM_WORKERS} restarts=$(IFS=+; echo "${restarts[*]}") rollouts_recorded=${n_all} successes=${ok_all}"
  sleep "${WATCH_S}"
done

# ---- 3. wrap up ------------------------------------------------------------------------------------
stop_all
"${PY}" "${ROOT}/scripts/roboelo_libero_common.py" "${OUT}" --num-trials "${NUM_TRIALS}" | tee -a "${LOGDIR}/supervisor.log"
# shellcheck disable=SC2086
n_all="$(rows_in ${all_suites})"
n_suites_all="$(wc -w <<<"${all_suites}")"
if (( gave_up )) || (( n_all < n_suites_all * expected_per_suite )); then
  status "finished_with_missing_rollouts rollouts_recorded=${n_all}"; say "stopped with rollouts missing; re-run to resume"; exit 1
fi
status "finished_ok rollouts_recorded=${n_all}"
say "all suites complete"

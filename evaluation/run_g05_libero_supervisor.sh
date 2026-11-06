#!/usr/bin/env bash
# G0.5 (GalaxeaVLA) on LIBERO: 4 suites x 10 tasks x 50 fixed initial states = 2,000 rollouts, one CSV row each.
#
# Waits until the checkpoint and the environment are in place and one GPU has enough free memory, then
# starts GalaxeaVLA's batched policy server on that GPU and runs several clients per suite against it,
# each on its own tasks. A server that dies is started again; the clients wait for it and redo the
# interrupted rollouts. Safe to re-run: finished rollouts are skipped.
#
#   setsid nohup ~/roboelo/scripts/run_g05_libero_supervisor.sh > ~/roboelo/logs/g05_libero_nohup.log 2>&1 &
#
# Pipeline test without a GPU (slow CPU inference, one policy call):
#   DEVICE=cpu SUITES=libero_spatial NUM_TRIALS=1 TASK_IDS=0 CLIENTS_PER_SUITE=1 MAX_STEPS_OVERRIDE=10 PORT=18576 \
#     OUT=~/roboelo/tmp/g05_smoke STATUS=~/roboelo/tmp/g05_smoke.status ~/roboelo/scripts/run_g05_libero_supervisor.sh
set -Euo pipefail

ROOT="${HOME}/roboelo"
PORT="${PORT:-18575}"
MIN_FREE_MIB="${MIN_FREE_MIB:-14000}"       # free memory required on one GPU before starting
POLL_S="${POLL_S:-60}"
STABLE_POLLS="${STABLE_POLLS:-3}"           # the GPU must look free this many polls in a row
MAX_WAIT_H="${MAX_WAIT_H:-336}"
MAX_SERVER_RESTARTS="${MAX_SERVER_RESTARTS:-500}"
MAX_BARREN_RESTARTS="${MAX_BARREN_RESTARTS:-3}"    # stop if the server dies this often in a row without a new rollout
WATCH_S="${WATCH_S:-30}"
NUM_TRIALS="${NUM_TRIALS:-50}"
CLIENTS_PER_SUITE="${CLIENTS_PER_SUITE:-2}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-20}"      # default of GalaxeaVLA's scripts/run/eval_libero.sh
MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-}"   # debug only
DEVICE="${DEVICE:-cuda}"                    # "cpu" = pipeline test without a GPU
SERVER_GDB="${SERVER_GDB:-1}"               # 1 = run the server under gdb: a crash leaves a native backtrace in server.log
MIN_SUCCESS_AFTER="${MIN_SUCCESS_AFTER:-20}"   # sanity gate: after this many rollouts ...
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-50}"     # ... stop if fewer than this percent succeeded
GPU_FIXED="${GPU_FIXED:-}"                  # set by the GPU queue: use this GPU and do not wait for one
HIDE_FLASH_ATTN="${HIDE_FLASH_ATTN:-1}"     # 1 = vision encoder uses PyTorch attention instead of FlashAttention-4 (see below)
OUT="${OUT:-${ROOT}/results/g05/libero_50}"
STATUS="${STATUS:-${ROOT}/state/g05_libero.status}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/logs/g05_libero_${RUN_ID}"
REPO="${ROOT}/third_party/GalaxeaVLA"
PY="${REPO}/.venv/bin/python"
SRC="${ROOT}/checkpoints/G05"               # files as downloaded from OpenGalaxea/G05
BUNDLE="${REPO}/checkpoints/g05-libero"     # layout that GalaxeaVLA's LIBERO evaluation expects
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

# ---- 0. checkpoint and environment must be there --------------------------------------------------
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
if [[ ! -f "${ROOT}/state/model_g05_libero.done" || ! -x "${PY}" ]]; then
  status "waiting_for_checkpoint_or_environment"
  say "waiting for the G0.5 checkpoint (state/model_g05_libero.done) and environment (${PY})"
  until [[ -f "${ROOT}/state/model_g05_libero.done" && -x "${PY}" ]]; do
    if (( $(date +%s) > deadline )); then status "gave_up_waiting_for_checkpoint"; exit 3; fi
    sleep "${POLL_S}"
  done
fi
# The checkpoint's config refers to checkpoints/g05-libero/{model.pt, action_tokenizer.pt, hf_processor, ...}
# relative to the repository root: build that folder from links to the downloaded files.
mkdir -p "${BUNDLE}/.hydra"
ln -sfn "${SRC}/g05-libero/model.pt" "${BUNDLE}/model.pt"
ln -sfn "${SRC}/g05-libero/dataset_stats.json" "${BUNDLE}/dataset_stats.json"
ln -sfn "${SRC}/action_tokenizer.pt" "${BUNDLE}/action_tokenizer.pt"
ln -sfn "${SRC}/qwen3_5_2b_base_processor" "${BUNDLE}/hf_processor"
cp -f "${SRC}/g05-libero/.hydra/config.yaml" "${BUNDLE}/.hydra/config.yaml"

# ---- 1. wait for a GPU (one supervisor at a time picks a GPU and loads its model) -------------------
GPU=0
exec 9> "${ROOT}/state/gpu_select.lock"
if [[ "${DEVICE}" != "cpu" && -n "${GPU_FIXED}" ]]; then
  GPU="${GPU_FIXED}"
  say "GPU ${GPU} was assigned by the queue"
elif [[ "${DEVICE}" != "cpu" ]]; then
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
  say "GPU ${GPU} has ${free_now} MiB free; starting"
fi

# ---- 2. policy server ------------------------------------------------------------------------------
cd "${REPO}"
read -r -a task_ids <<<"${TASK_IDS}"
n_suites="$(wc -w <<<"${SUITES}")"
n_clients=$(( n_suites * CLIENTS_PER_SUITE ))
common_env=(PYTHONPATH="${REPO}" LIBERO_CONFIG_PATH="${ROOT}/libero_config" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
            HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 MUJOCO_GL=egl
            PYOPENGL_PLATFORM=egl G05_REPO="${REPO}")
if [[ "${DEVICE}" == "cpu" ]]; then server_env=(CUDA_VISIBLE_DEVICES=); else server_env=(CUDA_VISIBLE_DEVICES="${GPU}"); fi
server_env+=(PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
# GalaxeaVLA's vision encoder uses FlashAttention-4 when it can be imported. Its kernel is written for
# H100/B200-class GPUs and does not compile on the RTX 5090 ("Operation creation failed" in
# flash_attn/cute/flash_fwd.py). Hiding the package makes the encoder take the fallback that the code
# provides for exactly this case, PyTorch's scaled_dot_product_attention: the same attention, another kernel.
if [[ "${HIDE_FLASH_ATTN}" == "1" ]]; then server_env+=(PYTHONPATH="${ROOT}/pydeps/no_flash_attn:${REPO}"); fi
# One request per client at a time: dispatch as soon as every client has asked, or after 50 ms.
server_cmd=("${PY}" scripts/serve_policy_batched.py --ckpt_path checkpoints/g05-libero/model.pt --host 127.0.0.1
            --port "${PORT}" --device "${DEVICE}" eval_embodiment=libero --action_steps 10
            --max_batch_size "${n_clients}" --max_wait_ms 50)
if [[ "${SERVER_GDB}" == "1" ]]; then
  # Symbols are read only after a crash, so start-up stays fast.
  server_cmd=(gdb -q -batch -ex "set pagination off" -ex "set print thread-events off" -ex "set auto-solib-add off"
              -ex run -ex "echo ===== NATIVE BACKTRACE =====" -ex 'x/6i $pc' -ex "info registers"
              -ex sharedlibrary -ex "bt 80" -ex "info sharedlibrary" --args "${server_cmd[@]}")
fi
SERVER_PID=""
pids=()
start_server() {
  printf '\n===== server start %s =====\n' "$(date -Is)" >> "${LOGDIR}/server.log"
  env "${common_env[@]}" "${server_env[@]}" "${server_cmd[@]}" >> "${LOGDIR}/server.log" 2>&1 9>&- &
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
  local pid kids
  for pid in "${pids[@]}"; do
    kids="$(pgrep -P "${pid}" || true)"
    # shellcheck disable=SC2086
    kill "${pid}" ${kids} 2>/dev/null || true
  done
  pids=()
}
trap 'stop_clients; stop_server' EXIT
trap 'exit 143' TERM INT HUP

cat > "${OUT}/run_config_${RUN_ID}.json" <<JSON
{"run_id": "${RUN_ID}", "model": "G0.5 (GalaxeaVLA)", "checkpoint": "OpenGalaxea/G05 g05-libero", "suites": "${SUITES}",
 "num_trials_per_task": ${NUM_TRIALS}, "initial_states": "LIBERO fixed initial states 0..N-1",
 "num_steps_wait": ${NUM_STEPS_WAIT}, "env_seed": 7, "action_steps": 10, "ensemble": false, "env_resolution": 256,
 "max_steps": "spatial 220, object 280, goal 300, libero_10 520", "clients_per_suite": ${CLIENTS_PER_SUITE},
 "server_batching": "max_batch_size ${n_clients}, max_wait_ms 50", "device": "${DEVICE}", "gpu": "${GPU}",
 "server_under_gdb": "${SERVER_GDB}",
 "vision_attention": "$([[ "${HIDE_FLASH_ATTN}" == "1" ]] && echo "PyTorch scaled_dot_product_attention (GalaxeaVLA's fallback; FlashAttention-4 does not compile on the RTX 5090)" || echo "FlashAttention-4")", "galaxeavla_commit": "$(git -C "${REPO}" rev-parse HEAD 2>/dev/null)",
 "libero_commit": "$(git -C "${ROOT}/third_party/LIBERO" rev-parse HEAD 2>/dev/null)"}
JSON

status "starting_server gpu=${GPU}"
start_server
say "policy server pid ${SERVER_PID}, port ${PORT}; waiting until it accepts connections"
ready=0
for _ in $(seq 1 360); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then break; fi
  if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then ready=1; break; fi
  sleep 5
done
flock -u 9   # the model is in memory (or failed): other supervisors may now look at the GPUs
if (( ! ready )); then status "server_not_ready"; say "the server did not come up; see ${LOGDIR}/server.log"; exit 5; fi
say "server is up"

# ---- 3. clients: CLIENTS_PER_SUITE per suite, each on every CLIENTS_PER_SUITE-th task -----------------
client_env=("${common_env[@]}" CUDA_VISIBLE_DEVICES= MUJOCO_EGL_DEVICE_ID="${EGL_GPU:-${GPU}}" OMP_NUM_THREADS=1
            MKL_NUM_THREADS=1)
extra=()
if [[ -n "${MAX_STEPS_OVERRIDE}" ]]; then extra+=(--max-steps-override "${MAX_STEPS_OVERRIDE}"); fi
run_client() {
  local suite="$1" c="$2" try; shift 2
  for try in 1 2 3; do
    env "${client_env[@]}" "${PY}" "${ROOT}/scripts/run_g05_libero_client.py" --suite "${suite}" --out "${OUT}" \
      --server-uri "ws://127.0.0.1:${PORT}" --num-trials "${NUM_TRIALS}" --num-steps-wait "${NUM_STEPS_WAIT}" \
      --task-ids "$@" "${extra[@]}" >> "${LOGDIR}/client_${suite}_${c}.log" 2>&1 9>&- && return 0
    say "client ${suite}/${c}: attempt ${try} ended with missing rollouts or an error; retrying"
    sleep 20
  done
  return 1
}
for suite in ${SUITES}; do
  for c in $(seq 0 $(( CLIENTS_PER_SUITE - 1 ))); do
    mine=()
    for j in "${!task_ids[@]}"; do if (( j % CLIENTS_PER_SUITE == c )); then mine+=("${task_ids[j]}"); fi; done
    if (( ${#mine[@]} == 0 )); then continue; fi
    run_client "${suite}" "${c}" "${mine[@]}" &
    pids+=("$!")
  done
done

expected=$(( n_suites * ${#task_ids[@]} * NUM_TRIALS ))
restarts=0 barren=0 n_at_restart=-1 gave_up=0
while :; do
  alive=0
  for pid in "${pids[@]}"; do if kill -0 "${pid}" 2>/dev/null; then alive=1; fi; done
  if (( ! alive )); then break; fi
  # shellcheck disable=SC2086
  n_all="$(rows_in ${SUITES})"; ok_all="$(ok_in ${SUITES})"
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    restarts=$(( restarts + 1 ))
    if (( n_all == n_at_restart )); then barren=$(( barren + 1 )); else barren=0; fi
    n_at_restart="${n_all}"
    if (( restarts > MAX_SERVER_RESTARTS || barren >= MAX_BARREN_RESTARTS )); then
      say "policy server died ${restarts} times (${barren} in a row without a new rollout); giving up, re-run to resume"
      gave_up=1; break
    fi
    say "policy server died with ${n_all} rollouts recorded (see ${LOGDIR}/server.log); restart ${restarts}"
    start_server
  fi
  if (( n_all >= MIN_SUCCESS_AFTER && ok_all * 100 < n_all * MIN_SUCCESS_RATE )); then
    say "only ${ok_all} of ${n_all} rollouts succeeded: something is wrong with the set-up; stopping"
    status "stopped_low_success rollouts_recorded=${n_all} successes=${ok_all}"
    exit 6
  fi
  status "running gpu=${GPU} server_restarts=${restarts} rollouts_recorded=${n_all} successes=${ok_all}"
  sleep "${WATCH_S}"
done
stop_clients
stop_server

# ---- 4. wrap up ------------------------------------------------------------------------------------
"${PY}" "${ROOT}/scripts/roboelo_libero_common.py" "${OUT}" --num-trials "${NUM_TRIALS}" | tee -a "${LOGDIR}/supervisor.log"
# shellcheck disable=SC2086
n_all="$(rows_in ${SUITES})"
if (( gave_up )) || (( n_all < expected )); then
  status "finished_with_missing_rollouts rollouts_recorded=${n_all}"; say "stopped with rollouts missing; re-run to resume"; exit 1
fi
status "finished_ok rollouts_recorded=${n_all}"
say "all suites complete"

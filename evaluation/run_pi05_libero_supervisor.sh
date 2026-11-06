#!/usr/bin/env bash
# pi0.5 on LIBERO: 4 suites x 10 tasks x 50 fixed initial states = 2,000 rollouts, one CSV row each.
#
# Waits until one GPU has enough free memory, starts the openpi policy server on that GPU, then runs
# one client per suite in parallel against it. Safe to re-run: finished rollouts are skipped.
#
#   nohup ~/roboelo/scripts/run_pi05_libero_supervisor.sh > ~/roboelo/logs/pi05_libero_nohup.log 2>&1 &
#
# Smoke test without a GPU (slow CPU inference, a few policy steps of one task):
#   JAX_PLATFORM=cpu SUITES=libero_spatial NUM_TRIALS=1 PORT=18556 OUT=~/roboelo/tmp/pi05_libero_smoke \
#     EXTRA_CLIENT_ARGS="--task-ids 0 --max-steps-override 10" ~/roboelo/scripts/run_pi05_libero_supervisor.sh
set -Euo pipefail

ROOT="${HOME}/roboelo"
PORT="${PORT:-18555}"                       # 8000 is taken by another user's service
MIN_FREE_MIB="${MIN_FREE_MIB:-14000}"       # free memory required on one GPU before starting
POLL_S="${POLL_S:-60}"
STABLE_POLLS="${STABLE_POLLS:-3}"           # the GPU must look free this many polls in a row
MAX_WAIT_H="${MAX_WAIT_H:-168}"
MAX_SERVER_RESTARTS="${MAX_SERVER_RESTARTS:-500}"  # restart a crashed policy server at most this many times
MAX_BARREN_RESTARTS="${MAX_BARREN_RESTARTS:-3}"    # ... and stop if it dies this often in a row without a new rollout
WATCH_S="${WATCH_S:-30}"                     # how often the run is checked and the status file refreshed
SERVER_MEM_FRACTION="${SERVER_MEM_FRACTION:-}"     # e.g. 0.5 = reserve half of the GPU up front; empty = grow on demand
EGL_GPU="${EGL_GPU:-}"                       # GPU that renders the simulation; empty = same as the policy, "other" = the other one
SERVER_GDB="${SERVER_GDB:-}"                 # 1 = run the policy server under gdb: a crash leaves a native backtrace in server.log
NUM_TRIALS="${NUM_TRIALS:-50}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
OUT="${OUT:-${ROOT}/results/pi05/libero_50}"
JAX_PLATFORM="${JAX_PLATFORM:-}"            # "cpu" = smoke test without a GPU
EXTRA_CLIENT_ARGS="${EXTRA_CLIENT_ARGS:-}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/logs/pi05_libero_${RUN_ID}"
STATUS="${STATUS:-${ROOT}/state/pi05_libero.status}"
CLIENT_PY="${ROOT}/micromamba/envs/roboelo-libero/bin/python"
SERVER_PY="${ROOT}/third_party/openpi/.venv/bin/python"
CKPT="${ROOT}/checkpoints/pi05_libero"

mkdir -p "${LOGDIR}" "${OUT}" "${ROOT}/state"
say() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOGDIR}/supervisor.log"; }
status() { printf 'time=%s\nrun_id=%s\nstate=%s\nlog_dir=%s\nout=%s\n' "$(date -Is)" "${RUN_ID}" "$*" "${LOGDIR}" "${OUT}" > "${STATUS}"; }

# ---- 1. wait for a GPU -----------------------------------------------------------------------------
GPU=0
if [[ "${JAX_PLATFORM}" != "cpu" ]]; then
  status "waiting_for_gpu min_free_mib=${MIN_FREE_MIB}"
  say "waiting for a GPU with >= ${MIN_FREE_MIB} MiB free (${STABLE_POLLS} polls of ${POLL_S}s in a row)"
  stable=0
  deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
  while :; do
    best="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)"
    gpu_now="$(cut -d, -f1 <<<"${best}" | tr -d ' ')"
    free_now="$(cut -d, -f2 <<<"${best}" | tr -d ' ')"
    if (( free_now >= MIN_FREE_MIB )); then
      stable=$(( stable + 1 ))
      if (( stable >= STABLE_POLLS )); then GPU="${gpu_now}"; break; fi
    else
      stable=0
    fi
    if (( $(date +%s) > deadline )); then status "gave_up_waiting_for_gpu"; say "no free GPU after ${MAX_WAIT_H} h"; exit 3; fi
    sleep "${POLL_S}"
  done
  say "GPU ${GPU} has ${free_now} MiB free; starting"
fi

# ---- 2. policy server ------------------------------------------------------------------------------
status "starting_server gpu=${GPU}"
cd "${ROOT}/third_party/openpi"
if [[ "${JAX_PLATFORM}" == "cpu" ]]; then
  server_env=(CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu)
elif [[ -n "${SERVER_MEM_FRACTION}" ]]; then
  server_env=(CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_MEM_FRACTION="${SERVER_MEM_FRACTION}")
else
  server_env=(CUDA_VISIBLE_DEVICES="${GPU}" XLA_PYTHON_CLIENT_PREALLOCATE=false)  # take only the memory needed
fi
server_env+=(PYTHONFAULTHANDLER=1)            # a crash leaves the Python stack of every thread in server.log
pids=()
count_rollouts() { cat "${OUT}"/*/episodes.csv 2>/dev/null | grep -vc '^model,' || true; }
stop_clients() {
  local pid kids
  for pid in "${pids[@]}"; do
    kids="$(pgrep -P "${pid}" || true)"
    # shellcheck disable=SC2086
    kill "${pid}" ${kids} 2>/dev/null || true
  done
}
server_cmd=("${SERVER_PY}" scripts/serve_policy.py --port "${PORT}" policy:checkpoint --policy.config=pi05_libero
            --policy.dir="${CKPT}")
if [[ -n "${SERVER_GDB}" ]]; then
  # Symbols are read only after the crash, so start-up stays fast.
  server_cmd=(gdb -q -batch -ex "set pagination off" -ex "set print thread-events off" -ex "set auto-solib-add off"
              -ex run -ex "echo ===== NATIVE BACKTRACE =====" -ex 'x/6i $pc' -ex "info registers"
              -ex sharedlibrary -ex "bt 80" -ex "info sharedlibrary" --args "${server_cmd[@]}")
fi
start_server() {
  printf '\n===== server start %s =====\n' "$(date -Is)" >> "${LOGDIR}/server.log"
  env "${server_env[@]}" "${server_cmd[@]}" >> "${LOGDIR}/server.log" 2>&1 &
  SERVER_PID=$!
}
stop_server() {  # under gdb the Python process is a child of SERVER_PID
  local kids
  kids="$(pgrep -P "${SERVER_PID}" || true)"
  # shellcheck disable=SC2086
  kill "${SERVER_PID}" ${kids} 2>/dev/null || true
}
start_server
trap 'stop_server; stop_clients' EXIT
trap 'exit 143' TERM INT HUP
say "policy server pid ${SERVER_PID}, port ${PORT}; waiting until it accepts connections"
for _ in $(seq 1 360); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then status "server_died"; say "server exited; see ${LOGDIR}/server.log"; exit 4; fi
  if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then exec 3>&- 3<&- || true; ready=1; break; fi
  sleep 5
done
if [[ "${ready:-0}" != 1 ]]; then status "server_not_ready"; say "server did not open port ${PORT} within 30 min"; exit 5; fi
say "server is up"

# ---- 3. one client per suite -----------------------------------------------------------------------
if [[ "${EGL_GPU}" == "other" ]]; then EGL_GPU=$(( 1 - GPU )); fi   # this machine has GPUs 0 and 1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="${EGL_GPU:-${GPU}}"
export LIBERO_CONFIG_PATH="${ROOT}/libero_config" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}/pydeps/pi05_libero_client:${ROOT}/third_party/openpi/packages/openpi-client/src:${ROOT}/third_party/LIBERO"
cat > "${OUT}/run_config_${RUN_ID}.json" <<JSON
{"run_id": "${RUN_ID}", "model": "pi05 (openpi)", "checkpoint": "${CKPT}", "config": "pi05_libero",
 "suites": "${SUITES}", "num_trials_per_task": ${NUM_TRIALS}, "seed": 7, "replan_steps": 5, "resize": 224,
 "gpu": "${GPU}", "egl_gpu": "${EGL_GPU:-${GPU}}", "server_mem_fraction": "${SERVER_MEM_FRACTION:-on demand}",
 "server_under_gdb": "${SERVER_GDB:-0}",
 "jax_platform": "${JAX_PLATFORM:-gpu}", "extra_client_args": "${EXTRA_CLIENT_ARGS}",
 "openpi_commit": "$(git -C "${ROOT}/third_party/openpi" rev-parse HEAD 2>/dev/null)",
 "libero_commit": "$(git -C "${ROOT}/third_party/LIBERO" rev-parse HEAD 2>/dev/null)"}
JSON

run_suite() {
  local suite="$1" try
  for try in 1 2 3; do
    # shellcheck disable=SC2086
    "${CLIENT_PY}" "${ROOT}/scripts/run_pi05_libero_client.py" --suite "${suite}" --out "${OUT}" --port "${PORT}" \
      --num-trials "${NUM_TRIALS}" ${EXTRA_CLIENT_ARGS} >> "${LOGDIR}/client_${suite}.log" 2>&1 && return 0
    say "suite ${suite}: attempt ${try} ended with missing rollouts or an error; retrying"
    sleep 20
  done
  return 1
}
status "running gpu=${GPU} suites=$(tr ' ' ',' <<<"${SUITES}")"
for suite in ${SUITES}; do run_suite "${suite}" & pids+=("$!"); done

# Watch the run. A client whose connection drops waits for the port and reconnects by itself, so a
# crashed policy server only has to be started again. On this machine it dies every few minutes with
# a general protection fault inside libcuda.so; the interrupted rollouts are redone, not counted.
restarts=0 barren=0 n_at_restart=-1
while :; do
  alive=0
  for pid in "${pids[@]}"; do if kill -0 "${pid}" 2>/dev/null; then alive=1; fi; done
  if (( ! alive )); then break; fi
  n_done="$(count_rollouts)"
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    restarts=$(( restarts + 1 ))
    if (( n_done == n_at_restart )); then barren=$(( barren + 1 )); else barren=0; fi
    n_at_restart="${n_done}"
    if (( restarts > MAX_SERVER_RESTARTS || barren >= MAX_BARREN_RESTARTS )); then
      say "policy server died ${restarts} times (${barren} in a row without a new rollout); giving up, re-run to resume"
      stop_clients
      break
    fi
    say "policy server died with ${n_done} rollouts recorded (see ${LOGDIR}/server.log); restart ${restarts}"
    start_server
  fi
  status "running gpu=${GPU} server_restarts=${restarts} rollouts_recorded=${n_done}"
  sleep "${WATCH_S}"
done
fail=0
for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done

# ---- 4. wrap up ------------------------------------------------------------------------------------
stop_server
"${CLIENT_PY}" - "${OUT}" <<'PY' | tee -a "${LOGDIR}/supervisor.log"
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
tot = [0, 0]
for f in sorted(out.glob("*/summary.json")):
    s = json.loads(f.read_text())
    tot[0] += s["successes"]; tot[1] += s["rollouts"]
    print(f"{s['suite']:16s} {s['successes']:4d}/{s['rollouts']:4d} = {s['success_rate']:.3f}")
print(f"{'all':16s} {tot[0]:4d}/{tot[1]:4d} = {tot[0] / max(1, tot[1]):.3f}")
PY
if (( fail )); then status "finished_with_missing_rollouts"; say "finished, but some rollouts are missing; re-run to resume"; exit 1; fi
status "finished_ok"
say "all suites complete"

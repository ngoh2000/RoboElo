#!/usr/bin/env bash
# A LeRobot policy on RoboTwin 2.0 (clean setting), one CSV row per episode (run_lerobot_robotwin.py).
#
# Runs NUM_WORKERS processes on one GPU, worker i taking every NUM_WORKERS-th task, all appending to
# <OUT>/episodes.csv. A worker that dies is started again and continues where it stopped; the run stops after 3
# restarts in a row without a new episode. Safe to re-run: recorded episodes are skipped. Started by
# roboelo_gpu_queue.sh with GPU_FIXED=<index>.
#
#   MODEL=smolvla CKPT=$HOME/roboelo/checkpoints/smolvla-robotwin-lerobot CKPT_LABEL=lerobot/smolvla_robotwin \
#   POLICY_ARGS="--env.action_mode=joint" NUM_WORKERS=2 ~/roboelo/scripts/run_robotwin_supervisor.sh
set -Euo pipefail
ROOT="${HOME}/roboelo"
RT="${ROOT}/third_party/RoboTwin"
MODEL="${MODEL:?MODEL is not set}"
CKPT="${CKPT:?CKPT is not set}"
CKPT_LABEL="${CKPT_LABEL:-${CKPT}}"
PY="${PY:-${ROOT}/envs/robotwin/bin/python}"
POLICY_ARGS="${POLICY_ARGS:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"
GPU_FIXED="${GPU_FIXED:-0}"
NUM_EPISODES="${NUM_EPISODES:-100}"
TASKS="${TASKS:-}"                       # default: the runner's list (49 tasks)
STEP_LIMITS="${STEP_LIMITS:-official}"
EXTRA_WORKER_ENV="${EXTRA_WORKER_ENV:-}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"     # put ahead of RoboTwin's root, e.g. a LeRobot branch's src/ for a policy not yet merged
LEROBOT_SRC="${LEROBOT_SRC:-${ROOT}/third_party/lerobot}"   # the LeRobot checkout in use (its commit goes into run_config)
RUN_NOTE="${RUN_NOTE:-}"
MAX_BARREN_RESTARTS="${MAX_BARREN_RESTARTS:-3}"
WATCH_S="${WATCH_S:-60}"
OUT="${OUT:-${ROOT}/results/${MODEL}/robotwin2_easy}"
STATUS="${STATUS:-${ROOT}/state/${MODEL}_robotwin.status}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/logs/${MODEL}_robotwin_${RUN_ID}"
mkdir -p "${LOGDIR}" "${OUT}"
say() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOGDIR}/supervisor.log"; }
status() { printf 'time=%s\nrun_id=%s\nstate=%s\nlog_dir=%s\nout=%s\n' "$(date -Is)" "${RUN_ID}" "$*" "${LOGDIR}" "${OUT}" > "${STATUS}"; }
json_str() { sed 's/["\\]/\\&/g' <<<"$1"; }
PYPATH="${EXTRA_PYTHONPATH}${EXTRA_PYTHONPATH:+:}${RT}"
n_tasks="$(cd "${RT}" && TASKS="${TASKS}" PYTHONPATH="${PYPATH}" "${PY}" -c "
import os; from lerobot.envs.robotwin import ROBOTWIN_TASKS
t = [x for x in os.environ.get('TASKS', '').split(',') if x] or [x for x in ROBOTWIN_TASKS if x != 'open_laptop']
print(len(t))" 2>/dev/null)"
expected=$(( n_tasks * NUM_EPISODES ))
rows() { if [[ -f "${OUT}/episodes.csv" ]]; then python3 -c "import csv,sys; print(len({(r['task'], r['episode_index']) for r in csv.DictReader(open(sys.argv[1]))}))" "${OUT}/episodes.csv"; else echo 0; fi; }
oks() { if [[ -f "${OUT}/episodes.csv" ]]; then python3 -c "import csv,sys; print(sum(int(r['success']) for r in csv.DictReader(open(sys.argv[1]))))" "${OUT}/episodes.csv"; else echo 0; fi; }

read -r -a extra_env <<<"${EXTRA_WORKER_ENV}"
read -r -a policy_args <<<"${POLICY_ARGS}"
cat > "${OUT}/run_config_${RUN_ID}.json" <<JSON
{"run_id": "${RUN_ID}", "model": "${MODEL}", "checkpoint": "${CKPT}", "checkpoint_label": "$(json_str "${CKPT_LABEL}")",
 "runner": "run_lerobot_robotwin.py: lerobot_eval.rollout on LeRobot's RoboTwinEnv, episode k = setup_demo(seed=k, is_test=True)",
 "benchmark": "RoboTwin 2.0 clean (demo_clean.yml, LeRobot's integration)", "robotwin_commit": "$(git -C "${RT}" rev-parse HEAD 2>/dev/null)",
 "tasks": ${n_tasks}, "episodes_per_task": ${NUM_EPISODES}, "expected_episodes": ${expected}, "step_limits": "${STEP_LIMITS}",
 "policy_args": "$(json_str "${POLICY_ARGS}")", "workers": ${NUM_WORKERS}, "gpu": "${GPU_FIXED}", "python": "${PY}",
 "extra_pythonpath": "${EXTRA_PYTHONPATH}", "lerobot_src": "${LEROBOT_SRC}",
 "lerobot_commit": "$(git -C "${LEROBOT_SRC}" rev-parse HEAD 2>/dev/null)", "note": "$(json_str "${RUN_NOTE}")"}
JSON

pids=() restarts=() barren=() rows_at_start=() finished=()
start_worker() {
  local i="$1"
  printf '\n===== worker %s start %s =====\n' "${i}" "$(date -Is)" >> "${LOGDIR}/worker_${i}.log"
  rows_at_start[i]="$(rows)"
  ( cd "${RT}" && exec env "${extra_env[@]}" CUDA_VISIBLE_DEVICES="${GPU_FIXED}" PYTHONPATH="${PYPATH}" HF_HOME="${ROOT}/hf_cache" \
      HF_HUB_OFFLINE=1 PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${WORKER_THREADS:-4}" \
      ROBOELO_OUT="${OUT}" ROBOELO_TASKS="${TASKS}" ROBOELO_TASK_SHARD="${i}/${NUM_WORKERS}" ROBOELO_NUM_EPISODES="${NUM_EPISODES}" \
      ROBOELO_STEP_LIMITS="${STEP_LIMITS}" ROBOELO_MODEL="${MODEL}" ROBOELO_CHECKPOINT="${CKPT_LABEL}" \
      "${PY}" "${ROOT}/scripts/run_lerobot_robotwin.py" --policy.path="${CKPT}" --env.type=robotwin \
      --policy.device=cuda --eval.batch_size=1 "${policy_args[@]}" ) >> "${LOGDIR}/worker_${i}.log" 2>&1 &
  pids[i]=$!
}
stop_all() { local pid; for pid in "${pids[@]}"; do kill "${pid}" $(pgrep -P "${pid}" || true) 2>/dev/null || true; done; }
trap 'stop_all' EXIT
trap 'exit 143' TERM INT HUP

say "GPU ${GPU_FIXED}; ${NUM_WORKERS} worker(s); ${n_tasks} tasks x ${NUM_EPISODES} episodes = ${expected} expected"
status "starting gpu=${GPU_FIXED} workers=${NUM_WORKERS}"
for ((i = 0; i < NUM_WORKERS; i++)); do restarts[i]=0; barren[i]=0; finished[i]=0; start_worker "${i}"; done
gave_up=0
while :; do
  n_finished=0
  for ((i = 0; i < NUM_WORKERS; i++)); do
    if (( finished[i] )); then n_finished=$(( n_finished + 1 )); continue; fi
    if kill -0 "${pids[i]}" 2>/dev/null; then continue; fi
    if wait "${pids[i]}"; then finished[i]=1; n_finished=$(( n_finished + 1 )); say "worker ${i} is complete"; continue; fi
    restarts[i]=$(( restarts[i] + 1 ))
    if [[ "$(rows)" == "${rows_at_start[i]}" ]]; then barren[i]=$(( barren[i] + 1 )); else barren[i]=0; fi
    if (( barren[i] >= MAX_BARREN_RESTARTS )); then say "worker ${i} died ${barren[i]} times in a row without a new episode; giving up"; gave_up=1; break; fi
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

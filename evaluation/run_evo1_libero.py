#!/usr/bin/env python
"""Evo-1 (official MINT-SJTU/Evo1_LIBERO) on LIBERO, one CSV row per rollout (record format of the RoboElo runners).

The per-episode logic is Evo-1's LIBERO_evaluation/libero_client_4tasks.py, talking to Evo-1's own server
(scripts/evo1_server_launcher.py = Evo_1/scripts/Evo1_server.py) over the same websocket JSON messages: 448 px
cameras, env.seed(42) per task environment, np.random / random seeded with 42 once, 10 no-op steps with the
all-zero action of that script, then queries of agent-view and wrist images (rotated 180 degrees, as nested lists)
plus an empty third image, state = eef position + axis-angle + gripper qpos, the task text as prompt, image mask
[1, 1, 0] and action mask 7 of 24; the first 14 actions of each returned chunk are executed (gripper -1 if > 0.5,
else +1), success checked after every step, at most 25 / 25 / 25 / 95 queries (spatial / object / goal / 10). An
action the env rejects (ValueError) ends the episode as a failure, as there.
What this runner adds: initial states 0..49 of every task (the script runs 10), one row per finished rollout
(resumable), partial progress from the goal predicates, and other exceptions are recorded separately and the
rollout run again.

Started by run_lerobot_libero_supervisor.sh (WORKER_SCRIPT=<this file>, SERVER_CMD=<the policy server>) with the
ROBOELO_* variables of run_lerobot_libero.py. Options: --port, --horizon, --seed.
"""
import argparse
import json
import math
import os
import pathlib
import random
import re
import sys
import time
import traceback

ROOT = os.path.expanduser("~/roboelo")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # LIBERO's pickled initial states
sys.path.insert(0, os.path.join(ROOT, "third_party", "LIBERO"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio  # noqa: E402
import numpy as np  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from websockets.sync.client import connect  # noqa: E402
from roboelo_libero_common import SUITES, EpisodeLog, goal_status  # noqa: E402

MAX_QUERIES = {"libero_spatial": 25, "libero_object": 25, "libero_goal": 25, "libero_10": 95}
DUMMY_ACTION = [0.0] * 6 + [0.0]  # libero_client_4tasks.py's LIBERO_DUMMY_ACTION
RESOLUTION = 448


def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_json_dict(obs, prompt, resize_size=448):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    dummy_proc = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)
    return {
        "image": [img.astype(np.uint8).tolist(), wrist_img.astype(np.uint8).tolist(), dummy_proc.tolist()],
        "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                 obs["robot0_gripper_qpos"])).tolist(),
        "prompt": prompt,
        "image_mask": [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17,
    }


def _int_list(name):
    raw = os.environ.get(name, "").strip()
    return [int(x) for x in re.split(r"[,\s]+", raw) if x] if raw else None


def _suite_specs(raw):
    """'libero_goal libero_10:0-4' -> [('libero_goal', None), ('libero_10', [0, 1, 2, 3, 4])]"""
    specs = []
    for token in re.split(r"[,\s]+", raw):
        if not token:
            continue
        name, _, ids = token.partition(":")
        task_ids = None
        if ids:
            task_ids = []
            for part in ids.split("+"):
                first, _, last = part.partition("-")
                task_ids.extend(range(int(first), int(last or first) + 1))
        specs.append((name, task_ids))
    return specs


def rollout(env, ws, task_description, init_state, max_queries, horizon, num_steps_wait):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    status = goal_status(env)
    goal_max, frames = sum(status), []
    prompt = str(task_description)
    done = False
    steps = n_infer = 0
    for _ in range(max_queries):
        ws.send(json.dumps(obs_to_json_dict(obs, prompt)))
        actions = np.array(json.loads(ws.recv()))
        n_infer += 1
        rejected = False
        for i in range(horizon):
            action = actions[i].tolist()
            action[6] = -1 if action[6] > 0.5 else 1
            try:
                obs, _, done, _ = env.step(action[:7])
            except ValueError as exc:  # the script ends the episode as a failure
                print(f"the action is not valid: {exc}", flush=True)
                done, rejected = False, True
                break
            steps += 1
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            status = goal_status(env)
            goal_max = max(goal_max, sum(status))
            if done:
                break
        if done or rejected:
            break
    return dict(success=bool(done), steps=steps, goal_total=len(status), goal_satisfied_final=sum(status),
                goal_satisfied_max=goal_max, n_infer=n_infer), frames


def main(args):
    out_root = os.environ["ROBOELO_OUT"]
    suites = _suite_specs(os.environ.get("ROBOELO_SUITES", " ".join(SUITES)))
    num_trials = int(os.environ.get("ROBOELO_NUM_TRIALS", "50"))
    task_ids_opt, init_ids_opt = _int_list("ROBOELO_TASK_IDS"), _int_list("ROBOELO_INIT_IDS")
    num_steps_wait = int(os.environ.get("ROBOELO_NUM_STEPS_WAIT", "10"))
    videos = os.environ.get("ROBOELO_VIDEOS", "failures")
    retries = int(os.environ.get("ROBOELO_RETRIES", "2"))
    model_label = os.environ.get("ROBOELO_MODEL", "evo1")
    label = os.environ.get("ROBOELO_CHECKPOINT", "MINT-SJTU/Evo1_LIBERO")

    np.random.seed(args.seed)  # once, as the official script
    random.seed(args.seed)
    deadline = time.time() + 3600
    while True:  # the server may still be loading
        try:
            ws = connect(f"ws://{args.host}:{args.port}", max_size=None, ping_interval=None, open_timeout=60)
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(5)
    n_missing = 0
    for suite_name, suite_task_ids in suites:
        max_queries = MAX_QUERIES[suite_name]
        suite = benchmark.get_benchmark_dict()[suite_name]()
        log = EpisodeLog(out_root, suite_name, model_label, label)
        task_ids = task_ids_opt if task_ids_opt is not None else list(range(suite.n_tasks))
        if suite_task_ids is not None:
            task_ids = [t for t in task_ids if t in suite_task_ids]
        init_ids = init_ids_opt if init_ids_opt is not None else list(range(num_trials))
        print(f"[setup] {suite_name}: at most {max_queries} queries x {args.horizon} actions, wait {num_steps_wait}, "
              f"seed {args.seed}, tasks {task_ids}, {len(log.done)} rollouts already recorded", flush=True)
        for task_id in task_ids:
            todo = log.todo(task_id, init_ids)
            if not todo:
                continue
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RESOLUTION, camera_widths=RESOLUTION)
            env.seed(args.seed)
            for init_id in todo:
                for attempt in range(1, retries + 2):
                    t0 = time.time()
                    try:
                        res, frames = rollout(env, ws, task.language, init_states[init_id], max_queries,
                                              args.horizon, num_steps_wait)
                    except Exception as exc:  # never record an exception as a task failure
                        log.error(task_id, init_id, attempt, exc)
                        traceback.print_exc()
                        if isinstance(exc, OSError) or "ConnectionClosed" in type(exc).__name__:
                            print("lost the policy server: exiting so this worker gets restarted", flush=True)
                            os._exit(3)
                        continue
                    video = ""
                    if videos == "all" or (videos == "failures" and not res["success"]):
                        video = log.video_path(task_id, init_id, res["success"])
                        imageio.mimwrite(video, [np.asarray(f) for f in frames[::2]], fps=10)
                    log.write(task_id=task_id, task_name=task.language, bddl_file=task.bddl_file,
                              init_state_id=init_id, seed=args.seed, max_steps=max_queries * args.horizon,
                              wall_time_s=time.time() - t0, video=video, **res)
                    break
            env.close()
        missing = log.missing(task_ids, init_ids)
        n_missing += len(missing)
        print(f"[{suite_name}] done: new {log.n_new}, exceptions {log.n_err}, missing {len(missing)}", flush=True)
    sys.stdout.flush()
    os._exit(1 if n_missing else 0)  # skip interpreter teardown: EGL contexts can hang or crash on exit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())

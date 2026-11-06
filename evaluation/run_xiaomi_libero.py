#!/usr/bin/env python
"""Xiaomi-Robotics-0 on LIBERO, one CSV row per rollout (same record format as the other RoboElo LIBERO runners).

The per-trial logic is Xiaomi-Robotics-0's eval_libero/main.py (run_trial), talking to the model through the
official deploy/client.py Client (served by scripts/xiaomi_policy_server.py, which is deploy/server.py for several
clients): a new OffScreenRenderEnv per trial (256 px cameras, env.seed(7)), np.random.seed(7) once, 10 no-op steps,
max steps 220 / 280 / 300 / 520, agent-view and wrist images rotated 180 degrees and sent as PIL images, state =
eef position + axis-angle + gripper qpos + 24 zeros, task_id "libero_all", instruction capitalized with a final
".", a per-query seed hashed from the inputs (hash_data_to_seed, before the "." is added), the first 10 actions of
each chunk ([0, :, :-1][0:10, 0:7]) executed before the next query, success checked after every step.
What this runner adds: one row per finished rollout (resumable), partial progress from the goal predicates, and an
exception is recorded separately and the rollout run again (the official run_trial counts it as a failure).

Started by run_lerobot_libero_supervisor.sh (WORKER_SCRIPT=<this file>, SERVER_CMD=<the policy server>) with the
ROBOELO_* variables of run_lerobot_libero.py. Options: --repo, --port, --replan-steps, --seed.
"""
import argparse
import collections
import os
import pathlib
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
from PIL import Image  # noqa: E402
from roboelo_libero_common import LIBERO_DUMMY_ACTION, SUITES, EpisodeLog, goal_status  # noqa: E402

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
LIBERO_ENV_RESOLUTION = 256


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


def rollout(main, client, task, init_state, max_steps, num_steps_wait, replan_steps, seed):
    """main.run_trial without its catch-all exception handler and video writing."""
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=LIBERO_ENV_RESOLUTION,
                             camera_widths=LIBERO_ENV_RESOLUTION)
    env.seed(seed)
    try:
        env.reset()
        obs = env.set_init_state(init_state)
        status = goal_status(env)
        goal_max, frames = sum(status), []
        action_plan = collections.deque()
        t = steps = n_infer = 0
        done = False
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue
            img = main.convert_to_uint8(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            wrist_img = main.convert_to_uint8(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
            frames.append(img)
            if len(action_plan) <= 0:
                state = np.concatenate([obs["robot0_eef_pos"], main._quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"], np.array([0.0] * 24)])
                model_inputs = {"task_id": "libero_all", "state": state, "base": Image.fromarray(img),
                                "wrist_left": Image.fromarray(wrist_img),
                                "language": str(task.language).capitalize()}
                model_inputs["seed"] = main.hash_data_to_seed(model_inputs)
                model_inputs["language"] = model_inputs["language"] + "."
                action_chunk = client(**model_inputs)[0, :, :-1]
                n_infer += 1
                action_plan.extend(action_chunk[0:replan_steps, 0:7])
            action = action_plan.popleft().tolist()
            obs, _, done, _ = env.step(action)
            steps += 1
            status = goal_status(env)
            goal_max = max(goal_max, sum(status))
            if done:
                break
            t += 1
        return dict(success=bool(done), steps=steps, goal_total=len(status), goal_satisfied_final=sum(status),
                    goal_satisfied_max=goal_max, n_infer=n_infer), frames
    finally:
        env.close()


def main_loop(args):
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, "eval_libero"))
    from deploy.client import Client
    import main as official  # eval_libero/main.py: convert_to_uint8, hash_data_to_seed, _quat2axisangle

    out_root = os.environ["ROBOELO_OUT"]
    suites = _suite_specs(os.environ.get("ROBOELO_SUITES", " ".join(SUITES)))
    num_trials = int(os.environ.get("ROBOELO_NUM_TRIALS", "50"))
    task_ids_opt, init_ids_opt = _int_list("ROBOELO_TASK_IDS"), _int_list("ROBOELO_INIT_IDS")
    num_steps_wait = int(os.environ.get("ROBOELO_NUM_STEPS_WAIT", "10"))
    episode_lengths = {name: int(n) for name, _, n in (
        item.partition("=") for item in re.split(r"[,\s]+", os.environ.get("ROBOELO_EPISODE_LENGTHS", "")) if item)}
    videos = os.environ.get("ROBOELO_VIDEOS", "failures")
    retries = int(os.environ.get("ROBOELO_RETRIES", "2"))
    model_label = os.environ.get("ROBOELO_MODEL", "xiaomi_r0")
    label = os.environ.get("ROBOELO_CHECKPOINT", "XiaomiRobotics/Xiaomi-Robotics-0-LIBERO")

    np.random.seed(args.seed)  # once, as the official script
    client = Client(args.host, args.port)
    n_missing = 0
    for suite_name, suite_task_ids in suites:
        max_steps = episode_lengths.get(suite_name, MAX_STEPS[suite_name])
        suite = benchmark.get_benchmark_dict()[suite_name]()
        log = EpisodeLog(out_root, suite_name, model_label, label)
        task_ids = task_ids_opt if task_ids_opt is not None else list(range(suite.n_tasks))
        if suite_task_ids is not None:
            task_ids = [t for t in task_ids if t in suite_task_ids]
        init_ids = init_ids_opt if init_ids_opt is not None else list(range(num_trials))
        print(f"[setup] {suite_name}: max steps {max_steps}, wait {num_steps_wait}, replan {args.replan_steps}, "
              f"seed {args.seed}, tasks {task_ids}, {len(log.done)} rollouts already recorded", flush=True)
        for task_id in task_ids:
            todo = log.todo(task_id, init_ids)
            if not todo:
                continue
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            for init_id in todo:
                for attempt in range(1, retries + 2):
                    t0 = time.time()
                    try:
                        res, frames = rollout(official, client, task, init_states[init_id], max_steps,
                                              num_steps_wait, args.replan_steps, args.seed)
                    except Exception as exc:  # never record an exception as a task failure
                        log.error(task_id, init_id, attempt, exc)
                        traceback.print_exc()
                        if isinstance(exc, (ConnectionError, OSError, EOFError)):
                            print("lost the policy server: exiting so this worker gets restarted", flush=True)
                            os._exit(3)
                        continue
                    video = ""
                    if videos == "all" or (videos == "failures" and not res["success"]):
                        video = log.video_path(task_id, init_id, res["success"])
                        imageio.mimwrite(video, [np.asarray(f) for f in frames], fps=10)
                    log.write(task_id=task_id, task_name=task.language, bddl_file=task.bddl_file,
                              init_state_id=init_id, seed=args.seed, max_steps=max_steps,
                              wall_time_s=time.time() - t0, video=video, **res)
                    break
        missing = log.missing(task_ids, init_ids)
        n_missing += len(missing)
        print(f"[{suite_name}] done: new {log.n_new}, exceptions {log.n_err}, missing {len(missing)}", flush=True)
    sys.stdout.flush()
    os._exit(1 if n_missing else 0)  # skip interpreter teardown: EGL contexts can hang or crash on exit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(ROOT, "third_party", "Xiaomi-Robotics-0"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    main_loop(ap.parse_args())

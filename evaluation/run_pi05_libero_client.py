#!/usr/bin/env python
"""pi0.5 (openpi) on LIBERO, one CSV row per rollout.

Evaluation protocol is openpi's examples/libero/main.py unchanged: 256 px render rotated 180 degrees,
resize-with-pad to 224, 10 wait steps, replan every 5 actions, env seed 7, LIBERO's fixed initial
states, per-suite step limits. This client only adds durable records:

  * <out>/<suite>/episodes.csv   one row per finished rollout (appended and flushed immediately)
  * <out>/<suite>/errors.csv     rollouts that raised an exception (never counted as failures)
  * <out>/<suite>/videos/        replay videos of failed rollouts only
  * <out>/<suite>/summary.json   per-task and suite success rates, written when the suite ends

Partial progress: LIBERO itself reports only success/failure, but every task goal is a conjunction of
BDDL predicates. We log how many are satisfied at the last step (goal_satisfied_final) and the most
that were satisfied at any step (goal_satisfied_max), out of goal_total.

Rollouts already present in episodes.csv are skipped, so the script can be re-run to resume.
"""
import argparse
import collections
import csv
import datetime
import json
import math
import pathlib
import sys
import time
import traceback

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
FIELDS = ["model", "checkpoint", "benchmark", "suite", "task_id", "task_name", "bddl_file", "init_state_id", "seed",
          "success", "steps", "max_steps", "goal_total", "goal_satisfied_final", "goal_satisfied_max",
          "partial_final", "partial_max", "n_infer", "wall_time_s", "finished_at", "video"]


def quat2axisangle(quat):
    """Copied from robosuite (same as openpi's LIBERO example)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def goal_status(env):
    """Truth value of each goal predicate of the task (success = all true)."""
    inner = env.env
    return [bool(inner._eval_predicate(state)) for state in inner.parsed_problem["goal_state"]]


def make_env(task, seed):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=LIBERO_ENV_RESOLUTION,
                             camera_widths=LIBERO_ENV_RESOLUTION)
    env.seed(seed)  # seed affects object positions even with a fixed initial state (openpi note)
    return env


def rollout(env, client, task_description, init_state, max_steps, args):
    env.reset()
    obs = env.set_init_state(init_state)
    action_plan = collections.deque()
    frames, t, n_infer, done = [], 0, 0, False
    status = goal_status(env)
    best = sum(status)
    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:  # let objects settle
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))
        frames.append(img)
        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                "prompt": str(task_description),
            }
            chunk = client.infer(element)["actions"]
            n_infer += 1
            assert len(chunk) >= args.replan_steps, f"policy returned {len(chunk)} actions < replan {args.replan_steps}"
            action_plan.extend(chunk[: args.replan_steps])
        obs, _, done, _ = env.step(action_plan.popleft().tolist())
        status = goal_status(env)
        best = max(best, sum(status))
        if done:
            break
        t += 1
    steps = max(0, t - args.num_steps_wait) + (1 if done else 0)
    return dict(success=int(bool(done)), steps=steps, goal_total=len(status), goal_satisfied_final=sum(status),
                goal_satisfied_max=best, n_infer=n_infer), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=sorted(MAX_STEPS))
    ap.add_argument("--out", required=True, help="results root; a <suite>/ folder is created inside")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18555)
    ap.add_argument("--num-trials", type=int, default=50, help="initial states 0..N-1 per task")
    ap.add_argument("--task-ids", type=int, nargs="*", help="default: every task of the suite")
    ap.add_argument("--init-ids", type=int, nargs="*", help="default: 0..num-trials-1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resize-size", type=int, default=224)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--max-steps-override", type=int, help="debug only: shorten rollouts")
    ap.add_argument("--videos", choices=["failures", "all", "none"], default="failures")
    ap.add_argument("--retries", type=int, default=20,
                    help="re-run a rollout that raised an exception (each crash of the policy server uses one)")
    ap.add_argument("--model", default="pi05")
    ap.add_argument("--checkpoint", default="pi05_libero")
    args = ap.parse_args()

    out = pathlib.Path(args.out) / args.suite
    (out / "videos").mkdir(parents=True, exist_ok=True)
    ep_csv, err_csv = out / "episodes.csv", out / "errors.csv"
    done_keys = set()
    if ep_csv.exists():
        with open(ep_csv, newline="") as f:
            done_keys = {(int(r["task_id"]), int(r["init_state_id"])) for r in csv.DictReader(f)}
    ep_f = open(ep_csv, "a", newline="")
    ep_w = csv.DictWriter(ep_f, fieldnames=FIELDS)
    if not done_keys and ep_f.tell() == 0:
        ep_w.writeheader()
        ep_f.flush()
    err_new = not err_csv.exists()
    err_f = open(err_csv, "a", newline="")
    err_w = csv.writer(err_f)
    if err_new:
        err_w.writerow(["suite", "task_id", "init_state_id", "attempt", "time", "error"])

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    max_steps = args.max_steps_override or MAX_STEPS[args.suite]
    task_ids = args.task_ids if args.task_ids else list(range(suite.n_tasks))
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"[{args.suite}] connected to policy server {args.host}:{args.port}; {len(done_keys)} rollouts already done",
          flush=True)

    n_new = n_err = 0
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        init_ids = args.init_ids if args.init_ids else list(range(min(args.num_trials, len(init_states))))
        todo = [i for i in init_ids if (task_id, i) not in done_keys]
        if not todo:
            continue
        env = make_env(task, args.seed)
        for init_id in todo:
            for attempt in range(1, args.retries + 2):
                t0 = time.time()
                try:
                    res, frames = rollout(env, client, task.language, init_states[init_id], max_steps, args)
                except Exception as exc:  # never record an exception as a task failure
                    n_err += 1
                    err_w.writerow([args.suite, task_id, init_id, attempt, datetime.datetime.now().isoformat(),
                                    f"{type(exc).__name__}: {exc}"])
                    err_f.flush()
                    traceback.print_exc()
                    time.sleep(5)
                    try:
                        client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
                    except Exception:
                        pass
                    continue
                video = ""
                if args.videos == "all" or (args.videos == "failures" and not res["success"]):
                    tag = "success" if res["success"] else "failure"
                    video = str(out / "videos" / f"task{task_id:02d}_init{init_id:02d}_{tag}.mp4")
                    imageio.mimwrite(video, [np.asarray(x) for x in frames], fps=10)
                ep_w.writerow(dict(
                    model=args.model, checkpoint=args.checkpoint, benchmark="libero", suite=args.suite, task_id=task_id,
                    task_name=task.language, bddl_file=task.bddl_file, init_state_id=init_id, seed=args.seed,
                    max_steps=max_steps, partial_final=round(res["goal_satisfied_final"] / res["goal_total"], 4),
                    partial_max=round(res["goal_satisfied_max"] / res["goal_total"], 4),
                    wall_time_s=round(time.time() - t0, 1), finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
                    video=video, **res))
                ep_f.flush()
                n_new += 1
                print(f"[{args.suite}] task {task_id} init {init_id}: success={res['success']} steps={res['steps']} "
                      f"goal {res['goal_satisfied_final']}/{res['goal_total']} ({time.time() - t0:.0f}s)", flush=True)
                break
        env.close()

    ep_f.close()
    err_f.close()
    with open(ep_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    per_task = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per_task[int(r["task_id"])][0] += int(r["success"])
        per_task[int(r["task_id"])][1] += 1
    expected = len(task_ids) * (len(args.init_ids) if args.init_ids else args.num_trials)
    summary = dict(suite=args.suite, rollouts=len(rows), successes=sum(int(r["success"]) for r in rows),
                   success_rate=round(sum(int(r["success"]) for r in rows) / max(1, len(rows)), 4),
                   expected_rollouts_this_call=expected, new_rollouts=n_new, exceptions=n_err,
                   per_task={k: dict(successes=v[0], rollouts=v[1]) for k, v in sorted(per_task.items())})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[{args.suite}] done: {summary['successes']}/{summary['rollouts']} = {summary['success_rate']:.3f}; "
          f"new {n_new}, exceptions {n_err}", flush=True)
    missing = [(t, i) for t in task_ids for i in (args.init_ids or range(args.num_trials))
               if (t, i) not in {(int(r["task_id"]), int(r["init_state_id"])) for r in rows}]
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

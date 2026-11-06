#!/usr/bin/env python
"""starVLA-family policies (starVLA, InternVLA-M1, ABot-M0) on LIBERO, one CSV row per rollout.

The three code bases share one LIBERO client (starVLA/examples/simBenchmarks/LIBERO/eval_files/eval_libero.py,
InternVLA-M1/examples/LIBERO/eval_libero.py, ABot-M0/examples/LIBERO/eval_files/eval_libero.py differ only in the
client class): the model runs in the repository's websocket policy server (deployment/model_server/server_policy.py),
and the LIBERO side uses the repository's own client class from model2libero_interface.py, imported here unchanged:
  starvla       ModelClient(host, port, unnorm_key)            server un-normalizes; PIL bilinear resize to 224
  internvla_m1  M1Inference(policy_ckpt_path, host, port)      client un-normalizes (min/max, gripper > 0.5); cv2 area resize
  abot          ModelClient(policy_ckpt_path, unnorm_key, ...) as InternVLA-M1, last 7 of the 14 action dims
  internvla_a15 InternVLA-A-series evaluation/LIBERO LiberoModelClient(host, port, replan_steps=8): the client sends
                both images and the state, the server preprocesses and un-normalizes, the client returns env actions
                (its eval_libero_server_client.py loop is the same as the others')
Loop as in those scripts: 256 px cameras, env.seed(7) per task environment, np.random.seed(7) once, 10 no-op steps,
max steps 220 / 280 / 300 / 520 (spatial / object / goal / 10), agent-view and wrist images rotated 180 degrees,
one action per step from the cached chunk (a new query every action_chunk_size steps), gripper +1 / -1 from
open > 0.5, success checked after every step. (The scripts also compute the robot state; none of them sends it.)
What this runner adds: one row per finished rollout (resumable), partial progress from the goal predicates, and
an exception is recorded separately and the rollout run again.

Started by run_lerobot_libero_supervisor.sh (WORKER_SCRIPT=<this file>, SERVER_CMD=<the policy server>) with the
ROBOELO_* variables of run_lerobot_libero.py. Options: --repo, --client, --checkpoint (.pt file, for the clients that
read its dataset_statistics.json), --port, --unnorm-key, --seed.
"""
import argparse
import importlib
import os
import pathlib
import re
import sys
import time
import traceback

ROOT = os.path.expanduser("~/roboelo")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # LIBERO's pickled initial states
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.join(ROOT, "third_party", "LIBERO"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio  # noqa: E402
import numpy as np  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from roboelo_libero_common import LIBERO_DUMMY_ACTION, SUITES, EpisodeLog, goal_status  # noqa: E402

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
LIBERO_ENV_RESOLUTION = 256
CLIENT_MODULES = {
    "starvla": "examples.simBenchmarks.LIBERO.eval_files.model2libero_interface",
    "internvla_m1": "examples.LIBERO.model2libero_interface",
    "abot": "examples.LIBERO.eval_files.model2libero_interface",
    "internvla_a15": "evaluation.LIBERO.model2libero_interface",
}


def _binarize_gripper_open(open_val):
    v = float(np.asarray(open_val, dtype=np.float32).reshape(-1)[0])
    return np.asarray([1.0 - 2.0 * (v > 0.5)], dtype=np.float32)


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


def make_client(args):
    sys.path.insert(0, args.repo)
    mod = importlib.import_module(CLIENT_MODULES[args.client])
    if args.client == "starvla":
        return mod.ModelClient(host=args.host, port=args.port, unnorm_key=args.unnorm_key)
    if args.client == "internvla_m1":
        return mod.M1Inference(policy_ckpt_path=args.checkpoint, host=args.host, port=args.port,
                               image_size=[args.resize, args.resize])
    if args.client == "internvla_a15":
        return mod.LiberoModelClient(host=args.host, port=args.port, rotate_images=True, binarize_gripper=True,
                                     replan_steps=args.replan_steps)
    return mod.ModelClient(policy_ckpt_path=args.checkpoint, unnorm_key=args.unnorm_key, host=args.host,
                           port=args.port, image_size=[args.resize, args.resize])


def query(client, kind, img, wrist_img, task_description, step):
    if kind == "internvla_m1":
        return client.step(images=[img, wrist_img], task_description=task_description, step=step)
    return client.step(example={"image": [img, wrist_img], "lang": task_description}, step=step)


def rollout(env, client, kind, task_description, init_state, max_steps, num_steps_wait):
    client.reset(task_description=task_description)
    env.reset()
    obs = env.set_init_state(init_state)
    status = goal_status(env)
    goal_max, frames = sum(status), []
    t = step = n_infer = 0
    done = False
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        frames.append(img)
        if kind == "internvla_a15":  # its client rotates the images, builds the state and returns env actions
            if step % client.replan_steps == 0:
                n_infer += 1
            action = np.asarray(client.step(obs, task_description))
        else:
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            if step % getattr(client, "action_chunk_size", 1) == 0:
                n_infer += 1
            raw_action = query(client, kind, img, wrist_img, task_description, step)["raw_action"]
            world_vector = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
            rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
            open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
            if not (world_vector.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                raise ValueError(f"invalid action sizes {world_vector.shape} {rotation_delta.shape} {open_gripper.shape}")
            action = np.concatenate([world_vector, rotation_delta, _binarize_gripper_open(open_gripper)], axis=0)
        obs, _, done, _ = env.step(action.tolist())
        status = goal_status(env)
        goal_max = max(goal_max, sum(status))
        if done:
            break
        t += 1
        step += 1
    return dict(success=bool(done), steps=step + int(done), goal_total=len(status),
                goal_satisfied_final=sum(status), goal_satisfied_max=goal_max, n_infer=n_infer), frames


def main(args):
    out_root = os.environ["ROBOELO_OUT"]
    suites = _suite_specs(os.environ.get("ROBOELO_SUITES", " ".join(SUITES)))
    num_trials = int(os.environ.get("ROBOELO_NUM_TRIALS", "50"))
    task_ids_opt, init_ids_opt = _int_list("ROBOELO_TASK_IDS"), _int_list("ROBOELO_INIT_IDS")
    num_steps_wait = int(os.environ.get("ROBOELO_NUM_STEPS_WAIT", "10"))
    episode_lengths = {name: int(n) for name, _, n in (
        item.partition("=") for item in re.split(r"[,\s]+", os.environ.get("ROBOELO_EPISODE_LENGTHS", "")) if item)}
    videos = os.environ.get("ROBOELO_VIDEOS", "failures")
    retries = int(os.environ.get("ROBOELO_RETRIES", "2"))
    model_label = os.environ.get("ROBOELO_MODEL", args.client)
    label = os.environ.get("ROBOELO_CHECKPOINT", args.checkpoint)

    np.random.seed(args.seed)  # once, as the official script
    client = make_client(args)
    print(f"[setup] client {args.client} ({type(client).__name__}) at {args.host}:{args.port}, "
          f"action_chunk_size {getattr(client, 'action_chunk_size', '?')}, unnorm_key {args.unnorm_key}", flush=True)
    n_missing = 0
    for suite_name, suite_task_ids in suites:
        max_steps = episode_lengths.get(suite_name, MAX_STEPS[suite_name])
        suite = benchmark.get_benchmark_dict()[suite_name]()
        log = EpisodeLog(out_root, suite_name, model_label, label)
        task_ids = task_ids_opt if task_ids_opt is not None else list(range(suite.n_tasks))
        if suite_task_ids is not None:
            task_ids = [t for t in task_ids if t in suite_task_ids]
        init_ids = init_ids_opt if init_ids_opt is not None else list(range(num_trials))
        print(f"[setup] {suite_name}: max steps {max_steps}, wait {num_steps_wait}, seed {args.seed}, "
              f"tasks {task_ids}, {len(log.done)} rollouts already recorded", flush=True)
        for task_id in task_ids:
            todo = log.todo(task_id, init_ids)
            if not todo:
                continue
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=LIBERO_ENV_RESOLUTION,
                                     camera_widths=LIBERO_ENV_RESOLUTION)
            env.seed(args.seed)
            for init_id in todo:
                for attempt in range(1, retries + 2):
                    t0 = time.time()
                    try:
                        res, frames = rollout(env, client, args.client, task.language, init_states[init_id],
                                              max_steps, num_steps_wait)
                    except Exception as exc:  # never record an exception as a task failure
                        log.error(task_id, init_id, attempt, exc)
                        traceback.print_exc()
                        if isinstance(exc, (ConnectionError, OSError)) or "ConnectionClosed" in type(exc).__name__:
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
            env.close()
        missing = log.missing(task_ids, init_ids)
        n_missing += len(missing)
        print(f"[{suite_name}] done: new {log.n_new}, exceptions {log.n_err}, missing {len(missing)}", flush=True)
    sys.stdout.flush()
    os._exit(1 if n_missing else 0)  # skip interpreter teardown: EGL contexts can hang or crash on exit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="starVLA / InternVLA-M1 / ABot-M0 checkout")
    ap.add_argument("--client", required=True, choices=sorted(CLIENT_MODULES))
    ap.add_argument("--checkpoint", default="", help=".pt checkpoint (its folder has dataset_statistics.json)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--unnorm-key", default=None)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--replan-steps", type=int, default=8, help="internvla_a15 only (its script's default)")
    ap.add_argument("--seed", type=int, default=7)
    main(ap.parse_args())

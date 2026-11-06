#!/usr/bin/env python
"""NORA on LIBERO, one CSV row per rollout (same record format as the other RoboElo LIBERO runners).

The per-step logic is declare-lab/nora's experiments/libero/run_libero_eval_nora.py. That script imports
libero_utils / robot_utils, which are empty files in the NORA repository: it runs inside OpenVLA's code base, so
OpenVLA's experiments/robot/libero/libero_utils.py and experiments/robot/robot_utils.py are used here
(get_libero_env with 256 px cameras, get_libero_image at the OpenVLA resize size 224, gripper normalize/invert).
The model is NORA's own Nora class (experiments/libero/nora_utils.py: Qwen2.5-VL-3B with FAST action tokens,
greedy decoding, the per-suite LIBERO normalization stats written in it). Settings as in the script: one fine-tuned
checkpoint per suite (declare-lab/nora-finetuned-libero-<suite>), 10 no-op steps, at most 500 policy queries,
gripper snapped to +1 when >= 0, seed 7. The script itself is written for the NORA-Long checkpoints
(time_horizon = 5); these are the single-step NORA checkpoints, so the FAST tokenizer keeps the Nora class's
default time_horizon = 1, and each query is one environment step.
What this runner adds: one row per finished rollout (resumable), partial progress from the goal predicates, and
an exception is recorded separately and the rollout run again (the official loop ends the episode as a failure).

Started by run_lerobot_libero_supervisor.sh (WORKER_SCRIPT=<this file>) with the same ROBOELO_* variables as
run_lerobot_libero.py. Command-line options: --checkpoint-pattern (with {suite}), --time-horizon, --seed.
"""
import argparse
import os
import re
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = os.path.expanduser("~/roboelo")
# LIBERO's stored initial states are pickled numpy arrays, which torch >= 2.6 refuses to load by default.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, os.path.join(ROOT, "third_party", "LIBERO"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "openvla"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from roboelo_libero_common import SUITES, EpisodeLog, goal_status  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "third_party", "nora", "experiments", "libero"))
from nora_utils import Nora  # noqa: E402

MAX_QUERIES = 500  # the script's max_steps for every suite; counts policy queries (one per chunk)


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


def rollout(env, model, suite_name, task_description, init_state, max_queries, num_steps_wait, resize_size):
    """One trial, as the official eval_libero loop runs it (without its catch-all exception handler)."""
    env.reset()
    obs = env.set_init_state(init_state)
    status = goal_status(env)
    goal_max, frames = sum(status), []
    t = steps = n_infer = 0
    done = False
    while t < max_queries + num_steps_wait:
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue
        img = get_libero_image(obs, resize_size)
        frames.append(img)
        action = model.inference(img, task_description, suite_name)
        n_infer += 1
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        action[..., -1] = np.where(action[..., -1] >= 0.0, 1.0, action[..., -1])
        for i in range(len(action)):
            obs, _, done, _ = env.step(action[i].tolist())
            steps += 1
            status = goal_status(env)
            goal_max = max(goal_max, sum(status))
        if done:  # as the official loop: checked after the whole chunk
            break
        t += 1
    return dict(success=bool(done), steps=steps, goal_total=len(status), goal_satisfied_final=sum(status),
                goal_satisfied_max=goal_max, n_infer=n_infer), frames


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
    model_label = os.environ.get("ROBOELO_MODEL", "nora")
    resize_size = get_image_resize_size(SimpleNamespace(model_family="openvla"))

    set_seed_everywhere(args.seed)  # once, as the official script
    n_missing = 0
    for suite_name, suite_task_ids in suites:
        short = suite_name.replace("libero_", "")
        ckpt = args.checkpoint_pattern.format(suite=short)
        model = Nora(model_path=ckpt)
        model.fast_tokenizer.time_horizon = args.time_horizon
        max_queries = episode_lengths.get(suite_name, MAX_QUERIES)
        suite = benchmark.get_benchmark_dict()[suite_name]()
        label = os.environ.get("ROBOELO_CHECKPOINT", "declare-lab/nora-finetuned-libero-{suite}").format(suite=short)
        log = EpisodeLog(out_root, suite_name, model_label, label)
        task_ids = task_ids_opt if task_ids_opt is not None else list(range(suite.n_tasks))
        if suite_task_ids is not None:
            task_ids = [t for t in task_ids if t in suite_task_ids]
        init_ids = init_ids_opt if init_ids_opt is not None else list(range(num_trials))
        print(f"[setup] {suite_name}: checkpoint {ckpt}, time_horizon {model.fast_tokenizer.time_horizon}, "
              f"resize {resize_size}, max queries {max_queries}, wait {num_steps_wait}, seed {args.seed}, "
              f"tasks {task_ids}, {len(log.done)} rollouts already recorded", flush=True)
        for task_id in task_ids:
            todo = log.todo(task_id, init_ids)
            if not todo:
                continue
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, "openvla", resolution=256)
            for init_id in todo:
                for attempt in range(1, retries + 2):
                    t0 = time.time()
                    try:
                        res, frames = rollout(env, model, suite_name, task_description, init_states[init_id],
                                              max_queries, num_steps_wait, resize_size)
                    except Exception as exc:  # never record an exception as a task failure
                        log.error(task_id, init_id, attempt, exc)
                        traceback.print_exc()
                        if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
                            print("CUDA error: this process cannot continue; exiting so it gets restarted", flush=True)
                            os._exit(3)
                        continue
                    video = ""
                    if videos == "all" or (videos == "failures" and not res["success"]):
                        video = log.video_path(task_id, init_id, res["success"])
                        imageio.mimwrite(video, [np.asarray(f) for f in frames], fps=20)
                    log.write(task_id=task_id, task_name=task.language, bddl_file=task.bddl_file,
                              init_state_id=init_id, seed=args.seed, max_steps=max_queries * args.time_horizon,
                              wall_time_s=time.time() - t0, video=video, **res)
                    break
            env.close()
        missing = log.missing(task_ids, init_ids)
        n_missing += len(missing)
        print(f"[{suite_name}] done: new {log.n_new}, exceptions {log.n_err}, missing {len(missing)}", flush=True)
        del model
        torch.cuda.empty_cache()
    sys.stdout.flush()
    os._exit(1 if n_missing else 0)  # skip interpreter teardown: EGL contexts can hang or crash on exit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-pattern",
                    default=os.path.join(ROOT, "checkpoints", "nora-finetuned-libero-{suite}"))
    ap.add_argument("--time-horizon", type=int, default=1, help="actions per FAST decode (5 for NORA-Long)")
    ap.add_argument("--seed", type=int, default=7)
    main(ap.parse_args())

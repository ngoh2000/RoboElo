#!/usr/bin/env python
"""MolmoAct on LIBERO, one CSV row per rollout (same record format as the other RoboElo LIBERO runners).

The per-step logic is allenai/molmoact's experiments/libero/run_libero_eval.py, the Hugging Face (non-vLLM)
evaluation; its step() (prompt, 90% center crop of both camera images, generation of depth tokens, trace and
action, action parsing with unnorm_key libero_<suite>_no_noops_modified), gripper functions and LIBERO utilities are
imported from the repository. Settings as there: one checkpoint per suite (allenai/MolmoAct-7B-D-LIBERO-<Suite>-0812),
bfloat16, agentview + wrist images at 256 px resized to the script's size, 10 no-op steps, step limits
220 / 280 / 300 / 520, the whole predicted action matrix executed before the next query, a step whose output cannot
be parsed executes one zero action (as in the script), seed 7 at the start of every task.
The repository's README replicates its results with vLLM 0.8.5, which has no kernels for the RTX 5090 (torch 2.6
+ cu124); the script used here is the one the README offers "with only huggingface".
What this runner adds: one row per finished rollout (resumable), partial progress from the goal predicates; a CUDA
error is not treated like an unparsable output but recorded, and the rollout run again.

Started by run_lerobot_libero_supervisor.sh (WORKER_SCRIPT=<this file>) with the same ROBOELO_* variables as
run_lerobot_libero.py. Command-line options: --checkpoint-pattern (with {Suite}), --seed.
"""
import argparse
import ast
import os
import re
import sys
import time
import traceback

ROOT = os.path.expanduser("~/roboelo")
EVAL_DIR = os.path.join(ROOT, "third_party", "molmoact", "experiments", "libero")
# LIBERO's stored initial states are pickled numpy arrays, which torch >= 2.6 refuses to load by default.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, os.path.join(ROOT, "third_party", "LIBERO"))
sys.path.insert(0, EVAL_DIR)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image, get_libero_wrist_image  # noqa: E402
from robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402
from run_libero_eval import invert_gripper_action, normalize_gripper_action, step  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402
from roboelo_libero_common import SUITES, EpisodeLog, goal_status  # noqa: E402

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
SUITE_NAMES = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}


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


def rollout(env, model, processor, model_family, task_description, init_state, max_steps, num_steps_wait,
            resize_size, unnorm_key):
    """One trial, as the official eval_libero loop runs it."""
    last_gripper_state = -1
    env.reset()
    obs = env.set_init_state(init_state)
    status = goal_status(env)
    goal_max, frames = sum(status), []
    t = steps = n_infer = n_unparsed = 0
    done = outer_done = False
    while t < max_steps + num_steps_wait and not outer_done:
        if t < num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(model_family))
            t += 1
            continue
        img = get_libero_image(obs, resize_size)
        wrist_img = get_libero_wrist_image(obs, resize_size)
        frames.append(img)
        wait = False
        n_infer += 1
        try:
            action_matrix, _, _ = step(img, wrist_img, task_description, model, processor, unnorm_key)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
                raise
            action_matrix = np.zeros((1, 7), dtype=float)  # as the official loop: one zero action
            action_matrix[:, -1] = last_gripper_state
            wait = True
            n_unparsed += 1
        action_num = 0
        for single_action in action_matrix:
            if isinstance(single_action, str):
                single_action = ast.literal_eval(single_action)
            single_action = normalize_gripper_action(single_action, binarize=True)
            single_action = invert_gripper_action(single_action)
            obs, _, done, _ = env.step(single_action)
            steps += 1
            status = goal_status(env)
            goal_max = max(goal_max, sum(status))
            action_num += 1
            if done:
                outer_done = True
                break
        if wait:
            action_num = 1
        t += action_num
        if done:
            break
    if n_unparsed:
        print(f"  {n_unparsed} of {n_infer} queries gave no parsable action", flush=True)
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
    model_label = os.environ.get("ROBOELO_MODEL", "molmoact")
    resize_size = get_image_resize_size()

    set_seed_everywhere(args.seed)
    n_missing = 0
    for suite_name, suite_task_ids in suites:
        name = SUITE_NAMES[suite_name]
        ckpt = args.checkpoint_pattern.format(Suite=name)
        processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True, torch_dtype="bfloat16",
                                                  device_map="auto", padding_side="left")
        model = AutoModelForImageTextToText.from_pretrained(ckpt, trust_remote_code=True, torch_dtype="bfloat16",
                                                            device_map="auto")
        model_family = f"allenai-MolmoAct-7B-D-LIBERO-{name}-0812"  # the script passes ckpt.replace("/", "-")
        unnorm_key = f"{suite_name}_no_noops_modified"
        max_steps = episode_lengths.get(suite_name, MAX_STEPS[suite_name])
        suite = benchmark.get_benchmark_dict()[suite_name]()
        label = os.environ.get("ROBOELO_CHECKPOINT", "allenai/MolmoAct-7B-D-LIBERO-{Suite}-0812").format(Suite=name)
        log = EpisodeLog(out_root, suite_name, model_label, label)
        task_ids = task_ids_opt if task_ids_opt is not None else list(range(suite.n_tasks))
        if suite_task_ids is not None:
            task_ids = [t for t in task_ids if t in suite_task_ids]
        init_ids = init_ids_opt if init_ids_opt is not None else list(range(num_trials))
        print(f"[setup] {suite_name}: checkpoint {ckpt}, unnorm_key {unnorm_key}, resize {resize_size}, "
              f"max_steps {max_steps}, wait {num_steps_wait}, seed {args.seed}, tasks {task_ids}, "
              f"{len(log.done)} rollouts already recorded", flush=True)
        for task_id in task_ids:
            todo = log.todo(task_id, init_ids)
            if not todo:
                continue
            set_seed_everywhere(args.seed)  # as eval_libero, at the start of every task
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, model_family, resolution=256)
            for init_id in todo:
                for attempt in range(1, retries + 2):
                    t0 = time.time()
                    try:
                        res, frames = rollout(env, model, processor, model_family, task_description,
                                              init_states[init_id], max_steps, num_steps_wait, resize_size,
                                              unnorm_key)
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
                              init_state_id=init_id, seed=args.seed, max_steps=max_steps,
                              wall_time_s=time.time() - t0, video=video, **res)
                    break
            env.close()
        missing = log.missing(task_ids, init_ids)
        n_missing += len(missing)
        print(f"[{suite_name}] done: new {log.n_new}, exceptions {log.n_err}, missing {len(missing)}", flush=True)
        del model, processor
        torch.cuda.empty_cache()
    sys.stdout.flush()
    os._exit(1 if n_missing else 0)  # skip interpreter teardown: EGL contexts can hang or crash on exit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-pattern",
                    default=os.path.join(ROOT, "checkpoints", "MolmoAct-7B-D-LIBERO-{Suite}-0812"))
    ap.add_argument("--seed", type=int, default=7)
    main(ap.parse_args())

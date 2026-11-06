#!/usr/bin/env python
"""VLA-Adapter on LIBERO, one CSV row per rollout (same record format as the other RoboElo LIBERO runners).

Model set-up, observation and action processing come from VLA-Adapter's experiments/robot/libero/run_libero_eval.py
itself (GenerateConfig defaults, initialize_model, check_unnorm_key, prepare_observation, process_action,
TASK_MAX_STEPS); the per-step loop is its run_episode: Qwen2.5-0.5B mini-VLM, 2 images + proprio, L1-regression
policy head, chunks of 8 actions executed open-loop, 256 px cameras, 10 no-op steps, step limits
220 / 280 / 300 / 520, seed 7. One checkpoint per suite (VLA-Adapter/LIBERO-{Spatial,Object,Goal,Long}, the
original release of 2025-09-12 behind the paper's numbers), so use_pro_version=False (the code's default, True,
is for the later -Pro checkpoints).
What this runner adds: one row per finished rollout (resumable), partial progress from the goal predicates, and
an exception is recorded separately and the rollout run again (the official loop counts it as a failure).

Started by run_lerobot_libero_supervisor.sh (WORKER_SCRIPT=<this file>) with the same ROBOELO_* variables as
run_lerobot_libero.py. The word "libero" must stay in this file's name: prismatic/vla/constants.py picks the
LIBERO constants (chunk 8, action dim 7, proprio dim 8) by looking for it on the command line.
"""
import argparse
import os
import re
import sys
import time
import traceback
from collections import deque

ROOT = os.path.expanduser("~/roboelo")
REPO = os.path.join(ROOT, "third_party", "VLA-Adapter")
# LIBERO's stored initial states are pickled numpy arrays, which torch >= 2.6 refuses to load by default.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, os.path.join(ROOT, "third_party", "LIBERO"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio  # noqa: E402
import numpy as np  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env  # noqa: E402
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    TASK_MAX_STEPS,
    GenerateConfig,
    initialize_model,
    prepare_observation,
    process_action,
)
from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere  # noqa: E402
from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # noqa: E402
import torch  # noqa: E402
from roboelo_libero_common import SUITES, EpisodeLog, goal_status  # noqa: E402


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


def rollout(cfg, env, parts, task_description, init_state, max_steps, resize_size):
    """One trial, as run_libero_eval.run_episode runs it (without its catch-all exception handler)."""
    model, action_head, proprio_projector, noisy_action_projector, processor = parts
    env.reset()
    obs = env.set_init_state(init_state)
    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    status = goal_status(env)
    goal_max, frames = sum(status), []
    t = steps = n_infer = 0
    done = False
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue
        observation, img = prepare_observation(obs, resize_size)
        frames.append(img)
        if len(action_queue) == 0:
            actions = get_action(cfg, model, observation, task_description, processor=processor,
                                 action_head=action_head, proprio_projector=proprio_projector,
                                 noisy_action_projector=noisy_action_projector, use_film=cfg.use_film,
                                 use_minivlm=cfg.use_minivlm)
            action_queue.extend(actions)
            n_infer += 1
        action = process_action(action_queue.popleft(), cfg.model_family)
        obs, _, done, _ = env.step(action.tolist())
        steps += 1
        status = goal_status(env)
        goal_max = max(goal_max, sum(status))
        if done:
            break
        t += 1
    return dict(success=bool(done), steps=steps, goal_total=len(status), goal_satisfied_final=sum(status),
                goal_satisfied_max=goal_max, n_infer=n_infer), frames


def main(args):
    out_root = os.environ["ROBOELO_OUT"]
    suites = _suite_specs(os.environ.get("ROBOELO_SUITES", " ".join(SUITES)))
    num_trials = int(os.environ.get("ROBOELO_NUM_TRIALS", "50"))
    task_ids_opt, init_ids_opt = _int_list("ROBOELO_TASK_IDS"), _int_list("ROBOELO_INIT_IDS")
    episode_lengths = {name: int(n) for name, _, n in (
        item.partition("=") for item in re.split(r"[,\s]+", os.environ.get("ROBOELO_EPISODE_LENGTHS", "")) if item)}
    videos = os.environ.get("ROBOELO_VIDEOS", "failures")
    retries = int(os.environ.get("ROBOELO_RETRIES", "2"))
    model_label = os.environ.get("ROBOELO_MODEL", "vla_adapter")
    num_steps_wait = int(os.environ.get("ROBOELO_NUM_STEPS_WAIT", "10"))

    set_seed_everywhere(args.seed)  # once, as the official script
    n_missing = 0
    for suite_name, suite_task_ids in suites:
        name = SUITE_NAMES[suite_name]
        ckpt = args.checkpoint_pattern.format(suite=name.lower())
        cfg = GenerateConfig(pretrained_checkpoint=ckpt, task_suite_name=suite_name, seed=args.seed,
                             num_steps_wait=num_steps_wait, use_pro_version=False, save_version="vla-adapter")
        parts = initialize_model(cfg)  # also sets cfg.unnorm_key (check_unnorm_key)
        resize_size = get_image_resize_size(cfg)
        print(f"[setup] {suite_name}: checkpoint {ckpt}; NUM_ACTIONS_CHUNK {NUM_ACTIONS_CHUNK}, "
              f"num_open_loop_steps {cfg.num_open_loop_steps}, images {cfg.num_images_in_input}, "
              f"proprio {cfg.use_proprio}, minivlm {cfg.use_minivlm}, pro {cfg.use_pro_version}, "
              f"version {cfg.save_version}, center_crop {cfg.center_crop}, resize {resize_size}, "
              f"env_img_res {cfg.env_img_res}, wait {cfg.num_steps_wait}, seed {cfg.seed}", flush=True)
        assert cfg.num_open_loop_steps == NUM_ACTIONS_CHUNK, "the LIBERO constants were not selected"
        ckpt_label = os.environ.get("ROBOELO_CHECKPOINT", "VLA-Adapter/LIBERO-{suite}").format(suite=name)
        max_steps = episode_lengths.get(suite_name, TASK_MAX_STEPS[suite_name])
        suite = benchmark.get_benchmark_dict()[suite_name]()
        log = EpisodeLog(out_root, suite_name, model_label, ckpt_label)
        task_ids = task_ids_opt if task_ids_opt is not None else list(range(suite.n_tasks))
        if suite_task_ids is not None:
            task_ids = [t for t in task_ids if t in suite_task_ids]
        init_ids = init_ids_opt if init_ids_opt is not None else list(range(num_trials))
        print(f"[setup] {suite_name}: unnorm_key {cfg.unnorm_key}, max_steps {max_steps}, tasks {task_ids}, "
              f"{len(log.done)} rollouts already recorded", flush=True)
        for task_id in task_ids:
            todo = log.todo(task_id, init_ids)
            if not todo:
                continue
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            for init_id in todo:
                for attempt in range(1, retries + 2):
                    t0 = time.time()
                    try:
                        res, frames = rollout(cfg, env, parts, task_description, init_states[init_id], max_steps,
                                              resize_size)
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
                              init_state_id=init_id, seed=cfg.seed, max_steps=max_steps,
                              wall_time_s=time.time() - t0, video=video, **res)
                    break
            env.close()
        missing = log.missing(task_ids, init_ids)
        n_missing += len(missing)
        print(f"[{suite_name}] done: new {log.n_new}, exceptions {log.n_err}, missing {len(missing)}", flush=True)
        del parts
        torch.cuda.empty_cache()
    sys.stdout.flush()
    os._exit(1 if n_missing else 0)  # skip interpreter teardown: EGL contexts can hang or crash on exit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-pattern", default=os.path.join(ROOT, "checkpoints", "vla-adapter-libero-{suite}"))
    ap.add_argument("--seed", type=int, default=7)
    main(ap.parse_args())

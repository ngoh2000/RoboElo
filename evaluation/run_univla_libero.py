#!/usr/bin/env python
"""UniVLA on LIBERO, one CSV row per rollout (same record format as the other RoboElo LIBERO runners).

The per-step logic is UniVLA's experiments/robot/libero/run_libero_eval.py (its GenerateConfig defaults and
ActionDecoder are imported from it): one fine-tuned checkpoint per suite (qwbu/univla-7b-224-sft-libero/
univla-libero-<suite>, each with its action_decoder.pt), a model query at every step, latent-action history of the
last step in the prompt, action decoder with window 12 and temporal ensembling, center crop, 256 px cameras,
10 no-op steps, step limits 240 / 300 / 320 / 550, gripper normalized and inverted, seed 7.
What this runner adds: one row per finished rollout (resumable), partial progress from the goal predicates, and
an exception is recorded separately and the rollout run again (the official loop ends the episode as a failure).

Started by run_lerobot_libero_supervisor.sh (WORKER_SCRIPT=<this file>) with the same ROBOELO_* variables as
run_lerobot_libero.py. Command-line options: --checkpoint-pattern (with {suite}), --seed.
"""
import argparse
import os
import re
import sys
import time
import traceback

ROOT = os.path.expanduser("~/roboelo")
UNIVLA = os.path.join(ROOT, "third_party", "UniVLA")
# LIBERO's stored initial states are pickled numpy arrays, which torch >= 2.6 refuses to load by default.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, os.path.join(ROOT, "third_party", "LIBERO"))
sys.path.insert(0, UNIVLA)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import ActionDecoder, GenerateConfig  # noqa: E402
from experiments.robot.openvla_utils import get_processor  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    get_image_resize_size,
    get_latent_action,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from roboelo_libero_common import SUITES, EpisodeLog, goal_status  # noqa: E402

MAX_STEPS = {"libero_spatial": 240, "libero_object": 300, "libero_goal": 320, "libero_10": 550}
LATENT_ACTION_DETOKENIZE = [f"<ACT_{i}>" for i in range(32)]


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


def rollout(cfg, env, model, processor, action_decoder, task_description, init_state, max_steps, resize_size):
    """One trial, as the official eval_libero loop runs it (without its catch-all exception handler)."""
    env.reset()
    action_decoder.reset()
    prev_hist_action = [""]
    obs = env.set_init_state(init_state)
    status = goal_status(env)
    goal_max, frames = sum(status), []
    t = steps = 0
    done = False
    action_norm_stats = model.get_action_stats(cfg.unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue
        img = get_libero_image(obs, resize_size)
        frames.append(img)
        observation = {
            "full_image": img,
            "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                     obs["robot0_gripper_qpos"])),
        }
        latent_action, visual_embed, generated_ids = get_latent_action(
            cfg, model, observation, task_description, processor=processor, hist_action=prev_hist_action[-1])
        hist_action = ""
        for latent_action_ids in generated_ids[0]:
            hist_action += LATENT_ACTION_DETOKENIZE[latent_action_ids.item() - 32001]
        prev_hist_action.append(hist_action)
        action = action_decoder(latent_action, visual_embed, mask, action_low, action_high)
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        obs, _, done, _ = env.step(action.tolist())
        steps += 1
        status = goal_status(env)
        goal_max = max(goal_max, sum(status))
        if done:
            break
        t += 1
    return dict(success=bool(done), steps=steps, goal_total=len(status), goal_satisfied_final=sum(status),
                goal_satisfied_max=goal_max, n_infer=steps), frames


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
    model_label = os.environ.get("ROBOELO_MODEL", "univla")

    set_seed_everywhere(args.seed)  # once, as the official script
    n_missing = 0
    for suite_name, suite_task_ids in suites:
        ckpt = args.checkpoint_pattern.format(suite=suite_name.replace("_", "-"))
        cfg = GenerateConfig(pretrained_checkpoint=ckpt, action_decoder_path=os.path.join(ckpt, "action_decoder.pt"),
                             task_suite_name=suite_name, num_steps_wait=num_steps_wait, seed=args.seed)
        cfg.unnorm_key = suite_name
        action_decoder = ActionDecoder(cfg.window_size)
        action_decoder.net.load_state_dict(torch.load(cfg.action_decoder_path))
        action_decoder.eval().cuda()
        model = get_model(cfg)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"action un-normalization key {cfg.unnorm_key} not in {ckpt}"
        processor = get_processor(cfg)
        resize_size = get_image_resize_size(cfg)
        max_steps = episode_lengths.get(suite_name, MAX_STEPS[suite_name])

        suite = benchmark.get_benchmark_dict()[suite_name]()
        label = os.environ.get("ROBOELO_CHECKPOINT", "qwbu/univla-7b-224-sft-libero/univla-{suite}").format(
            suite=suite_name.replace("_", "-"))
        log = EpisodeLog(out_root, suite_name, model_label, label)
        task_ids = task_ids_opt if task_ids_opt is not None else list(range(suite.n_tasks))
        if suite_task_ids is not None:
            task_ids = [t for t in task_ids if t in suite_task_ids]
        init_ids = init_ids_opt if init_ids_opt is not None else list(range(num_trials))
        print(f"[setup] {suite_name}: checkpoint {ckpt}, unnorm_key {cfg.unnorm_key}, window {cfg.window_size}, "
              f"center_crop {cfg.center_crop}, resize {resize_size}, max_steps {max_steps}, wait {cfg.num_steps_wait}, "
              f"seed {cfg.seed}, tasks {task_ids}, {len(log.done)} rollouts already recorded", flush=True)
        for task_id in task_ids:
            todo = log.todo(task_id, init_ids)
            if not todo:
                continue
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
            for init_id in todo:
                for attempt in range(1, retries + 2):
                    t0 = time.time()
                    try:
                        res, frames = rollout(cfg, env, model, processor, action_decoder, task_description,
                                              init_states[init_id], max_steps, resize_size)
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
        del model, action_decoder
        torch.cuda.empty_cache()
    sys.stdout.flush()
    os._exit(1 if n_missing else 0)  # skip interpreter teardown: EGL contexts can hang or crash on exit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-pattern",
                    default=os.path.join(ROOT, "checkpoints", "univla-7b-224-sft-libero", "univla-{suite}"))
    ap.add_argument("--seed", type=int, default=7)
    main(ap.parse_args())

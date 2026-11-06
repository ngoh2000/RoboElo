#!/usr/bin/env python
"""A LeRobot policy on RoboTwin 2.0 (clean / "easy" setting), one CSV row per episode, resumable.

The command line is lerobot-eval's (--policy.path, --policy.*, --env.type=robotwin, --env.action_mode ...). Policy,
pre/post-processors, LeRobot's RoboTwinEnv and the per-step loop (lerobot_eval.rollout) are LeRobot's own, set up as
lerobot_eval.eval_main does (same as run_lerobot_libero.py). What differs from lerobot-eval:

  * episode k of a task is RoboTwin's setup_demo(seed=k, is_test=True), k = 0..N-1, whatever happened before (the
    env's own step() resets itself after an episode and moves on to the next index; that reset is skipped here);
  * the step limit per task is RoboTwin's own evaluation limit (task_config/_eval_step_limit.yml) when
    ROBOELO_STEP_LIMITS=official, else --env.episode_length (LeRobot's default 1200 for every task);
  * one row per finished episode, written immediately; finished episodes are skipped when re-run.

Environment: ROBOELO_OUT (required; <out>/episodes.csv), ROBOELO_TASKS (comma separated; default LeRobot's
ROBOTWIN_TASKS without open_laptop, whose check_success is broken upstream), ROBOELO_TASK_SHARD k/N (every N-th task,
so that several processes share the list), ROBOELO_NUM_EPISODES (default 100), ROBOELO_STEP_LIMITS (official |
config), ROBOELO_VIDEOS (failures | all | none), ROBOELO_RETRIES (default 2), ROBOELO_MODEL, ROBOELO_CHECKPOINT.
Run from RoboTwin's root with it on PYTHONPATH.
"""
import csv
import datetime
import fcntl
import os
import sys
import time
import traceback
from contextlib import nullcontext

import gymnasium as gym
import imageio
import numpy as np
import torch
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import make_env_pre_post_processors
from lerobot.envs.robotwin import ROBOTWIN_TASKS, RoboTwinEnv
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

FIELDS = ["model", "checkpoint", "benchmark", "setting", "task", "episode_index", "seed", "success", "steps",
          "max_steps", "n_infer", "instruction", "wall_time_s", "finished_at", "video"]


class LoggedRoboTwinEnv(RoboTwinEnv):
    """RoboTwinEnv whose step() does not reset itself at the end of an episode, keeping the frames."""

    def reset(self, seed=None, **kwargs):
        self.frames = []
        observation, info = super().reset(seed=seed, **kwargs)
        self.frames.append(observation["pixels"].get("head_camera"))
        return observation, info

    def step(self, action):
        reset, self.reset = self.reset, (lambda *a, **k: None)  # the parent calls self.reset() on termination
        try:
            observation, reward, terminated, truncated, info = super().step(action)
        finally:
            self.reset = reset
        self.frames.append(observation["pixels"].get("head_camera"))
        return observation, reward, terminated, truncated, info


def official_step_limits():
    import yaml

    for rel in ("task_config/_eval_step_limit.yml", "envs/task_config/_eval_step_limit.yml"):
        if os.path.exists(rel):
            return {k: int(v) for k, v in yaml.safe_load(open(rel)).items()}
    raise FileNotFoundError("RoboTwin's _eval_step_limit.yml not found (run from RoboTwin's root)")


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    out = os.environ["ROBOELO_OUT"]
    os.makedirs(os.path.join(out, "videos"), exist_ok=True)
    # the supervisor always sets ROBOELO_TASKS, empty when no list is given: empty means the default list
    tasks = [t for t in (os.environ.get("ROBOELO_TASKS") or ",".join(t for t in ROBOTWIN_TASKS if t != "open_laptop")).split(",") if t]
    k, n = (int(x) for x in os.environ.get("ROBOELO_TASK_SHARD", "0/1").split("/"))
    tasks = tasks[k::n]
    num_episodes = int(os.environ.get("ROBOELO_NUM_EPISODES", "100"))
    limits = official_step_limits() if os.environ.get("ROBOELO_STEP_LIMITS", "official") == "official" else {}
    videos = os.environ.get("ROBOELO_VIDEOS", "failures")
    retries = int(os.environ.get("ROBOELO_RETRIES", "2"))
    model = os.environ.get("ROBOELO_MODEL", cfg.policy.type)
    checkpoint = os.environ.get("ROBOELO_CHECKPOINT", str(cfg.policy.pretrained_path))
    ep_csv, err_csv = os.path.join(out, "episodes.csv"), os.path.join(out, "errors.csv")

    def append(path, fields, row):
        with open(path, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            w = csv.DictWriter(f, fieldnames=fields)
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)

    def recorded():
        if not os.path.exists(ep_csv):
            return set()
        return {(r["task"], int(r["episode_index"])) for r in csv.DictReader(open(ep_csv))}

    # ---- same set-up as lerobot_eval.eval_main ----
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                                "rename_observations_processor": {"rename_map": cfg.rename_map}})
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    cams = [c.strip() for c in cfg.env.camera_names.split(",") if c.strip()]
    done = recorded()
    print(f"[setup] model={model} checkpoint={checkpoint} action_mode={cfg.env.action_mode} tasks={len(tasks)} "
          f"episodes={num_episodes} step_limits={'official' if limits else cfg.env.episode_length}; "
          f"{len(done)} episodes already recorded", flush=True)

    n_missing = 0
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        for task in tasks:
            todo = [e for e in range(num_episodes) if (task, e) not in done]
            if not todo:
                continue
            limit = limits.get(task, cfg.env.episode_length)
            venv = gym.vector.SyncVectorEnv([lambda task=task, limit=limit: LoggedRoboTwinEnv(
                task_name=task, episode_index=0, camera_names=cams, observation_height=cfg.env.observation_height,
                observation_width=cfg.env.observation_width, episode_length=limit, action_mode=cfg.env.action_mode)])
            inner = venv.envs[0]
            for e in todo:
                for attempt in range(1, retries + 2):
                    t0 = time.time()
                    try:
                        inner.episode_index = e  # setup_demo(seed=e)
                        data = rollout(env=venv, policy=policy, env_preprocessor=env_preprocessor,
                                       env_postprocessor=env_postprocessor, preprocessor=preprocessor,
                                       postprocessor=postprocessor, seeds=None)
                    except Exception as exc:  # never record an exception as a task failure
                        append(err_csv, ["task", "episode_index", "attempt", "time", "error"],
                               dict(task=task, episode_index=e, attempt=attempt, time=datetime.datetime.now().isoformat(),
                                    error=f"{type(exc).__name__}: {exc}"[:500]))
                        traceback.print_exc()
                        if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
                            print("CUDA error: exiting so that the process gets restarted", flush=True)
                            os._exit(3)
                        continue
                    done_index = int(torch.argmax(data["done"][0].to(int)))
                    steps = done_index + 1
                    success = bool(data["success"][0, : done_index + 2].any())
                    video = ""
                    if videos == "all" or (videos == "failures" and not success):
                        video = os.path.join(out, "videos", f"{task}_ep{e:03d}_{'success' if success else 'failure'}.mp4")
                        frames = [np.ascontiguousarray(f) for f in inner.frames if f is not None]
                        if frames:
                            imageio.mimwrite(video, frames, fps=25)
                    n_action_steps = int(getattr(policy.config, "n_action_steps", 1) or 1)
                    append(ep_csv, FIELDS, dict(
                        model=model, checkpoint=checkpoint, benchmark="robotwin2", setting="easy", task=task,
                        episode_index=e, seed=e, success=int(success), steps=steps, max_steps=limit,
                        n_infer=-(-steps // n_action_steps), instruction=inner.task_description,
                        wall_time_s=round(time.time() - t0, 1),
                        finished_at=datetime.datetime.now().isoformat(timespec="seconds"), video=video))
                    print(f"[{task}] episode {e}: success={int(success)} steps={steps}/{limit} ({time.time() - t0:.0f}s)",
                          flush=True)
                    break
            venv.close()
    rec = recorded()
    n_missing = sum(1 for t in tasks for e in range(num_episodes) if (t, e) not in rec)
    print(f"[done] missing {n_missing}", flush=True)
    sys.stdout.flush()
    os._exit(1 if n_missing else 0)


if __name__ == "__main__":
    init_logging()
    register_third_party_plugins()
    main()

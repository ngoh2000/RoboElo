#!/usr/bin/env python
"""A LeRobot policy on LIBERO, one CSV row per rollout.

The command line is lerobot-eval's (--policy.path, --policy.*, --env.type=libero, --seed ...). Policy,
pre/post-processors, the LIBERO env and the per-step loop are LeRobot's own (lerobot_eval.rollout,
envs.libero.LiberoEnv), set up exactly as lerobot_eval.eval_main does. What differs from lerobot-eval:

  * rollout k of a task starts from LIBERO's fixed initial state k (k = 0..49). lerobot-eval's env resets
    itself after a success and is reset again by the evaluation loop, so it skips an initial state after
    every successful rollout and never visits some of them;
  * the number of no-op steps after a reset can be set (ROBOELO_NUM_STEPS_WAIT). LeRobot's env class
    defaults to 10 and its config cannot set it; the MolmoAct2 authors report their results with 50;
  * one row per finished rollout, written immediately; finished rollouts are skipped when re-run.

Options that are not part of lerobot-eval come from the environment:
  ROBOELO_OUT              results root (required); a <suite>/ folder is created inside
  ROBOELO_SUITES           suites to run, space or comma separated (default: the four standard suites);
                           "libero_10:0-4" or "libero_10:0+2+5" limits a suite to some of its tasks, so
                           that several processes can share one suite
  ROBOELO_NUM_TRIALS       initial states 0..N-1 per task (default 50)
  ROBOELO_TASK_IDS         default: every task of the suite
  ROBOELO_INIT_IDS         default: 0..NUM_TRIALS-1
  ROBOELO_NUM_STEPS_WAIT   no-op steps after each reset (default 10)
  ROBOELO_EPISODE_LENGTHS  per-suite horizon, e.g. "libero_spatial=300,libero_10=600"; suites not listed use
                           --env.episode_length, or LeRobot's per-suite default when that is not set
  ROBOELO_VIDEOS           failures (default) | all | none
  ROBOELO_RETRIES          re-run a rollout that raised an exception (default 2)
  ROBOELO_MODEL, ROBOELO_CHECKPOINT   labels written to the CSV
  ROBOELO_LOAD_WEIGHTS_ON_CPU  1 = read model.safetensors on the CPU (lower GPU peak while loading)
A shorter horizon for debugging is lerobot-eval's own --env.episode_length.
"""
import math
import os
import re
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
from lerobot.envs.libero import LiberoEnv, _get_suite
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roboelo_libero_common import SUITES, EpisodeLog, goal_status  # noqa: E402


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


class LoggedLiberoEnv(LiberoEnv):
    """LeRobot's LiberoEnv with a chosen initial state per reset and per-step bookkeeping.

    step() is the parent's without its `self.reset()` on termination: the caller resets before every
    rollout anyway, and that automatic reset would consume the next initial state.
    """

    def _note(self, pixels, first=False):
        status = goal_status(self._env)
        if first:
            self.goal_total, self.goal_max, self.frames = len(status), 0, []
        self.goal_last = sum(status)
        self.goal_max = max(self.goal_max, self.goal_last)
        agentview = pixels[self.camera_name_mapping[self.camera_name[0]]]
        self.frames.append(agentview[::-1, ::-1])  # upright, as LiberoEnv.render() shows it

    def reset(self, seed=None, **kwargs):
        observation, info = super().reset(seed=seed, **kwargs)
        self._note(observation["pixels"], first=True)
        return observation, info

    def step(self, action):
        self._ensure_env()
        if action.ndim != 1:
            raise ValueError(f"Expected a 1-D action, got shape {action.shape}")
        raw_obs, reward, done, info = self._env.step(action)
        is_success = self._env.check_success()
        terminated = done or is_success
        info.update({"task": self.task, "task_id": self.task_id, "done": done, "is_success": is_success})
        observation = self._format_raw_obs(raw_obs)
        self._note(observation["pixels"])
        return observation, reward, terminated, False, info


class _EqualsAnything:
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    __hash__ = None


def set_policy_episode(policy, k):
    """Make a policy that seeds its sampler per episode (MolmoAct2's per_episode_seed) use episode index k.

    The policy counts episodes of a task itself, starting at 0 whenever the task text changes; here the
    index is tied to the initial state so that a resumed run draws the same noise for the same rollout.
    """
    if hasattr(policy, "_rollout_index_for_task"):
        policy._rollout_index_for_task = k - 1  # the policy adds one when the next episode starts
        policy._rollout_task_key = _EqualsAnything()


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    out_root = os.environ["ROBOELO_OUT"]
    suites = _suite_specs(os.environ.get("ROBOELO_SUITES", " ".join(SUITES)))
    num_trials = int(os.environ.get("ROBOELO_NUM_TRIALS", "50"))
    task_ids_opt, init_ids_opt = _int_list("ROBOELO_TASK_IDS"), _int_list("ROBOELO_INIT_IDS")
    num_steps_wait = int(os.environ.get("ROBOELO_NUM_STEPS_WAIT", "10"))
    episode_lengths = {name: int(n) for name, _, n in (
        item.partition("=") for item in re.split(r"[,\s]+", os.environ.get("ROBOELO_EPISODE_LENGTHS", "")) if item)}
    videos = os.environ.get("ROBOELO_VIDEOS", "failures")
    retries = int(os.environ.get("ROBOELO_RETRIES", "2"))
    model = os.environ.get("ROBOELO_MODEL", cfg.policy.type)
    checkpoint = os.environ.get("ROBOELO_CHECKPOINT", str(cfg.policy.pretrained_path))

    if os.environ.get("ROBOELO_LOAD_WEIGHTS_ON_CPU") == "1":
        # LeRobot reads model.safetensors straight onto the GPU next to the already built model, which doubles the
        # peak (FastWAM: ~24 GB of modules + 12 GB of weights > 32 GB). Read it on the CPU instead; load_state_dict
        # then copies into the GPU parameters, so the result is the same.
        from lerobot.policies.pretrained import PreTrainedPolicy

        _load = PreTrainedPolicy._load_as_safetensor.__func__
        PreTrainedPolicy._load_as_safetensor = classmethod(
            lambda cls, model, model_file, map_location, strict: _load(cls, model, model_file, "cpu", strict))

    # ---- same set-up as lerobot_eval.eval_main ----
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    gym_kwargs = {k: v for k, v in cfg.env.gym_kwargs.items() if k != "task_ids"}
    n_action_steps = int(getattr(policy.config, "n_action_steps", 1))
    print(f"[setup] model={model} checkpoint={checkpoint} device={device} suites={suites} trials={num_trials} "
          f"num_steps_wait={num_steps_wait} seed={cfg.seed} n_action_steps={n_action_steps} "
          f"episode_lengths={episode_lengths or cfg.env.episode_length}", flush=True)

    n_missing = 0
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        for suite_name, suite_task_ids in suites:
            suite = _get_suite(suite_name)
            log = EpisodeLog(out_root, suite_name, model, checkpoint)
            task_ids = task_ids_opt if task_ids_opt is not None else list(range(len(suite.tasks)))
            if suite_task_ids is not None:
                task_ids = [t for t in task_ids if t in suite_task_ids]
            init_ids = init_ids_opt if init_ids_opt is not None else list(range(num_trials))
            print(f"[{suite_name}] {len(log.done)} rollouts already recorded", flush=True)
            for task_id in task_ids:
                todo = log.todo(task_id, init_ids)
                if not todo:
                    continue
                task = suite.get_task(task_id)

                def make(task_id=task_id, suite=suite, suite_name=suite_name):
                    return LoggedLiberoEnv(
                        task_suite=suite, task_id=task_id, task_suite_name=suite_name,
                        camera_name=cfg.env.camera_name, init_states=True,
                        episode_length=episode_lengths.get(suite_name, cfg.env.episode_length),
                        episode_index=0, n_envs=1, control_mode=cfg.env.control_mode,
                        camera_name_mapping=cfg.env.camera_name_mapping, is_libero_plus=False,
                        num_steps_wait=num_steps_wait, **gym_kwargs)

                venv = gym.vector.SyncVectorEnv([make])  # what lerobot-eval builds for eval.batch_size=1
                inner = venv.envs[0]
                for init_id in todo:
                    for attempt in range(1, retries + 2):
                        t0 = time.time()
                        seed = None if cfg.seed is None else cfg.seed + init_id
                        try:
                            inner.init_state_id = init_id
                            set_policy_episode(policy, init_id)
                            data = rollout(
                                env=venv, policy=policy, env_preprocessor=env_preprocessor,
                                env_postprocessor=env_postprocessor, preprocessor=preprocessor,
                                postprocessor=postprocessor, seeds=None if seed is None else [seed])
                        except Exception as exc:  # never record an exception as a task failure
                            log.error(task_id, init_id, attempt, exc)
                            traceback.print_exc()
                            if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
                                print("CUDA error: this process cannot continue; exiting so it gets restarted", flush=True)
                                os._exit(3)
                            continue
                        # as lerobot_eval.eval_policy: everything up to the first done counts
                        done_index = int(torch.argmax(data["done"][0].to(int)))
                        steps = done_index + 1
                        success = bool(data["success"][0, : done_index + 2].any())
                        video = ""
                        if videos == "all" or (videos == "failures" and not success):
                            video = log.video_path(task_id, init_id, success)
                            imageio.mimwrite(video, [np.ascontiguousarray(f) for f in inner.frames], fps=20)
                        log.write(
                            task_id=task_id, task_name=task.language, bddl_file=task.bddl_file, init_state_id=init_id,
                            seed="" if seed is None else seed, success=success, steps=steps,
                            max_steps=inner._max_episode_steps, goal_total=inner.goal_total,
                            goal_satisfied_final=inner.goal_last, goal_satisfied_max=inner.goal_max,
                            n_infer=math.ceil(steps / n_action_steps), wall_time_s=time.time() - t0, video=video)
                        break
                venv.close()
            missing = log.missing(task_ids, init_ids)
            n_missing += len(missing)
            print(f"[{suite_name}] done: new {log.n_new}, exceptions {log.n_err}, missing {len(missing)}", flush=True)
    sys.stdout.flush()
    os._exit(1 if n_missing else 0)  # skip interpreter teardown: EGL contexts can hang or crash on exit


if __name__ == "__main__":
    init_logging()
    register_third_party_plugins()
    main()

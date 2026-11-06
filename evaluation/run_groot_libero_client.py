#!/usr/bin/env python
"""GR00T N1.7 on LIBERO, one CSV row per rollout (client side).

The policy runs in GR00T's own server (gr00t/eval/run_gr00t_server.py --use-sim-policy-wrapper). This
client is run with GR00T's LIBERO venv (gr00t/eval/sim/LIBERO/libero_uv) and uses GR00T's LIBERO env,
MultiStepWrapper and rollout loop (gr00t/eval/rollout_policy.py, n_envs=1) with the settings of GR00T's
LIBERO instructions (n_action_steps 8, max_episode_steps 720). What differs from rollout_policy.py:

  * rollout k of a task starts from LIBERO's fixed initial state k (k = 0..49), followed by
    num_steps_wait no-op steps so the objects settle. GR00T's script calls a plain env.reset(), which
    places the objects at random, so its rollouts cannot be matched across models;
  * one row per finished rollout, written immediately; finished rollouts are skipped when re-run;
  * a policy-server outage is waited out and the interrupted rollout is run again.
"""
import argparse
import collections
import os
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from gr00t.eval._horizon_contract import PolicyHorizonSpec  # noqa: E402
from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv  # noqa: E402
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper  # noqa: E402
from gr00t.policy.server_client import PolicyClient  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roboelo_libero_common import LIBERO_DUMMY_ACTION, EpisodeLog, goal_status  # noqa: E402


class FixedInitLiberoEnv(LiberoEnv):
    """GR00T's LIBERO env, started from LIBERO's fixed initial states, with per-step bookkeeping.

    `queue` holds the initial-state ids of the coming rollouts; every reset takes the next one. The
    vector env resets inside step() when a rollout ends, so the statistics of the rollout that just
    ended are kept in `last_episode`.
    """

    def __init__(self, task_bddl_file, task_description, init_states, num_steps_wait, env_seed):
        super().__init__(task_bddl_file, task_description)
        self._env.seed(env_seed)  # seeded once per task env, as in openpi's LIBERO evaluation
        self.init_states, self.num_steps_wait = init_states, num_steps_wait
        self.queue, self.cur, self.last_episode = collections.deque(), None, None

    def reset(self, seed=None, options=None):
        self.last_episode, self.cur = self.cur, None
        if seed is not None:
            self._env.seed(int(seed))
        observation = self._env.reset()
        if self.queue:
            init_id = self.queue.popleft()
            observation = self._env.set_init_state(self.init_states[init_id])
            for _ in range(self.num_steps_wait):
                observation, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)
            status = goal_status(self._env)
            self.cur = dict(init_state_id=init_id, steps=0, goal_total=len(status), goal_last=sum(status),
                            goal_max=sum(status), frames=[observation["agentview_image"][::-1, ::-1]])
        observation = self._process_observation(observation)
        return observation, {"success": self._env.check_success()}

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        if self.cur is not None:
            status = goal_status(self._env)
            self.cur["steps"] += 1
            self.cur["goal_last"] = sum(status)
            self.cur["goal_max"] = max(self.cur["goal_max"], sum(status))
            self.cur["frames"].append(observation["video.image"])
        return observation, reward, done, truncated, info


def _any_success(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return bool(np.any(value))
    return bool(value)


class _ZeroPolicy:
    """--fake-policy: all-zero actions, to exercise the env side and the records without a policy server."""

    def __init__(self, n_action_steps):
        self.n = n_action_steps

    def reset(self):
        pass

    def close(self):
        pass

    def get_action(self, observations):
        keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        return {f"action.{k}": np.zeros((1, self.n, 1), dtype=np.float32) for k in keys}, {}


def connect(args):
    """Wait until the policy server answers."""
    if args.fake_policy:
        return _ZeroPolicy(args.n_action_steps)
    waited = 0
    while True:
        policy = PolicyClient(host=args.host, port=args.port, timeout_ms=args.timeout_ms)
        if policy.ping():
            return policy
        policy.close()
        waited += 1
        if waited % 6 == 1:
            print(f"[{args.suite}] waiting for the policy server at {args.host}:{args.port}", flush=True)
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True, help="results root; a <suite>/ folder is created inside")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18565)
    ap.add_argument("--num-trials", type=int, default=50, help="initial states 0..N-1 per task")
    ap.add_argument("--task-ids", type=int, nargs="*", help="default: every task of the suite")
    ap.add_argument("--init-ids", type=int, nargs="*", help="default: 0..num-trials-1")
    ap.add_argument("--max-episode-steps", type=int, default=720)
    ap.add_argument("--n-action-steps", type=int, default=8)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--env-seed", type=int, default=7)
    ap.add_argument("--timeout-ms", type=int, default=120000, help="per request to the policy server")
    ap.add_argument("--videos", choices=["failures", "all", "none"], default="failures")
    ap.add_argument("--retries", type=int, default=20,
                    help="re-run a rollout that raised an exception (each crash of the policy server uses one)")
    ap.add_argument("--model", default="gr00t_n1.7")
    ap.add_argument("--checkpoint", default="GR00T-N1.7-LIBERO")
    ap.add_argument("--fake-policy", action="store_true", help="debug only: zero actions instead of a policy server")
    args = ap.parse_args()

    suite = benchmark.get_benchmark_dict()[args.suite]()
    log = EpisodeLog(args.out, args.suite, args.model, f"{args.checkpoint}/{args.suite}")
    task_ids = args.task_ids if args.task_ids else list(range(suite.get_num_tasks()))
    init_ids = args.init_ids if args.init_ids else list(range(args.num_trials))
    policy = connect(args)
    if args.fake_policy:
        contract = PolicyHorizonSpec(n_action_steps=args.n_action_steps, action_horizon=args.n_action_steps,
                                     video_delta_indices=(0,), state_delta_indices=(0,))
    else:
        contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=args.n_action_steps)
    print(f"[{args.suite}] connected to {args.host}:{args.port}; {contract}; tasks {task_ids}; "
          f"{len(log.done)} rollouts already recorded", flush=True)
    try:
        import imageio
    except ImportError:
        imageio = None
        print(f"[{args.suite}] imageio is not installed: no videos will be written", flush=True)

    for task_id in task_ids:
        todo = collections.deque(log.todo(task_id, init_ids))
        if not todo:
            continue
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(task_id)

        def make(bddl=bddl, task=task, init_states=init_states):
            env = FixedInitLiberoEnv(bddl, task.language, init_states, args.num_steps_wait, args.env_seed)
            return MultiStepWrapper(env, contract=contract, max_episode_steps=args.max_episode_steps,
                                    terminate_on_success=True)

        venv = gym.vector.SyncVectorEnv([make])  # what rollout_policy.py builds for n_envs=1
        inner = venv.envs[0].unwrapped
        attempts = collections.Counter()
        while todo:
            init_id = todo[0]
            try:
                # ---- the loop of rollout_policy._collect_rollout_episodes, for one env ----
                inner.queue, inner.cur = collections.deque(todo), None
                observations, _ = venv.reset()
                policy.reset()
                success, n_infer, t0 = False, 0, time.time()
                while todo:
                    init_id = todo[0]
                    actions, _ = policy.get_action(observations)
                    n_infer += 1
                    observations, _, terminations, truncations, infos = venv.step(actions)
                    if "success" in infos:
                        success |= _any_success(infos["success"][0])
                    if "final_info" in infos and infos["final_info"][0] is not None:
                        success |= _any_success(infos["final_info"][0]["success"])
                    if terminations[0] or truncations[0]:
                        ep = inner.last_episode  # the vector env has already reset the env for the next rollout
                        assert ep is not None and ep["init_state_id"] == init_id, (ep and ep["init_state_id"], init_id)
                        video = ""
                        if imageio is not None and (args.videos == "all" or (args.videos == "failures" and not success)):
                            video = log.video_path(task_id, init_id, success)
                            imageio.mimwrite(video, [np.ascontiguousarray(f) for f in ep["frames"]], fps=20)
                        log.write(
                            task_id=task_id, task_name=task.language, bddl_file=task.bddl_file, init_state_id=init_id,
                            seed=args.env_seed, success=success, steps=ep["steps"], max_steps=args.max_episode_steps,
                            goal_total=ep["goal_total"], goal_satisfied_final=ep["goal_last"],
                            goal_satisfied_max=ep["goal_max"], n_infer=n_infer, wall_time_s=time.time() - t0,
                            video=video)
                        todo.popleft()
                        success, n_infer, t0 = False, 0, time.time()
            except Exception as exc:  # never record an exception as a task failure
                attempts[init_id] += 1
                log.error(task_id, init_id, attempts[init_id], exc)
                traceback.print_exc()
                if attempts[init_id] > args.retries:
                    print(f"[{args.suite}] task {task_id} init {init_id}: giving up after {attempts[init_id]} errors",
                          flush=True)
                    todo.popleft()
                time.sleep(5)
                policy.close()
                policy = connect(args)
        venv.close()

    missing = log.missing(task_ids, init_ids)
    print(f"[{args.suite}] done: new {log.n_new}, exceptions {log.n_err}, missing {len(missing)}", flush=True)
    sys.stdout.flush()
    os._exit(1 if missing else 0)  # skip interpreter teardown: EGL contexts can hang or crash on exit


if __name__ == "__main__":
    main()

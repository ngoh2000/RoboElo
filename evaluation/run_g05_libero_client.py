#!/usr/bin/env python
"""G0.5 (GalaxeaVLA) on LIBERO, one CSV row per rollout (client side).

The policy runs in GalaxeaVLA's own server (scripts/serve_policy_batched.py ... eval_embodiment=libero
--action_steps 10). This client is run with GalaxeaVLA's environment from the repository root and uses
its LIBERO utilities (observation and action conversion, gripper command state) and its WebSocket
client. The per-step logic follows experiments/libero/eval_libero_parallel.py for one environment, with
the settings of scripts/run/eval_libero.sh: 20 no-op steps after the initial state, 256 px cameras,
chunks of 10 actions, step limits 220 / 280 / 300 / 520, trial k = LIBERO's fixed initial state k.
What this client adds:

  * one row per finished rollout, written immediately; finished rollouts are skipped when re-run;
  * the number of satisfied goal predicates (partial progress);
  * a policy-server outage is waited out and the interrupted rollout is run again.
"""
import argparse
import asyncio
import os
import sys
import time
import traceback

REPO = os.environ.get("G05_REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio  # noqa: E402
import numpy as np  # noqa: E402
from experiments.libero.libero_eval_utils import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    LiberoGripperCommandState,
    build_libero_raw_obs,
    ensure_libero_config,
    ensure_libero_import,
    extract_libero_images,
    get_libero_env,
    get_max_steps,
    single_action_to_libero_action,
)
from roboelo_libero_common import EpisodeLog, goal_status  # noqa: E402
from scripts.utils.policy_ws_client import MultiPolicyWSClient  # noqa: E402


async def connect(args):
    """Wait until the policy server answers; one connection, as one env of the official client."""
    waited = 0
    while True:
        client = MultiPolicyWSClient(args.server_uri, 1, timeout_s=args.timeout_s)
        try:
            await client.connect_all()
            return client
        except Exception:
            waited += 1
            if waited % 6 == 1:
                print(f"[{args.suite}] waiting for the policy server at {args.server_uri}", flush=True)
            await asyncio.sleep(5)


async def rollout(env, client, task_description, init_state, max_steps, args):
    """One trial, as run_parallel_task does it for a single environment."""
    env.reset()  # zero velocities from the previous episode before applying the initial state
    obs = env.set_init_state(init_state)
    gripper = LiberoGripperCommandState.from_obs(obs)
    await client.reset_all()
    status = goal_status(env)
    goal_max, frames = sum(status), []
    step_count = action_step = n_infer = 0
    done = False
    total_steps = max_steps + args.num_steps_wait
    while step_count < total_steps:
        if step_count < args.num_steps_wait:  # warm-up: no-op steps
            gripper.set_env_command(LIBERO_DUMMY_ACTION[-1])
            action = LIBERO_DUMMY_ACTION
        else:
            if client.needs_recompute(0):
                raw_obs, _ = build_libero_raw_obs(obs=obs, task_description=task_description,
                                                  embodiment_type=args.embodiment_type)
                await client.infer_batch([0], [raw_obs], [action_step])
                n_infer += 1
            chunk_action = client.get_cached_action(0)
            if chunk_action is None:
                raise RuntimeError("the policy server returned no action")
            action = single_action_to_libero_action(chunk_action, obs=obs, gripper_state=gripper)
            frames.append(extract_libero_images(obs)["image"])
        obs, _, done, _ = env.step(action)
        step_count += 1
        if step_count > args.num_steps_wait:
            action_step += 1
        status = goal_status(env)
        goal_max = max(goal_max, sum(status))
        if done:
            break
    await client.reset_connection(0)
    return dict(success=bool(done), steps=action_step, goal_total=len(status), goal_satisfied_final=sum(status),
                goal_satisfied_max=goal_max, n_infer=n_infer), frames


async def main(args):
    ensure_libero_import()
    ensure_libero_config()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    max_steps = args.max_steps_override or get_max_steps(args.suite)
    log = EpisodeLog(args.out, args.suite, args.model, args.checkpoint)
    task_ids = args.task_ids if args.task_ids else list(range(suite.n_tasks))
    init_ids = args.init_ids if args.init_ids else list(range(args.num_trials))
    client = await connect(args)
    print(f"[{args.suite}] connected to {args.server_uri}; server says {client.metadata[0]}; tasks {task_ids}; "
          f"{len(log.done)} rollouts already recorded", flush=True)

    for task_id in task_ids:
        todo = log.todo(task_id, init_ids)
        if not todo:
            continue
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, args.env_resolution, args.seed)
        for init_id in todo:
            for attempt in range(1, args.retries + 2):
                t0 = time.time()
                try:
                    res, frames = await rollout(env, client, task_description, init_states[init_id], max_steps, args)
                except Exception as exc:  # never record an exception as a task failure
                    log.error(task_id, init_id, attempt, exc)
                    traceback.print_exc()
                    await asyncio.sleep(5)
                    try:
                        await client.close_all()
                    except Exception:
                        pass
                    client = await connect(args)
                    continue
                video = ""
                if args.videos == "all" or (args.videos == "failures" and not res["success"]):
                    video = log.video_path(task_id, init_id, res["success"])
                    imageio.mimwrite(video, [np.asarray(f) for f in frames], fps=20)
                log.write(task_id=task_id, task_name=task.language, bddl_file=task.bddl_file, init_state_id=init_id,
                          seed=args.seed, max_steps=max_steps, wall_time_s=time.time() - t0, video=video, **res)
                break
        env.close()

    missing = log.missing(task_ids, init_ids)
    print(f"[{args.suite}] done: new {log.n_new}, exceptions {log.n_err}, missing {len(missing)}", flush=True)
    sys.stdout.flush()
    os._exit(1 if missing else 0)  # skip interpreter teardown: EGL contexts can hang or crash on exit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True, help="results root; a <suite>/ folder is created inside")
    ap.add_argument("--server-uri", default="ws://127.0.0.1:18575")
    ap.add_argument("--num-trials", type=int, default=50, help="initial states 0..N-1 per task")
    ap.add_argument("--task-ids", type=int, nargs="*", help="default: every task of the suite")
    ap.add_argument("--init-ids", type=int, nargs="*", help="default: 0..num-trials-1")
    ap.add_argument("--num-steps-wait", type=int, default=20)
    ap.add_argument("--env-resolution", type=int, default=LIBERO_ENV_RESOLUTION)
    ap.add_argument("--embodiment-type", default="libero")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--max-steps-override", type=int, help="debug only: shorten rollouts")
    ap.add_argument("--videos", choices=["failures", "all", "none"], default="failures")
    ap.add_argument("--retries", type=int, default=20,
                    help="re-run a rollout that raised an exception (each crash of the policy server uses one)")
    ap.add_argument("--model", default="g05")
    ap.add_argument("--checkpoint", default="g05-libero")
    asyncio.run(main(ap.parse_args()))

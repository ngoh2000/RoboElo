#!/usr/bin/env python
"""A policy on SimplerEnv's official episodes, one CSV row per episode (resumable).

The episodes are the lines of state/simpler/episodes_official.jsonl (simpler_episode_grid.py: every setting of
SimplerEnv's official scripts, expanded as maniskill2_evaluator does). Each episode runs the loop of SimplerEnv's
run_maniskill2_eval_single_episode (simpler_env/evaluation/maniskill2_evaluator.py): same env construction, reset
options, model.reset / model.step, early stop when the policy predicts termination on the final subtask, success =
`done` of the last step. What this runner adds: one row per episode with the steps taken, the number of policy
calls and the env's episode_stats (partial progress such as grasped / moved near / drawer opened), finished episodes
skipped on a re-run, an exception recorded separately and the episode run again, videos of failed episodes only.

The policy comes from a factory, "<python file>:<function>", called as factory(policy_setup, **policy_kwargs) with
policy_setup "google_robot" or "widowx_bridge"; it returns an object with SimplerEnv's policy interface
(reset(task_description); step(image, task_description) -> (raw_action, action) with action["world_vector"],
["rot_axangle"], ["gripper"], ["terminate_episode"]). The factory is called again when the robot family changes.
A policy with `needs_obs = True` gets reset(task_description, obs=, env=) and step(image, task_description, obs=,
env=) instead, for models that also read the robot state; a policy with a `control_modes` dict (robot family ->
controller name) gets its env built with that controller instead of the setting's.

  python run_simpler_env.py --factory scripts/simpler_policies.py:rt1 --policy-kwargs '{"ckpt": "..."}' \
      --out results/rt1x/simplerenv --model rt1x --checkpoint rt_1_x_tf_trim [--families widowx] [--shard 0/4]
Run from SimplerEnv's repository root (relative asset paths of the official settings).
"""
import argparse
import csv
import datetime
import fcntl
import importlib.util
import json
import os
import sys
import time
import traceback

ROOT = os.path.expanduser("~/roboelo")
FIELDS = ["model", "checkpoint", "benchmark", "robot_family", "task", "protocol", "setting_id", "env_name",
          "scene_name", "episode_index", "robot_init_x", "robot_init_y", "robot_init_quat", "obj_init_x", "obj_init_y",
          "obj_episode_id", "additional_env_build_kwargs", "rgb_overlay_path", "instruction", "success", "steps",
          "max_steps", "n_infer", "episode_stats", "wall_time_s", "finished_at", "video"]
ERROR_FIELDS = ["episode_index", "attempt", "time", "error"]
SETUP = {"google_robot": "google_robot", "widowx": "widowx_bridge"}
# Comparison runs under a model's own protocol (not the uniform one): ROBOELO_MAX_EPISODE_STEPS = a number of steps or
# "x<factor>" of the official limit (X-VLA: 1200 on WidowX, x2 on Google), ROBOELO_STOP_ON_SUCCESS=1 ends an episode
# at its first success instead of judging the last step.
MAX_STEPS_SETTING = os.environ.get("ROBOELO_MAX_EPISODE_STEPS", "")
STOP_ON_SUCCESS = os.environ.get("ROBOELO_STOP_ON_SUCCESS") == "1"


def max_steps_for(ep):
    if not MAX_STEPS_SETTING:
        return ep["max_episode_steps"]
    if MAX_STEPS_SETTING.startswith("x"):
        return int(round(ep["max_episode_steps"] * float(MAX_STEPS_SETTING[1:])))
    return int(MAX_STEPS_SETTING)


def append_row(path, fields, row):
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
        f.flush()


def load_factory(spec):
    path, _, name = spec.rpartition(":")
    module_spec = importlib.util.spec_from_file_location("roboelo_simpler_policies", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, name)


def run_episode(model, ep, video_path_fn):
    """SimplerEnv's run_maniskill2_eval_single_episode for one episode, returning its outcome."""
    import numpy as np
    from simpler_env.utils.env.env_builder import build_maniskill2_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    additional = dict(ep["additional_env_build_kwargs"] or {})
    # A policy that acts in another action space (X-VLA: absolute end-effector poses) names its controller per family.
    control_mode = getattr(model, "control_modes", {}).get(ep["robot_family"], ep["control_mode"])
    kwargs = dict(obs_mode="rgbd", robot=ep["robot"], sim_freq=ep["sim_freq"], control_mode=control_mode,
                  control_freq=ep["control_freq"], max_episode_steps=max_steps_for(ep),
                  scene_name=ep["scene_name"], camera_cfgs={"add_segmentation": True},
                  rgb_overlay_path=ep["rgb_overlay_path"])
    if ep["enable_raytracing"]:
        additional = {"shader_dir": "rt", **additional}
    env = build_maniskill2_env(ep["env_name"], **additional, **kwargs)
    try:
        options = {"robot_init_options": {"init_xy": np.array([ep["robot_init_x"], ep["robot_init_y"]]),
                                          "init_rot_quat": np.array(ep["robot_init_quat"])}}
        if "obj_init_x" in ep:
            options["obj_init_options"] = {"init_xy": np.array([ep["obj_init_x"], ep["obj_init_y"]])}
        else:
            options["obj_init_options"] = {"episode_id": ep["obj_episode_id"]}
        obs, _ = env.reset(options=options)
        is_final_subtask = env.is_final_subtask()
        task_description = env.get_language_instruction()
        instruction = task_description
        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=ep["obs_camera_name"])
        images = [image]
        # Policies that also need the robot state (GR00T, X-VLA) declare needs_obs and get the env's obs dict and env.
        needs_obs = getattr(model, "needs_obs", False)
        if needs_obs:
            model.reset(task_description, obs=obs, env=env)
        else:
            model.reset(task_description)
        predicted_terminated = truncated = done = False
        info, timestep = {}, 0
        while not (predicted_terminated or truncated):
            if needs_obs:
                raw_action, action = model.step(image, task_description, obs=obs, env=env)
            else:
                raw_action, action = model.step(image, task_description)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated and not is_final_subtask:
                predicted_terminated = False
                env.advance_to_next_subtask()
            obs, reward, done, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]))
            if done and STOP_ON_SUCCESS:  # a model's own protocol (comparison runs only)
                predicted_terminated = True
            new_description = env.get_language_instruction()
            if new_description != task_description:
                task_description = new_description
            is_final_subtask = env.is_final_subtask()
            image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=ep["obs_camera_name"])
            images.append(image)
            timestep += 1
        success = bool(done)
        video = ""
        path = video_path_fn(success)
        if path:
            # imageio brings its own ffmpeg (SimplerEnv's write_video needs a system ffmpeg); a failed video never
            # costs the episode's record
            try:
                import imageio

                imageio.mimwrite(path, [np.asarray(im) for im in images], fps=5)
                video = path
            except Exception as exc:
                print(f"video not written: {type(exc).__name__}: {exc}", flush=True)
        stats = {k: (v.item() if hasattr(v, "item") else v) for k, v in (info.get("episode_stats", {}) or {}).items()}
        return dict(success=success, steps=timestep, n_infer=getattr(model, "n_infer", timestep),
                    episode_stats=stats, instruction=instruction, video=video)
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=os.path.join(ROOT, "state", "simpler", "episodes_official.jsonl"))
    ap.add_argument("--factory", required=True, help="<python file>:<function>")
    ap.add_argument("--policy-kwargs", default="{}", help="JSON object passed to the factory")
    ap.add_argument("--families", default="google_robot,widowx", help="robot families to run (comma separated)")
    ap.add_argument("--protocols", default="visual_matching,variant_aggregation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard", default="0/1", help="k/N: run the episodes whose index % N == k")
    ap.add_argument("--videos", choices=["failures", "all", "none"], default="failures")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    families, protocols = args.families.split(","), args.protocols.split(",")
    k, n = (int(x) for x in args.shard.split("/"))
    episodes = [json.loads(line) for line in open(args.episodes)]
    episodes = [e for e in episodes if e["robot_family"] in families and e["protocol"] in protocols
                and e["episode_index"] % n == k]
    episodes.sort(key=lambda e: (families.index(e["robot_family"]), e["episode_index"]))
    os.makedirs(os.path.join(args.out, "videos"), exist_ok=True)
    ep_csv, err_csv = os.path.join(args.out, "episodes.csv"), os.path.join(args.out, "errors.csv")
    done_ids = set()
    if os.path.exists(ep_csv):
        with open(ep_csv, newline="") as f:
            done_ids = {int(r["episode_index"]) for r in csv.DictReader(f)}
    todo = [e for e in episodes if e["episode_index"] not in done_ids]
    print(f"[setup] model={args.model} checkpoint={args.checkpoint} families={families} protocols={protocols} "
          f"shard={args.shard}: {len(episodes)} episodes, {len(episodes) - len(todo)} already recorded", flush=True)

    factory = load_factory(args.factory)
    policy_kwargs = json.loads(args.policy_kwargs)
    model, model_family = None, None
    n_new = n_err = 0
    for ep in todo:
        if ep["robot_family"] != model_family:
            model = None
            model = factory(SETUP[ep["robot_family"]], **policy_kwargs)
            model_family = ep["robot_family"]
        for attempt in range(1, args.retries + 2):
            t0 = time.time()

            def video_path_fn(success, ep=ep):
                if args.videos == "all" or (args.videos == "failures" and not success):
                    tag = "success" if success else "failure"
                    return os.path.join(args.out, "videos", f"ep{ep['episode_index']:04d}_{ep['env_name']}_{tag}.mp4")
                return None

            try:
                if hasattr(model, "n_infer"):
                    model.n_infer = 0
                res = run_episode(model, ep, video_path_fn)
            except Exception as exc:  # never record an exception as a task failure
                n_err += 1
                append_row(err_csv, ERROR_FIELDS, dict(episode_index=ep["episode_index"], attempt=attempt,
                                                      time=datetime.datetime.now().isoformat(),
                                                      error=f"{type(exc).__name__}: {exc}"[:500]))
                traceback.print_exc()
                if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
                    print("CUDA error: this process cannot continue; exiting so it gets restarted", flush=True)
                    os._exit(3)
                continue
            row = dict(model=args.model, checkpoint=args.checkpoint, benchmark="simplerenv",
                       robot_family=ep["robot_family"], task=ep["task"], protocol=ep["protocol"],
                       setting_id=ep["setting_id"], env_name=ep["env_name"], scene_name=ep["scene_name"],
                       episode_index=ep["episode_index"], robot_init_x=ep["robot_init_x"],
                       robot_init_y=ep["robot_init_y"], robot_init_quat=json.dumps(ep["robot_init_quat"]),
                       obj_init_x=ep.get("obj_init_x", ""), obj_init_y=ep.get("obj_init_y", ""),
                       obj_episode_id=ep.get("obj_episode_id", ""),
                       additional_env_build_kwargs=json.dumps(ep["additional_env_build_kwargs"] or {}),
                       rgb_overlay_path=ep["rgb_overlay_path"] or "", instruction=res["instruction"],
                       success=int(res["success"]), steps=res["steps"], max_steps=max_steps_for(ep),
                       n_infer=res["n_infer"], episode_stats=json.dumps(res["episode_stats"]),
                       wall_time_s=round(time.time() - t0, 1),
                       finished_at=datetime.datetime.now().isoformat(timespec="seconds"), video=res["video"])
            append_row(ep_csv, FIELDS, row)
            n_new += 1
            print(f"[{ep['task']}/{ep['protocol']}] episode {ep['episode_index']} {ep['env_name']}: "
                  f"success={int(res['success'])} steps={res['steps']} stats={res['episode_stats']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            break
    recorded = set()
    if os.path.exists(ep_csv):
        with open(ep_csv, newline="") as f:
            recorded = {int(r["episode_index"]) for r in csv.DictReader(f)}
    missing = [e["episode_index"] for e in episodes if e["episode_index"] not in recorded]
    print(f"[done] new {n_new}, exceptions {n_err}, missing {len(missing)}", flush=True)
    sys.stdout.flush()
    os._exit(1 if missing else 0)


if __name__ == "__main__":
    main()

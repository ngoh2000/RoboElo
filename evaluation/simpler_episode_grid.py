#!/usr/bin/env python
"""The episode list of SimplerEnv's official evaluation, one JSON line per episode.

SimplerEnv's scripts/rt1_*.sh (Google Robot: visual matching and variant aggregation) and scripts/rt1x_bridge.sh
(WidowX) call simpler_env/main_inference.py once per setting, and maniskill2_evaluator loops over the robot and
object initializations of that setting. Here every script is run with a stand-in `python` that only records its
arguments; the policy-specific ones (--policy-model, --ckpt-path, --policy-setup) are dropped, identical settings are
merged, and each setting is expanded with SimplerEnv's own argument parser exactly as maniskill2_evaluator does.
Run with SimplerEnv's environment from the repository root:

  python simpler_episode_grid.py <SimplerEnv root> <out.jsonl> [fork]

With "fork", the task scripts of the DelinQu/SimplerEnv-OpenVLA fork (bridge.sh, drawer_visual_matching.sh, ...,
called with <ckpt> <model> <action_ensemble_temp> <logging_dir> <gpu>) are expanded instead, to compare the two.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

root, out_path = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
sys.path.insert(0, root)
from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.utils.env.env_builder import get_robot_control_mode  # noqa: E402

SCRIPTS = [
    ("google_robot", "pick_coke_can", "visual_matching", "rt1_pick_coke_can_visual_matching.sh"),
    ("google_robot", "pick_coke_can", "variant_aggregation", "rt1_pick_coke_can_variant_agg.sh"),
    ("google_robot", "move_near", "visual_matching", "rt1_move_near_visual_matching.sh"),
    ("google_robot", "move_near", "variant_aggregation", "rt1_move_near_variant_agg.sh"),
    ("google_robot", "drawer", "visual_matching", "rt1_drawer_visual_matching.sh"),
    ("google_robot", "drawer", "variant_aggregation", "rt1_drawer_variant_agg.sh"),
    ("google_robot", "put_in_drawer", "visual_matching", "rt1_put_in_drawer_visual_matching.sh"),
    ("google_robot", "put_in_drawer", "variant_aggregation", "rt1_put_in_drawer_variant_agg.sh"),
    ("widowx", "bridge", "visual_matching", "rt1x_bridge.sh"),
]
POLICY_ARGS = {"--policy-model", "--ckpt-path", "--policy-setup", "--logging-dir", "--action-ensemble-temp"}
FORK = len(sys.argv) > 3 and sys.argv[3] == "fork"
SCRIPT_ARGS = []
if FORK:
    SCRIPTS = [(f, t, p, s.replace("rt1_", "").replace("rt1x_", "")) for f, t, p, s in SCRIPTS]
    SCRIPT_ARGS = ["CKPT", "openvla", "-0.8", "/tmp/simpler_grid_logs", "0"]

shim_dir = tempfile.mkdtemp()
record = os.path.join(shim_dir, "calls.jsonl")
with open(os.path.join(shim_dir, "python"), "w") as f:
    f.write("#!/usr/bin/env bash\n"
            f"/usr/bin/env python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \"$@\" >> {record}\n")
os.chmod(os.path.join(shim_dir, "python"), 0o755)

settings = []  # (robot_family, task, protocol, script, argv)
seen = set()
for family, task, protocol, script in SCRIPTS:
    if os.path.exists(record):
        os.remove(record)
    env = dict(os.environ, PATH=shim_dir + os.pathsep + os.environ["PATH"])
    subprocess.run(["bash", os.path.join(root, "scripts", script)] + SCRIPT_ARGS, cwd=root, env=env, check=True,
                   stdout=subprocess.DEVNULL)
    for line in open(record):
        argv = json.loads(line)
        if argv and argv[0] == "-u":
            argv = argv[1:]
        assert argv[0].endswith("main_inference.py"), argv[:2]
        argv, kept, skip = argv[1:], [], False
        for a in argv:
            if skip:
                skip = False
                continue
            if a in POLICY_ARGS:
                skip = True
                continue
            kept.append(a)
        key = (family, task, protocol, tuple(kept))
        if key in seen:
            continue
        seen.add(key)
        settings.append((family, task, protocol, script, kept))

n = 0
with open(out_path, "w") as out:
    for setting_id, (family, task, protocol, script, argv) in enumerate(settings):
        sys.argv = ["main_inference.py"] + argv
        a = get_args()
        common = dict(setting_id=setting_id, robot_family=family, task=task, protocol=protocol, script=script,
                      argv=argv, env_name=a.env_name, scene_name=a.scene_name, robot=a.robot,
                      control_mode=get_robot_control_mode(a.robot, "any"), control_freq=a.control_freq,
                      sim_freq=a.sim_freq, max_episode_steps=a.max_episode_steps,
                      rgb_overlay_path=a.rgb_overlay_path, additional_env_build_kwargs=a.additional_env_build_kwargs,
                      enable_raytracing=a.enable_raytracing, additional_env_save_tags=a.additional_env_save_tags,
                      obs_camera_name=a.obs_camera_name)
        for rx in a.robot_init_xs:
            for ry in a.robot_init_ys:
                for quat in a.robot_init_quats:
                    base = dict(common, robot_init_x=float(rx), robot_init_y=float(ry),
                                robot_init_quat=[float(q) for q in np.asarray(quat)])
                    if a.obj_variation_mode == "xy":
                        objs = [dict(obj_init_x=float(ox), obj_init_y=float(oy)) for ox in a.obj_init_xs
                                for oy in a.obj_init_ys]
                    else:
                        objs = [dict(obj_episode_id=i) for i in range(a.obj_episode_range[0], a.obj_episode_range[1])]
                    for o in objs:
                        out.write(json.dumps(dict(base, episode_index=n, **o)) + "\n")
                        n += 1
print(f"{len(settings)} settings, {n} episodes -> {out_path}")

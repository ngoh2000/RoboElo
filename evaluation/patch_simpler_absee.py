#!/usr/bin/env python3
"""Add X-VLA's absolute end-effector controllers to a copy of ManiSkill2_real2sim (third_party/SimplerEnv-absee).

X-VLA predicts absolute end-effector poses and is evaluated on 255isWhite/SimplerEnv, whose ManiSkill2_real2sim
(commit 3c3590a "enable abs eef control") adds two controllers and reads their rotation as Euler xyz. Only those
changes are applied here; the rest of that commit (default control modes switched to the new controllers, the WidowX
sink camera and robot pose moved for the eggplant task, debug prints) is left out, so scenes, cameras and episodes
stay the official ones. The new controllers are only used when a run asks for them (run_simpler_env.py, the policy's
`control_modes`).

  patch_simpler_absee.py <ManiSkill2_real2sim folder of the copy>
Each file is rewritten as a new inode, so a hard-linked copy (cp -al) leaves the official checkout untouched.
"""
import os
import sys

root = os.path.join(sys.argv[1], "mani_skill2_real2sim")


def patch(rel, edits):
    path = os.path.join(root, rel)
    text = open(path).read()
    for old, new in edits:
        if new in text:
            continue  # already applied
        assert text.count(old) == 1, (rel, old[:60])
        text = text.replace(old, new)
    os.unlink(path)  # break the hard link before writing
    with open(path, "w") as f:
        f.write(text)
    print("patched", rel)


GOOGLE_NEW = '''        arm_pd_ee_base_pose_align_interpolate_by_planner = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="base",
            interpolate=True,
            use_delta=False,
            interpolate_by_planner=True,
            interpolate_planner_vlim=self.arm_vel_limit,
            interpolate_planner_alim=self.arm_acc_limit,
            interpolate_planner_jerklim=self.arm_jerk_limit,
            **arm_common_kwargs,
        )
        _C["arm"] = dict(
            arm_pd_ee_delta_pose=arm_pd_ee_delta_pose,'''
patch("agents/configs/google_robot/defaults.py", [
    ('''        _C["arm"] = dict(
            arm_pd_ee_delta_pose=arm_pd_ee_delta_pose,''', GOOGLE_NEW),
    ("            arm_pd_ee_target_delta_pose_align_interpolate_by_planner=arm_pd_ee_target_delta_pose_align_interpolate_by_planner,\n",
     "            arm_pd_ee_target_delta_pose_align_interpolate_by_planner=arm_pd_ee_target_delta_pose_align_interpolate_by_planner,\n"
     "            arm_pd_ee_base_pose_align_interpolate_by_planner=arm_pd_ee_base_pose_align_interpolate_by_planner,\n"),
])

patch("agents/configs/widowx/defaults.py", [
    ('''        arm_pd_ee_target_delta_pose_align2 = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee_align2", use_target=True, **arm_common_kwargs
        )
''', '''        arm_pd_ee_target_delta_pose_align2 = PDEEPoseControllerConfig(
            *arm_common_args, frame="ee_align2", use_target=True, **arm_common_kwargs
        )
        arm_pd_ee_target_base_pose = PDEEPoseControllerConfig(
            *arm_common_args, frame="base", use_target=True, use_delta=False, **arm_common_kwargs
        )
'''),
    ("            arm_pd_ee_target_delta_pose_align2=arm_pd_ee_target_delta_pose_align2,\n",
     "            arm_pd_ee_target_delta_pose_align2=arm_pd_ee_target_delta_pose_align2,\n"
     "            arm_pd_ee_target_base_pose=arm_pd_ee_target_base_pose,\n"),
])

patch("agents/controllers/pd_ee_pose.py", [
    ("            target_quat = Rotation.from_rotvec(target_rot).as_quat()[[3, 0, 1, 2]]\n            target_pose",
     "            target_quat = Rotation.from_euler('xyz', target_rot).as_quat()[[3, 0, 1, 2]]  # X-VLA: Euler xyz\n"
     "            target_pose"),
])

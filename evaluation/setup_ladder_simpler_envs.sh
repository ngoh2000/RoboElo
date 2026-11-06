#!/usr/bin/env bash
# One-time set-up for the remaining SimplerEnv models of the policy ladder: GR00T N1.6 / N1.7, X-VLA,
# VLA-JEPA, MolmoAct. Safe to re-run; every step checks whether it is done. New things only, nothing shared changes:
#   envs/simpler-hf440      + pyzmq, msgpack (client side of the GR00T / LeRobot policy servers)
#   envs/groot-n16          venv (uv) whose base is Isaac-GR00T's .venv site-packages, with GR00T N1.6's pin
#                           transformers 4.51.3 on top; run with PYTHONPATH=third_party/Isaac-GR00T-n1.6
#                           (tag n1.6.1-release: the N1.6 checkpoints are Gr00tN1d6, which the N1.7 code cannot load)
#   third_party/SimplerEnv-absee  hard-linked copy of the official SimplerEnv with X-VLA's absolute-EE controllers
#   checkpoints/*-local     copies whose JSON name local folders instead of Hub repositories (offline runs)
#   checkpoints/vla-jepa-simplerenv-lerobot/original_dataset_statistics.json  oxe_bridge / oxe_rt1 statistics of
#                           the original checkpoint (ginwind/VLA-JEPA, SimplerEnv/dataset_statistics.json)
#   envs/simpler-molmoact   copy of envs/molmoact + SimplerEnv's simulation packages
#   MolmoAct-7B-D-Pretrain-RT-1-0812 stored as bf16 (its SimplerEnv evaluation loads bf16)
set -euo pipefail
R="${HOME}/roboelo"
T="${R}/third_party"
UV="${R}/micromamba/envs/roboelo-download/bin/uv"
cd "${R}"
step() { printf '\n==== %s\n' "$*"; }

step "client packages in envs/simpler-hf440"
envs/simpler-hf440/bin/python -c "import zmq, msgpack" 2>/dev/null || envs/simpler-hf440/bin/python -m pip install --progress-bar off "pyzmq==27.0.1" "msgpack==1.1.0"
envs/simpler-hf440/bin/python -c "import zmq, msgpack; print('client ok', zmq.__version__, msgpack.version)"

step "GR00T N1.6 server env"
if [[ ! -x envs/groot-n16/bin/python ]]; then
  "${UV}" venv --python /usr/bin/python3.12 envs/groot-n16
  echo "${T}/Isaac-GR00T/.venv/lib/python3.12/site-packages" > envs/groot-n16/lib/python3.12/site-packages/zz_groot_n17_venv.pth
fi
# N1.6's pins that differ from the N1.7 venv and matter for inference (av: gr00t.utils.video_utils; setuptools: the distutils shim transformers 4.51 imports on Python 3.12)
UV_HTTP_TIMEOUT=1800 "${UV}" pip install --python envs/groot-n16/bin/python "transformers==4.51.3" "numpy==1.26.4" "av==16.1.0" "setuptools<80"
(cd "${T}/Isaac-GR00T-n1.6" && PYTHONPATH="${PWD}" HF_HUB_OFFLINE=1 "${R}/envs/groot-n16/bin/python" -c "
import transformers, torch, gr00t
from gr00t.model.gr00t_n1d6 import gr00t_n1d6  # noqa: F401
print('n1.6 ok', gr00t.__file__, 'transformers', transformers.__version__, 'torch', torch.__version__)")

step "SimplerEnv copy with X-VLA's absolute end-effector controllers"
if [[ ! -f "${T}/SimplerEnv-absee/PATCHED" ]]; then
  rm -rf "${T}/SimplerEnv-absee"
  cp -al "${T}/SimplerEnv" "${T}/SimplerEnv-absee"
  rm -rf "${T}/SimplerEnv-absee/.git" "${T}/SimplerEnv-absee/ManiSkill2_real2sim/.git"
  find "${T}/SimplerEnv-absee" -name __pycache__ -type d -prune -exec rm -rf {} +
  python3 "${R}/scripts/patch_simpler_absee.py" "${T}/SimplerEnv-absee/ManiSkill2_real2sim"
  date -Is > "${T}/SimplerEnv-absee/PATCHED"
fi
(cd "${T}/SimplerEnv-absee" && PYTHONPATH="${PWD}:${PWD}/ManiSkill2_real2sim" "${R}/envs/simpler-hf440/bin/python" -c "
from simpler_env.utils.env.env_builder import build_maniskill2_env
for env_name, robot, mode, scene in [('PutSpoonOnTableClothInScene-v0', 'widowx', 'arm_pd_ee_target_base_pose_gripper_pd_joint_pos', 'bridge_table_1_v1'),
        ('GraspSingleOpenedCokeCanInScene-v0', 'google_robot_static', 'arm_pd_ee_base_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner', 'google_pick_coke_can_1_v4')]:
    env = build_maniskill2_env(env_name, obs_mode='rgbd', robot=robot, control_mode=mode, scene_name=scene)
    env.reset(); env.close(); print('absee ok', env_name)")

step "local checkpoint copies"
[[ -f checkpoints/xvla-widowx-lerobot-local/config.json ]] || python3 scripts/prepare_local_checkpoint.py \
  checkpoints/xvla-widowx-lerobot checkpoints/xvla-widowx-lerobot-local facebook/bart-large=checkpoints/bart-large-tokenizer
[[ -f checkpoints/xvla-google-robot-lerobot-local/config.json ]] || python3 scripts/prepare_local_checkpoint.py \
  checkpoints/xvla-google-robot-lerobot checkpoints/xvla-google-robot-lerobot-local facebook/bart-large=checkpoints/bart-large-tokenizer
[[ -f checkpoints/vla-jepa-simplerenv-lerobot-local/config.json ]] || python3 scripts/prepare_local_checkpoint.py \
  checkpoints/vla-jepa-simplerenv-lerobot checkpoints/vla-jepa-simplerenv-lerobot-local \
  Qwen/Qwen3-VL-2B-Instruct=checkpoints/Qwen3-VL-2B-Instruct facebook/vjepa2-vitl-fpc64-256=checkpoints/vjepa2-vitl-fpc64-256
stats=checkpoints/vla-jepa-simplerenv-lerobot/original_dataset_statistics.json
if [[ ! -s "${stats}" ]]; then
  curl -fsSL --retry 5 -o "${stats}.part" https://huggingface.co/ginwind/VLA-JEPA/resolve/main/SimplerEnv/dataset_statistics.json
  mv "${stats}.part" "${stats}"
fi
python3 -c "import json; d=json.load(open('${stats}')); print('vla-jepa stats', sorted(d)); [print(k, d[k]['action']['q01'], d[k]['action']['q99']) for k in ('oxe_bridge', 'oxe_rt1')]"

step "MolmoAct SimplerEnv env and bf16 weights"
[[ -d envs/simpler-molmoact ]] || cp -a envs/molmoact envs/simpler-molmoact
envs/simpler-molmoact/bin/python -c "import sapien" 2>/dev/null || scripts/make_simpler_overlay_env.sh envs/simpler-molmoact none
micromamba/envs/roboelo-libero/bin/python scripts/convert_safetensors_bf16.py checkpoints/MolmoAct-7B-D-Pretrain-RT-1-0812
du -sh checkpoints/MolmoAct-7B-D-Pretrain-RT-1-0812
echo "SETUP_LADDER_SIMPLER_DONE"

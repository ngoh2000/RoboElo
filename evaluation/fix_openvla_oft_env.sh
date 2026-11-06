#!/usr/bin/env bash
# OpenVLA-OFT's environment (micromamba/envs/roboelo-openvla-oft) for its LIBERO evaluation. It already had
# openvla-oft (editable), the transformers fork moojink/transformers-openvla-oft and torch 2.7.1+cu128 (the
# repository pins torch 2.2.0, which has no RTX 5090 kernels). This script adds:
#   * LIBERO's packages (experiments/robot/libero/libero_requirements.txt) with MuJoCo 3.3.2,
#   * tensorflow 2.15 (experiments/robot/openvla_utils.py imports it), numpy < 2 for it,
#   * the RLDS data packages that prismatic imports even for evaluation (dlimp, tensorflow_datasets,
#     tensorflow_graphics); tensorflow-metadata 1.15 keeps protobuf at 3.20 (newer ones need protobuf 5).
# flash-attn is not needed: openvla-oft loads the model without attn_implementation="flash_attention_2".
# LIBERO itself comes from third_party/LIBERO on sys.path (run_openvla_oft_libero.py).
set -euo pipefail
ROOT="${HOME}/roboelo"
P="${ROOT}/micromamba/envs/roboelo-openvla-oft/bin/python"
"${P}" -m pip freeze > "${ROOT}/logs/openvla_oft_env_before_fix_$(date +%Y%m%d_%H%M%S).txt"
"${P}" -m pip install --progress-bar off "numpy==1.26.4" "tensorflow==2.15.0" "mujoco==3.3.2" "robosuite==1.4.1" \
  "bddl==3.6.0" "easydict==1.13" "cloudpickle==2.1.0" "gym==0.25.2" "imageio-ffmpeg==0.6.0" "wandb==0.16.6" \
  json-numpy jsonlines rich "sentencepiece==0.1.99" "opencv-python==4.11.0.86" "future==1.0.0"
"${P}" -m pip install --progress-bar off "tensorflow_datasets==4.9.3" "tensorflow-metadata==1.15.0" \
  "git+https://github.com/moojink/dlimp_openvla"
"${P}" -m pip install --progress-bar off --no-deps "tensorflow_graphics==2021.12.3"
cd "${ROOT}/third_party/openvla-oft"
LIBERO_CONFIG_PATH="${ROOT}/libero_config" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "${P}" - < /dev/null <<'PY'
import os, sys
sys.path.insert(0, os.path.expanduser("~/roboelo/third_party/LIBERO"))
sys.argv = ["run_openvla_oft_libero.py"]
import experiments.robot.libero.run_libero_eval  # noqa: F401
import mujoco
print("OFT_ENV_OK mujoco", mujoco.__version__)
PY

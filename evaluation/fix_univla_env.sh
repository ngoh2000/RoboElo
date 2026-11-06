#!/usr/bin/env bash
# UniVLA's environment (micromamba/envs/roboelo-univla) for its LIBERO evaluation:
#   * LIBERO's packages with MuJoCo 3.3.2 (it had 3.12, which changes some LIBERO start scenes),
#   * tensorflow 2.15 (experiments/robot/openvla_utils.py imports it), numpy < 2 for it,
#   * flash-attn 2.8.3 (the model loads with attn_implementation="flash_attention_2"), copied from roboelo-libero:
#     same Python 3.10 and torch 2.7.1+cu128 there, and no prebuilt wheel to download (a build takes hours).
# Waits until no other pip install of this user is running, so that a tensorflow wheel downloaded for another
# environment comes from pip's cache.
set -euo pipefail
ROOT="${HOME}/roboelo"
P="${ROOT}/micromamba/envs/roboelo-univla/bin/python"
SRC="${ROOT}/micromamba/envs/roboelo-libero/lib/python3.10/site-packages"
DST="${ROOT}/micromamba/envs/roboelo-univla/lib/python3.10/site-packages"

while pgrep -u "$(id -u)" -f '[p]ip install' > /dev/null; do sleep 20; done
"${P}" -m pip freeze > "${ROOT}/logs/univla_env_before_fix_$(date +%Y%m%d_%H%M%S).txt"
"${P}" -m pip install --progress-bar off "numpy==1.26.4" "tensorflow==2.15.0" "mujoco==3.3.2" "robosuite==1.4.1" \
  "bddl==3.6.0" "easydict==1.13" "cloudpickle==2.1.0" "gym==0.25.2" "imageio-ffmpeg==0.6.0" "wandb==0.16.6" einops
# The prismatic package imports its RLDS data code (dlimp, tensorflow_datasets, tensorflow_graphics) even for
# evaluation. tensorflow-metadata 1.15 keeps protobuf at 3.20 (newer ones need protobuf 5, which tensorflow 2.15
# does not accept); tensorflow_graphics without its dependencies (OpenEXR would have to be compiled; not imported).
"${P}" -m pip install --progress-bar off "tensorflow_datasets==4.9.3" "tensorflow-metadata==1.15.0" \
  "git+https://github.com/moojink/dlimp_openvla"
"${P}" -m pip install --progress-bar off --no-deps "tensorflow_graphics==2021.12.3"
if ! "${P}" -c "import flash_attn" 2> /dev/null; then
  cp -a "${SRC}/flash_attn" "${SRC}/flash_attn-2.8.3.dist-info" "${SRC}/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so" "${DST}/"
fi
"${P}" - <<'PY'
import flash_attn, mujoco, numpy, robosuite, tensorflow, torch
print("UNIVLA_ENV_OK", "mujoco", mujoco.__version__, "robosuite", robosuite.__version__, "numpy", numpy.__version__,
      "tensorflow", tensorflow.__version__, "torch", torch.__version__, "flash_attn", flash_attn.__version__)
PY

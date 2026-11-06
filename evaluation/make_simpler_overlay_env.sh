#!/usr/bin/env bash
# A SimplerEnv evaluation environment for one model: a venv on top of the model's own environment
# (--system-site-packages) plus the simulation packages of ManiSkill2_real2sim/requirements.txt (SAPIEN 2.2.2 ...).
# simpler_env and mani_skill2_real2sim themselves are not installed: run_simpler_supervisor.sh puts the official
# SimplerEnv checkout on PYTHONPATH, so every model runs the same simulation code.
#
#   make_simpler_overlay_env.sh <new env folder> <base python> [extra pip requirements ...]
set -euo pipefail
dest="$(realpath -m "$1")"; base="$2"; shift 2
[[ -d "${dest}" ]] || "${base}" -m venv --system-site-packages "${dest}"
"${dest}/bin/python" -m pip install --progress-bar off "setuptools<70" "sapien==2.2.2" "gymnasium==0.29.1" h5py pyyaml tqdm \
  GitPython tabulate "gdown>=4.6.0" transforms3d imageio "imageio-ffmpeg" "trimesh==5.0.0" rtree ruckig mediapy "$@"
cd "${HOME}/roboelo/third_party/SimplerEnv"
PYTHONPATH="${PWD}:${PWD}/ManiSkill2_real2sim" "${dest}/bin/python" - <<'PY'
import numpy, sapien, simpler_env
from simpler_env.utils.env.env_builder import build_maniskill2_env
print("SIMPLER_OVERLAY_OK numpy", numpy.__version__, "sapien", sapien.__version__)
PY

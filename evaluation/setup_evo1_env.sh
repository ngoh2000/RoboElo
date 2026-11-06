#!/usr/bin/env bash
# envs/evo1: Evo-1's policy server (scripts/evo1_server_launcher.py) and its LIBERO client
# (scripts/run_evo1_libero.py). A venv on roboelo-libero (--system-site-packages: Python 3.10, torch 2.7.1+cu128,
# flash-attn 2.8.3, LIBERO with MuJoCo 3.3.2) with transformers 4.39.0, fvcore and websockets from
# Evo_1/requirements.txt (which pins torch 2.5.1; torch stays 2.7.1+cu128 for the RTX 5090).
# Evo-1 loads its backbone as "OpenGVLab/InternVL3-1B" (remote code): the server runs in run_cwd/evo1, where that
# name is a link to the local copy checkpoints/InternVL3-1B.
set -euo pipefail
ROOT="${HOME}/roboelo"
ENV="${ROOT}/envs/evo1"
[[ -d "${ENV}" ]] || "${ROOT}/micromamba/envs/roboelo-libero/bin/python" -m venv --system-site-packages "${ENV}"
for a in 1 2 3 4 5; do
  "${ENV}/bin/python" -m pip install --progress-bar off --retries 10 --timeout 120 "transformers==4.39.0" fvcore \
    "websockets>=12" einops pandas "numpy==1.26.4" && break
  sleep 30
done
mkdir -p "${ROOT}/run_cwd/evo1/OpenGVLab"
ln -sfn "${ROOT}/checkpoints/InternVL3-1B" "${ROOT}/run_cwd/evo1/OpenGVLab/InternVL3-1B"
"${ENV}/bin/python" -c "import transformers, torch, fvcore, websockets, mujoco; from websockets.sync.client import connect; print('evo1', transformers.__version__, torch.__version__, websockets.__version__, mujoco.__version__)"
cd "${ROOT}/third_party/Evo-1/Evo_1/scripts" && CUDA_VISIBLE_DEVICES= "${ENV}/bin/python" -c "import Evo1_server; print('Evo1_server imports ok')"
touch "${ROOT}/state/evo1_env.done"

#!/usr/bin/env bash
# envs/xiaomi-r0: Xiaomi-Robotics-0 policy server (scripts/xiaomi_policy_server.py) and its LIBERO client.
# A venv on roboelo-libero (--system-site-packages: Python 3.10, torch 2.7.1+cu128, flash-attn 2.8.3, LIBERO with
# MuJoCo 3.3.2) with transformers 4.57.1 as Xiaomi-Robotics-0's README asks (its model code comes with the checkpoint,
# trust_remote_code) and tyro for eval_libero/main.py. The README's own environment is Python 3.12 / torch 2.8.0; only
# deploy/server.py needs Python 3.12 (an f-string), and xiaomi_policy_server.py carries that code with the quotes changed.
set -euo pipefail
ROOT="${HOME}/roboelo"
ENV="${ROOT}/envs/xiaomi-r0"
[[ -d "${ENV}" ]] || "${ROOT}/micromamba/envs/roboelo-libero/bin/python" -m venv --system-site-packages "${ENV}"
for a in 1 2 3 4 5; do
  "${ENV}/bin/python" -m pip install --progress-bar off --retries 10 --timeout 120 "transformers==4.57.1" tyro \
    "numpy==1.26.4" && break
  sleep 30
done
"${ENV}/bin/python" -c "import transformers, torch, tyro, mujoco; print('xiaomi-r0', transformers.__version__, torch.__version__, mujoco.__version__)"
touch "${ROOT}/state/xiaomi_r0_env.done"

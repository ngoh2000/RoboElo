#!/usr/bin/env bash
# Environment for VLA-Adapter's LIBERO evaluation: envs/vla-adapter, a venv on top of roboelo-univla
# (--system-site-packages: transformers 4.40.1, torch 2.7.1+cu128, tensorflow 2.15, dlimp, flash-attn 2.8.3,
# LIBERO packages with MuJoCo 3.3.2; see fix_univla_env.sh), plus the few packages VLA-Adapter imports that
# UniVLA does not. VLA-Adapter's own prismatic/experiments packages come first on sys.path (run_vla_adapter_libero.py),
# ahead of UniVLA's editable install.
set -euo pipefail
ROOT="${HOME}/roboelo"
[[ -d "${ROOT}/envs/vla-adapter" ]] || "${ROOT}/micromamba/envs/roboelo-univla/bin/python" -m venv --system-site-packages "${ROOT}/envs/vla-adapter"
"${ROOT}/envs/vla-adapter/bin/python" -m pip install --progress-bar off json-numpy jsonlines rich "diffusers==0.30.3" uvicorn fastapi \
  msgpack msgpack_numpy

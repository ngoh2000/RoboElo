#!/usr/bin/env bash
# envs/internvla-a15: InternVLA-A1.5's LIBERO policy server (InternVLA-A-series
# evaluation/LIBERO/policy_server/server_policy.py). Its tutorial (tutorials/installation.md) asks for Python 3.11,
# torch 2.10.0, transformers 5.2.0 patched with the repository's model files, flash-attn 2.8.3,
# flash-linear-attention 0.5.0 and causal-conv1d 1.6.1. Here: a venv on roboelo-libero (--system-site-packages:
# Python 3.10, torch 2.7.1+cu128 and a flash-attn 2.8.3 build that runs on the RTX 5090) with transformers 5.2.0,
# flash-linear-attention 0.5.0 and the repository (pip install -e, its own LeRobot fork) with its dependencies, torch
# and numpy kept as they are (constraints). causal-conv1d is not built (Qwen3.5's code then uses its PyTorch path).
# The tutorial's step 7 (copy pi0 / pi05 / internvla_a1_5 transformers_replace/models into Transformers) is applied
# to this venv's own copy of Transformers only.
# Base VLM: the policy builds Qwen3.5-2B with from_pretrained before LeRobot's (non-strict) checkpoint load, as in the
# official set-up, so the full Qwen/Qwen3.5-2B is downloaded (checkpoints/Qwen3.5-2B, scripts/start_download_batch2.sh)
# and passed as --vlm_model_path.
set -euo pipefail
ROOT="${HOME}/roboelo"
ENV="${ROOT}/envs/internvla-a15"
REPO="${ROOT}/third_party/InternVLA-A-series"
[[ -d "${ENV}" ]] || "${ROOT}/micromamba/envs/roboelo-libero/bin/python" -m venv --system-site-packages "${ENV}"
CONSTRAINTS="${ENV}/constraints.txt"
cat > "${CONSTRAINTS}" <<'TXT'
torch==2.7.1+cu128
torchvision==0.22.1+cu128
numpy==1.26.4
torchcodec==0.4.0
TXT
for a in 1 2 3 4 5; do
  PIP_CONSTRAINT="${CONSTRAINTS}" "${ENV}/bin/python" -m pip install --progress-bar off --retries 10 --timeout 120 \
    "transformers==5.2.0" "flash-linear-attention==0.5.0" "peft>=0.17" -e "${REPO}" && break
  sleep 30
done
SITE="$("${ENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for p in pi0 pi05 internvla_a1_5; do cp -r "${REPO}/src/lerobot/policies/${p}/transformers_replace/models" "${SITE}/transformers/"; done

cd "${REPO}" && CUDA_VISIBLE_DEVICES= "${ENV}/bin/python" -c "
import sys; sys.path.insert(0, '.')
import torch, transformers, fla
from evaluation.LIBERO.policy_server.backends.policy_backend_internvla_a1_5 import *  # noqa
print('internvla-a15', transformers.__version__, torch.__version__, 'fla', fla.__version__)"
touch "${ROOT}/state/internvla_a15_env.done"

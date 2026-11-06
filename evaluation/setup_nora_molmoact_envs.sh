#!/usr/bin/env bash
# Environments for NORA's and MolmoAct's LIBERO evaluations. Both are venvs on top of
# roboelo-libero (--system-site-packages: torch 2.7.1+cu128, flash-attn 2.8.3, LIBERO packages with MuJoCo 3.3.2,
# robosuite 1.4.1) with the transformers version each repository asks for:
#   envs/nora      transformers 4.50.0, accelerate 1.5.2, qwen_vl_utils   (nora/inference/requirements.txt)
#   envs/molmoact  transformers 4.52.1, accelerate, einops                (molmoact README, section 5.2 LIBERO)
# (torch stays 2.7.1+cu128 for the RTX 5090; the repositories pin 2.4.0 / >=2.3.1.)
# NORA's code also fetches physical-intelligence/fast (its action tokenizer, remote code) and
# declare-lab/nora/norm_stats.json from the Hub; both are put into the Hugging Face cache here.
set -euo pipefail
ROOT="${HOME}/roboelo"
BASE="${ROOT}/micromamba/envs/roboelo-libero/bin/python"
for e in nora molmoact; do [[ -d "${ROOT}/envs/${e}" ]] || "${BASE}" -m venv --system-site-packages "${ROOT}/envs/${e}"; done
"${ROOT}/envs/nora/bin/python" -m pip install --progress-bar off "transformers==4.50.0" "accelerate==1.5.2" qwen_vl_utils
"${ROOT}/envs/molmoact/bin/python" -m pip install --progress-bar off "transformers==4.52.1" accelerate einops
HF_HOME="${ROOT}/hf_cache" "${ROOT}/envs/nora/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
print(snapshot_download("physical-intelligence/fast"))
print(hf_hub_download("declare-lab/nora", "norm_stats.json"))
PY
for e in nora molmoact; do "${ROOT}/envs/${e}/bin/python" -c "import transformers, torch, mujoco; print('${e}', transformers.__version__, torch.__version__, mujoco.__version__)"; done

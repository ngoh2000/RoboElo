#!/usr/bin/env bash
# Environment for RT-1-X and Octo on SimplerEnv: envs/simpler-tf, a venv on top of roboelo-simpler (SAPIEN 2.2.2,
# ManiSkill2_real2sim, simpler_env, numpy 1.24.4) with the inference packages of SimplerEnv's "Full Installation"
# (requirements_full_install.txt, tensorflow 2.15) and Octo 1.0 (commit 653c54a, as SimplerEnv's README says).
# TensorFlow 2.15 and jax 0.4.20 have no RTX 5090 kernels, and both models are small (RT-1 ~35M, Octo-base 93M
# parameters), so they run on the CPU; SAPIEN still renders on the GPU.
set -euo pipefail
ROOT="${HOME}/roboelo"
E="${ROOT}/envs/simpler-tf"
[[ -d "${E}" ]] || "${ROOT}/micromamba/envs/roboelo-simpler/bin/python" -m venv --system-site-packages "${E}"
P="${E}/bin/python"
"${P}" -m pip install --progress-bar off "setuptools<70" "numpy==1.24.4" "tensorflow==2.15.0" "tensorflow_hub==0.16.0" \
  "tf_agents==0.19.0" "tensorflow_datasets==4.9.4" "jax==0.4.20" "jaxlib==0.4.20" "flax==0.8.1" "distrax==0.1.5" \
  "chex==0.1.85" "optax==0.1.7" "orbax-checkpoint==0.4.4" ml_collections einops "transformers==4.40.1" \
  "scipy==1.12.0" mediapy "git+https://github.com/kvablack/dlimp@d08da3852c149548aaa8551186d619d87375df08"
if [[ ! -d "${ROOT}/third_party/octo" ]]; then
  git clone -q https://github.com/octo-models/octo.git "${ROOT}/third_party/octo"
  git -C "${ROOT}/third_party/octo" checkout -q 653c54acde686fde619855f2eac0dd6edad7116b
fi
"${P}" -m pip install --progress-bar off --no-deps -e "${ROOT}/third_party/octo"
"${P}" - <<'PY'
import jax, numpy, tensorflow, tf_agents, octo, simpler_env, sapien
print("SIMPLER_TF_OK numpy", numpy.__version__, "tf", tensorflow.__version__, "jax", jax.__version__, jax.devices())
PY

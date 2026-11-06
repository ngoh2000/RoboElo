#!/usr/bin/env bash
# One-time setup of the LIBERO simulation venv used by GR00T's rollout client.
#
# Same steps as third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/setup_libero.sh, with two differences:
#   * it does not delete ~/.libero; the LIBERO config is written to ~/roboelo/libero_config_groot instead
#   * uv is taken from the roboelo-download environment and uses the workspace cache
# Nothing outside third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv and the config folder is changed.
set -euxo pipefail

ROOT="${HOME}/roboelo"
UV="${ROOT}/micromamba/envs/roboelo-download/bin/uv"
PROJECT_REPO="${ROOT}/third_party/Isaac-GR00T"
LIBERO_REPO="${PROJECT_REPO}/external_dependencies/LIBERO"
LIBERO_UV_ENV="${PROJECT_REPO}/gr00t/eval/sim/LIBERO/libero_uv"
export UV_CACHE_DIR="${ROOT}/uv_cache"
export LIBERO_CONFIG_PATH="${ROOT}/libero_config_groot"

test -f "${LIBERO_REPO}/requirements.txt"   # the submodule is already checked out

rm -rf "${LIBERO_UV_ENV}"
mkdir -p "${LIBERO_UV_ENV}" "${LIBERO_CONFIG_PATH}"
"${UV}" venv "${LIBERO_UV_ENV}/.venv" --python 3.12
# shellcheck disable=SC1091
source "${LIBERO_UV_ENV}/.venv/bin/activate"

# LIBERO's pinned requirements predate Python 3.12 (same patch as the official script).
PATCHED_REQUIREMENTS="${LIBERO_UV_ENV}/requirements-py312.txt"
python - <<PY
from pathlib import Path

replacements = {
    "hydra-core": "hydra-core==1.3.2",
    "numpy": "numpy==1.26.4",
    "transformers": "transformers==4.57.3",
    "opencv-python": "opencv-python==4.10.0.84",
    "matplotlib": "matplotlib==3.9.4",
    "wandb": "wandb==0.18.7",
}
src = Path("${LIBERO_REPO}/requirements.txt")
dst = Path("${PATCHED_REQUIREMENTS}")
lines = []
for raw in src.read_text().splitlines():
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        lines.append(raw)
        continue
    name = stripped.split("==", 1)[0].strip().lower()
    lines.append(replacements.get(name, raw))
dst.write_text("\n".join(lines) + "\n")
PY
"${UV}" pip install --requirements "${PATCHED_REQUIREMENTS}"
"${UV}" pip install -e "${LIBERO_REPO}" --config-settings editable_mode=compat
"${UV}" pip install torch==2.9.0 torchvision==0.24.0 pydantic av tianshou==0.5.1 numba==0.65.1 llvmlite==0.47.0 tyro pandas dm_tree einops==0.8.1 albumentations==1.4.18 zmq
"${UV}" pip install transformers==4.57.3 msgpack==1.1.0 msgpack-numpy==0.4.8 gymnasium==0.29.1
"${UV}" pip install numpy==1.26.4 mujoco==3.3.1

# Expose gr00t from the repo root via a .pth (as the official script does).
python -c "import sysconfig, pathlib; pathlib.Path(sysconfig.get_path('purelib'), 'gr00t.pth').write_text(pathlib.Path('${PROJECT_REPO}').resolve().as_posix() + '\n')"

# First import asks whether to use a custom dataset folder; answer no. Writes ${LIBERO_CONFIG_PATH}/config.yaml.
rm -f "${LIBERO_CONFIG_PATH}/config.yaml"
printf 'n\n' | python -c "from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs"
cat "${LIBERO_CONFIG_PATH}/config.yaml"

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python - <<'PY'
from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
register_libero_envs()
import gymnasium as gym
env = gym.make("libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate")
env.reset()
env.close()
print("Env OK:", type(env))
PY
python - <<'PY'
import importlib.metadata as m
for p in ["torch", "numpy", "mujoco", "robosuite", "gymnasium", "transformers", "imageio", "imageio-ffmpeg", "av", "pyzmq", "msgpack"]:
    try:
        print(p, m.version(p))
    except Exception:
        print(p, "MISSING")
PY
echo "SETUP_DONE"

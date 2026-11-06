#!/usr/bin/env bash
# One-time set-up for RoboTwin 2.0 evaluations of LeRobot policies (SmolVLA, LingBot-VA).
# Follows LeRobot's docker/Dockerfile.benchmark.robotwin; differences are noted. Safe to re-run.
#   third_party/RoboTwin           at the commit LeRobot pins (0aeea2d)
#   envs/cuda-12.8                 nvcc 12.8 (micromamba), only to build CuRobo: torch here is cu128 and the system
#                                  toolkit is 13.2, which torch's extension builder refuses
#   envs/robotwin                  venv on roboelo-lerobot (Python 3.12, LeRobot with smolvla / lingbot_va) +
#                                  sapien 3.0.0b1, mplib 0.2.1, open3d 0.19, warp-lang 1.3.1 ... (Dockerfile pins);
#                                  CuRobo v0.7.8 built for the RTX 5090 (TORCH_CUDA_ARCH_LIST 12.0); pytorch3d is
#                                  skipped (RoboTwin imports it inside try/except, only for point clouds)
#   third_party/RoboTwin/assets    embodiments + objects (what the clean / easy setting needs, ~4 GB)
#   checkpoints/smolvla-robotwin-lerobot, checkpoints/lingbot-va-robotwin-lerobot
set -euo pipefail
R="${HOME}/roboelo"
RT="${R}/third_party/RoboTwin"
SHA=0aeea2d669c0f8516f4d5785f0aa33ba812c14b4
CUDA="${R}/envs/cuda-12.8"
PY="${R}/envs/robotwin/bin/python"
export MAMBA_ROOT_PREFIX="${R}/micromamba" HF_HOME="${R}/hf_cache" HF_HUB_DISABLE_XET=1
step() { printf '\n==== %s (%s)\n' "$*" "$(date -Is)"; }

step "RoboTwin at LeRobot's commit"
if [[ "$(git -C "${RT}" rev-parse HEAD)" != "${SHA}" ]]; then
  timeout 600 git -C "${RT}" fetch -q --depth 1 origin "${SHA}"
  git -C "${RT}" checkout -q "${SHA}"
fi
git -C "${RT}" log -1 --format='%h %ad %s' --date=short

step "nvcc 12.8"
[[ -x "${CUDA}/bin/nvcc" ]] || "${R}/tools/micromamba" create -y -p "${CUDA}" -c nvidia/label/cuda-12.8.1 -c conda-forge \
  cuda-nvcc cuda-cudart-dev cuda-cccl cuda-libraries-dev
"${CUDA}/bin/nvcc" --version | tail -1

step "envs/robotwin"
[[ -x "${PY}" ]] || "${R}/micromamba/envs/roboelo-lerobot/bin/python" -m venv --system-site-packages "${R}/envs/robotwin"
"${PY}" -c "import sapien, mplib, open3d, warp" 2>/dev/null || "${PY}" -m pip install --progress-bar off \
  "sapien==3.0.0b1" "mplib==0.2.1" "transforms3d==0.4.2" "trimesh==4.4.3" "open3d==0.19.0" "imageio==2.34.2" \
  termcolor zarr pydantic h5py "warp-lang==1.3.1"

step "CuRobo v0.7.8"
if ! "${PY}" -c "import torch, curobo.curobolib.kinematics_fused_cu" 2>/dev/null; then
  [[ -d "${RT}/envs/curobo" ]] || timeout 600 git clone -q --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "${RT}/envs/curobo"
  # conda's CUDA packages keep headers and libraries under targets/x86_64-linux, where torch's extension builder
  # (CUDA_HOME/include, CUDA_HOME/lib64) does not look
  (cd "${RT}/envs/curobo" && PATH="${CUDA}/bin:${PATH}" CUDA_HOME="${CUDA}" TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 \
     CPATH="${CUDA}/targets/x86_64-linux/include" LIBRARY_PATH="${CUDA}/targets/x86_64-linux/lib:${CUDA}/lib" \
     "${PY}" -m pip install --progress-bar off -e . --no-build-isolation)
fi
"${PY}" -c "import torch, curobo, curobo.curobolib.kinematics_fused_cu, numpy; print('curobo ok', curobo.__file__, 'numpy', numpy.__version__)"

step "mplib / sapien patches (as the Dockerfile)"
"${PY}" - <<'PY'
import pathlib, re, site
for d in site.getsitepackages():
    p = pathlib.Path(d) / "mplib" / "planner.py"
    if p.exists() and re.search(r"\bor collide\b", p.read_text()):
        p.write_text(re.sub(r"\bor collide\b", "", p.read_text(), count=1)); print("mplib patch applied:", p)
    p = pathlib.Path(d) / "sapien" / "wrapper" / "urdf_loader.py"
    if p.exists() and "with open(srdf_path) as f:" in p.read_text():
        p.write_text(p.read_text().replace("with open(srdf_path) as f:", 'with open(srdf_path, encoding="utf-8") as f:')
                     .replace('"srdf"', '".srdf"')); print("sapien patch applied:", p)
PY

step "numpy 1.26.4"
# CuRobo's install pulls numpy 2.5, and mplib 0.2.1 (old pybind11) segfaults on numpy 2 when it converts the robot's
# collision shapes (RoboTwin's requirements also pin numpy 1.26.4 for its native extensions). LeRobot declares
# numpy>=2 but its RoboTwin path, SmolVLA and LingBot-VA run on 1.26 (checked by the pipeline tests).
"${PY}" -c "import numpy, sys; sys.exit(0 if numpy.__version__ == '1.26.4' else 1)" || "${PY}" -m pip install --progress-bar off "numpy==1.26.4"

step "assets (embodiments, objects)"
for z in embodiments objects; do
  if [[ ! -d "${RT}/assets/${z}" ]]; then
    "${PY}" - "${RT}/assets" "${z}.zip" <<'PY'
import pathlib, sys, zipfile
from huggingface_hub import hf_hub_download
assets, name = pathlib.Path(sys.argv[1]), sys.argv[2]
assets.mkdir(parents=True, exist_ok=True)
for attempt in range(30):
    try:
        local = hf_hub_download(repo_id="TianxingChen/RoboTwin2.0", repo_type="dataset", filename=name, local_dir=str(assets))
        break
    except Exception as exc:
        print("retry", attempt, type(exc).__name__, str(exc)[:200], flush=True)
with zipfile.ZipFile(local) as z:
    z.extractall(str(assets))
pathlib.Path(local).unlink()
print("extracted", name)
PY
  fi
done
(cd "${RT}" && for s in script scripts; do [[ -f "${s}/update_embodiment_config_path.py" ]] && "${PY}" "${s}/update_embodiment_config_path.py" && break; done)

step "checkpoints"
python3 "${R}/scripts/download_hf_resume.py" lerobot/smolvla_robotwin=smolvla-robotwin-lerobot=smolvla_robotwin \
  lerobot/lingbot_va_robotwin=lingbot-va-robotwin-lerobot=lingbot_va_robotwin

step "import test"
(cd "${RT}" && PYTHONPATH="${RT}" "${PY}" -c "
from lerobot.envs.robotwin import RoboTwinEnv
import numpy as np
env = RoboTwinEnv('beat_block_hammer', episode_index=0)
obs, info = env.reset()
print('robotwin ok', {k: v.shape for k, v in obs['pixels'].items()}, obs['agent_pos'].shape)
obs, r, term, trunc, info = env.step(np.zeros(14, dtype=np.float32))
print('step ok', info)
env.close()")
date -Is > "${R}/state/robotwin_env.done"
echo "SETUP_ROBOTWIN_DONE"

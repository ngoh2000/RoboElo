#!/usr/bin/env bash
# One-time setup of the G0.5 (GalaxeaVLA) code and its Python environment, as its README describes:
# uv sync from the repository's lock file, then LIBERO installed without dependencies.
# Everything stays under ~/roboelo/third_party/GalaxeaVLA; no existing environment is touched.
set -euxo pipefail
ROOT="${HOME}/roboelo"
UV="${ROOT}/micromamba/envs/roboelo-download/bin/uv"
REPO="${ROOT}/third_party/GalaxeaVLA"
export UV_CACHE_DIR="${ROOT}/uv_cache"

if [[ ! -d "${REPO}/.git" ]]; then
  git clone --depth 1 https://github.com/OpenGalaxea/GalaxeaVLA.git "${REPO}"
fi
cd "${REPO}"
git rev-parse HEAD
# --frozen: install exactly what uv.lock lists. Without it this uv version re-resolves the project and
# fails, because it finds a wrong "packaging" on the PyTorch index before looking at PyPI.
# The link is slow and large wheels time out, so: long timeout, few downloads at once, and try again
# (wheels that are already in the cache are not downloaded twice).
export UV_HTTP_TIMEOUT=1800 UV_CONCURRENT_DOWNLOADS=3
synced=0
for try in $(seq 1 40); do
  echo "[$(date -Is)] uv sync attempt ${try}"
  if "${UV}" sync --frozen; then synced=1; break; fi
  sleep 20
done
[[ "${synced}" == "1" ]]
"${UV}" pip install -e "${ROOT}/third_party/LIBERO" --no-deps
# LIBERO's top-level folder has no __init__.py, so the editable install does not make "libero" importable:
# put the repository on this environment's path instead.
echo "${ROOT}/third_party/LIBERO" > "$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/roboelo_libero_repo.pth"
.venv/bin/python - <<'PY'
import importlib.metadata as m
import libero, torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
for p in ["transformers", "mujoco", "robosuite", "websockets", "hydra-core", "numpy", "flash-attn-4"]:
    try:
        print(p, m.version(p))
    except Exception:
        print(p, "MISSING")
PY
echo SETUP_DONE

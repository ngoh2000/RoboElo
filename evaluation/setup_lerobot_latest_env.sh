#!/usr/bin/env bash
# envs/lerobot-latest: LeRobot main (third_party/lerobot-latest, d3793b7 of 2026-10-23) for policies merged
# after our LeRobot checkout (30da8e6, 2026-07-06), first DM0.5 (lerobot/dm05_libero). Like envs/lerobot-mujoco332 it
# is a venv on roboelo-lerobot (--system-site-packages: torch 2.7.1+cu128, transformers 5.5.4, hf-libero 0.1.4) with
# MuJoCo 3.3.2; in addition draccus 0.11.6 (LeRobot main needs >= 0.11.6) and LeRobot main's src/ ahead of the
# installed LeRobot on sys.path (a .pth file, so the old checkout and its environments stay untouched).
set -euo pipefail
ROOT="${HOME}/roboelo"
ENV="${ROOT}/envs/lerobot-latest"
[[ -d "${ENV}" ]] || "${ROOT}/micromamba/envs/roboelo-lerobot/bin/python" -m venv --system-site-packages "${ENV}"
for a in 1 2 3 4 5; do
  "${ENV}/bin/python" -m pip install --progress-bar off --retries 10 --timeout 120 --no-deps "mujoco==3.3.2" \
    && "${ENV}/bin/python" -m pip install --progress-bar off --retries 10 --timeout 120 "draccus==0.11.6" && break
  sleep 30
done
SITE="$("${ENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "${ROOT}/third_party/lerobot-latest/src" > "${SITE}/00_lerobot_latest.pth"
LIBERO_CONFIG_PATH="${ROOT}/libero_config_lerobot" "${ENV}/bin/python" - <<'PY'
import lerobot, mujoco, transformers, draccus, importlib.metadata as m
print("lerobot", lerobot.__file__, "| mujoco", mujoco.__version__, "| transformers", transformers.__version__,
      "| draccus", m.version("draccus"))
from lerobot.policies.dm05 import DM05Policy  # noqa: F401
from lerobot.envs.libero import LiberoEnv  # noqa: F401
print("dm05 + libero import ok")
PY
touch "${ROOT}/state/lerobot_latest_env.done"

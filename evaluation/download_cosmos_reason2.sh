#!/usr/bin/env bash
# GR00T N1.7 loads the gated backbone nvidia/Cosmos-Reason2-2B (4.9 GB). The Hugging Face token must be
# in the environment as HF_TOKEN; it is not stored on this machine. Files go to the workspace cache, so
# the policy server can then run with HF_HUB_OFFLINE=1 and needs no token. Safe to re-run (resumes).
set -uo pipefail
ROOT="${HOME}/roboelo"
export HF_HOME="${ROOT}/hf_cache"
: "${HF_TOKEN:?HF_TOKEN is not set}"
for try in 1 2 3 4 5 6 7 8; do
  echo "[$(date -Is)] attempt ${try}"
  if "${ROOT}/micromamba/envs/roboelo-download/bin/hf" download nvidia/Cosmos-Reason2-2B; then
    echo "[$(date -Is)] exit 0"; date -Is > "${ROOT}/state/model_cosmos_reason2_2b.done"; exit 0
  fi
  sleep 30
done
echo "[$(date -Is)] failed"; exit 1

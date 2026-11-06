#!/usr/bin/env bash
# G0.5 LIBERO checkpoint from the gated repository OpenGalaxea/G05: g05-libero (11.4 GB), the action
# tokenizer and the processor files. The Hugging Face token must be in the environment as HF_TOKEN; it
# is not stored on this machine. Safe to re-run (resumes).
set -uo pipefail
ROOT="${HOME}/roboelo"
export HF_HOME="${ROOT}/hf_cache"
: "${HF_TOKEN:?HF_TOKEN is not set}"
for try in 1 2 3 4 5 6 7 8; do
  echo "[$(date -Is)] attempt ${try}"
  if "${ROOT}/micromamba/envs/roboelo-download/bin/hf" download OpenGalaxea/G05 \
       --include "g05-libero/*" --include "action_tokenizer.pt" --include "qwen3_5_2b_base_processor/*" \
       --include "README.md" --include "licenses/*" --local-dir "${ROOT}/checkpoints/G05"; then
    echo "[$(date -Is)] exit 0"; date -Is > "${ROOT}/state/model_g05_libero.done"; exit 0
  fi
  sleep 30
done
echo "[$(date -Is)] failed"; exit 1

#!/usr/bin/env bash
# Download Hugging Face model repositories one after another, each into its own folder, and write a
# marker file when one is complete. Safe to re-run (resumes; finished ones are skipped).
#
#   download_hf_models.sh <repo>=<folder under ~/roboelo/checkpoints, or "cache">=<marker name>[=<include glob,...>] ...
#
# "cache" stores the repository in the Hugging Face cache (HF_HOME) instead of a folder. A token is only
# used if HF_TOKEN is already in the environment (gated repositories); it is not stored on this machine.
set -uo pipefail
ROOT="${HOME}/roboelo"
HF="${ROOT}/micromamba/envs/roboelo-download/bin/hf"
export HF_HOME="${ROOT}/hf_cache"
# Plain HTTPS instead of the Xet transfer, which stalled for good on this machine's slow link; a read
# that hangs then times out, and the next attempt continues the partly downloaded file.
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
MAX_TRIES="${MAX_TRIES:-300}"
mkdir -p "${ROOT}/state"
failed=0
for spec in "$@"; do
  IFS='=' read -r repo folder marker include <<<"${spec}"
  done_file="${ROOT}/state/model_${marker}.done"
  if [[ -f "${done_file}" ]]; then echo "[$(date -Is)] ${repo}: already complete"; continue; fi
  args=(download "${repo}")
  if [[ "${folder}" != "cache" ]]; then args+=(--local-dir "${ROOT}/checkpoints/${folder}"); fi
  if [[ -n "${include:-}" ]]; then
    IFS=',' read -r -a globs <<<"${include}"
    for g in "${globs[@]}"; do args+=(--include "${g}"); done
  fi
  ok=0
  for try in $(seq 1 "${MAX_TRIES}"); do
    echo "[$(date -Is)] ${repo}: attempt ${try}"
    if "${HF}" "${args[@]}"; then ok=1; break; fi
    sleep 30
  done
  if [[ "${ok}" == "1" ]]; then
    date -Is > "${done_file}"; echo "[$(date -Is)] ${repo}: complete"
  else
    echo "[$(date -Is)] ${repo}: FAILED"; failed=1
  fi
done
echo "[$(date -Is)] ALL_DOWNLOADS_END failed=${failed}"
exit "${failed}"

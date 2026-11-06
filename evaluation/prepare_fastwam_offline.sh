#!/usr/bin/env bash
# FastWAM (lerobot/fastwam_libero_uncond_2cam224) offline. LeRobot's FastWAM (commit 30da8e68) loads
#   * the Wan2.2 VAE from Wan-AI/Wan2.2-TI2V-5B-Diffusers (subfolder vae), a repository name fixed in its code,
#   * the UMT5-XXL text encoder from config.text_encoder_model_id (same repository, subfolder text_encoder),
#   * the UMT5 tokenizer from config.tokenizer_model_id (google/umt5-xxl).
# The checkpoint repository itself does not contain them (its README says it does; the file list does not).
# The files were downloaded with download_hf_resume.py into checkpoints/; this script
#   1. makes a Hugging Face cache entry for Wan-AI/Wan2.2-TI2V-5B-Diffusers whose snapshot links to that folder,
#      so that the VAE is found with HF_HUB_OFFLINE=1,
#   2. makes checkpoints/fastwam-libero-lerobot-local, whose config.json names the text encoder and the tokenizer
#      by the local folders.
set -euo pipefail
ROOT="${HOME}/roboelo"
WAN="${ROOT}/checkpoints/Wan2.2-TI2V-5B-Diffusers"
SHA="b8fff7315c768468a5333511427288870b2e9635"   # Hub revision of Wan-AI/Wan2.2-TI2V-5B-Diffusers when downloaded
CACHE="${ROOT}/hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"
[[ -f "${WAN}/vae/diffusion_pytorch_model.safetensors" && -f "${WAN}/text_encoder/config.json" ]]
mkdir -p "${CACHE}/refs" "${CACHE}/snapshots/${SHA}"
printf '%s' "${SHA}" > "${CACHE}/refs/main"
for f in model_index.json vae text_encoder tokenizer; do ln -sfn "${WAN}/${f}" "${CACHE}/snapshots/${SHA}/${f}"; done
python3 "${ROOT}/scripts/prepare_local_checkpoint.py" "${ROOT}/checkpoints/fastwam-libero-lerobot" \
  "${ROOT}/checkpoints/fastwam-libero-lerobot-local" \
  "Wan-AI/Wan2.2-TI2V-5B-Diffusers=${WAN}" "google/umt5-xxl=${ROOT}/checkpoints/umt5-xxl-tokenizer"

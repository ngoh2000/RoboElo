#!/usr/bin/env bash
# Environments for the starVLA-family LIBERO runs (starVLA, InternVLA-M1, ABot-M0). Both are venvs
# on top of roboelo-libero (--system-site-packages: Python 3.10, torch 2.7.1+cu128, flash-attn 2.8.3, LIBERO with
# MuJoCo 3.3.2, robosuite 1.4.1) with what each repository's requirements.txt asks for at inference:
#   envs/starvla       transformers 4.57.0 (starVLA and ABot-M0 requirements), accelerate 1.5.2, qwen-vl-utils, ...;
#                      also the LIBERO client environment of all three (websockets, msgpack for their clients)
#   envs/internvla-m1  the same with transformers 4.52.3 (InternVLA-M1 requirements)
#   vggt (ABot-M0's spatial encoder) without its dependencies: it pins torch 2.3.1
# The repositories pin torch 2.6 / torchvision 0.21; torch stays 2.7.1+cu128 for the RTX 5090.
# Base VLMs: config, tokenizer and processor files only (checkpoints/base/...); see starvla_server_launcher.py.
set -euo pipefail
ROOT="${HOME}/roboelo"
BASE="${ROOT}/micromamba/envs/roboelo-libero/bin/python"
for e in starvla internvla-m1; do [[ -d "${ROOT}/envs/${e}" ]] || "${BASE}" -m venv --system-site-packages "${ROOT}/envs/${e}"; done
COMMON=("accelerate==1.5.2" qwen-vl-utils omegaconf diffusers timm einops websockets msgpack tyro typing_extensions
        "numpy==1.26.4" "transformers_stream_generator==0.0.4" tiktoken matplotlib "peft>=0.17")
pip_install() { local py="$1"; shift; for a in 1 2 3 4 5; do "${py}" -m pip install --progress-bar off --retries 10 --timeout 120 "$@" && return 0; sleep 30; done; return 1; }
pip_install "${ROOT}/envs/starvla/bin/python" "transformers==4.57.0" "${COMMON[@]}"
pip_install "${ROOT}/envs/starvla/bin/python" --no-deps "git+https://github.com/facebookresearch/vggt.git"
# imported by starVLA's server through its data loader (policy_norm_processor reuses the training transforms);
# pipablepytorch3d (pure-Python pytorch3d.transforms) without dependencies: it would pull another torch
pip_install "${ROOT}/envs/starvla/bin/python" "pydantic==2.10.6" "numpydantic==1.6.9" "albumentations==1.4.18"
pip_install "${ROOT}/envs/starvla/bin/python" --no-deps "pipablepytorch3d==0.7.6"
pip_install "${ROOT}/envs/internvla-m1/bin/python" "transformers==4.52.3" "${COMMON[@]}"

mkdir -p "${ROOT}/checkpoints/base"
base_files() {  # $1 = repo, then the files
  local repo="$1" dir="${ROOT}/checkpoints/base/$(basename "$1")" f; shift
  mkdir -p "${dir}"
  for f in "$@"; do
    [[ -s "${dir}/${f}" ]] || curl -sSfL --retry 10 -o "${dir}/${f}" "https://huggingface.co/${repo}/resolve/main/${f}"
  done
}
QWEN_FILES=(config.json generation_config.json chat_template.json merges.txt preprocessor_config.json tokenizer.json
            tokenizer_config.json vocab.json)
base_files Qwen/Qwen2.5-VL-3B-Instruct "${QWEN_FILES[@]}"
base_files Qwen/Qwen3-VL-4B-Instruct "${QWEN_FILES[@]}" video_preprocessor_config.json
# base_vlm of InternVLA-M1-Pretrain-RT-1-Bridge (SimplerEnv): Qwen2.5-VL-3B with InternVLA-M1's tokenizer files
base_files InternRobotics/InternVLA-M1 config.json generation_config.json chat_template.json merges.txt \
           preprocessor_config.json added_tokens.json special_tokens_map.json tokenizer_config.json vocab.json

# InternVLA-M1 builds DINOv2 ViT-S/14 with torch.hub.load("facebookresearch/dinov2", "dinov2_vits14") (its weights are
# then replaced by the checkpoint's): the hub code and weights go to TORCH_HOME=~/roboelo/torch_home in advance.
HUB="${ROOT}/torch_home/hub"
mkdir -p "${HUB}/checkpoints"
[[ -f "${HUB}/facebookresearch_dinov2_main/hubconf.py" ]] || \
  git clone --depth 1 https://github.com/facebookresearch/dinov2.git "${HUB}/facebookresearch_dinov2_main"
[[ -s "${HUB}/checkpoints/dinov2_vits14_pretrain.pth" ]] || curl -sSfL --retry 10 -o "${HUB}/checkpoints/dinov2_vits14_pretrain.pth" \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
grep -qx "facebookresearch_dinov2" "${HUB}/trusted_list" 2>/dev/null || echo "facebookresearch_dinov2" >> "${HUB}/trusted_list"

for e in starvla internvla-m1; do
  "${ROOT}/envs/${e}/bin/python" -c "import transformers, torch, mujoco, websockets, msgpack, omegaconf, qwen_vl_utils, diffusers; print('${e}', transformers.__version__, torch.__version__, mujoco.__version__)"
done
"${ROOT}/envs/starvla/bin/python" -c "from vggt.models.vggt import VGGT; print('vggt ok')"
touch "${ROOT}/state/starvla_envs.done"

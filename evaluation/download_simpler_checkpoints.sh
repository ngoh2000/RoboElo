#!/usr/bin/env bash
# SimplerEnv checkpoints of the policy ladder, one repository after another,
# with download_hf_resume.py (curl -C -, SHA-256 check).
cd ~/roboelo
exec python3 scripts/download_hf_resume.py \
  'openvla/openvla-7b=openvla-7b=openvla_7b' \
  'IPEC-COMMUNITY/spatialvla-4b-224-pt=spatialvla-4b-224-pt=spatialvla_4b_224_pt' \
  'CogACT/CogACT-Base=CogACT-Base=cogact_base' \
  'qwbu/univla-7b-224-sft-simpler-bridge=univla-7b-224-sft-simpler-bridge=univla_simpler_bridge' \
  'lerobot/xvla-widowx=xvla-widowx-lerobot=xvla_widowx' \
  'lerobot/xvla-google-robot=xvla-google-robot-lerobot=xvla_google_robot' \
  'nvidia/GR00T-N1.6-bridge=GR00T-N1.6-bridge=groot_n16_bridge' \
  'nvidia/GR00T-N1.6-fractal=GR00T-N1.6-fractal=groot_n16_fractal' \
  'lerobot/VLA-JEPA-SimplerEnv=vla-jepa-simplerenv-lerobot=vla_jepa_simplerenv' \
  'allenai/MolmoAct-7B-D-Pretrain-RT-1-0812=MolmoAct-7B-D-Pretrain-RT-1-0812=molmoact_pretrain_rt1'

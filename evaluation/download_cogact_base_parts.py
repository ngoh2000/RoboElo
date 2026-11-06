#!/usr/bin/env python
"""Files that CogACT's prismatic loader fetches from the Hub even when the weights come from the CogACT checkpoint,
put into the Hugging Face cache (HF_HOME) so that runs work with HF_HUB_OFFLINE=1:
  * the timm vision backbones, created with pretrained=True and then overwritten by the checkpoint: timm looks for the
    safetensors files first (model.safetensors, open_clip_model.safetensors), so only those are fetched;
  * meta-llama/Llama-2-7b-hf config and tokenizer. The repository is gated: while the access request is still being
    reviewed, it is tried again every 20 minutes (up to 3 days). HF_TOKEN comes from the environment, never stored.
Run with HF_HUB_DISABLE_XET=1 (Xet transfers stall on the server's link)."""
import datetime
import os
import time

from huggingface_hub import hf_hub_download, snapshot_download

for repo, files in [("timm/vit_large_patch14_reg4_dinov2.lvd142m", ["config.json", "model.safetensors"]),
                    ("timm/ViT-SO400M-14-SigLIP", ["open_clip_config.json", "open_clip_model.safetensors"])]:
    for name in files:
        print(hf_hub_download(repo, name), flush=True)
print("TIMM_BACKBONES_DONE", flush=True)

tok = os.environ.get("HF_TOKEN")
for attempt in range(1, 3 * 24 * 3 + 1):
    try:
        print(snapshot_download("meta-llama/Llama-2-7b-hf", token=tok, allow_patterns=[
            "config.json", "generation_config.json", "tokenizer.model", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json"]), flush=True)
        print("COGACT_PARTS_DONE", flush=True)
        break
    except Exception as exc:
        print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] Llama-2 attempt {attempt}: "
              f"{type(exc).__name__}: {str(exc).splitlines()[-1][:160]}", flush=True)
        time.sleep(20 * 60)

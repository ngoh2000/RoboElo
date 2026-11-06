#!/usr/bin/env python
"""Store a Hugging Face checkpoint's float32 weights as bfloat16, shard by shard, in place.

For checkpoints that are always loaded with torch_dtype=bfloat16 (MolmoAct's LIBERO evaluation does so), the
weights the model sees are the same, and the files take half the space. Each shard is rewritten to a temporary
file next to it and then renamed over the original, so only one extra shard's worth of disk is needed. Other
dtypes (integer buffers, bf16/fp16 tensors) are left as they are. model.safetensors.index.json gets the new total
size and config.json torch_dtype "bfloat16"; a marker file records the conversion.

  python convert_safetensors_bf16.py <checkpoint folder> [...]
"""
import glob
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

for folder in sys.argv[1:]:
    marker = os.path.join(folder, "CONVERTED_FP32_TO_BF16")
    if os.path.exists(marker):
        print(f"{folder}: already converted")
        continue
    shards = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    assert shards, f"no .safetensors in {folder}"
    total = 0
    for shard in shards:
        tensors, n_cast = {}, 0
        with safe_open(shard, framework="pt") as f:
            metadata = f.metadata()
            for name in f.keys():
                t = f.get_tensor(name)
                if t.dtype == torch.float32:
                    t, n_cast = t.to(torch.bfloat16), n_cast + 1
                tensors[name] = t.contiguous()
        tmp = shard + ".bf16tmp"
        save_file(tensors, tmp, metadata=metadata)
        before, after = os.path.getsize(shard), os.path.getsize(tmp)
        os.replace(tmp, shard)
        total += sum(t.numel() * t.element_size() for t in tensors.values())
        print(f"{os.path.basename(shard)}: {n_cast}/{len(tensors)} tensors cast, {before / 1e9:.2f} -> {after / 1e9:.2f} GB",
              flush=True)
        del tensors
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        data = json.load(open(index))
        data.setdefault("metadata", {})["total_size"] = total
        json.dump(data, open(index, "w"), indent=2)
    config = os.path.join(folder, "config.json")
    if os.path.exists(config):
        cfg = json.load(open(config))
        if cfg.get("torch_dtype") == "float32":
            cfg["torch_dtype"] = "bfloat16"
            json.dump(cfg, open(config, "w"), indent=2)
    with open(marker, "w") as f:
        f.write("float32 tensors of every *.safetensors file were cast to bfloat16 (convert_safetensors_bf16.py)\n")
    print(f"{folder}: done, weights now {total / 1e9:.2f} GB")

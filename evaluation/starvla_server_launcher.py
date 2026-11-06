#!/usr/bin/env python
"""Run a starVLA-family policy server (deployment/model_server/server_policy*.py of starVLA, InternVLA-M1 or ABot-M0)
with the loading changes this machine needs; the server code itself runs unchanged.

1. Base VLM: the framework builds the Qwen VLM from config.yaml's `base_vlm` and then loads the whole checkpoint with
   load_state_dict(strict=True), which replaces every weight. A base folder with only the config, tokenizer and
   processor files is therefore enough: the VLM is then built from its config without initializing weights, as
   from_pretrained itself does before loading. ROBOELO_VLM_VOCAB sets another vocabulary size (ABot-M0's
   Qwen3-VL-4B-Instruct-Action: Qwen3-VL-4B-Instruct plus 2048 FAST action tokens, 153984 rows in its checkpoint).
   ROBOELO_VLM_DIRS="name=dir;name=dir" maps a base_vlm written in a checkpoint config (by full name or folder
   name) to a local folder.
2. ROBOELO_VGGT_DEFAULT=1: ABot-M0's spatial encoder VGGT.from_pretrained('facebook/VGGT-1B') is built as VGGT()
   (the Hub config is the class defaults), for the same reason: its weights are in the checkpoint.
3. Attention stays what each code base asks for (flash_attention_2 or sdpa), unless flash-attn fails a small test
   on this GPU at start-up (or ROBOELO_ATTN is set): then sdpa is used and the server log says so.

  python starvla_server_launcher.py deployment/model_server/server_policy.py --ckpt_path <.pt> --port N [--use_bf16]
Run from the repository root (it is put on sys.path, as the official launch scripts do with PYTHONPATH).
"""
import os
import runpy
import sys

import torch
import transformers
from transformers.modeling_utils import no_init_weights

DIRS = dict(item.split("=", 1) for item in os.environ.get("ROBOELO_VLM_DIRS", "").split(";") if "=" in item)
VOCAB = int(os.environ.get("ROBOELO_VLM_VOCAB", "0") or 0)


def _flash_attn_works():
    try:
        from flash_attn import flash_attn_func
        q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
        flash_attn_func(q, q, q)
        torch.cuda.synchronize()
        return True
    except Exception as exc:  # not installed, or no kernels for this GPU
        print(f"[launcher] flash-attn unusable here ({type(exc).__name__}: {str(exc)[:200]}); using sdpa", flush=True)
        return False


ATTN = os.environ.get("ROBOELO_ATTN") or (None if _flash_attn_works() else "sdpa")


def _resolve(name):
    """A key of ROBOELO_VLM_DIRS is a full name or a folder name (base_vlm is written as an HF id or a path)."""
    name = str(name)
    return DIRS.get(name) or DIRS.get(os.path.basename(name.rstrip("/"))) or name


def _has_weights(path):
    return any(f.endswith((".safetensors", ".bin", ".pt")) for f in os.listdir(path))


def _patch_vlm(cls):
    orig = cls.from_pretrained

    def from_pretrained(model_id, *args, **kwargs):
        model_id = _resolve(model_id)
        if ATTN:
            kwargs["attn_implementation"] = ATTN
        if not (os.path.isdir(model_id) and not _has_weights(model_id)):
            return orig(model_id, *args, **kwargs)
        config = transformers.AutoConfig.from_pretrained(model_id)
        if VOCAB:
            for sub in (config, getattr(config, "text_config", None)):
                if sub is not None and hasattr(sub, "vocab_size"):
                    sub.vocab_size = VOCAB
        dtype = kwargs.get("dtype", kwargs.get("torch_dtype", torch.bfloat16))
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16
        attn = kwargs.get("attn_implementation")
        print(f"[launcher] {cls.__name__} from the config in {model_id} (dtype {dtype}, attention {attn}, vocab "
              f"{VOCAB or 'as configured'}); its weights come from the checkpoint", flush=True)
        with no_init_weights():
            return cls._from_config(config, torch_dtype=dtype, attn_implementation=attn)

    cls.from_pretrained = from_pretrained


def _patch_path_only(cls):
    orig = cls.from_pretrained
    cls.from_pretrained = lambda name, *a, **k: orig(_resolve(name), *a, **k)


for name in ("Qwen2_5_VLForConditionalGeneration", "Qwen3VLForConditionalGeneration"):
    if hasattr(transformers, name):
        _patch_vlm(getattr(transformers, name))
for name in ("AutoProcessor", "AutoTokenizer"):
    _patch_path_only(getattr(transformers, name))

if os.environ.get("ROBOELO_VGGT_DEFAULT") == "1":
    from vggt.models.vggt import VGGT

    def _vggt(*args, **kwargs):
        print("[launcher] VGGT() with the default configuration; its weights come from the checkpoint", flush=True)
        with no_init_weights():
            return VGGT()

    VGGT.from_pretrained = _vggt

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.getcwd())
runpy.run_path(script, run_name="__main__")

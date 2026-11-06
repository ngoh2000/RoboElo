#!/usr/bin/env python
"""MolmoAct inference with vLLM for many SimplerEnv workers at once.

This is allenai/SimplerEnv's MolmoActInferenceVLLM (simpler_env/policies/molmoact/molmoact_model_vllm.py; the MolmoAct
README replicates its results with vLLM) as a server: the same LLM(dtype bfloat16), SamplingParams(max_tokens=256,
temperature=0), chat-template prompt, MolmoActParser and unnorm_key fractal20220817_data. The README pins vLLM 0.8.5,
which has no RTX 5090 kernels; this runs vLLM 0.9.2 built for CUDA 12.8 (envs/molmoact-vllm) on its V0 engine, for which
MolmoAct's vLLM model code (written for 0.8.5) needs one change: vLLM >= 0.9 renames a config's llm_config to
text_config, which the model code does not expect. What the server adds: requests of several clients that arrive
together are generated as one batch (vLLM batches anyway); single and batched queries can decode differently, as any
two vLLM / Hugging Face runs can.

  python molmoact_vllm_server.py --checkpoint <dir> --port 18620 [--gpu-util 0.6] [--max-batch 16] [--wait-ms 50]

Requests (multiprocessing.connection, authkey b"roboelo"): {"cmd": "infer", "image": .npy bytes of a 256x256x3 uint8
image, "instruction": str} -> {"text": generated text, "action": parsed action (list) or None}; {"cmd": "info"}.
"""
import argparse
import importlib.util
import io
import os
import queue
import sys
import threading
import time
import traceback
from multiprocessing.connection import Listener

os.environ.setdefault("VLLM_USE_V1", "0")
REPO = os.path.expanduser("~/roboelo/third_party/SimplerEnv-MolmoAct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpu-util", type=float, default=0.6)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--wait-ms", type=float, default=50.0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, ModelRegistry
    from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
    from vllm.sampling_params import SamplingParams
    from vllm.transformers_utils import config as vllm_config_utils

    spec = importlib.util.spec_from_file_location("molmoact", os.path.join(REPO, "simpler_env", "policies", "molmoact", "molmoact.py"))
    mm = importlib.util.module_from_spec(spec)
    sys.modules["molmoact"] = mm
    spec.loader.exec_module(mm)
    ModelRegistry.register_model("MolmoActForActionReasoning", mm.MolmoActForActionReasoning)
    _MULTIMODAL_MODELS["MolmoActForActionReasoning"] = ("molmoact", "MolmoActForActionReasoning")
    vllm_config_utils._CONFIG_ATTRS_MAPPING.pop("llm_config", None)

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, torch_dtype="bfloat16",
                                              device_map="auto", padding_side="left")
    llm = LLM(model=args.checkpoint, trust_remote_code=True, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len)
    sampling = SamplingParams(max_tokens=256, temperature=0)
    parser = mm.MolmoActParser.from_pretrained(args.checkpoint)

    def prompt(instruction):
        text = (f"The task is {instruction}. What is the action that the robot should take. "
                f"To figure out the action that the robot should take to {instruction}, let's think through it step by step. "
                "First, what is the depth map for this image? Second, what is the trajectory of the end effector? "
                "Based on the depth map of the image and the trajectory of the end effector, what is the action that the "
                "robot should take?")
        return processor.apply_chat_template([{"role": "user", "content": [dict(type="text", text=text)]}],
                                             tokenize=False, add_generation_prompt=True)

    pending = queue.Queue()
    stats = {"queries": 0, "batches": 0}

    def batch_loop():
        while True:
            items = [pending.get()]
            deadline = time.time() + args.wait_ms / 1000.0
            while len(items) < args.max_batch:
                try:
                    items.append(pending.get(timeout=max(0.0, deadline - time.time())))
                except queue.Empty:
                    break
            try:
                inputs = [{"prompt": prompt(req["instruction"]),
                           "multi_modal_data": {"image": [Image.fromarray(np.load(io.BytesIO(req["image"]), allow_pickle=False))]}}
                          for req, _, _ in items]
                outs = llm.generate(inputs, sampling_params=sampling, use_tqdm=False)
                for (req, done, slot), out in zip(items, outs):
                    text = out.outputs[0].text
                    try:
                        action = np.asarray(parser.parse_action(text, unnorm_key="fractal20220817_data"), dtype=float).tolist()
                    except Exception:
                        action = None
                    slot["reply"] = {"text": text, "action": action}
                    done.set()
                stats["queries"] += len(items)
                stats["batches"] += 1
                if stats["batches"] % 200 == 0:
                    print(f"[server] {stats['queries']} queries in {stats['batches']} batches "
                          f"(mean batch {stats['queries'] / stats['batches']:.1f})", flush=True)
            except Exception as exc:
                traceback.print_exc()
                for _, done, slot in items:
                    slot["reply"] = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
                    done.set()
                if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
                    print("[server] CUDA error: exiting so that the supervisor restarts the server", flush=True)
                    os._exit(3)

    def serve(conn):
        with conn:
            while True:
                try:
                    req = conn.recv()
                except (EOFError, OSError):
                    return
                if req.get("cmd") == "info":
                    conn.send({"model": "molmoact", "checkpoint": args.checkpoint, "max_batch": args.max_batch, **stats})
                    continue
                done, slot = threading.Event(), {}
                pending.put((req, done, slot))
                done.wait()
                conn.send(slot["reply"])

    threading.Thread(target=batch_loop, daemon=True).start()
    listener = Listener(("127.0.0.1", args.port), authkey=b"roboelo")
    print(f"[server] MolmoAct vLLM server on 127.0.0.1:{args.port}, max batch {args.max_batch}", flush=True)
    while True:
        try:
            conn = listener.accept()
        except Exception:  # a bare port probe fails the handshake
            continue
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()

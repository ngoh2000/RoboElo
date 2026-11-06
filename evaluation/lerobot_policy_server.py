#!/usr/bin/env python
"""A LeRobot policy behind a local socket, for evaluation clients that run in another Python.

SimplerEnv needs Python 3.10 (SAPIEN 2.2.2), the LeRobot ports (X-VLA, VLA-JEPA) need Python 3.12, so the policy runs
here and the SimplerEnv adapters in simpler_policies.py are its clients. Started by run_simpler_supervisor.sh
(SERVER_CMD) in envs/lerobot-mujoco332:

  python lerobot_policy_server.py --checkpoint <LeRobot checkpoint folder> --port 18605 [--device cuda]

Requests over multiprocessing.connection (127.0.0.1, authkey b"roboelo"; arrays travel as .npy bytes, so the two
sides need not share numpy versions):
  {"cmd": "info"}  -> {"type", "chunk_size", "input_features"}
  {"cmd": "chunk", "images": {feature key: uint8 HxWx3}, "state": float vector or None, "task": str,
   "extra": {batch key: number}, "postprocess": bool}
      -> {"chunk": float32 (chunk_size, action_dim)}: the policy's pre-processor (tokenizer, normalizer, device), then
         predict_action_chunk, then the post-processor unless postprocess is False (normalized model output).
Images are given as the camera delivers them (HWC uint8) and become float CHW in [0, 1], as LeRobot's env processors
hand them to a policy. One request at a time; several clients may stay connected.
"""
import argparse
import io
import threading
import traceback
from multiprocessing.connection import Listener

import numpy as np
import torch


def npy(a):
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
    return buf.getvalue()


def from_npy(b):
    return np.load(io.BytesIO(b), allow_pickle=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pipeline", choices=["saved", "code"], default="saved",
                    help="pre/post-processors saved with the checkpoint, or the policy code's current defaults (the "
                         "X-VLA WidowX / Google Robot uploads saved an older pipeline without ImageNet normalization)")
    ap.add_argument("--image-range", type=float, default=1.0,
                    help="images go to the pre-processor as float CHW in [0, image-range] (X-VLA's code pipeline "
                         "divides by 255 itself)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="override a policy config attribute after loading, before the model and processors are "
                         "built (X-VLA: the original processor pads to num_views 3 and language to 50 tokens)")
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.factory import get_policy_class

    import json

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = args.device
    cfg.pretrained_path = args.checkpoint
    for item in args.set:
        key, _, value = item.partition("=")
        assert hasattr(cfg, key), f"the {cfg.type} config has no attribute {key}"
        print(f"[server] config {key}: {getattr(cfg, key)!r} -> {json.loads(value)!r}", flush=True)
        setattr(cfg, key, json.loads(value))
    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint, config=cfg)
    policy.to(args.device).eval()
    if args.pipeline == "saved":
        pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.checkpoint,
                                             preprocessor_overrides={"device_processor": {"device": args.device}})
    else:
        pre, post = make_pre_post_processors(policy_cfg=policy.config)
    info = {"type": cfg.type, "chunk_size": int(getattr(cfg, "chunk_size", 1)), "pipeline": args.pipeline,
            "preprocessor": [type(s).__name__ for s in pre.steps],
            "input_features": {k: list(v.shape) for k, v in cfg.input_features.items()}}
    print(f"[server] {cfg.type} from {args.checkpoint} on {args.device}: {info}", flush=True)
    lock = threading.Lock()

    def chunk(req):
        batch = {}
        for key, data in req["images"].items():
            img = from_npy(data)
            batch[key] = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().mul(args.image_range / 255.0)[None]
        if req.get("state") is not None:
            batch["observation.state"] = torch.from_numpy(from_npy(req["state"]).astype(np.float32))[None]
        batch["task"] = [req["task"]]
        batch = pre(batch)
        for key, value in (req.get("extra") or {}).items():
            batch[key] = torch.as_tensor([value], device=args.device)
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch)
            if req.get("postprocess", True):
                actions = post(actions)
        return actions[0].float().cpu().numpy()

    def serve(conn):
        with conn:
            while True:
                try:
                    req = conn.recv()
                except (EOFError, OSError):
                    return
                try:
                    if req.get("cmd") == "info":
                        reply = info
                    elif req.get("cmd") == "chunk":
                        with lock:
                            reply = {"chunk": npy(chunk(req))}
                    else:
                        reply = {"error": f"unknown command {req.get('cmd')!r}"}
                except Exception as exc:  # the client records it as a failed attempt
                    traceback.print_exc()
                    reply = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
                    if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
                        conn.send(reply)
                        print("[server] CUDA error: exiting so that the supervisor restarts the server", flush=True)
                        import os

                        os._exit(3)
                conn.send(reply)

    listener = Listener(("127.0.0.1", args.port), authkey=b"roboelo")
    print(f"[server] listening on 127.0.0.1:{args.port}", flush=True)
    while True:
        try:
            conn = listener.accept()
        except Exception:  # a bare port probe fails the handshake
            continue
        threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()

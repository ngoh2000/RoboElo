#!/usr/bin/env python
"""Xiaomi-Robotics-0 policy server for several evaluation clients at once.

The official deploy/server.py serves one connection at a time (the official set-up starts one server per client,
8 servers on 8 GPUs). This server loads the model once and serves every connection in its own thread; requests are
answered one at a time under a lock. Everything else is deploy/server.py: the same model loading
(AutoModel.from_pretrained(trust_remote_code=True, attn_implementation="flash_attention_2", dtype=bfloat16), the
processor with use_fast=False), the same request handling (prompt template, processor call, action mask, state,
model forward, processor.decode_action) and the same wire format (4-byte length prefix + pickle), so the official
deploy/client.py Client talks to it unchanged. If flash-attn fails a small test on this GPU, sdpa is used (logged).

  python xiaomi_policy_server.py --model <checkpoint dir> --port 18841
"""
import argparse
import os
import pickle
import socket
import struct
import threading
import time
import traceback

import torch
from transformers import AutoModel, AutoProcessor


def flash_attn_works():
    try:
        from flash_attn import flash_attn_func
        q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
        flash_attn_func(q, q, q)
        torch.cuda.synchronize()
        return True
    except Exception as exc:
        print(f"[server] flash-attn unusable here ({type(exc).__name__}: {str(exc)[:200]}); using sdpa", flush=True)
        return False


def recv_all(conn, length):
    data = b""
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()

    attn = "flash_attention_2" if flash_attn_works() else "sdpa"
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, attn_implementation=attn,
                                      dtype=torch.bfloat16).cuda().to(torch.bfloat16)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    lock = threading.Lock()
    stats = {"requests": 0, "seconds": 0.0}

    def handle(input_data):
        """deploy/server.py Server.run, the body of its request loop."""
        robot_type = input_data["task_id"]
        data = {key: (value.to(model.device, model.dtype) if isinstance(value, torch.Tensor) else value)
                for key, value in input_data.items()}
        if "bridge" in robot_type or "fractal" in robot_type:
            instruction = data["language"]
            vl_inputs = processor(text=[instruction], images=[data["base"]], videos=None, padding=True,
                                  return_tensors="pt")
        else:
            instruction = ("<|im_start|>user\nThe following observations are captured from multiple views.\n"
                           "# Base View\n<|vision_start|><|image_pad|><|vision_end|>\n"
                           "# Left-Wrist View\n<|vision_start|><|image_pad|><|vision_end|>\n"
                           f"Generate robot actions for the task:\n{data['language']} /no_cot<|im_end|>\n"
                           "<|im_start|>assistant\n<cot></cot><|im_end|>\n")
            vl_inputs = processor(text=[instruction], images=[data["base"], data["wrist_left"]], videos=None,
                                  padding=True, return_tensors="pt")
        vl_inputs = vl_inputs.to(model.device)
        data.update(vl_inputs)
        data["action_mask"] = processor.get_action_mask(robot_type).to(model.device, model.dtype)
        data["state"] = torch.from_numpy(data["state"]).to(model.device, model.dtype).view(1, 1, -1)
        outputs = model(**data)
        return processor.decode_action(outputs.actions, robot_type=robot_type)

    def serve(conn):
        with conn:
            while True:
                head = recv_all(conn, 4)
                if not head:
                    return
                payload = recv_all(conn, struct.unpack(">I", head)[0])
                if not payload:
                    return
                tic = time.time()
                try:
                    with lock, torch.inference_mode():
                        action = handle(pickle.loads(payload))
                except Exception:
                    traceback.print_exc()
                    if "CUDA" in traceback.format_exc():
                        print("[server] CUDA error: exiting so that the supervisor restarts the server", flush=True)
                        os._exit(3)
                    return  # the client sees the connection close and records an exception
                response = pickle.dumps(action.cpu())
                conn.sendall(struct.pack(">I", len(response)) + response)
                stats["requests"] += 1
                stats["seconds"] += time.time() - tic
                if stats["requests"] % 500 == 0:
                    print(f"[server] {stats['requests']} requests, {1000 * stats['seconds'] / stats['requests']:.0f} ms "
                          f"each on average", flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((args.host, args.port))
        server_socket.listen(16)
        print(f"[server] Xiaomi-Robotics-0 ({args.model}, attention {attn}) on {args.host}:{args.port}", flush=True)
        while True:
            conn, _ = server_socket.accept()
            threading.Thread(target=serve, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()

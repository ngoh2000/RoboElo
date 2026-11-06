#!/usr/bin/env python
"""Evo-1 policy server: Evo-1's Evo_1/scripts/Evo1_server.py (load_model_and_normalizer, handle_request ->
infer_from_json_dict, the websockets.serve call of its __main__ block) with the three settings that block leaves to
the user: checkpoint folder, port, and arm / dataset key. The released MINT-SJTU/Evo1_LIBERO is an "old version"
checkpoint whose norm_stats.json has the key "libero_robot"; the README says to set only arm_key for those
(dataset_key stays "libero_4_datasets"). Evo-1 loads its backbone as "OpenGVLab/InternVL3-1B": run this from a folder
that holds OpenGVLab/InternVL3-1B as a local copy (transformers then reads it instead of the Hub).

  python evo1_server_launcher.py --checkpoint <Evo1_LIBERO folder> --port 18851 [--arm-key libero_robot]
"""
import argparse
import asyncio
import os
import sys

ROOT = os.path.expanduser("~/roboelo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(ROOT, "third_party", "Evo-1"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--arm-key", default="libero_robot")
    ap.add_argument("--dataset-key", default="libero_4_datasets")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, "Evo_1", "scripts"))
    import Evo1_server as server  # puts Evo_1/ on sys.path itself
    import websockets

    print("Loading EVO_1 model...", flush=True)
    model, normalizer = server.load_model_and_normalizer(args.checkpoint)

    async def serve():
        print(f"EVO_1 server running at ws://{args.host}:{args.port} (arm_key {args.arm_key}, dataset_key "
              f"{args.dataset_key})", flush=True)
        async with websockets.serve(
                lambda ws: server.handle_request(ws, model, normalizer, args.arm_key, args.dataset_key),
                args.host, args.port, max_size=100_000_000, ping_interval=None, ping_timeout=None):
            await asyncio.Future()

    asyncio.run(serve())


if __name__ == "__main__":
    main()

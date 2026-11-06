#!/usr/bin/env python3
"""A small subset of SimplerEnv's official episode list for a pipeline test: the first N episodes of every
(robot family, task, protocol, env) group, keeping their episode_index so that the test rows match the full run.

  make_simpler_smoke_episodes.py <out.jsonl> [--families widowx,google_robot] [--per-group 1] [--protocols ...]
"""
import argparse
import collections
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--episodes", default=os.path.expanduser("~/roboelo/state/simpler/episodes_official.jsonl"))
ap.add_argument("--families", default="widowx,google_robot")
ap.add_argument("--protocols", default="visual_matching,variant_aggregation")
ap.add_argument("--per-group", type=int, default=1)
args = ap.parse_args()

families, protocols = args.families.split(","), args.protocols.split(",")
taken, out = collections.Counter(), []
for line in open(args.episodes):
    ep = json.loads(line)
    if ep["robot_family"] not in families or ep["protocol"] not in protocols:
        continue
    key = (ep["robot_family"], ep["task"], ep["protocol"], ep["env_name"])
    if taken[key] < args.per_group:
        taken[key] += 1
        out.append(line)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as f:
    f.writelines(out)
print(f"{len(out)} episodes in {len(taken)} groups -> {args.out}")

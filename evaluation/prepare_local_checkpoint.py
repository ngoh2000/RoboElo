#!/usr/bin/env python3
"""Make a copy of a LeRobot checkpoint folder whose JSON files name Hub repositories by local folders.

Runs are offline (HF_HUB_OFFLINE=1). Several checkpoints name a base model or tokenizer by its Hub repository
(X-VLA: facebook/bart-large in policy_preprocessor.json; VLA-JEPA: Qwen/Qwen3-VL-2B-Instruct, ...), and
transformers cannot always resolve those from the cache offline. The copy links every file of the source
folder except the top-level *.json files; in those, every string equal to <repo> is replaced by <folder>.

  prepare_local_checkpoint.py <source folder> <destination folder> <repo>=<local folder> [...]

Every <repo> must be found at least once, and every <local folder> must exist.
"""
import json
import os
import sys

src, dst = (os.path.abspath(os.path.expanduser(p)) for p in sys.argv[1:3])
mapping = {}
for item in sys.argv[3:]:
    repo, _, folder = item.partition("=")
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        sys.exit(f"missing local folder {folder}")
    mapping[repo] = folder
if not os.path.isfile(os.path.join(src, "config.json")):
    sys.exit(f"no config.json in {src}")

found = dict.fromkeys(mapping, 0)


def replace(x):
    if isinstance(x, dict):
        return {k: replace(v) for k, v in x.items()}
    if isinstance(x, list):
        return [replace(v) for v in x]
    if isinstance(x, str) and x in mapping:
        found[x] += 1
        return mapping[x]
    return x


os.makedirs(dst, exist_ok=True)
for name in sorted(os.listdir(src)):
    s, d = os.path.join(src, name), os.path.join(dst, name)
    if name.endswith(".json") and os.path.isfile(s):
        before = dict(found)
        data = replace(json.load(open(s)))
        if os.path.islink(d):
            os.remove(d)
        with open(d, "w") as f:
            json.dump(data, f, indent=2)
        changed = {k: found[k] - before[k] for k in found if found[k] != before[k]}
        print(f"{name}: {changed or 'unchanged'}")
    else:
        if os.path.lexists(d):
            os.remove(d)
        os.symlink(s, d)
missing = [k for k, n in found.items() if n == 0]
if missing:
    sys.exit(f"not found in any JSON file: {missing}")

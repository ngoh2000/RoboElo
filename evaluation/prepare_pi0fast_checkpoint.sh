#!/usr/bin/env bash
# pi0-FAST (lerobot/pi0fast-libero): make a copy of the checkpoint folder (links to the same files) whose
# config.json and policy_preprocessor.json name the action tokenizer by a local folder instead of the
# Hub repository jadechoghari/tokenizer-lib-mean. The folder holds the same files; with the repository
# name, transformers still asks the Hub for bpe_tokenizer/config.json and fails when offline.
set -euo pipefail
ROOT="${HOME}/roboelo"
SRC="${ROOT}/checkpoints/pi0fast-libero-lerobot"
DST="${ROOT}/checkpoints/pi0fast-libero-lerobot-local"
TOK="${ROOT}/checkpoints/pi0fast-action-tokenizer"
[[ -f "${TOK}/processor_config.json" && -f "${SRC}/config.json" ]]
mkdir -p "${DST}"
for f in "${SRC}"/*; do
  case "$(basename "${f}")" in
    config.json|policy_preprocessor.json) ;;
    *) ln -sfn "${f}" "${DST}/$(basename "${f}")" ;;
  esac
done
python3 - "${SRC}" "${DST}" "${TOK}" <<'PY'
import json, sys
src, dst, tok = sys.argv[1:]
cfg = json.load(open(f"{src}/config.json"))
old = cfg["action_tokenizer_name"]
cfg["action_tokenizer_name"] = tok
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
pre = json.load(open(f"{src}/policy_preprocessor.json"))
n = 0
for step in pre["steps"]:
    if step.get("config", {}).get("action_tokenizer_name") == old:
        step["config"]["action_tokenizer_name"] = tok
        n += 1
assert n == 1, n
json.dump(pre, open(f"{dst}/policy_preprocessor.json", "w"), indent=2)
print(f"action tokenizer {old} -> {tok}")
PY

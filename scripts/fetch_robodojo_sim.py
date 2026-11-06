"""Fetch RoboDojo simulation per-task results from the official leaderboard site.

The leaderboard (https://robodojo-benchmark.com/leaderboard) ships its per-task
table ("RoboDojo Sim Per-Task Scores") inside the front-end JS bundle as a
JSON.parse('...') literal, together with the task -> capability-dimension map.
This script downloads the bundle, extracts both, and writes a tidy CSV.

Usage:  python scripts/fetch_robodojo_sim.py [--bundle local_index.js]
"""
import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SITE = "https://robodojo-benchmark.com"
ROOT = Path(__file__).resolve().parents[1]  # repository root
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

# Episodes per test case in the RoboDojo sim protocol (paper Sec. 5): 50 per task;
# Generalization tasks are split into 25 standard + 25 randomized episodes.
N_EPISODES_DEFAULT = 50
N_EPISODES_GEN_SPLIT = 25

DIM_VARS = {"generalization", "precision", "longHorizon", "memory", "open"}


def http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 RoboElo-research"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8")


def find_bundle_url() -> str:
    html = http_get(f"{SITE}/leaderboard")
    m = re.search(r'src="(/assets/index-[^"]+\.js)"', html)
    if not m:
        sys.exit("could not locate index bundle in leaderboard HTML")
    return SITE + m.group(1)


def extract_per_task(js: str) -> tuple[dict, str]:
    # const <var>="<iso date>",<var>=JSON.parse('{"A1":{"align_blocks":{"score":..
    m = re.search(r'"(\d{4}-\d\d-\d\dT[^"]+)",\w+=JSON\.parse\(\'(\{"[^\']*?"successRate"[^\']*)\'\)', js)
    if not m:
        sys.exit("per-task JSON literal not found; bundle layout changed")
    return json.loads(m.group(2)), m.group(1)


def extract_dimensions(js: str) -> dict[str, list[str]]:
    # ue("generalization",Pr) ... where Pr=["stack_bowls",...]
    arrays = dict(re.findall(r'(\w+)=(\[(?:"[A-Za-z0-9_]+",?)+\])', js))
    pairs = re.findall(r'\w+\("(generalization|precision|longHorizon|memory|open)",(\w+)\)', js)
    dims = {}
    for dim, var in pairs:
        if var in arrays:
            dims[dim] = json.loads(arrays[var])
    if set(dims) != DIM_VARS:
        sys.exit(f"dimension map incomplete: {sorted(dims)}")
    return dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", help="use a local copy of the JS bundle instead of downloading")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.bundle:
        js = Path(args.bundle).read_text(encoding="utf-8")
        src = args.bundle
    else:
        src = find_bundle_url()
        js = http_get(src)
        (RAW / "leaderboard_bundle.js").write_text(js, encoding="utf-8")

    per_task, data_date = extract_per_task(js)
    dims = extract_dimensions(js)
    (RAW / "per_task.json").write_text(json.dumps(per_task, indent=1), encoding="utf-8")
    (RAW / "dimensions.json").write_text(json.dumps(dims, indent=1), encoding="utf-8")

    task2dim = {t: d for d, ts in dims.items() for t in ts}
    gen_tasks = set(dims["generalization"])
    rows = []
    for policy, tasks in per_task.items():
        for tc, v in tasks.items():
            base = tc[: -len("_random")] if tc.endswith("_random") else tc
            variant = "random" if tc.endswith("_random") else "standard"
            n = N_EPISODES_GEN_SPLIT if base in gen_tasks else N_EPISODES_DEFAULT
            rows.append(dict(
                policy=policy, test_case=tc, task=base, variant=variant,
                dimension=task2dim.get(base, "unknown"),
                success_rate=v.get("successRate"), score=v.get("score"), n_episodes=n,
            ))
    df = pd.DataFrame(rows)
    df["successes"] = (df.success_rate / 100 * df.n_episodes).round().astype("Int64")
    df["data_date"] = data_date
    df["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df["source"] = src
    df.sort_values(["policy", "dimension", "test_case"]).to_csv(OUT / "robodojo_sim_per_task.csv", index=False)
    print(f"{df.policy.nunique()} policies x {df.test_case.nunique()} test cases -> {len(df)} rows "
          f"(data date {data_date})")
    unk = df[df.dimension == "unknown"].test_case.unique()
    if len(unk):
        print("WARNING: test cases without dimension:", unk)


if __name__ == "__main__":
    main()

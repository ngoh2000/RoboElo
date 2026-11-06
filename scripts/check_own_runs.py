"""Check our own LIBERO and SimplerEnv runs (data/own_runs/) for completeness before they enter the fit.

For every run folder it reports episodes recorded against episodes expected, duplicate and invalid rows, logged
errors, and episodes finished inside a given outage window (default: the computenode-01 GPU fault, 29 Oct 22:00
to 30 Oct 03:00, server time).

  python scripts/check_own_runs.py                       # all runs
  python scripts/check_own_runs.py ecot xiaomi_r0        # only folders starting with these prefixes
  python scripts/check_own_runs.py --outage 2026-10-29T22:00 2026-10-30T03:00
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OWN = ROOT / "data" / "own_runs"
args = [a for a in sys.argv[1:]]
outage = ("2026-10-29T22:00", "2026-10-30T03:00")
if "--outage" in args:
    i = args.index("--outage")
    outage = (args[i + 1], args[i + 2])
    del args[i:i + 3]
prefixes = args
LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SIMPLER_EXPECTED = {"google_robot": 2856, "widowx": 96}   # official episode lists: both protocols, all tasks


def in_outage(ts):
    t = pd.to_datetime(ts, errors="coerce")
    return int(((t >= pd.Timestamp(outage[0])) & (t < pd.Timestamp(outage[1]))).sum())


def check_simpler(folder):
    e = pd.read_csv(folder / "episodes.csv", on_bad_lines="skip")
    configs = [json.load(open(f)) for f in sorted(glob.glob(str(folder / "run_config_*.json")))]
    # robot families the run was started for (older configs do not record it: then the families in the data)
    fams = {f for c in configs for f in str(c.get("families", "")).split(",") if f} or set(e.robot_family)
    expected = {f: SIMPLER_EXPECTED.get(f) for f in fams}
    rows = []
    for fam, g in e.groupby("robot_family"):
        valid = g[g.success.isin([0, 1])]
        rows.append(dict(part=fam, expected=expected.get(fam), recorded=valid.episode_index.nunique(),
                         duplicates=int(valid.episode_index.duplicated().sum()),
                         invalid=int((~g.success.isin([0, 1])).sum()), in_outage=in_outage(g.finished_at),
                         last=str(g.finished_at.max())[:16]))
    for fam in expected:
        if fam not in set(e.robot_family):
            rows.append(dict(part=fam, expected=expected[fam], recorded=0, duplicates=0, invalid=0, in_outage=0,
                             last=""))
    errors = folder / "errors.csv"
    n_err = len(pd.read_csv(errors)) if errors.exists() else 0
    if n_err:
        err_ids = set(pd.read_csv(errors).episode_index)
        done = set(e[e.success.isin([0, 1])].episode_index)
        left = len(err_ids - done)
        rows.append(dict(part="errors.csv", expected=0, recorded=left, duplicates=0, invalid=0, in_outage=0,
                         last=f"{n_err} logged errors; {left} episodes never completed", retried=True))
    return rows


def check_libero(folder):
    rows = []
    for suite in LIBERO_SUITES:
        f = folder / suite / "episodes.csv"
        if not f.exists():
            continue
        e = pd.read_csv(f, on_bad_lines="skip")
        s = folder / suite / "summary.json"
        exp = json.load(open(s)).get("expected_rollouts") if s.exists() else None
        exp = exp or 500                                  # 10 tasks x 50 initial states
        valid = e[e.success.isin([0, 1])]
        key = ["task_id", "init_state_id"]
        rows.append(dict(part=suite, expected=exp, recorded=len(valid.drop_duplicates(key)),
                         duplicates=int(valid.duplicated(key).sum()), invalid=int((~e.success.isin([0, 1])).sum()),
                         in_outage=in_outage(e.finished_at), last=str(e.finished_at.max())[:16]))
    return rows


def main():
    out = []
    for folder in sorted(p for p in OWN.iterdir() if p.is_dir()):
        if prefixes and not any(folder.name.startswith(p) for p in prefixes):
            continue
        rows = check_simpler(folder) if folder.name.endswith("_simplerenv") else check_libero(folder)
        for r in rows:
            r["run"] = folder.name
            exp = r["expected"]
            if r.pop("retried", False):
                r["status"] = "ok" if r["recorded"] == 0 else f"MISSING {r['recorded']}"
            elif "_mj332_" in folder.name:
                r["status"] = "re-run of selected tasks"
            else:
                r["status"] = ("ok" if exp and r["recorded"] == exp and not r["duplicates"] else
                           "check" if exp is None else f"MISSING {exp - r['recorded']}" if r["recorded"] < exp
                           else "check")
            if r["in_outage"]:
                r["status"] += ", outage"
            out.append(r)
    df = pd.DataFrame(out)[["run", "part", "expected", "recorded", "duplicates", "invalid", "in_outage", "last",
                            "status"]]
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    bad = df[~df.status.isin(["ok", "re-run of selected tasks"])]
    print(f"\n{len(bad)} of {len(df)} parts need a look" if len(bad) else f"\nall {len(df)} parts complete")


if __name__ == "__main__":
    main()

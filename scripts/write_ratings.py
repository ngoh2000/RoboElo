"""Write the published ratings (ratings/) from the outputs of scripts/run_unified_scale.py (results/unified/).

Run after `run_unified_scale.py report`:
  python scripts/write_ratings.py
"""
import sys
from pathlib import Path

import pandas as pd

from roboelo import unified as U
from roboelo.names import display_name

ROOT = Path(__file__).resolve().parents[1]  # repository root
RES, OUT = ROOT / "results/unified", ROOT / "ratings"
SETTINGS = {"robodojo": "RoboDojo", "robotwin2/easy": "RoboTwin clean", "robotwin2/hard": "RoboTwin randomized",
            "libero": "LIBERO", "simplerenv/google": "SimplerEnv Google Robot", "simplerenv/widowx": "SimplerEnv WidowX"}


def fmt(x):
    return "" if pd.isna(x) else f"{x:.0f}"


def level_cell(r, fam):
    if not pd.isna(r[f"level_{fam}"]):
        return fmt(r[f"level_{fam}"])
    if fam in str(r.predicted_benchmarks):
        return f"*({fmt(r[f'predicted_level_{fam}'])})*"
    return "–"


def main():
    t = pd.read_csv(RES / "policy_ratings.csv").sort_values("overall_elo", ascending=False).reset_index(drop=True)
    if "overall_lo" not in t:
        sys.exit("results/unified/policy_ratings.csv has no intervals: run `run_unified_scale.py report` first")
    s = pd.read_csv(RES / "settings.csv")
    OUT.mkdir(exist_ok=True)

    cols = ["policy", "overall_elo", "overall_lo", "overall_hi", "full_coverage_elo", "full_coverage_lo",
            "full_coverage_hi", "benchmarks", "predicted_benchmarks"]
    cols += [f"level_{f}" for f in U.MAIN_FAMILIES] + [f"predicted_level_{f}" for f in U.MAIN_FAMILIES]
    cols += ["rollouts", "partial_run"]
    t[cols].round(1).to_csv(OUT / "policy_ratings.csv", index=False)
    s[["setting", "a", "a_lo", "a_hi", "anchors"]].round(3).to_csv(OUT / "settings.csv", index=False)
    d = pd.read_csv(RES / "test_difficulty.csv")
    d.assign(difficulty_elo=d.difficulty_elo.round(1)).to_csv(OUT / "test_difficulty.csv", index=False)

    rows = []
    for i, r in t.iterrows():
        name = display_name(r.policy) + (" ¹" if r.partial_run else "")
        rows.append(f"| {i + 1} | {name} | **{fmt(r.overall_elo)}** | {fmt(r.overall_lo)}–{fmt(r.overall_hi)} | "
                    + (f"{fmt(r.full_coverage_elo)} | {fmt(r.full_coverage_lo)}–{fmt(r.full_coverage_hi)} | "
                       if not pd.isna(r.full_coverage_elo) else "– | | ")
                    + " | ".join(level_cell(r, f) for f in U.MAIN_FAMILIES) + " |")
    factors = [f"| {SETTINGS[x.setting]} | {x.a:.2f} | {x.a_lo:.2f}–{x.a_hi:.2f} | "
               f"{', '.join(display_name(p) for p in str(x.anchors).split(', '))} |" for x in s.itertuples()]
    n_full = int(t.full_coverage_elo.notna().sum())
    text = f"""# RoboElo ratings

{len(t)} robot policies on one Elo scale, from four simulation benchmarks: RoboDojo, RoboTwin 2.0, LIBERO and SimplerEnv. The fit uses only the milestone policies ([data/milestone_policies.csv](../data/milestone_policies.csv)) that have results. How the scores are computed is described in [docs/ELO_METHOD.md](../docs/ELO_METHOD.md); the data are in [data/](../data/README.md) (public leaderboards: snapshot of 2026-10-24).

## How to read the table

- **Scale.** The reference panel (π0, π0.5, GR00T N1.7, SmolVLA, OpenVLA-OFT, X-VLA, ACT) averages 1500. On a benchmark with conversion factor a = 1, 100 points multiply the odds of success by 1.78, and 400 points by 10.
- **Overall score:** the mean of the policy's levels on the benchmarks it ran. It depends on which benchmarks those are: of two policies equally strong on LIBERO, the one that also ran a benchmark where it is weaker gets the lower score. A policy that ran one benchmark is placed against policies of other benchmarks only by the anchoring convention (method, section 5.2), which no test can confirm.
- **Full-coverage score:** the mean over all four benchmarks. A benchmark the policy did not run is filled with its predicted level, shown *(in italics)*. It is given only where the linking test shows that a benchmark the policy ran predicts the missing one (method, section 6.1); {n_full} of {len(t)} policies have one. Its interval includes the error of the predicted levels.
- **Intervals:** 95 %, from 1,000 bootstrap replicates with test cases resampled within every setting. They do not include uncertainty about which policies are in the data, so cross-benchmark intervals are too narrow.
- **Levels per benchmark:** μ + γ, averaged over a benchmark's settings. "–" = not run.
- **A policy is a model recipe:** most checkpoints are fine-tuned for each benchmark (method, section 1.4).

## Policies

| # | Policy | Overall score | 95 % interval | Full-coverage score | 95 % interval | RoboDojo | RoboTwin | LIBERO | SimplerEnv |
|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + ("""

¹ Run still in progress; partial results.""" if t.partial_run.any() else "") + """

## Conversion factors

How strongly a rating gap turns into success on each setting (geometric mean 1). Anchors: the policies whose mean offset on the setting is 0 (method, section 5.2).

| Setting | a | 95 % interval | Anchors |
|---|---|---|---|
""" + "\n".join(factors) + """

## Files

| File | Contents |
|---|---|
| `policy_ratings.csv` | the table above, with intervals and predicted levels |
| `settings.csv` | conversion factors with intervals, and the anchors of every setting |
| `test_difficulty.csv` | difficulty of every test case (Elo: the level that gives a 50 % chance of success) |

Reproduce: run `linking`, `fit`, `sources`, `bootstrap` and `report` of `scripts/run_unified_scale.py`, then `python scripts/write_ratings.py`.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print(f"wrote {OUT}: {len(t)} policies, {n_full} with a full-coverage score")


if __name__ == "__main__":
    main()

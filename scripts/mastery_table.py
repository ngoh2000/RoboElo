"""Gap to mastery per benchmark.

  mastery level = the level at which the fit predicts 90 % mean success over the benchmark's test cases
  best now      = the highest level on the benchmark among the fitted policies
  gap           = mastery level - best now
  years         = gap / trend, where the trend is the straight-line fit of dated policies' levels on that benchmark
                  against release date; the interval comes from resampling the dated policies (2,000 replicates)

  python scripts/mastery_table.py              # reads results/unified/ (run `run_unified_scale.py fit` first)
Writes results/unified/mastery_table.csv and prints a table.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_unified_figures as M  # noqa: E402  (release dates, ratings table)
from roboelo import ELO_PER_LOGIT  # noqa: E402
from roboelo import unified as U  # noqa: E402
from roboelo.names import display_name  # noqa: E402

TARGET = 0.9
N_BOOT = 2000
BENCHMARKS = ["RoboDojo", "RoboTwin", "SimplerEnv", "LIBERO"]


def main():
    if len(sys.argv) > 1:
        M.RES = Path(sys.argv[1])
    f = pickle.load(open(M.RES / "fits.pkl", "rb"))["fit"]
    beta = f.elo(f.beta)
    a_test = np.asarray(f.a)[f.unit_of_test]
    fam_test = np.array([U.FAMILY[f.units[j]] for j in f.unit_of_test])
    t = M.rated()
    raw = pd.read_csv(M.RES / "policy_ratings.csv").set_index("policy")
    rng = np.random.default_rng(0)
    rows = []
    for bench in BENCHMARKS:
        m = fam_test == bench
        b, a = beta[m], a_test[m]

        def mean_success(level):
            return np.mean(1 / (1 + np.exp(-a * (level - b) / ELO_PER_LOGIT)))
        mastery = brentq(lambda L: mean_success(L) - TARGET, -5000, 10000)
        col = f"level_{bench}"
        d = t.dropna(subset=[col])
        i = d[col].idxmax()
        best, who = d.loc[i, col], d.loc[i, "policy"]
        dated = d.dropna(subset=["year"])
        x, y = dated.year.to_numpy(), dated[col].to_numpy()
        slope = np.polyfit(x, y, 1)[0]
        boots = []
        for _ in range(N_BOOT):
            k = rng.integers(0, len(x), len(x))
            if np.ptp(x[k]) > 0:
                boots.append(np.polyfit(x[k], y[k], 1)[0])
        boots = np.array(boots)
        gap = mastery - best
        row = dict(benchmark=bench, best_policy=display_name(who), best_now=best, mastery=mastery, gap=gap,
                   predicted_success_at_best=mean_success(best), dated_policies=len(x), trend_per_year=slope,
                   trend_lo=np.percentile(boots, 2.5), trend_hi=np.percentile(boots, 97.5))
        if gap > 0 and slope > 0:
            yrs = gap / boots[boots > 0]
            row.update(years=gap / slope, years_lo=np.percentile(yrs, 2.5), years_hi=np.percentile(yrs, 97.5),
                       share_of_replicates_with_positive_trend=np.mean(boots > 0))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(M.RES / "mastery_table.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    print("\nBenchmark | Best now | Mastery | Gap | Years")
    for r in df.itertuples():
        gap = f"{r.gap:.0f}" if r.gap > 0 else "passed"
        yrs = f"{r.years:.1f} ({r.years_lo:.1f}–{r.years_hi:.1f})" if r.gap > 0 else "—"
        print(f"{r.benchmark} | {r.best_now:.0f} ({r.best_policy}) | {r.mastery:.0f} | {gap} | {yrs}")


if __name__ == "__main__":
    main()

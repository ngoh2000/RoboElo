"""How far each benchmark is from mastery: for the best policy on each setting, the share of test cases on which
its predicted success is at least 90 % (mastered), 50-90 %, 10-50 % or below 10 %, and the rating gain it would need
for 90 % success on the setting's median test case.

  python scripts/plot_mastery_gap.py            # reads results/unified/fits.pkl (run `run_unified_scale.py fit`)
Writes figures/mastery_gap.png.
"""
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roboelo import ELO_PER_LOGIT
from roboelo.names import display_name

ROOT = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/unified"
FIG = ROOT / "figures"
NAMES = {"robodojo": "RoboDojo", "libero": "LIBERO", "robotwin2/easy": "RoboTwin clean",
         "robotwin2/hard": "RoboTwin randomized", "simplerenv/google": "SimplerEnv Google Robot",
         "simplerenv/widowx": "SimplerEnv WidowX"}
BINS = [(0.9, 1.01, "≥ 90 % (mastered)", "#1baf7a"), (0.5, 0.9, "50–90 %", "#9fd8bd"),
        (0.1, 0.5, "10–50 %", "#f2c46d"), (0.0, 0.1, "< 10 %", "#d9d6cf")]
INK, INK2 = "#1d1d1b", "#52514e"
MASTERY = 0.9


def main():
    f = pickle.load(open(RES / "fits.pkl", "rb"))["fit"]
    lev, beta, a = f.elo(f.level()), f.elo(f.beta), np.asarray(f.a)
    need = ELO_PER_LOGIT * np.log(MASTERY / (1 - MASTERY))          # logit gap for 90 %, before dividing by a
    rows = []
    for j, u in enumerate(f.units):
        ran = f.observed_unit[:, j]
        best = int(np.argmax(np.where(ran, lev[:, j], -np.inf)))
        lv = lev[best, j]
        b = beta[f.unit_of_test == j]
        p = 1 / (1 + np.exp(-a[j] * (lv - b) / ELO_PER_LOGIT))
        row = dict(setting=NAMES.get(u, u), best=display_name(f.policies[best]), level=lv, tasks=len(b),
                   gap_median=np.median(b) + need / a[j] - lv, mean_p=p.mean())
        for lo, hi, lab, _ in BINS:
            row[lab] = np.mean((p >= lo) & (p < hi))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["≥ 90 % (mastered)", "mean_p"], ascending=True).reset_index(drop=True)

    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 11})
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    y = np.arange(len(df))
    left = np.zeros(len(df))
    for _, _, lab, col in BINS:
        ax.barh(y, df[lab] * 100, left=left, color=col, height=0.62, label=lab, edgecolor="white", lw=1)
        for yi, l, w in zip(y, left, df[lab] * 100):
            if w >= 7:
                ax.text(l + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=9.5, color=INK)
        left += df[lab] * 100
    ax.set_yticks(y, [f"{r.setting}\n{r.tasks} tasks" for r in df.itertuples()], fontsize=10.5)
    for yi, r in zip(y, df.itertuples()):
        gap = ("already above 90 % on the median task" if r.gap_median <= 0
               else f"+{r.gap_median:.0f} points for 90 % on the median task")
        ax.text(102, yi, f"best: {r.best} ({r.level:.0f})\n{gap}", va="center", fontsize=9.8, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the setting's test cases, by the best policy's predicted success")
    ax.set_title("How far each benchmark is from mastery (best policy on each setting)", loc="left",
                 fontsize=13.5, fontweight="bold", color=INK, pad=28)
    ax.legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    FIG.mkdir(exist_ok=True)
    out = FIG / "mastery_gap.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {out}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

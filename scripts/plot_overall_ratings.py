"""Overall rating plot: task-difficulty curves per benchmark, with the policies measured on two or more benchmarks
as dots at their overall score (reference panel in red).

  python scripts/plot_overall_ratings.py                  # reads results/unified/ (else ratings/)
  python scripts/plot_overall_ratings.py results/unified_min2bench
  python scripts/plot_overall_ratings.py --split-robotwin   # RoboTwin clean / randomized as two curves
Writes figures/overall_ratings.png (or overall_ratings_robotwin_split.png).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from roboelo import unified as U
from roboelo.names import display_name

ROOT = Path(__file__).resolve().parents[1]
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
RES = Path(ARGS[0]) if ARGS else ROOT / "results/unified"
if not (RES / "policy_ratings.csv").exists():
    RES = ROOT / "ratings"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SPLIT = "--split-robotwin" in sys.argv          # RoboTwin clean and randomized as separate curves
if SPLIT:
    BENCH = [("LIBERO", "#1baf7a"), ("SimplerEnv", "#7b5ea7"), ("RoboTwin clean", "#f0904f"),
             ("RoboTwin randomized", "#b5401a"), ("RoboDojo", "#2a78d6")]
else:
    BENCH = [("LIBERO", "#1baf7a"), ("SimplerEnv", "#7b5ea7"), ("RoboTwin", "#eb6834"), ("RoboDojo", "#2a78d6")]
BW = 0.35          # KDE bandwidth factor (narrower than the default, so the shape of each benchmark shows)
BG, INK, INK2, GRID, RED, DOT = "#f9f8f5", "#1d1d1b", "#5f5e5a", "#dcdbd6", "#c8372d", "#1d1d1b"
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 10.5, "axes.edgecolor": GRID,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "savefig.dpi": 200})


def main():
    pol = pd.read_csv(RES / "policy_ratings.csv")
    tests = pd.read_csv(RES / "test_difficulty.csv")
    tests["benchmark"] = tests.setting.map(U.FAMILY)
    if SPLIT:
        tests.loc[tests.setting == "robotwin2/easy", "benchmark"] = "RoboTwin clean"
        tests.loc[tests.setting == "robotwin2/hard", "benchmark"] = "RoboTwin randomized"
    tests = tests[tests.benchmark.isin([b for b, _ in BENCH])]
    pol["n_bench"] = pol.benchmarks.str.split(", ").str.len()
    dots = pol[pol.n_bench >= 2].sort_values("overall_elo", ascending=False).reset_index(drop=True)

    lo = min(tests.difficulty_elo.min(), dots.overall_elo.min()) - 220
    hi = max(tests.difficulty_elo.max(), dots.overall_elo.max()) + 380
    xs = np.linspace(lo, hi, 800)

    fig, ax = plt.subplots(figsize=(11.6, 8.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ymax = 0
    for name, col in BENCH:
        d = tests.loc[tests.benchmark == name, "difficulty_elo"].to_numpy()
        k = gaussian_kde(d, bw_method=BW)(xs)
        ax.fill_between(xs, k, color=col, alpha=0.13, lw=0)
        ax.plot(xs, k, color=col, lw=2.3)
        ymax = max(ymax, k.max())
    ax.set_ylim(0, ymax * (1.55 if SPLIT else 1.42))
    ax.set_xlim(lo, hi)

    # benchmark headers above the curves, at each median, pushed apart where they would overlap
    meds = [(name, col, tests.loc[tests.benchmark == name, "difficulty_elo"]) for name, col in BENCH]
    order = sorted(meds, key=lambda m: m[2].median())
    pos, gap = [], (hi - lo) * (0.165 if SPLIT else 0.19)
    for m in order:
        x = m[2].median()
        pos.append(max(x, pos[-1] + gap) if pos else x)
    shift = max(0, pos[-1] - (hi - (hi - lo) * 0.09))
    for (name, col, d), x in zip(order, pos):
        x = max(x - shift, lo + (hi - lo) * 0.08)
        ax.text(x, 0.985, name.replace("RoboTwin ", "RoboTwin\n") if SPLIT else f"{name} tasks", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=13.5, fontweight="bold", color=col)
        ax.text(x, 0.905 if SPLIT else 0.94, f"median {d.median():.0f}\n{len(d)} tasks", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, color=INK2, linespacing=1.25)

    # models: stacked from the top in rating order, at their overall score
    ys = np.linspace(0.81, 0.04, len(dots)) if len(dots) > 1 else np.array([0.5])
    tr = ax.get_xaxis_transform()
    for r, y in zip(dots.itertuples(), ys):
        panel = r.policy in U.PANEL
        ax.scatter([r.overall_elo], [y], transform=tr, s=40, color=RED if panel else DOT, zorder=5,
                   edgecolor=BG, lw=0.6)
        ax.text(r.overall_elo + (hi - lo) * 0.004, y, f"{display_name(r.policy)}  {r.overall_elo:.0f}",
                transform=tr, va="center", ha="left", fontsize=9.5, color=INK, zorder=6,
                bbox=dict(boxstyle="square,pad=0.12", fc=BG, ec="none", alpha=0.75))

    # legend
    lx, ly = 0.715, 0.82
    ax.text(lx, ly, "Curves = tasks (difficulty)", transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color=INK)
    for i, (name, col) in enumerate(BENCH):
        y = ly - 0.045 * (i + 1)
        ax.plot([lx, lx + 0.03], [y + 0.006] * 2, transform=ax.transAxes, color=col, lw=2.3)
        ax.text(lx + 0.04, y, name, transform=ax.transAxes, fontsize=9.5, color=INK2)
    y = ly - 0.045 * (len(BENCH) + 1) - 0.02
    ax.text(lx, y, "Dots = policies on 2+ benchmarks", transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color=INK)
    for j, (lab, col) in enumerate([("reference-panel policy", RED), ("other policy", DOT)]):
        yy = y - 0.045 * (j + 1)
        ax.scatter([lx + 0.015], [yy + 0.006], transform=ax.transAxes, s=40, color=col)
        ax.text(lx + 0.04, yy, lab, transform=ax.transAxes, fontsize=9.5, color=INK2)

    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    ax.set_ylabel("share of tasks")
    ax.set_xlabel("Elo scale: policy overall score (dots) and task difficulty (curves); reference panel averages 1500")
    fig.tight_layout()
    out = FIG / ("overall_ratings_robotwin_split.png" if SPLIT else "overall_ratings.png")
    fig.savefig(out, facecolor=BG)
    print(f"wrote {out} from {RES}: {len(dots)} policies on 2+ benchmarks, {len(tests)} tasks")


if __name__ == "__main__":
    main()

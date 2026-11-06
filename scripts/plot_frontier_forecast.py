"""Frontier forecast of policy capability, in the style of Ho et al. (2025), "A Rosetta Stone for AI benchmarks",
Figure 5: take the policies that were the best at their release (top 1 at release), fit a straight line to their
overall scores against release date, and extend it three years with a 95 % confidence band.

By default only policies with RoboDojo or RoboTwin results are used: comparisons between them and policies seen only
on LIBERO or SimplerEnv are not supported by the linking test (docs/ELO_METHOD.md, section 8.2).

  python scripts/plot_frontier_forecast.py                 # reads results/unified/ (else ratings/)
  python scripts/plot_frontier_forecast.py --all           # all dated policies
  python scripts/plot_frontier_forecast.py results/unified_all_policies   # from the all-policy fit
  python scripts/plot_frontier_forecast.py --by-benchmark  # one panel per benchmark (+ RoboDojo alone)
Writes figures/frontier_forecast[_all][_allpolicyfit].png.
"""
import datetime as dt
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_unified_figures as M  # noqa: E402  (release dates, ratings, frontier)
from roboelo.names import display_name  # noqa: E402

ALL = "--all" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
YEARS_AHEAD = 3
TODAY = dt.date.today()
NOW = TODAY.year + (TODAY.timetuple().tm_yday - 0.5) / 365.25
TEAL, TEAL_LIGHT, PINK, INK, INK2, GRID = "#14a3a0", "#9ed8d6", "#e0408a", "#1d1d1b", "#52514e", "#ecebe7"


def main():
    if ARGS:
        M.RES = Path(ARGS[0]) if Path(ARGS[0]).is_absolute() else M.ROOT / ARGS[0]
    elif not (M.RES / "policy_ratings.csv").exists():
        M.RES = M.ROOT / "ratings"
    allfit = "all_policies" in M.RES.name
    t = M.rated().dropna(subset=["year"])
    if not ALL:
        t = t[t.hard]
    x, y = t.year.to_numpy(), t.overall_elo.to_numpy()
    _, _, idx = M.frontier(x, y)                       # policies that were top 1 at their release
    fx, fy = x[idx], y[idx]
    fit = stats.linregress(fx, fy)
    n = len(fx)
    tq = stats.t.ppf(0.975, n - 2)
    slope_lo, slope_hi = fit.slope - tq * fit.stderr, fit.slope + tq * fit.stderr

    def band(xs):                                      # 95 % confidence band of the fitted line
        se = np.sqrt(np.sum((fy - (fit.intercept + fit.slope * fx)) ** 2) / (n - 2))
        s = se * np.sqrt(1 / n + (xs - fx.mean()) ** 2 / np.sum((fx - fx.mean()) ** 2))
        mid = fit.intercept + fit.slope * xs
        return mid, mid - tq * s, mid + tq * s

    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 11})
    fig, ax = plt.subplots(figsize=(11, 6.2))
    rest = np.setdiff1d(np.arange(len(x)), idx)
    ax.scatter(x[rest], y[rest], s=46, color=TEAL_LIGHT, alpha=0.75, lw=0, label=f"All dated policies ({len(x)})",
               zorder=2)
    ax.scatter(fx, fy, s=70, color=TEAL, lw=0, label="Policies that were top 1 at release", zorder=4)
    prev = None
    for xi, yi, p in zip(fx, fy, t.policy.to_numpy()[idx]):
        off = (6, -12)
        if prev is not None and abs(xi - prev[0]) < 0.15 and abs(yi - prev[1]) < 60:
            off = (-8, 8)                               # two records released close together: label above-left
        ax.annotate(display_name(p), (xi, yi), xytext=off, textcoords="offset points", fontsize=9.5, color=INK2,
                    ha="right" if off[0] < 0 else "left")
        prev = (xi, yi)
    xs = np.linspace(fx.min(), NOW, 50)
    ax.plot(xs, fit.intercept + fit.slope * xs, ls="--", color=TEAL, lw=2, label="Frontier trend", zorder=3)
    xf = np.linspace(NOW, NOW + YEARS_AHEAD, 50)
    mid, lo, hi = band(xf)
    ax.fill_between(xf, lo, hi, color=PINK, alpha=0.25, lw=0, label="95 % confidence interval")
    ax.plot(xf, mid, color=PINK, lw=2.2, label="Forecast")
    ax.axvline(NOW, color=INK, ls=":", lw=1.4, label="Present")
    ax.annotate(f"{mid[-1]:.0f}\n({lo[-1]:.0f}–{hi[-1]:.0f})", (xf[-1], mid[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=10, color=PINK)
    who = "all dated policies" if ALL else "policies with RoboDojo / RoboTwin results"
    fitname = "all-policy fit" if allfit else "milestone fit"
    ax.set_title(f"{YEARS_AHEAD}-year frontier forecast of the overall rating ({who}; {fitname})", fontsize=13,
                 color=INK, pad=12)
    ax.set_xlabel("first public release (paper, else weights)", color=INK2)
    ax.set_ylabel("overall rating (Elo, reference panel = 1500)", color=INK2)
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(min(x.min(), fx.min()) - 0.25, NOW + YEARS_AHEAD + 0.45)
    ax.legend(loc="upper left", fontsize=9.5, frameon=True, framealpha=0.9)
    ax.text(0.99, 0.02, f"Frontier trend: {fit.slope:.0f} points / year (95 % CI {slope_lo:.0f}–{slope_hi:.0f}), "
            f"from {n} frontier policies", transform=ax.transAxes, ha="right", fontsize=9.5, color=INK2)
    fig.tight_layout()
    out = M.FIG / ("frontier_forecast" + ("_all" if ALL else "") + ("_allpolicyfit" if allfit else "") + ".png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    print(f"frontier policies: {[display_name(p) for p in t.policy.to_numpy()[idx]]}")
    print(f"slope {fit.slope:.1f} points/year (95 % CI {slope_lo:.1f} to {slope_hi:.1f}), r2 {fit.rvalue ** 2:.3f}")
    print(f"at present ({NOW:.2f}): {mid[0]:.0f} ({lo[0]:.0f}-{hi[0]:.0f}); "
          f"in {YEARS_AHEAD} years: {mid[-1]:.0f} ({lo[-1]:.0f}-{hi[-1]:.0f})")


def trend(fx, fy):
    """Straight-line fit with a 95 % confidence band; None when fewer than 3 points."""
    n = len(fx)
    if n < 3:
        return None
    fit = stats.linregress(fx, fy)
    tq = stats.t.ppf(0.975, n - 2)
    se = np.sqrt(np.sum((fy - (fit.intercept + fit.slope * fx)) ** 2) / (n - 2))

    def band(xs):
        s = se * np.sqrt(1 / n + (xs - fx.mean()) ** 2 / np.sum((fx - fx.mean()) ** 2))
        mid = fit.intercept + fit.slope * xs
        return mid, mid - tq * s, mid + tq * s
    return fit, tq, band


def panel(ax, t, tests, bench, end, big=False):
    """One benchmark: its policies' levels by release date, the frontier, its trend and forecast."""
    d = t.dropna(subset=[f"level_{bench}"])
    x, y = d.year.to_numpy(), d[f"level_{bench}"].to_numpy()
    _, _, idx = M.frontier(x, y)
    fx, fy = x[idx], y[idx]
    med = tests.loc[tests.benchmark == bench, "difficulty_elo"].median()
    rest = np.setdiff1d(np.arange(len(x)), idx)
    fs = 10.5 if big else 8.5
    ax.scatter(x[rest], y[rest], s=50 if big else 34, color=TEAL_LIGHT, alpha=0.8, lw=0, zorder=2)
    ax.scatter(fx, fy, s=80 if big else 56, color=TEAL, lw=0, zorder=4)
    for xi, yi, p in zip(fx, fy, d.policy.to_numpy()[idx]):
        ax.annotate(display_name(p), (xi, yi), xytext=(6, -12), textcoords="offset points", fontsize=fs, color=INK2)
    ax.axhline(med, color="#7b5ea7", lw=1.5, ls=(0, (5, 3)), zorder=1)
    ax.text(min(x.min(), 2023) - 0.15, med, f" median task {med:.0f}", color="#7b5ea7", fontsize=fs + 0.5,
            va="bottom")
    ax.axvline(NOW, color=INK, ls=":", lw=1.2)
    tr = trend(fx, fy)
    note, row = f"{len(d)} policies, {len(idx)} frontier", dict(benchmark=bench, policies=len(d), frontier=len(idx),
                                                                best=fy.max(), median_task=med)
    if tr:
        fit, tq, band = tr
        xs = np.linspace(fx.min(), NOW, 30)
        ax.plot(xs, fit.intercept + fit.slope * xs, ls="--", color=TEAL, lw=1.9, zorder=3)
        xf = np.linspace(NOW, end, 60)
        mid, lo, hi = band(xf)
        ax.fill_between(xf, lo, hi, color=PINK, alpha=0.22, lw=0)
        ax.plot(xf, mid, color=PINK, lw=2.2)
        slo, shi = fit.slope - tq * fit.stderr, fit.slope + tq * fit.stderr
        note += f": {fit.slope:+.0f} / year (95 % CI {slo:.0f} to {shi:.0f})"
        if fy.max() >= med:
            note += "\nfrontier already above the median task"
        elif fit.slope > 0:
            cx = (med - fit.intercept) / fit.slope
            lo_x = xf[np.argmax(hi >= med)] if (hi >= med).any() else None
            hi_x = xf[np.argmax(lo >= med)] if (lo >= med).any() else None
            rng = (f" (95 % CI {lo_x:.1f} to " + (f"{hi_x:.1f})" if hi_x else f"after {end:.1f})")) if lo_x else ""
            note += f"\ntrend reaches the median task around {cx:.1f}{rng}"
            ax.scatter([cx], [med], s=60, marker="D", color="#7b5ea7", zorder=5)
        row.update(slope=fit.slope, lo=slo, hi=shi)
    else:
        note += ": too few frontier policies for a trend"
    top_left = bench in ("RoboDojo", "RoboTwin", "LIBERO")
    ax.text(0.01 if top_left else 0.99, 0.97 if top_left else 0.03, note, transform=ax.transAxes,
            ha="left" if top_left else "right", va="top" if top_left else "bottom", fontsize=fs + 0.5, color=INK2)
    ax.set_xlim(min(x.min(), 2023) - 0.25, end + 0.2)
    ax.grid(color=GRID, lw=0.8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_ylabel(f"level on {bench} (Elo)", color=INK2)
    return row


def legend_handles():
    return [plt.Line2D([], [], marker="o", ls="", color=TEAL_LIGHT, ms=7, label="policy"),
            plt.Line2D([], [], marker="o", ls="", color=TEAL, ms=8, label="top 1 at release"),
            plt.Line2D([], [], ls="--", color=TEAL, label="frontier trend"),
            plt.Line2D([], [], color=PINK, lw=2, label="forecast (95 % CI shaded)"),
            plt.Line2D([], [], color="#7b5ea7", ls=(0, (5, 3)), label="median task difficulty"),
            plt.Line2D([], [], color=INK, ls=":", label="present")]


def main_by_benchmark():
    """Frontier and forecast per benchmark, on policies' levels there, with the benchmark's median task.
    Also writes a single RoboDojo panel for slides."""
    from roboelo import unified as U
    if ARGS:
        M.RES = Path(ARGS[0]) if Path(ARGS[0]).is_absolute() else M.ROOT / ARGS[0]
    elif not (M.RES / "policy_ratings.csv").exists():
        M.RES = M.ROOT / "ratings"
    t = M.rated().dropna(subset=["year"])
    tests = pd.read_csv(M.RES / "test_difficulty.csv")
    tests["benchmark"] = tests.setting.map(U.FAMILY)
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 10.5})
    end = NOW + YEARS_AHEAD
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    rows = []
    for ax, bench in zip(axes.ravel(), ["RoboDojo", "RoboTwin", "LIBERO", "SimplerEnv"]):
        rows.append(panel(ax, t, tests, bench, end))
        ax.set_title(bench, loc="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.suptitle(f"Frontier per benchmark and {YEARS_AHEAD}-year forecast (policies' levels on each benchmark)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=6, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = M.FIG / "frontier_forecast_by_benchmark.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    fig, ax = plt.subplots(figsize=(11, 6.2))
    panel(ax, t, tests, "RoboDojo", end, big=True)
    ax.set_title(f"RoboDojo: frontier level and {YEARS_AHEAD}-year forecast", fontsize=13.5, color=INK, pad=12)
    ax.set_xlabel("first public release (paper, else weights)", color=INK2)
    ax.legend(handles=legend_handles(), loc="lower right", fontsize=9.5, frameon=True, framealpha=0.9)
    fig.tight_layout()
    out = M.FIG / "frontier_forecast_robodojo.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main_by_benchmark() if "--by-benchmark" in sys.argv else main()

"""Figures for the unified-scale analysis (results/unified/ -> figures/unified_*.png)."""
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roboelo import ELO_PER_LOGIT
from roboelo import unified as U
from roboelo.names import display_name

ROOT = Path(__file__).resolve().parents[1]  # repository root

warnings.filterwarnings("ignore")
RES, FIG = ROOT / "results/unified", ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": "#bdbcb7", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
nm = display_name


def release_dates():
    rd = pd.read_csv(ROOT / "data/policy_release_dates.csv")
    out = {}
    for r in rd.itertuples():
        d = r.paper_first_public if isinstance(r.paper_first_public, str) else r.weights_public
        if isinstance(d, str):
            y, m = map(int, d.split("-"))
            out[r.policy.lower()] = y + (m - 0.5) / 12
    out["octo_small"] = out.get("octo")
    return out


def rated():
    t = pd.read_csv(RES / "policy_ratings.csv")
    dates = release_dates()
    t["year"] = [dates.get(p.lower()) for p in t.policy]
    t["hard"] = t.benchmarks.str.contains("RoboDojo|RoboTwin")
    b = RES / "bootstrap_policies.csv"
    if b.exists() and pd.read_csv(b).b.nunique() >= 20:
        bs = pd.read_csv(b).groupby("policy").overall_elo.quantile([0.025, 0.975]).unstack()
        t["lo"] = np.minimum(t.policy.map(bs[0.025]), t.overall_elo)
        t["hi"] = np.maximum(t.policy.map(bs[0.975]), t.overall_elo)
    return t


def frontier(x, y):
    order = np.argsort(x)
    xs, ys, best, pts = [], [], -np.inf, []
    for i in order:
        if y[i] > best:
            best = y[i]
            pts.append(i)
        xs.append(x[i])
        ys.append(best)
    return np.array(xs), np.array(ys), pts


def year_axis(ax):
    ax.set_xlim(2022.75, 2027.0)
    ax.set_xticks([2023, 2024, 2025, 2026, 2027])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_time_overall():
    t = rated().dropna(subset=["year"])
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for hard, col, lab in [(True, BLUE, "has RoboDojo / RoboTwin results"),
                           (False, ORANGE, "LIBERO / SimplerEnv only")]:
        d = t[t.hard == hard]
        if "lo" in d:
            ax.errorbar(d.year, d.overall_elo, yerr=[d.overall_elo - d.lo, d.hi - d.overall_elo], fmt="none",
                        ecolor=col, alpha=0.35, lw=1, zorder=1)
        ax.scatter(d.year, d.overall_elo, s=34, color=col if hard else "white", edgecolor=col, lw=1.6,
                   zorder=3, label=lab)
    for d, col, ls in [(t, "#7a7974", (0, (4, 3))), (t[t.hard], BLUE, "-")]:
        x, y, pts = frontier(d.year.to_numpy(), d.overall_elo.to_numpy())
        ax.step(np.r_[x, 2027], np.r_[y, y[-1]], where="post", color=col, lw=1.6, ls=ls, zorder=2)
    label = {"RT_1", "DP", "ACT", "RT-1-X", "DP3", "Octo", "OpenVLA", "Pi_0", "CogACT", "Pi_05", "UniVLA", "X_VLA",
             "LingBot_VA", "G05", "MolmoACT2", "GR00T_N17", "SmolVLA", "PatchWAM", "SpatialVLA"}
    offs = {"LingBot_VA": (-62, 5), "Octo": (5, -12), "PatchWAM": (-58, 6), "CogACT": (-44, 6), "UniVLA": (-40, 6),
            "SpatialVLA": (5, -11), "Pi_05": (-30, 6), "G05": (5, 5)}
    for r in t[t.policy.isin(label)].itertuples():
        ax.annotate(nm(r.policy) + (" (partial)" if r.partial_run else ""), (r.year, r.overall_elo),
                    xytext=offs.get(r.policy, (5, 5)), textcoords="offset points", fontsize=8.5, color=INK2)
    k = np.polyfit(t.year, t.overall_elo, 1)[0]
    kh = np.polyfit(t[t.hard].year, t[t.hard].overall_elo, 1)[0]
    gain = t.overall_elo.max() - t[t.year < 2024].overall_elo.max()
    fig.text(0.01, 0.975, f"Overall rating by release date: the best model gained ~{gain:.0f} points since 2023",
             fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.925, f"Linear trend over all {len(t)} dated models: {k:.0f} points per year ({kh:.0f} for "
             f"models with RoboDojo / RoboTwin results); 174 points = success odds x2.7.\n"
             f"Lines: best so far (dashed: including LIBERO / SimplerEnv-only models). "
             f"Bars: 95% interval (tasks resampled).",
             fontsize=8.5, color=INK2, va="top")
    year_axis(ax)
    ax.set_xlim(2022.75, 2026.95)
    ax.set_ylabel("overall rating (Elo, reference panel = 1500)")
    ax.set_xlabel("first public release (paper, else weights)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.875))
    fig.savefig(FIG / "unified_time_overall.png")
    plt.close(fig)
    return k, kh


def main_fit():
    return pickle.load(open(RES / "fits.pkl", "rb"))["fit"]


def unit_levels():
    f = main_fit()
    return pd.DataFrame(np.where(f.observed_unit, f.elo(f.level()), np.nan), index=f.policies, columns=f.units)


def fig_time_by_benchmark():
    t = rated().dropna(subset=["year"])
    lev = unit_levels()
    t["level_SimplerEnv Google Robot"] = t.policy.map(lev["simplerenv/google"])
    fams = ["RoboDojo", "RoboTwin", "LIBERO", "SimplerEnv Google Robot"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
    slopes = {}
    for ax, fam in zip(axes.ravel(), fams):
        d = t.dropna(subset=[f"level_{fam}"])
        y = d[f"level_{fam}"].to_numpy()
        ax.scatter(d.year, y, s=26, color=BLUE, zorder=3)
        x_, y_, pts = frontier(d.year.to_numpy(), y)
        ax.step(np.r_[x_, 2027], np.r_[y_, y_[-1]], where="post", color=INK2, lw=1.4, zorder=2)
        k = np.polyfit(d.year, y, 1)[0]
        slopes[fam] = (k, len(d))
        offs = {"ME_Dex_1_0": (-62, -2), "PatchWAM": (-52, 4), "OpenDM05": (-60, 6), "G05": (6, -11)}
        for i in set(pts) | set(np.argsort(y)[:1]):
            p = d.policy.iloc[i]
            ax.annotate(nm(p) + (" (partial)" if d.partial_run.iloc[i] else ""), (d.year.iloc[i], y[i]),
                        xytext=offs.get(p, (4, 4)), textcoords="offset points", fontsize=8, color=INK2)
        ax.set_title(f"{fam}: {len(d)} models, {k:+.0f} points / year", loc="left", fontsize=11)
        year_axis(ax)
        ax.set_xlim(2022.75, 2026.95)
    for ax in axes[:, 0]:
        ax.set_ylabel("level on the benchmark (Elo)")
    fig.suptitle("Within every benchmark the best model keeps improving (line: best so far)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "unified_time_by_benchmark.png")
    plt.close(fig)
    return slopes


def fig_wright():
    f = main_fit()
    t = rated()
    tests = pd.read_csv(RES / "test_difficulty.csv")
    a = dict(zip(f.units, f.a))
    cols = [("RoboDojo", ["robodojo"], BLUE), ("RoboTwin", ["robotwin2/easy", "robotwin2/hard"], ORANGE),
            ("LIBERO", ["libero"], AQUA), ("SimplerEnv", ["simplerenv/google", "simplerenv/widowx"], YELLOW)]
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    rng = np.random.default_rng(1)
    x0 = 0
    ax.scatter(x0 + rng.uniform(-0.28, 0.28, len(t)), t.overall_elo, s=18, color=INK2, zorder=3)
    dy = {"X_VLA": 14, "Pi_05": -14, "OpenVLA": 14, "Octo": -14, "Simate-beta": 0, "PatchWAM": 10, "G05": -6}
    for p in ["Pi_05", "X_VLA", "OpenVLA", "Octo", "G05", "Simate-beta", "DP", "PatchWAM"]:
        r = t[t.policy == p]
        if len(r):
            ax.annotate(nm(p), (x0 + 0.32, r.overall_elo.iloc[0] + dy.get(p, 0)), fontsize=7.5, color=INK2,
                        va="center")
    for j, (name, units, col) in enumerate(cols, start=1):
        d = tests[tests.setting.isin(units)]
        half = np.log(3) / d.setting.map(a) * ELO_PER_LOGIT
        xs = j * 1.25 + rng.uniform(-0.42, 0.42, len(d))
        ax.vlines(xs, d.difficulty_elo - half, d.difficulty_elo + half, color=col, alpha=0.45, lw=1.6)
        ax.scatter(xs, d.difficulty_elo, s=7, color=col, zorder=3)
        aa = ", ".join(f"{a[u]:.2f}" for u in units)
        ax.text(j * 1.25, 1.02, f"{name}\n{len(d)} tasks, a = {aa}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(x0, 1.02, f"Models\n{len(t)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=9, color=INK)
    ax.set_xticks([])
    ax.set_xlim(-0.6, 5.7)
    ax.spines["bottom"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("rating (Elo)")
    ax.set_title("Models and tasks of four benchmarks on one scale", loc="left", pad=38)
    fig.text(0.01, 0.01, "Task bar: centre = rating a typical model needs for 50% success; ends = 25% and 75%. "
             "Longer bars = smaller conversion factor a.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "unified_wright_map.png")
    plt.close(fig)


def fig_factors():
    s = pd.read_csv(RES / "settings.csv")
    b = RES / "bootstrap_settings.csv"
    labels = {"robodojo": "RoboDojo", "robotwin2/easy": "RoboTwin clean", "robotwin2/hard": "RoboTwin randomized",
              "libero": "LIBERO", "simplerenv/google": "SimplerEnv Google Robot", "simplerenv/widowx": "SimplerEnv WidowX"}
    s = s.set_index("setting").loc[list(labels)]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    y = np.arange(len(s))[::-1]
    if b.exists() and pd.read_csv(b).b.nunique() >= 20:
        q = pd.read_csv(b).groupby("setting").a.quantile([0.025, 0.975]).unstack().loc[s.index]
        ax.hlines(y, q[0.025], q[0.975], color=BLUE, lw=2.2, alpha=0.45)
    ax.scatter(s.a, y, s=46, color=BLUE, zorder=3)
    for yi, (u, r) in zip(y, s.iterrows()):
        ax.text(s.a.max() * 1.0 + 0.55, yi, f"{r.a:.2f}   ({int(r.n_anchors)} anchor{'s' if r.n_anchors != 1 else ''})",
                va="center", fontsize=9, color=INK2)
    ax.axvline(1, color="#7a7974", lw=1, ls=(0, (4, 3)))
    ax.set_yticks(y, [labels[u] for u in s.index])
    ax.set_xlim(0, s.a.max() + 1.6)
    ax.set_xlabel("conversion factor a (log-odds per logit of rating gap; geometric mean = 1)")
    ax.set_title(f"{labels[s.a.idxmax()]} turns a rating gap into success most steeply", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "unified_conversion_factors.png")
    plt.close(fig)


def fig_linking():
    f = RES / "linking_summary.csv"
    if not f.exists():
        return None
    c = pd.read_csv(f).set_index(["config", "benchmark"])
    shared = U.config_name(U.SIGMA_GAMMA, U.SIGMA_BETA, None)
    order = [b for b in U.MAIN_FAMILIES if (U.config_name(), b) in c.index]
    main = c.loc[U.config_name()].loc[order]
    names = [(main.loss_task_mean, "task average of other models", "#bdbcb7")]
    if shared != U.config_name() and (shared, order[0]) in c.index:
        names.append((c.loc[shared].loc[order].loss, "joint scale, one conversion", ORANGE))
    names.append((main.loss, "joint scale (main configuration)", BLUE))
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    w = 0.8 / len(names)
    for k, (vals, lab, colr) in enumerate(names):
        xs = np.arange(len(order)) + (k - (len(names) - 1) / 2) * w
        ax.bar(xs, vals, w * 0.92, color=colr, label=lab)
        for x_, v in zip(xs, vals):
            ax.text(x_, v + 0.005, f"{v:.3f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(np.arange(len(order)), [f"{b}\n({int(main.loc[b, 'n_policies'])} models held out)" for b in order])
    ax.set_ylabel("log loss per hidden trial")
    better = [b for b in order if main.loc[b, "loss"] < main.loc[b, "loss_task_mean"]]
    ax.set_title(f"Held-out benchmark: the joint scale beats the task average on {len(better)} of {len(order)}",
                 loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_ylim(0, max(v.max() for v, _, _ in names) * 1.3)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "unified_linking.png")
    plt.close(fig)
    return main


if __name__ == "__main__":
    print("overall trend:", fig_time_overall())
    print("by benchmark:", fig_time_by_benchmark())
    fig_wright()
    fig_factors()
    r = fig_linking()
    if r is not None:
        print(r.round(4))

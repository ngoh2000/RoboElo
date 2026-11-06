"""Rating stability over the share of data (the joint-fit counterpart of AGI-Elo's Figure 6).

AGI-Elo updates ratings match by match, so its ratings have a path over the % of matches. The joint fit has no
order: it fits all results at once. The counterpart here: put every trial (episode) in a random order, refit on the
first x % for x = 1 ... 100, and follow each policy's rating. The shares are nested (the first 10 % are part of the
first 20 %), and at 100 % the fit is the main fit. Every fit starts from scratch, so it cannot lean on the final
answer. Several random orders give the spread shown in the last panel.

  python scripts/rating_stability.py                 # 5 orders; writes results/stability/, figures/rating_stability.png
  python scripts/rating_stability.py --orders 1      # one order (quicker)
  python scripts/rating_stability.py --plot          # redraw from saved results
  python scripts/rating_stability.py --slide         # compact 16:9 version -> figures/rating_stability_slide.png
Respects ROBOELO_POLICIES and ROBOELO_MIN_BENCHMARKS like run_unified_scale.py; ROBOELO_WORKERS sets the processes.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_unified_scale as R  # noqa: E402  (data_main, output folders)
from roboelo import unified as U  # noqa: E402
from roboelo.names import display_name  # noqa: E402

PCTS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
OUT = R.ROOT / "results" / ("stability" + R.OUT.name.replace("unified", ""))
FIG = R.ROOT / "figures"
WORKERS = int(os.environ.get("ROBOELO_WORKERS", max(1, (os.cpu_count() or 2) - 2)))
DATA = None


def trial_order(data, seed):
    """A random position in [0, 1) for every trial: per cell, sorted positions of its successes and failures."""
    rng = np.random.default_rng(seed)
    succ, fail = {}, {}
    for i, t in zip(*np.nonzero(data.observed)):
        k, n = int(data.k_sr[i, t]), int(data.n[i, t])
        succ[i, t] = np.sort(rng.random(k))
        fail[i, t] = np.sort(rng.random(n - k))
    return succ, fail


def thinned(data, order, share):
    """Data with only the trials whose position is below share (nested in share)."""
    succ, fail = order
    k, n = np.zeros_like(data.n), np.zeros_like(data.n)
    for (i, t), s in succ.items():
        ks, kf = np.searchsorted(s, share), np.searchsorted(fail[i, t], share)
        k[i, t], n[i, t] = ks, ks + kf
    d = replace(data, n=n, k_sr=k, k_score=np.where(n > 0, data.k_score * n / np.maximum(data.n, 1), 0.0))
    keep_p = d.policies[d.observed.any(1)]
    d = d.subset(policies=list(keep_p))
    return d.subset(tests=list(d.tests[d.observed.any(0)]))


def _init():
    global DATA
    DATA = R.data_main()[1]


def _task(seed, pct):
    d = thinned(DATA, trial_order(DATA, seed), pct / 100)
    f = U.fit(d)
    F, fams = f.family_levels()
    rows = pd.DataFrame(f.elo(F), columns=fams)
    rows.insert(0, "overall", f.elo(f.overall()))
    rows.insert(0, "policy", f.policies)
    return rows.assign(order=seed, pct=pct, trials=int(d.n.sum()), converged=f.extras["converged"])


def run(n_orders):
    _init()
    tasks = [(s, p) for s in range(n_orders) for p in PCTS]
    rows = []
    with ProcessPoolExecutor(WORKERS, initializer=_init) as pool:
        futs = [pool.submit(_task, *t) for t in tasks]
        for k, fut in enumerate(as_completed(futs), start=1):
            rows.append(fut.result())
            print(f"{k}/{len(tasks)} fits done", flush=True)
    df = pd.concat(rows).sort_values(["order", "pct", "policy"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "levels.csv", index=False)
    print(f"{(~df.groupby(['order', 'pct']).converged.first()).sum()} of {len(tasks)} fits did not report convergence")
    return df


def summary(df):
    final = df[df.pct == 100].set_index(["order", "policy"]).overall
    d = df.join(final.rename("final"), on=["order", "policy"])
    d["err"] = (d.overall - d.final).abs()
    g = d.groupby(["order", "pct"]).err.median().unstack(0)
    from scipy.stats import spearmanr
    rho = d.dropna(subset=["overall", "final"]).groupby(["order", "pct"]).apply(
        lambda x: spearmanr(x.overall, x.final)[0]).unstack(0)
    s = pd.DataFrame({"trials": df.groupby("pct").trials.first(), "median_abs_change_elo": g.median(axis=1),
                      "worst_order": g.max(axis=1), "spearman_with_final": rho.median(axis=1)})
    return s


def plot(df):
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": "#bdbcb7",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})
    one = df[df.order == 0]
    panels = [("overall", "Overall score")] + [(b, f"{b}: level on the benchmark")
                                               for b in U.MAIN_FAMILIES if b in df.columns]
    fig = plt.figure(figsize=(12, 19))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.25, 1, 1, 0.85], hspace=0.5, wspace=0.22)
    axes = [fig.add_subplot(gs[0, :])] + [fig.add_subplot(gs[1 + k // 2, k % 2]) for k in range(len(panels) - 1)]
    cmap = plt.get_cmap("tab20")
    for ax, (col, title) in zip(axes, panels):
        w = one.pivot(index="pct", columns="policy", values=col)
        w = w.loc[:, w.loc[100].notna()].sort_values(100, axis=1, ascending=False)
        for j, p in enumerate(w.columns):
            ax.plot(w.index, w[p], lw=1.3, color=cmap(j % 20), ls="-" if j < 20 else "--",
                    label=f"{display_name(p)} {w.loc[100, p]:.0f}")
        ax.set_title(f"{title} ({w.shape[1]} policies)", loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of trials used")
        ax.set_ylabel("rating (Elo)")
        ax.grid(color="#e4e3df", lw=0.7)
        if col == "overall":
            ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      handlelength=1.8, title="final score, highest first", title_fontsize=7.5)
        else:
            ax.legend(fontsize=6.5, ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.14),
                      handlelength=1.6, columnspacing=0.8)
    s = summary(df)
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(s.index, s.median_abs_change_elo, color="#2a78d6", lw=2, marker="o", ms=3.5,
            label="median over policies" + (" (median of the orders)" if df.order.nunique() > 1 else ""))
    if df.order.nunique() > 1:
        ax.plot(s.index, s.worst_order, color="#2a78d6", lw=1, ls="--", label="worst of the random orders")
    k = df.order.nunique()
    ax.set_title(f"Distance from the final overall score ({k} random order{'s' if k > 1 else ''})", loc="left",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.set_xlabel("% of trials used")
    ax.set_ylabel("|score − final score| (Elo)")
    ax.grid(color="#e4e3df", lw=0.7)
    ax.legend(frameon=False, fontsize=8)
    note = fig.add_subplot(gs[3, 1])
    note.axis("off")
    rows = s.loc[[p for p in (5, 10, 25, 50) if p in s.index]]
    txt = "\n".join(f"{p:>3d} % of trials ({int(r.trials):,}): median change {r.median_abs_change_elo:.0f} Elo, "
                     f"Spearman {r.spearman_with_final:.3f}" for p, r in rows.iterrows())
    k = df.order.nunique()
    how = (f"Lines: one random order of all trials.\nLeft panel: median and worst of {k} random orders."
           if k > 1 else "One random order of all trials.")
    head = "Overall score against the final one" + (f" (median of {k} orders)" if k > 1 else "") + ":\n"
    note.text(0, 0.95, "Each fit starts from scratch on the first x % of the trials\n(shares are nested; 100 % = "
              "main fit).\n" + how + "\n\n" + head + txt,
              va="top", fontsize=8.5, color="#52514e", linespacing=1.5)
    fig.suptitle("Policy ratings over the share of trials used in the joint fit", x=0.01, y=0.99, ha="left",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.955, bottom=0.03)
    axes[0].set_position([0.07, axes[0].get_position().y0, 0.6, axes[0].get_position().height])
    FIG.mkdir(exist_ok=True)
    out = FIG / "rating_stability.png"
    fig.savefig(out)
    print(f"wrote {out}")
    print(s.round(3).to_string())
    s.to_csv(OUT / "summary.csv")


def plot_slide(df):
    """Compact two-panel version for a 16:9 slide: overall scores (a few policies highlighted) and the distance
    from the final score."""
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 12, "axes.edgecolor": "#bdbcb7",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "savefig.dpi": 220, "axes.spines.top": False, "axes.spines.right": False})
    BLUE, INK, INK2, GREY = "#2a78d6", "#1d1d1b", "#52514e", "#c9c8c3"
    # two slow settlers (a ceiling and a floor case) and four policies spread over the range for reference
    HIGHLIGHT = [("LingBot_VA", "#7b5ea7"), ("X_VLA", "#2a78d6"), ("GR00T_N17", "#1baf7a"), ("Pi_0", "#eda100"),
                 ("RDT_1B", "#c8372d"), ("DP", "#eb6834")]
    w = df[df.order == 0].pivot(index="pct", columns="policy", values="overall")
    s = summary(df)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.33, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})
    for p in w.columns:
        if p not in dict(HIGHLIGHT):
            a1.plot(w.index, w[p], color=GREY, lw=1.0, zorder=1)
    shown = [(p, col) for p, col in HIGHLIGHT if p in w.columns]
    ys = {p: w.loc[100, p] for p, _ in shown}
    gap = (w.max().max() - w.min().min()) * 0.055                 # keep end labels apart
    placed = {}
    for p in sorted(ys, key=ys.get):
        y = ys[p]
        if placed:
            y = max(y, max(placed.values()) + gap)
        placed[p] = y
    for p, col in shown:
        a1.plot(w.index, w[p], color=col, lw=2.4, zorder=3)
        a1.text(101.5, placed[p], f"{display_name(p)} {w.loc[100, p]:.0f}", color=col, va="center",
                fontsize=11.5, fontweight="medium")
    if "RDT_1B" in w.columns:
        y0, y1 = w.loc[25, "RDT_1B"], w.loc[30, "RDT_1B"]
        a1.annotate("RDT-1B: few RoboDojo successes,\njumps at 30 %, settles by ~60 %", ((25 + 30) / 2, (y0 + y1) / 2),
                    xytext=(40, w.min().min() + 40), fontsize=10.5, color="#c8372d",
                    arrowprops=dict(arrowstyle="-", color="#c8372d", lw=0.9))
    if "LingBot_VA" in w.columns:
        a1.annotate("LingBot-VA: 500 LIBERO trials near 100 %\nsuccess, drifts down until ~80 %",
                    (30, w.loc[30, "LingBot_VA"]), xytext=(34, w.max().max() - 20), fontsize=10.5, color="#7b5ea7",
                    va="top", arrowprops=dict(arrowstyle="-", color="#7b5ea7", lw=0.9))
    a1.set_xlim(0, 100)
    a1.set_xlabel("% of trials used")
    a1.set_ylabel("overall score (Elo)")
    a1.set_title(f"Overall score of the {w.shape[1]} policies (6 highlighted)", loc="left", fontsize=13,
                 fontweight="bold", color=INK)
    a1.grid(color="#ecebe7", lw=0.8)
    k = df.order.nunique()
    a2.plot(s.index, s.median_abs_change_elo, color=BLUE, lw=2.6, marker="o", ms=4.5, zorder=3,
            label="median over policies" + (f" (median of {k} orders)" if k > 1 else ""))
    if k > 1:
        a2.plot(s.index, s.worst_order, color=BLUE, lw=1.4, ls="--", label="worst random order")
    r = s.loc[25]
    a2.axvline(25, color=INK2, lw=1, ls=":")
    a2.annotate(f"25 % of trials:\nmedian change {r.median_abs_change_elo:.0f} Elo\n"
                f"ranking ρ = {r.spearman_with_final:.2f}", (25, r.median_abs_change_elo),
                xytext=(38, s.median_abs_change_elo.max() * 0.55), fontsize=11.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    a2.set_xlim(0, 100)
    a2.set_ylim(0, None)
    a2.set_xlabel("% of trials used")
    a2.set_ylabel("|score − final score| (Elo)")
    a2.set_title("Distance from the final overall score", loc="left", fontsize=13, fontweight="bold", color=INK)
    a2.grid(color="#ecebe7", lw=0.8)
    a2.legend(frameon=False, fontsize=11, loc="upper right")
    fig.tight_layout(w_pad=4)
    out = FIG / "rating_stability_slide.png"
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    if "--slide" in sys.argv:
        plot_slide(pd.read_csv(OUT / "levels.csv"))
    elif "--plot" in sys.argv:
        plot(pd.read_csv(OUT / "levels.csv"))
    else:
        n = int(sys.argv[sys.argv.index("--orders") + 1]) if "--orders" in sys.argv else 5
        plot(run(n))

"""AGI-Elo-style figures and tables from the main fit (Sun et al., NeurIPS 2025, arXiv 2505.12844).

  1. figures/agielo_calibration.png        predicted vs observed success by rating gap   (AGI-Elo Fig. 5)
  2. agielo_consistency.csv / .png         rank consistency and prediction error          (AGI-Elo Table 2)
  3. figures/agielo_difficulty_percentiles.png  task difficulty histogram, cumulative %
                                                curve and policy levels per setting       (AGI-Elo Fig. 3)
     agielo_tasks_beaten.csv               share of tasks each policy beats at >= 50 % / >= 90 %
  4. agielo_competency_gap.csv / .png      gap from the best policy to an "oracle" that succeeds on the
                                           hardest task with 50 / 90 / 99 % chance        (AGI-Elo Table 1)

  python scripts/agielo_figures.py            # needs results/unified/fits.pkl (run `run_unified_scale.py fit`)
Tables go to results/unified/, figures to figures/.
"""
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_unified_scale as R  # noqa: E402  (data_main, output folder)
from roboelo import ELO_PER_LOGIT as K  # noqa: E402
from roboelo.names import display_name  # noqa: E402

RES, FIG = R.OUT, R.ROOT / "figures"
FIG.mkdir(exist_ok=True)
NAMES = {"robodojo": "RoboDojo", "robotwin2/easy": "RoboTwin clean", "robotwin2/hard": "RoboTwin randomized",
         "libero": "LIBERO", "simplerenv/google": "SimplerEnv Google Robot", "simplerenv/widowx": "SimplerEnv WidowX"}
ORDER = ["robodojo", "robotwin2/easy", "robotwin2/hard", "libero", "simplerenv/google", "simplerenv/widowx"]
COL = {"robodojo": "#2a78d6", "robotwin2/easy": "#f0904f", "robotwin2/hard": "#b5401a", "libero": "#1baf7a",
       "simplerenv/google": "#7b5ea7", "simplerenv/widowx": "#a48cc9"}
INK, INK2, GRID = "#1d1d1b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#bdbcb7", "savefig.dpi": 200})


def load():
    f = pickle.load(open(RES / "fits.pkl", "rb"))["fit"]
    _, d = R.data_main()
    assert list(f.policies) == list(d.policies) and list(f.tests) == list(d.tests), "fit and data do not match"
    units = list(f.units)
    u_t = np.asarray(f.unit_of_test)
    lev = np.asarray(f.level())                          # logits, policies x settings
    beta = np.asarray(f.beta)                            # logits, per test case
    a = np.asarray(f.a)[u_t]
    gap = lev[:, u_t] - beta[None, :]                    # logits
    p = 1 / (1 + np.exp(-a * gap))
    obs = d.observed
    return dict(f=f, d=d, units=units, u_t=u_t, lev=lev, beta=beta, a=a, gap=gap, p=p, obs=obs,
                k=np.where(obs, d.k_sr, 0.0), n=np.where(obs, d.n, 0.0))


def calibration(D, slide=False):
    """Observed success against the rating gap, in 100-Elo bins, with the model's curve.
    slide=True: a compact version with larger text for slides (figures/agielo_calibration_slide.png)."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2) if slide else (13, 7.6), sharey=True)
    for ax, u in zip(axes.ravel(), ORDER):
        j = D["units"].index(u)
        m = D["obs"] & (D["u_t"][None, :] == j)
        g = D["gap"][m] * K                                  # Elo
        k, n = D["k"][m], D["n"][m]
        a = D["f"].a[j]
        edges = np.arange(np.floor(g.min() / 100) * 100, g.max() + 100, 100)
        idx = np.digitize(g, edges)
        xs, ys, ws = [], [], []
        for b in np.unique(idx):
            s = idx == b
            if n[s].sum() > 0:
                xs.append(np.average(g[s], weights=n[s]))
                ys.append(k[s].sum() / n[s].sum())
                ws.append(n[s].sum())
        ws = np.array(ws)
        xx = np.linspace(g.min() - 50, g.max() + 50, 300)
        ax.plot(xx, 1 / (1 + np.exp(-a * xx / K)), color=INK2, lw=1.6, label="model prediction")
        ax.scatter(xs, ys, s=12 + 160 * ws / ws.max(), color=COL[u], alpha=0.85, lw=0, zorder=3,
                   label="observed (bins of 100 Elo; size = trials)")
        cell_err = np.abs(k / np.maximum(n, 1) - D["p"][m])
        ax.set_title(f"{NAMES[u]}  (a = {a:.2f})", loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.text(0.03, 0.95, f"{int(n.sum()):,} trials" + ("" if slide else "\nmean |observed − predicted| per cell: "
                f"{np.average(cell_err, weights=n):.3f}"), transform=ax.transAxes, va="top",
                fontsize=9 if slide else 8.5, color=INK2)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(color=GRID, lw=0.7)
        ax.axvline(0, color=GRID, lw=1)
        ax.set_xlabel("policy level − task difficulty (Elo)", fontsize=9 if slide else 10)
    for ax in axes[:, 0]:
        ax.set_ylabel("success rate")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9.5)
    if not slide:
        fig.suptitle("Predicted vs observed success by rating gap", x=0.01, ha="left", fontsize=13.5,
                     fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1 if slide else 0.96))
    fig.savefig(FIG / ("agielo_calibration_slide.png" if slide else "agielo_calibration.png"))
    plt.close(fig)


def consistency(D):
    """AGI-Elo Table 2: rank consistency with raw success rates, and binned prediction error."""
    rows = []
    for u in ORDER:
        j = D["units"].index(u)
        cols = D["u_t"] == j
        o = D["obs"][:, cols]
        k, n, p = D["k"][:, cols], D["n"][:, cols], D["p"][:, cols]
        ran = o.any(1)
        t_succ = k.sum(0) / np.maximum(n.sum(0), 1)
        a_succ = k[ran].sum(1) / np.maximum(n[ran].sum(1), 1)
        rho_t = spearmanr(D["beta"][cols], t_succ).statistic
        rho_a = spearmanr(D["lev"][ran, j], a_succ).statistic
        # AGI-Elo's binned error: for each policy, test cases grouped into 100-Elo bins of difficulty;
        # mean observed vs mean predicted success in each (policy, bin)
        bins = np.floor(D["beta"][cols] * K / 100)
        err = []
        for i in np.where(ran)[0]:
            for b in np.unique(bins[o[i]]):
                s = o[i] & (bins == b)
                err.append(k[i, s].sum() / n[i, s].sum() - np.average(p[i, s], weights=n[i, s]))
        err = np.array(err)
        rows.append(dict(setting=NAMES[u], test_cases=int(cols.sum()), policies=int(ran.sum()),
                         trials=int(n.sum()), rho_t=rho_t, rho_a=rho_a, MAE=np.abs(err).mean(),
                         MSE=(err ** 2).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "agielo_consistency.csv", index=False)
    table_png(df.assign(rho_t=df.rho_t.map("{:.3f}".format), rho_a=df.rho_a.map("{:.3f}".format),
                        MAE=df.MAE.map("{:.3f}".format), MSE=df.MSE.map("{:.4f}".format),
                        trials=df.trials.map("{:,}".format)),
              ["Setting", "Test cases", "Policies", "Trials", "ρ_t ↓", "ρ_a ↑", "MAE ↓", "MSE ↓"],
              FIG / "agielo_consistency_table.png",
              "ρ_t: task difficulty vs the task's mean success; ρ_a: policy level vs its mean success; "
              "MAE / MSE: observed vs predicted success per policy and 100-Elo difficulty bin.")
    return df


def percentiles(D):
    """AGI-Elo Fig. 3: difficulty histogram, cumulative share of tasks and policy levels per setting."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    beaten = []
    for ax, u in zip(axes.ravel(), ORDER):
        j = D["units"].index(u)
        cols = D["u_t"] == j
        b = np.sort(D["beta"][cols] * K + 1500)
        ran = D["f"].observed_unit[:, j]
        lv = D["lev"][ran, j] * K + 1500
        pol = np.asarray(D["f"].policies)[ran]
        a = D["f"].a[j]
        lo, hi = min(b.min(), lv.min()) - 150, max(b.max(), lv.max()) + 150
        ax.hist(b, bins=np.arange(np.floor(lo / 100) * 100, hi + 100, 100), color=COL[u], alpha=0.35,
                edgecolor="white")
        ax.set_ylabel("test cases", color=INK2)
        ax2 = ax.twinx()
        xs = np.linspace(lo, hi, 400)
        ax2.plot(xs, [100 * np.mean(b <= x) for x in xs], color=COL[u], lw=2.2)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("% of test cases below", color=INK2)
        ax2.spines["top"].set_visible(False)
        order = np.argsort(-lv)
        for r, i in enumerate(order):
            ax.axvline(lv[i], color=INK if r == 0 else "#9a9893", lw=1.6 if r == 0 else 0.7,
                       ls="-" if r == 0 else ":")
        for i, (name, l) in enumerate(zip(pol, lv)):
            beaten.append(dict(setting=NAMES[u], policy=display_name(name), level=l,
                               beats_50=np.mean(b < l), beats_90=np.mean(b < l - K * np.log(9) / a)))
        top = order[0]
        s50 = np.mean(b < lv[top])
        s90 = np.mean(b < lv[top] - K * np.log(9) / a)
        ax.set_title(NAMES[u], loc="left", fontsize=11.5, fontweight="bold", color=INK)
        ax.text(0.02, 0.97, f"best: {display_name(pol[top])} ({lv[top]:.0f})\n"
                f"beats {100 * s50:.0f} % of test cases with ≥ 50 % chance, {100 * s90:.0f} % with ≥ 90 %",
                transform=ax.transAxes, va="top", fontsize=8.8, color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
        ax.set_xlim(lo, hi)
        ax.grid(color=GRID, lw=0.6, axis="x")
        ax.set_xlabel("rating (Elo)")
    fig.suptitle("Task difficulty per setting, with policy levels (solid: best policy; dotted: the others)",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "agielo_difficulty_percentiles.png")
    plt.close(fig)
    df = pd.DataFrame(beaten).sort_values(["setting", "level"], ascending=[True, False])
    df.to_csv(RES / "agielo_tasks_beaten.csv", index=False)
    return df


def competency_gap(D):
    """AGI-Elo Table 1: rating an oracle needs for S chance on the hardest test case, minus the best policy.
    With a conversion factor a: R_oracle(S) = R_t,max + (173.72 / a) * ln(S / (1 - S))."""
    k_tot = D["k"].sum(0)
    rows = []
    for u in ORDER:
        j = D["units"].index(u)
        cols = D["u_t"] == j
        a = D["f"].a[j]
        b = D["beta"][cols] * K + 1500
        solved = k_tot[cols] > 0
        ran = D["f"].observed_unit[:, j]
        lv = D["lev"][ran, j] * K + 1500
        best = lv.max()
        who = display_name(np.asarray(D["f"].policies)[ran][np.argmax(lv)])
        for label, pool in (("all test cases", b), ("solved test cases only", b[solved])):
            tmax = pool.max()
            row = dict(setting=NAMES[u], hardest=label, best_policy=who, best_level=best, a=a, R_t_max=tmax,
                       best_on_hardest=1 / (1 + np.exp(-a * (best - tmax) / K)))
            for s in (0.5, 0.9, 0.99):
                row[f"gap_{int(s * 100)}"] = tmax + K / a * np.log(s / (1 - s)) - best
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "agielo_competency_gap.csv", index=False)
    show = df[df.hardest == "all test cases"].copy()
    sol = df[df.hardest == "solved test cases only"].set_index("setting")
    show["note"] = [("same" if abs(sol.loc[s, "R_t_max"] - r) < 1 else f"{sol.loc[s, 'gap_50']:.0f}")
                    for s, r in zip(show.setting, show.R_t_max)]
    table_png(pd.DataFrame({
        "setting": show.setting, "best": [f"{p} ({l:.0f})" for p, l in zip(show.best_policy, show.best_level)],
        "hardest": show.R_t_max.map("{:.0f}".format), "p": show.best_on_hardest.map("{:.3f}".format),
        "g50": show.gap_50.map(fmt_gap), "g90": show.gap_90.map(fmt_gap),
        "g99": show.gap_99.map(fmt_gap), "note": show.note}),
        ["Setting", "Best policy", "Hardest task", "Best on hardest", "Gap @50 %", "Gap @90 %", "Gap @99 %",
         "Gap @50 % (solved only)"],
        FIG / "agielo_competency_gap_table.png",
        "Gap = rating an oracle needs for that chance of success on the hardest test case, minus the best policy's "
        "level. Unsolved tasks are placed mostly by the prior, hence the last column.")
    return df


def fmt_gap(g):
    return "passed" if g <= 0 else f"{g:.0f}"


def table_png(df, header, out, note):
    fig, ax = plt.subplots(figsize=(13, 0.45 * (len(df) + 1)))
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=header, loc="upper left", cellLoc="left", colLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.55)
    t.auto_set_column_width(list(range(len(header))))
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d9d4c7")
        if r == 0:
            cell.set_facecolor("#1c2230")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#fbfaf7" if r % 2 else "#ece9e2")
    fig.text(0.01, 0.0, note, fontsize=8.5, color=INK2, va="top")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    D = load()
    calibration(D)
    calibration(D, slide=True)
    c = consistency(D)
    b = percentiles(D)
    g = competency_gap(D)
    pd.set_option("display.width", 200)
    print("Consistency and prediction error (AGI-Elo Table 2)")
    print(c.round(4).to_string(index=False))
    print("\nBest policy per setting: share of test cases beaten")
    print(b.groupby("setting").head(1).round(3).to_string(index=False))
    print("\nCompetency gap to an oracle (AGI-Elo Table 1)")
    print(g.round(1).to_string(index=False))
    print(f"\nwrote {FIG}/agielo_calibration.png, agielo_difficulty_percentiles.png, agielo_consistency_table.png, "
          f"agielo_competency_gap_table.png and the CSVs in {RES}")


if __name__ == "__main__":
    main()

"""Fit the RoboElo rating (roboelo/unified.py, docs/ELO_METHOD.md) on everything in data/ and check it.

Data: RoboDojo and RoboTwin 2.0 (official leaderboards), LIBERO (our own runs plus the Manipulation Benchmark
Audit's per-episode records for Pi_05_SF and SimVLA) and the standard SimplerEnv tasks (our own runs). The audit's
shifted SimplerEnv conditions are linked to the other benchmarks by only three policies, so they are left out of
the main fit; `fit` also writes a sensitivity fit that includes them.
Policies: the milestone policies of data/milestone_policies.csv that have results. ROBOELO_POLICIES=all fits every
policy with results instead and writes to results/unified_all_policies/. The configuration in roboelo/unified.py was
chosen by the linking test on the milestone policies; for every policy, run `linking` again and set the one it chooses.
ROBOELO_MIN_BENCHMARKS=2 keeps only policies measured on at least two benchmarks and writes to <output folder>_min2bench/.

  python scripts/run_unified_scale.py linking     linking test over the grid of prior widths and conversion-factor
                                                  granularities: hide one policy's results on one benchmark and
                                                  predict them from its results elsewhere
  python scripts/run_unified_scale.py fit         main fit, with the configuration set in roboelo/unified.py
  python scripts/run_unified_scale.py sources     linking test from one source benchmark at a time
  python scripts/run_unified_scale.py bootstrap   1,000 replicates: test cases resampled within every setting
  python scripts/run_unified_scale.py cv          random-cell cross-validation (5 folds) of the model variants
  python scripts/run_unified_scale.py recovery    parameter recovery on data simulated from the main fit
  python scripts/run_unified_scale.py report      ratings with 95 % intervals (needs linking, fit, sources,
                                                  bootstrap)
Outputs in results/unified/. Runs on ROBOELO_WORKERS processes (default: number of CPUs - 2).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # the fits are small; parallelism comes from processes
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import itertools
import json
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

from roboelo import unified as U
from roboelo.data import EXTRA_CSV
from roboelo.own_runs import combined_extra

ROOT = Path(__file__).resolve().parents[1]  # repository root

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # policy names are not all ASCII
POLICIES = os.environ.get("ROBOELO_POLICIES", "milestones")    # "milestones" | "all"
if POLICIES not in ("milestones", "all"):
    raise SystemExit(f"ROBOELO_POLICIES must be 'milestones' or 'all', not {POLICIES!r}")
MIN_BENCHMARKS = int(os.environ.get("ROBOELO_MIN_BENCHMARKS", "1"))   # keep policies measured on >= this many benchmarks
OUT = ROOT / ("results/unified" if POLICIES == "milestones" else "results/unified_all_policies")
if MIN_BENCHMARKS > 1:
    OUT = OUT.with_name(OUT.name + f"_min{MIN_BENCHMARKS}bench")
OUT.mkdir(parents=True, exist_ok=True)
WORKERS = int(os.environ.get("ROBOELO_WORKERS", max(1, (os.cpu_count() or 2) - 2)))
B = 1000
FOLDS = 5
RECOVERY_REPLICATES = 20
# linking-test grid: prior widths in logits, conversion-factor granularity (None = one shared a = 1)
GRID = [dict(sigma_gamma=sg, sigma_beta=sb, a_by=ab) for sg, sb, ab in
        itertools.product([0.25, 0.5, 1.0, 2.0], [0.5, 1.0, 2.0, 4.0, 8.0], ["setting", "benchmark", None])]
MIN_LINKED = 5      # a benchmark link counts only with at least this many held-out policies
DATA = None


def data_main(include_audit=False):
    ex = combined_extra(EXTRA_CSV)
    if not include_audit:
        ex = ex[ex.benchmark != "simplerenv_audit"]
    data = U.load_all(ex)
    if POLICIES == "milestones":
        keep = set(pd.read_csv(ROOT / "data/milestone_policies.csv").policy)
        ex = ex[ex.policy.isin(keep)]
        data = data.subset(policies=[p for p in data.policies if p in keep])
        data = data.subset(tests=data.tests[data.observed.any(0)])      # drop test cases no milestone policy ran
    if MIN_BENCHMARKS > 1:
        fm, _ = family_matrix(data)
        keep = [p for p, k in zip(data.policies, fm.sum(1)) if k >= MIN_BENCHMARKS]
        lost = [p for p in U.PANEL if p in data.policies and p not in keep]
        if lost:
            raise SystemExit(f"reference-panel policies below {MIN_BENCHMARKS} benchmarks: {lost}")
        ex = ex[ex.policy.isin(keep)]
        data = data.subset(policies=keep)
        data = data.subset(tests=data.tests[data.observed.any(0)])
    return ex, data


def _init_worker():
    global DATA
    DATA = data_main()[1]


def pmap(fn, tasks, label):
    """Run fn(*task) for every task on WORKERS processes; yields (task, result) as they finish."""
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS, initializer=_init_worker) as pool:
        futs = {pool.submit(fn, *t): t for t in tasks}
        for k, fut in enumerate(as_completed(futs), start=1):
            yield futs[fut], fut.result()
            if k % max(1, len(tasks) // 20) == 0 or k == len(tasks):
                print(f"{label}: {k}/{len(tasks)} done, {time.time() - t0:.0f}s", flush=True)


def config_key(cfg):
    return U.config_name(**cfg)


MAIN = dict(sigma_gamma=U.SIGMA_GAMMA, sigma_beta=U.SIGMA_BETA, a_by=U.A_BY)


def family_of_tests(data):
    return np.array([U.FAMILY[u] for u in data.offset_unit])


def family_matrix(data):
    """(P, F) policy has data on benchmark family F, with the family names."""
    fam = family_of_tests(data)
    fams = list(dict.fromkeys(fam))
    return np.array([[data.observed[i, fam == f].any() for f in fams] for i in range(data.P)]), fams


def log_loss(p, y, n):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -n * (y * np.log(p) + (1 - y) * np.log(1 - p))


def task_mean_baseline(data, i, t_cols):
    """Pooled success rate of the other policies on each test case."""
    others = data.observed[:, t_cols].copy()
    others[i] = False
    k = np.where(others, data.k_sr[:, t_cols], 0).sum(0)
    n = np.where(others, data.n[:, t_cols], 0).sum(0)
    return np.where(n > 0, k / np.maximum(n, 1), np.nan)


def observed_level(f, data, i, family):
    """The level policy i shows on its cells of a benchmark family, on the scale of fit f (difficulties and
    conversion factors held fixed): the mean over the family's settings of the 1-D maximum-likelihood level.
    NaN if the policy solved all or none of its trials there (the level is then unbounded)."""
    levels = []
    for j, u in enumerate(f.units):
        if U.FAMILY[u] != family:
            continue
        t = np.where((f.unit_of_test == j) & data.observed[i])[0]
        if not len(t):
            continue
        n, k, a, b = data.n[i, t], data.k_sr[i, t], f.a[j], f.beta[t]
        if k.sum() == 0 or k.sum() == n.sum():
            return np.nan
        nll = lambda L: (n * np.logaddexp(0, a * (L - b)) - k * a * (L - b)).sum()
        levels.append(minimize_scalar(nll, bounds=(b.min() - 30 / a, b.max() + 30 / a), method="bounded").x)
    return f.elo(np.mean(levels)) if levels else np.nan


# ---------------------------------------------------------------- linking test

def _hidden_prediction(cfg, raw, i, target, keep=None):
    """Hide policy i's results on the target family (and, if keep is given, on every family except keep),
    fit, and return per-cell predictions and the predicted / shown level on the target."""
    d = DATA
    fam = family_of_tests(d)
    hide = np.zeros_like(d.observed)
    hide[i, fam == target] = True
    if keep is not None:
        hide[i, fam != keep] = True
    f = U.fit(d.mask_cells(hide), init=raw, **cfg)
    t_cols = np.where((fam == target) & d.observed[i])[0]
    rate = d.rate("sr")
    F, fams = f.family_levels(fill=True)
    cells = pd.DataFrame({"policy": d.policies[i], "benchmark": target, "test_case": d.tests[t_cols],
                          "n": d.n[i, t_cols], "observed": rate[i, t_cols], "pred": f.predict()[i, t_cols],
                          "pred_task_mean": task_mean_baseline(d, i, t_cols)})
    level = dict(policy=d.policies[i], benchmark=target, level_pred=f.elo(F[i, fams.index(target)]),
                 level_shown=observed_level(f, d, i, target))
    return cells, level, f.extras["converged"]


def _linking_task(c, cfg, raw, i, target):
    cells, level, conv = _hidden_prediction(cfg, raw, i, target)
    return cells.assign(config=c), dict(level, config=c, converged=conv)


def summarise_linking(cells, by):
    """Pooled log loss per hidden trial for the model and the task-mean baseline, and the share of
    policies whose own loss improves."""
    c = cells.assign(ll=log_loss(cells.pred, cells.observed, cells.n),
                     ll_base=log_loss(cells.pred_task_mean, cells.observed, cells.n))
    per_pol = c.groupby(by + ["policy"])[["ll", "ll_base", "n"]].sum()
    g = c.groupby(by)
    out = pd.DataFrame({"n_policies": g.policy.nunique(), "trials": g.n.sum(),
                        "loss": g.ll.sum() / g.n.sum(), "loss_task_mean": g.ll_base.sum() / g.n.sum()})
    out["share_improved"] = (per_pol.ll < per_pol.ll_base).groupby(level=by).mean()
    return out.reset_index()


def cmd_linking():
    ex, data = data_main()
    fm, fams = family_matrix(data)
    pairs = [(i, fams[j]) for i in range(data.P) for j in range(len(fams)) if fm[i, j] and fm[i].sum() >= 2]
    grid = GRID if "--main-only" not in sys.argv else [MAIN]
    print(f"{len(pairs)} (policy, benchmark) pairs x {len(grid)} configurations, {WORKERS} workers")
    full, rows_p, rows_a = {}, [], []
    for (k,), f in pmap(_full_fit_task, [(k,) for k in range(len(grid))], "full fits"):
        full[k] = f.extras["raw"]
        rows_p.append(pd.DataFrame({"config": config_key(grid[k]), "policy": f.policies,
                                    "overall_elo": f.elo(f.overall())}))
        rows_a.append(pd.DataFrame({"config": config_key(grid[k]), "setting": f.units, "a": f.a}))
    pd.concat(rows_p).to_csv(OUT / "linking_full_fits.csv", index=False)
    pd.concat(rows_a).to_csv(OUT / "linking_full_factors.csv", index=False)
    tasks = [(config_key(grid[k]), grid[k], full[k], i, fam) for k in range(len(grid)) for i, fam in pairs]
    cells, levels = [], []
    for _, (c, lv) in pmap(_linking_task, tasks, "linking"):
        cells.append(c)
        levels.append(lv)
    cells, levels = pd.concat(cells, ignore_index=True), pd.DataFrame(levels)
    cells.to_csv(OUT / "linking_cells.csv", index=False)
    levels.to_csv(OUT / "linking_levels.csv", index=False)
    s = summarise_linking(cells, ["config", "benchmark"])
    err = levels.assign(sq=(levels.level_pred - levels.level_shown) ** 2).groupby(["config", "benchmark"])
    s["level_rmse_elo"] = s.set_index(["config", "benchmark"]).index.map(np.sqrt(err.sq.mean()))
    s["level_n"] = s.set_index(["config", "benchmark"]).index.map(err.sq.count())
    s.to_csv(OUT / "linking_summary.csv", index=False)
    score = s[s.benchmark.isin(U.MAIN_FAMILIES)].groupby("config").loss.mean().sort_values()
    score.rename("mean_loss_over_benchmarks").to_csv(OUT / "linking_configs.csv")
    print(f"{(~levels.converged).sum()} of {len(levels)} fits did not report convergence")
    print("\nbest configurations (mean over the four benchmarks of the log loss per hidden trial):")
    print(score.head(8).round(4).to_string())
    best = score.index[0]
    print(f"\n{best}:")
    print(s[s.config == best].drop(columns="config").round(4).to_string(index=False))
    print("\nSet SIGMA_GAMMA, SIGMA_BETA and A_BY in roboelo/unified.py to the chosen configuration.")


def _full_fit_task(k):
    grid = GRID if "--main-only" not in sys.argv else [MAIN]
    return U.fit(DATA, **grid[k])


def level_errors():
    """Root-mean-square error (Elo) of a predicted benchmark level, per hidden benchmark, from the linking test
    of the configuration set in roboelo/unified.py."""
    s = pd.read_csv(OUT / "linking_summary.csv")
    s = s[s.config == config_key(MAIN)]
    if s.empty:
        raise SystemExit(f"linking_summary.csv has no rows for {config_key(MAIN)}: run `linking` again")
    return dict(zip(s.benchmark, s.level_rmse_elo))


def source_links():
    """(source, target) benchmark pairs along which a level may be predicted: in the single-source linking test,
    at least MIN_LINKED held-out policies and a lower log loss than the task mean."""
    s = pd.read_csv(OUT / "sources_summary.csv")
    ok = s[(s.n_policies >= MIN_LINKED) & (s.loss < s.loss_task_mean)]
    return set(zip(ok.source, ok.target))


# ---------------------------------------------------------------- main fit

def ratings_table(f, data, ex):
    F, fams = f.family_levels()
    Ff, _ = f.family_levels(fill=True)
    df = pd.DataFrame({"policy": f.policies, "overall_elo": f.elo(f.overall())})
    for j, fam in enumerate(fams):
        df[f"level_{fam}"] = f.elo(F[:, j])
    for fam in U.MAIN_FAMILIES:
        j = fams.index(fam)
        df[f"predicted_level_{fam}"] = np.where(np.isnan(F[:, j]), f.elo(Ff[:, j]), np.nan)
    df["benchmarks"] = [", ".join(fam for j, fam in enumerate(fams) if not np.isnan(F[i, j])) for i in range(len(df))]
    df["n_benchmarks"] = (~np.isnan(F)).sum(1)
    df["rollouts"] = data.n.sum(1).astype(int)
    partial = set(ex[ex.setting == "own run (partial)"].policy)
    df["partial_run"] = df.policy.isin(partial)
    return df.sort_values("overall_elo", ascending=False)


def cmd_fit():
    ex, data = data_main()
    t0 = time.time()
    f = U.fit(data)
    print(f"{config_key(MAIN)}: {time.time() - t0:.0f}s, converged {f.extras['converged']}, "
          f"max gradient {f.extras['grad_max']:.1e}")
    pickle.dump({"fit": f, "config": MAIN}, open(OUT / "fits.pkl", "wb"))
    ratings_table(f, data, ex).to_csv(OUT / "policy_ratings.csv", index=False)
    pd.DataFrame({"test_case": f.tests, "setting": [f.units[i] for i in f.unit_of_test],
                  "difficulty_elo": f.elo(f.beta),
                  "n_policies": data.observed.sum(0)}).to_csv(OUT / "test_difficulty.csv", index=False)
    pd.DataFrame({"setting": f.units, "a": f.a,
                  "anchors": [", ".join(map(str, f.extras["anchors"][u])) for u in f.units],
                  "n_anchors": [len(f.extras["anchors"][u]) for u in f.units]}).to_csv(OUT / "settings.csv", index=False)
    # sensitivity: with the audit's shifted SimplerEnv tasks
    ex2, d2 = data_main(include_audit=True)
    g = U.fit(d2, init=f.extras["raw"])
    ratings_table(g, d2, ex2).to_csv(OUT / "policy_ratings_with_audit.csv", index=False)
    pd.DataFrame({"setting": g.units, "a": g.a}).to_csv(OUT / "settings_with_audit.csv", index=False)
    print(pd.read_csv(OUT / "settings.csv").round(3).to_string(index=False))


def main_fit():
    d = pickle.load(open(OUT / "fits.pkl", "rb"))
    if d["config"] != MAIN:
        raise SystemExit("results/unified/fits.pkl was fitted with another configuration: run `fit` again")
    return d["fit"]


# ---------------------------------------------------------------- linking test from one source

def _source_task(raw, i, target, source):
    cells, level, conv = _hidden_prediction(MAIN, raw, i, target, keep=source)
    return cells.assign(source=source), dict(level, source=source, converged=conv)


def cmd_sources():
    ex, data = data_main()
    f0 = main_fit()
    fm, fams = family_matrix(data)
    tasks = [(f0.extras["raw"], i, fams[t], fams[s]) for i in range(data.P) for t in range(len(fams))
             for s in range(len(fams)) if s != t and fm[i, t] and fm[i, s]]
    cells, levels = [], []
    for _, (c, lv) in pmap(_source_task, tasks, "sources"):
        cells.append(c)
        levels.append(lv)
    cells, levels = pd.concat(cells, ignore_index=True), pd.DataFrame(levels)
    cells.to_csv(OUT / "sources_cells.csv", index=False)
    s = summarise_linking(cells.rename(columns={"benchmark": "target"}), ["source", "target"])
    err = levels.assign(sq=(levels.level_pred - levels.level_shown) ** 2).groupby(["source", "benchmark"]).sq
    s["level_rmse_elo"] = [np.sqrt(err.mean().get((a, b), np.nan)) for a, b in zip(s.source, s.target)]
    s.to_csv(OUT / "sources_summary.csv", index=False)
    print(f"{(~levels.converged).sum()} of {len(levels)} fits did not report convergence")
    print(s.round(4).to_string(index=False))


# ---------------------------------------------------------------- bootstrap

def _boot_task(b, raw):
    rng = np.random.default_rng([20261027, b])
    units = np.asarray(DATA.offset_unit)
    idx = np.concatenate([rng.choice(np.where(units == u)[0], size=(units == u).sum(), replace=True)
                          for u in dict.fromkeys(units)])
    f = U.fit(DATA.take_tests(idx), init=raw)
    F, fams = f.family_levels()
    pol = pd.DataFrame({"b": b, "policy": f.policies, "overall_elo": f.elo(f.overall()),
                        "full_coverage_elo": f.elo(f.overall_full()),       # every benchmark filled; see report
                        **{f"level_{fam}": f.elo(F[:, j]) for j, fam in enumerate(fams)}})
    return pol, pd.DataFrame({"b": b, "setting": f.units, "a": f.a, "converged": f.extras["converged"]})


def cmd_bootstrap():
    f0 = main_fit()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else B
    rows_p, rows_a = [], []
    for _, (p, a) in pmap(_boot_task, [(b, f0.extras["raw"]) for b in range(n)], "bootstrap"):
        rows_p.append(p)
        rows_a.append(a)
    rows_a = pd.concat(rows_a).sort_values(["b", "setting"])
    pd.concat(rows_p).sort_values(["b", "policy"]).to_csv(OUT / "bootstrap_policies.csv", index=False)
    rows_a.to_csv(OUT / "bootstrap_settings.csv", index=False)
    print(f"{(~rows_a.groupby('b').converged.first()).sum()} of {n} fits did not report convergence")


# ---------------------------------------------------------------- random-cell cross-validation

VARIANTS = {"main": {}, "offsets per benchmark": {"per_benchmark_offsets": True},
            **{name: {"a_by": a} for name, a in [("conversion factor per setting", "setting"),
                                                  ("conversion factor per benchmark", "benchmark"),
                                                  ("one shared conversion factor", None)] if a != U.A_BY}}


def _cv_task(variant, fold, hidden_cells):
    opts = dict(VARIANTS[variant])
    d = U.offsets_per_benchmark(DATA) if opts.pop("per_benchmark_offsets", False) else DATA
    cfg = {**MAIN, **opts}
    hide = np.zeros_like(d.observed)
    hide[hidden_cells[:, 0], hidden_cells[:, 1]] = True
    f = U.fit(d.mask_cells(hide), **cfg)
    if not f.extras["converged"]:
        raise RuntimeError(f"cross-validation fit did not converge: {variant}, fold {fold}")
    p = f.predict()[hidden_cells[:, 0], hidden_cells[:, 1]]
    rate = d.rate("sr")[hidden_cells[:, 0], hidden_cells[:, 1]]
    n = d.n[hidden_cells[:, 0], hidden_cells[:, 1]]
    fam = family_of_tests(DATA)[hidden_cells[:, 1]]
    return pd.DataFrame({"variant": variant, "fold": fold, "benchmark": fam, "n": n,
                         "ll": log_loss(p, rate, n)})


def cmd_cv():
    _, data = data_main()
    cells = np.argwhere(data.observed)
    fold_of = np.random.default_rng(20261027).permutation(len(cells)) % FOLDS
    tasks = [(v, k, cells[fold_of == k]) for v in VARIANTS for k in range(FOLDS)]
    rows = pd.concat([r for _, r in pmap(_cv_task, tasks, "cv")], ignore_index=True)
    g = rows.groupby(["variant", "benchmark"])
    s = (g.ll.sum() / g.n.sum()).unstack()
    s.insert(0, "all cells", rows.groupby("variant").ll.sum() / rows.groupby("variant").n.sum())
    s = s.loc[list(VARIANTS)]
    s.to_csv(OUT / "cv_summary.csv")
    print("held-out log loss per trial:")
    print(s.round(4).to_string())


# ---------------------------------------------------------------- parameter recovery

def _recovery_task(r, truth):
    rng = np.random.default_rng([20261028, r])
    k = np.where(DATA.observed, rng.binomial(DATA.n.astype(int), truth.predict()), 0).astype(float)
    g = U.fit(replace(DATA, k_sr=k, k_score=k), init=truth.extras["raw"])
    if not g.extras["converged"]:
        raise RuntimeError(f"recovery fit did not converge: replicate {r}")
    F0, fams = truth.family_levels()
    F1, _ = g.family_levels()
    o0, o1 = truth.elo(truth.overall()), g.elo(g.overall())
    row = {"replicate": r, "overall_spearman": spearmanr(o0, o1)[0], "overall_bias_elo": np.mean(o1 - o0),
           "overall_rmse_elo": np.sqrt(np.mean((o1 - o0) ** 2)),
           "difficulty_rmse_elo": np.sqrt(np.mean((g.elo(g.beta) - truth.elo(truth.beta)) ** 2))}
    for j, fam in enumerate(fams):
        m = ~np.isnan(F0[:, j]) & ~np.isnan(F1[:, j])
        row[f"level_rmse_elo {fam}"] = np.sqrt(np.mean((g.elo(F1[m, j]) - truth.elo(F0[m, j])) ** 2))
    for j, u in enumerate(truth.units):
        row[f"a_ratio {u}"] = g.a[j] / truth.a[j]
    return row


def cmd_recovery():
    f0 = main_fit()
    rows = pd.DataFrame([r for _, r in pmap(_recovery_task, [(r, f0) for r in range(RECOVERY_REPLICATES)],
                                            "recovery")]).sort_values("replicate")
    rows.to_csv(OUT / "recovery_replicates.csv", index=False)
    s = rows.drop(columns="replicate").agg(["mean", "std"]).T
    s.to_csv(OUT / "recovery_summary.csv")
    print(s.round(3).to_string())


# ---------------------------------------------------------------- report

def cmd_report():
    ex, data = data_main()
    f = main_fit()
    rmse, links = level_errors(), source_links()
    t = ratings_table(f, data, ex)
    bs = pd.read_csv(OUT / "bootstrap_policies.csv")
    q = bs.groupby("policy").overall_elo.quantile([0.025, 0.975]).unstack()
    t["overall_lo"], t["overall_hi"] = t.policy.map(q[0.025]), t.policy.map(q[0.975])
    # full-coverage score where every benchmark the policy did not run is linked from one it ran; interval from
    # the bootstrap spread plus the linking test's error of each predicted benchmark level
    full = pd.Series(f.elo(f.overall_full(links)), index=f.policies)
    t["full_coverage_elo"] = t.policy.map(full)
    ran = {p: set(b.split(", ")) for p, b in zip(t.policy, t.benchmarks)}
    t["predicted_benchmarks"] = [", ".join(x for x in U.MAIN_FAMILIES if x not in ran[p]) if not np.isnan(v) else ""
                                 for p, v in zip(t.policy, t.full_coverage_elo)]
    var = t.policy.map(bs.groupby("policy").full_coverage_elo.var()).to_numpy()
    pred_var = np.array([sum((rmse[x] / len(U.MAIN_FAMILIES)) ** 2 for x in s.split(", ") if x)
                         for s in t.predicted_benchmarks])
    sd = np.sqrt(var + pred_var)
    t["full_coverage_lo"], t["full_coverage_hi"] = t.full_coverage_elo - 1.96 * sd, t.full_coverage_elo + 1.96 * sd
    measured = t.full_coverage_elo.notna() & (t.predicted_benchmarks == "")   # ran all four: same as overall score
    t.loc[measured, "full_coverage_lo"] = t.loc[measured, "overall_lo"]
    t.loc[measured, "full_coverage_hi"] = t.loc[measured, "overall_hi"]
    t["bootstrap_replicates"] = t.policy.map(bs.groupby("policy").b.nunique())
    t.to_csv(OUT / "policy_ratings.csv", index=False)
    a = pd.read_csv(OUT / "bootstrap_settings.csv").groupby("setting").a.quantile([0.025, 0.975]).unstack()
    s = pd.read_csv(OUT / "settings.csv")
    s["a_lo"], s["a_hi"] = s.setting.map(a[0.025]), s.setting.map(a[0.975])
    s.to_csv(OUT / "settings.csv", index=False)
    # face validity: level against raw success on each benchmark
    fam = family_of_tests(data)
    F, fams = f.family_levels()
    rows = []
    for j, b in enumerate(fams):
        m = ~np.isnan(F[:, j])
        raw = data.k_sr[:, fam == b].sum(1) / np.maximum(data.n[:, fam == b].sum(1), 1)
        rows.append(dict(benchmark=b, policies=int(m.sum()), spearman=spearmanr(F[m, j], raw[m])[0]))
    pd.DataFrame(rows).to_csv(OUT / "face_validity.csv", index=False)
    cols = ["policy", "overall_elo", "overall_lo", "overall_hi", "full_coverage_elo", "full_coverage_lo",
            "full_coverage_hi", "benchmarks", "predicted_benchmarks"]
    print(t[cols].round(0).to_string(index=False))
    print("benchmark links used for the full-coverage score:", sorted(links))
    print(s.round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    {"linking": cmd_linking, "fit": cmd_fit, "sources": cmd_sources, "bootstrap": cmd_bootstrap, "cv": cmd_cv,
     "recovery": cmd_recovery, "report": cmd_report}[sys.argv[1]]()

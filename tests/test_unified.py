"""Checks of roboelo/unified.py on a small simulated data set (python -m pytest tests, or python tests/test_unified.py)."""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roboelo import unified as U  # noqa: E402
from roboelo.data import RatingData  # noqa: E402

UNITS = ["robodojo", "robotwin2/easy", "robotwin2/hard", "libero", "simplerenv/google"]


def simulated(seed=0, P=14, per_unit=8):
    rng = np.random.default_rng(seed)
    policies = np.array([f"p{i}" for i in range(P)])
    policies[:4] = ["Pi_0", "Pi_05", "X_VLA", "ACT"]                     # reference-panel members
    unit = np.repeat(UNITS, per_unit)
    T = len(unit)
    theta = rng.normal(0, 1.5, P)
    gamma = rng.normal(0, 0.7, (P, len(UNITS)))
    beta = rng.normal(0, 1.5, T)
    a = np.array([1.8, 0.9, 0.9, 0.8, 1.1])
    u = np.array([UNITS.index(x) for x in unit])
    p = 1 / (1 + np.exp(-a[u] * (theta[:, None] + gamma[:, u] - beta)))
    ran = rng.random((P, len(UNITS))) < 0.6                              # which settings each policy ran
    ran[:, 2] = ran[:, 1]                                                # RoboTwin clean and randomized together
    ran[:4, :2] = ran[:4, 2] = True
    n = np.where(ran[:, u], 50.0, 0.0)
    k = rng.binomial(n.astype(int), p).astype(float)
    return RatingData(policies, np.array([f"t{j}" for j in range(T)]),
                      dimension=np.array([x.split("/")[0] for x in unit]),
                      variant=np.array([x.split("/")[-1] for x in unit]),
                      n=n, k_sr=k, k_score=k.copy(), benchmark=np.array([x.split("/")[0] for x in unit]),
                      offset_unit=unit)


def renormalised(f, panel):
    """The fit's raw parameters put on the scale of another reference panel."""
    g = copy.deepcopy(f)
    r = g.extras["raw"]
    g.theta, g.gamma, g.beta, g.a = r["theta"].copy(), r["gamma"].copy(), r["beta"].copy(), r["a"].copy()
    return U.normalise(g, panel)


def test_conventions_change_no_prediction():
    data = simulated()
    f = U.fit(data, panel=["Pi_0", "Pi_05", "X_VLA", "ACT"])
    assert (~f.observed_unit).any(), "the test needs policies without data on some setting"
    base = f.predict()
    for panel in (["Pi_0", "X_VLA"], ["p5", "p6", "p7", "p8", "p9"], []):
        assert np.allclose(renormalised(f, panel).predict(), base, atol=1e-10), panel


def test_raw_fit_predicts_like_the_normalised_one():
    data = simulated(1)
    f = U.fit(data)
    r = f.extras["raw"]
    u = f.unit_of_test
    raw = 1 / (1 + np.exp(-r["a"][u] * ((r["theta"][:, None] + r["gamma"])[:, u] - r["beta"])))
    assert np.allclose(f.predict(), raw, atol=1e-10)


def test_anchors_are_informative_panel_policies():
    data = simulated(2)
    f = U.fit(data, panel=["Pi_0", "Pi_05", "X_VLA", "ACT"])
    for j, u in enumerate(f.units):
        b = U.anchor_mask(f, j, ["Pi_0", "Pi_05", "X_VLA", "ACT"])
        assert b.sum() >= 1
        assert f.observed_unit[b, j].all()
        assert abs(f.gamma[b, j].mean()) < 1e-9                           # convention 1


def test_full_coverage_score_equals_overall_when_everything_ran():
    data = simulated(3)
    f = U.fit(data)
    fams = f.families()
    ran_all = f.measured_families()[:, [fams.index(x) for x in ["RoboDojo", "RoboTwin", "LIBERO", "SimplerEnv"]]]
    full = f.overall_full(links=set())
    assert np.allclose(full[ran_all.all(1)], f.overall()[ran_all.all(1)])
    assert np.isnan(full[~ran_all.all(1)]).all()
    assert not np.isnan(f.overall_full()).any()                          # no links given: everything is filled


def test_full_coverage_score_needs_a_link_from_a_benchmark_the_policy_ran():
    f = U.fit(simulated(4))
    fams = f.families()
    ran = f.measured_families()[:, [fams.index(x) for x in U.MAIN_FAMILIES]]
    only_robodojo = ran[:, 0] & ~ran[:, 1:].any(1)
    links = {("RoboDojo", "RoboTwin"), ("RoboDojo", "LIBERO"), ("RoboDojo", "SimplerEnv")}
    assert only_robodojo.any()
    assert not np.isnan(f.overall_full(links)[only_robodojo]).any()
    assert np.isnan(f.overall_full(links - {("RoboDojo", "LIBERO")})[only_robodojo]).all()


def test_cold_fit_converges_on_a_cross_validation_fold():
    """On this fold of the repository data, an unbounded line search once pushed log a to ~2500 and the
    objective overflowed to NaN."""
    from roboelo.data import EXTRA_CSV
    from roboelo.own_runs import combined_extra
    ex = combined_extra(EXTRA_CSV)
    data = U.load_all(ex[ex.benchmark != "simplerenv_audit"])
    cells = np.argwhere(data.observed)
    hidden = cells[np.random.default_rng(20261027).permutation(len(cells)) % 5 == 4]
    hide = np.zeros_like(data.observed)
    hide[hidden[:, 0], hidden[:, 1]] = True
    f = U.fit(data.mask_cells(hide), sigma_gamma=1.0, sigma_beta=4.0, a_by="setting")
    assert f.extras["converged"] and np.isfinite(f.extras["objective"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

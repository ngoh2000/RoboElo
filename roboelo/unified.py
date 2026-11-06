"""One rating scale across benchmarks (the method is described in docs/ELO_METHOD.md).

    logit P(policy p solves test case t) = a_u * (theta_p + gamma_{p,u} - beta_t),   u = setting of t

theta = overall competency, gamma = competency offset of the policy on setting u (every setting gets one;
no setting is the reference), beta = test-case difficulty, a_u = conversion factor of setting u (how much
the success log-odds move per point of rating gap; one per setting, one per benchmark, or one shared a = 1).
Settings: RoboDojo; RoboTwin clean / randomized; LIBERO; SimplerEnv Google Robot / WidowX (our runs);
SimplerEnv shifted variants (audit).

MAP fit with priors theta ~ N(mu, 2^2), gamma ~ N(0, sigma_gamma^2), beta ~ N(group mean, sigma_beta^2),
group mean ~ N(0, 3^2), log a ~ N(0, sigma_loga^2). A policy without data on a setting has no gamma there in
the fit: its offset is the prior mean, i.e. that of a typical policy. The likelihood is unchanged by three
moves, which the conventions below fix after the fit, so that they change no prediction:
  1. per setting u, a shift of every gamma_{.,u} (also of the policies without data on u) and every beta of
     u: the mean gamma of the setting's anchors is set to 0. Anchors are the first of these groups that has
     at least MIN_ANCHORS members: informative reference-panel policies, informative policies that also ran
     another benchmark (RoboTwin clean and randomized, or SimplerEnv Google Robot and WidowX, count as one
     benchmark), informative policies; else every policy with data on u. Informative = at least
     INFORMATIVE_SHARE of the policy's test cases on u lie strictly between 0 % and 100 % success;
  2. a common scale factor on all a and all ratings: the geometric mean of the conversion factors is 1;
  3. a common shift: the mean overall score of the reference panel is set to 1500.
Overall score of a policy = mean over the benchmarks it ran of its level theta + gamma on that benchmark
(a benchmark with several settings is first averaged over its settings). The full-coverage score averages
over all four main benchmarks and fills a benchmark the policy did not run with its predicted level.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from . import ELO_PER_LOGIT
from .data import RatingData, load

PANEL = ["Pi_0", "Pi_05", "GR00T_N17", "SmolVLA", "OpenVLA_OFT", "X_VLA", "ACT"]
FAMILY = {"robodojo": "RoboDojo", "robotwin2/easy": "RoboTwin", "robotwin2/hard": "RoboTwin", "libero": "LIBERO",
          "simplerenv/google": "SimplerEnv", "simplerenv/widowx": "SimplerEnv",
          "simplerenv_audit": "SimplerEnv (audit, shifted)",
          "robotwin2": "RoboTwin", "simplerenv": "SimplerEnv"}   # last two: one offset unit per benchmark
MAIN_FAMILIES = ["RoboDojo", "RoboTwin", "LIBERO", "SimplerEnv"]
# prior widths (logits) and conversion-factor granularity chosen by the linking test
# (scripts/run_unified_scale.py linking; docs/ELO_METHOD.md section 8)
SIGMA_GAMMA, SIGMA_BETA, A_BY = 2.0, 8.0, "benchmark"
INFORMATIVE_SHARE = 0.25
MIN_ANCHORS = 3
LOGA_BOUND = 4.0                    # |log a| <= 4: a between 0.018 and 55


def config_name(sigma_gamma=SIGMA_GAMMA, sigma_beta=SIGMA_BETA, a_by=A_BY):
    """Label of a configuration in the linking-test outputs."""
    return f"sigma_gamma={sigma_gamma:g} sigma_beta={sigma_beta:g} a_by={a_by or 'shared'}"


def unit_of(benchmark, group):
    if benchmark == "robotwin2":
        return group                                  # robotwin2/easy | robotwin2/hard
    if benchmark == "simplerenv":
        return "simplerenv/widowx" if group.endswith("widowx") else "simplerenv/google"
    return benchmark


def load_all(extra) -> RatingData:
    """RoboDojo plus the given per-test-case rows of other benchmarks, with one offset unit per setting."""
    base = load()
    meta = extra.drop_duplicates("test_case").sort_values(["benchmark", "group", "test_case"])
    new_pol = sorted(set(extra.policy) - set(base.policies), key=str.lower)
    policies = np.array(list(base.policies) + new_pol)
    tests = np.array(list(base.tests) + list(meta.test_case))
    P, T, T0 = len(policies), len(tests), base.T
    n, ks = np.zeros((P, T)), np.zeros((P, T))
    n[:base.P, :T0], ks[:base.P, :T0] = base.n, base.k_sr
    pidx = {p: i for i, p in enumerate(policies)}
    tidx = {t: j for j, t in enumerate(tests)}
    for r in extra.itertuples():
        i, j = pidx[r.policy], tidx[r.test_case]
        n[i, j] += r.n_episodes
        ks[i, j] += r.successes
    return RatingData(
        policies, tests,
        dimension=np.array(list(base.dimension) + [g.split("/")[0] for g in meta.group]),
        variant=np.array(list(base.variant) + [g.split("/")[-1] for g in meta.group]),
        n=n, k_sr=ks, k_score=ks.copy(),
        benchmark=np.array(["robodojo"] * T0 + list(meta.benchmark)),
        offset_unit=np.array(["robodojo"] * T0 + [unit_of(b, g) for b, g in zip(meta.benchmark, meta.group)]))


def offsets_per_benchmark(data: RatingData) -> RatingData:
    """Copy with one offset unit per benchmark instead of per setting (RoboTwin and SimplerEnv merged)."""
    from dataclasses import replace
    merge = {"robotwin2/easy": "robotwin2", "robotwin2/hard": "robotwin2",
             "simplerenv/google": "simplerenv", "simplerenv/widowx": "simplerenv"}
    return replace(data, offset_unit=np.array([merge.get(u, u) for u in data.offset_unit]))


@dataclass
class UnifiedFit:
    policies: np.ndarray
    tests: np.ndarray
    units: list
    unit_of_test: np.ndarray        # (T,) index into units
    observed_unit: np.ndarray       # (P, U) policy has data on the setting
    informative: np.ndarray         # (P, U) policy is informative on the setting (module docstring)
    theta: np.ndarray               # (P,) logits, after the conventions
    gamma: np.ndarray               # (P, U); for a setting without data, the typical policy's offset
    beta: np.ndarray                # (T,)
    a: np.ndarray                   # (U,)
    a_group: np.ndarray             # (U,) index of the conversion factor each setting uses
    a_by: str | None = "setting"
    extras: dict = field(default_factory=dict)

    def level(self):
        """(P, U) level theta + gamma, logits; predicted where the policy has no data on the setting."""
        return self.theta[:, None] + self.gamma

    def logits(self):
        u = self.unit_of_test
        return self.a[u][None, :] * (self.level()[:, u] - self.beta[None, :])

    def predict(self):
        return expit(self.logits())

    def families(self):
        return list(dict.fromkeys(FAMILY[u] for u in self.units))

    def family_levels(self, fill=False):
        """(P, F) level per benchmark family: mean over the family's settings the policy ran, NaN if it ran
        none -- or, with fill=True, its predicted level there (mean over all the family's settings)."""
        fams = self.families()
        L = self.level()
        Lm = np.where(self.observed_unit, L, np.nan)
        out = np.full((len(self.policies), len(fams)), np.nan)
        for j, f in enumerate(fams):
            cols = [i for i, u in enumerate(self.units) if FAMILY[u] == f]
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[:, j] = np.nanmean(Lm[:, cols], axis=1)
            if fill:
                miss = np.isnan(out[:, j])
                out[miss, j] = L[miss][:, cols].mean(1)
        return out, fams

    def measured_families(self):
        """(P, F) policy ran the benchmark family."""
        F, _ = self.family_levels()
        return ~np.isnan(F)

    def overall(self):
        """Mean over the benchmarks the policy ran (score A)."""
        F, _ = self.family_levels()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(F, axis=1)

    def overall_full(self, links=None):
        """Mean over the four main benchmarks, a benchmark the policy did not run filled with its predicted
        level (score B). links: the (source, target) benchmark pairs along which a level may be predicted; a
        policy gets NaN if a benchmark it did not run is not linked from any benchmark it ran. None: fill all."""
        fams = self.families()
        cols = [fams.index(x) for x in MAIN_FAMILIES]
        F = self.family_levels(fill=True)[0][:, cols]
        ran = self.measured_families()[:, cols]
        ok = np.ones(len(self.policies), bool)
        if links is not None:
            for t, target in enumerate(MAIN_FAMILIES):
                linked = np.array([(s, target) in links for s in MAIN_FAMILIES])
                ok &= ran[:, t] | (ran & linked[None, :]).any(1)
        return np.where(ok, F.mean(1), np.nan)

    def elo(self, x):
        return 1500.0 + ELO_PER_LOGIT * x


def informative_units(data: RatingData, uidx, U):
    """(P, U) at least INFORMATIVE_SHARE of the policy's test cases on the setting are strictly between 0 and 1."""
    strict = data.observed & (data.k_sr > 0) & (data.k_sr < data.n)
    out = np.zeros((data.P, U), bool)
    for j in range(U):
        m = uidx == j
        n_obs, n_strict = data.observed[:, m].sum(1), strict[:, m].sum(1)
        out[:, j] = (n_obs > 0) & (n_strict >= INFORMATIVE_SHARE * n_obs)
    return out


def fit(data: RatingData, a_by=A_BY, sigma_gamma=SIGMA_GAMMA, sigma_beta=SIGMA_BETA, sigma_theta=2.0,
        sigma_loga=1.0, sigma_group=3.0, panel=PANEL, max_iter=20000, init=None) -> UnifiedFit:
    """a_by: "setting" (one conversion factor per setting), "benchmark" (one per benchmark family) or None
    (one shared a = 1). init: the "raw" parameters of an earlier fit (fit.extras["raw"]) to start from;
    policies, test cases and settings are matched by name, so a resampled data set can start from the full fit."""
    P, T = data.P, data.T
    units = list(dict.fromkeys(data.offset_unit))
    U = len(units)
    uidx = np.array([units.index(x) for x in data.offset_unit])
    grp, group_names = data.group, data.group_names
    G = grp.max() + 1
    if a_by == "setting":
        a_names = units
    elif a_by == "benchmark":
        a_names = list(dict.fromkeys(FAMILY[u] for u in units))
    elif a_by is None:
        a_names = []
    else:
        raise ValueError(a_by)
    a_group = np.array([a_names.index(u if a_by == "setting" else FAMILY[u]) for u in units]) if a_names else \
        np.zeros(U, int)
    A = len(a_names)
    n, k = data.n, data.k_sr
    observed_unit = np.zeros((P, U), bool)
    for j in range(U):
        observed_unit[:, j] = data.observed[:, uidx == j].any(1)
    ou = observed_unit.astype(float)
    a_of_test = a_group[uidx]
    cp, ct = np.nonzero(data.observed)              # the likelihood only involves observed cells
    cn, ck, cu = n[cp, ct], k[cp, ct], uidx[ct]
    cpu, ca = cp * U + cu, a_of_test[ct]

    sl, o = {}, 0
    for name, size in [("theta", P), ("mu", 1), ("gamma", P * U), ("beta", T), ("mu_g", G), ("loga", A)]:
        sl[name] = slice(o, o + size)
        o += size
    x0 = np.zeros(o)
    rate = (k.sum(1) + 0.5) / (n.sum(1) + 1)
    x0[sl["theta"]] = np.log(rate / (1 - rate)) + 2.0
    if init is not None:
        pol = {p: i for i, p in enumerate(init["policies"])}
        tst = {t: i for i, t in enumerate(init["tests"])}
        un = {x: i for i, x in enumerate(init["units"])}
        th, gm = x0[sl["theta"]], x0[sl["gamma"]].reshape(P, U)
        for i, p in enumerate(data.policies):
            j = pol.get(p)
            if j is not None:
                th[i] = init["theta"][j]
                for uu, x in enumerate(units):
                    if x in un:
                        gm[i, uu] = init["gamma"][j, un[x]]
        be = x0[sl["beta"]]
        for t, name in enumerate(data.tests):
            if name in tst:
                be[t] = init["beta"][tst[name]]
        mg = x0[sl["mu_g"]]
        for g, name in enumerate(group_names):
            mg[g] = init.get("mu_g", {}).get(name, 0.0)
        la = x0[sl["loga"]]
        for g in range(A):
            logs = [np.log(init["a"][un[u]]) for uu, u in enumerate(units) if a_group[uu] == g and u in un]
            la[g] = np.mean(logs) if logs else 0.0
        x0[sl["mu"]] = init["mu_theta"]
        x0[sl["gamma"]] = (gm * ou).ravel()

    def objective(x):
        theta, mu, beta, mu_g = x[sl["theta"]], x[sl["mu"]][0], x[sl["beta"]], x[sl["mu_g"]]
        gamma = x[sl["gamma"]].reshape(P, U) * ou
        loga = x[sl["loga"]]
        a = np.exp(loga)[ca] if A else 1.0
        z = a * (theta[cp] + gamma.ravel()[cpu] - beta[ct])
        e = np.exp(-np.abs(z))
        f = (cn * (np.maximum(z, 0) + np.log1p(e)) - ck * z).sum()      # binomial NLL, softplus(z) = log(1 + e^z)
        r = cn * np.where(z >= 0, 1 / (1 + e), e / (1 + e)) - ck        # d f / d z = n sigmoid(z) - k
        ra = r * a
        dt, db = theta - mu, beta - mu_g[grp]
        f += 0.5 * (gamma ** 2).sum() / sigma_gamma ** 2 + 0.5 * (dt ** 2).sum() / sigma_theta ** 2
        f += 0.5 * (db ** 2).sum() / sigma_beta ** 2 + 0.5 * (mu_g ** 2).sum() / sigma_group ** 2
        g = np.empty_like(x)
        g[sl["theta"]] = np.bincount(cp, weights=ra, minlength=P) + dt / sigma_theta ** 2
        g[sl["mu"]] = -dt.sum() / sigma_theta ** 2
        g[sl["gamma"]] = (np.bincount(cpu, weights=ra, minlength=P * U).reshape(P, U)
                          + gamma / sigma_gamma ** 2).ravel() * ou.ravel()
        g[sl["beta"]] = -np.bincount(ct, weights=ra, minlength=T) + db / sigma_beta ** 2
        g[sl["mu_g"]] = -np.bincount(grp, weights=db, minlength=G) / sigma_beta ** 2 + mu_g / sigma_group ** 2
        if A:
            f += 0.5 * (loga ** 2).sum() / sigma_loga ** 2
            g[sl["loga"]] = np.bincount(ca, weights=r * z, minlength=A) + loga / sigma_loga ** 2
        return f, g

    # log a is boxed so that a line search cannot overflow exp(log a); the optimum lies far inside the box
    bounds = [(None, None)] * o
    bounds[sl["loga"]] = [(-LOGA_BOUND, LOGA_BOUND)] * A
    x = x0
    for _ in range(3):              # restart from where an abnormal stop left off
        res = minimize(objective, x, jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=max_iter, maxfun=2 * max_iter, maxcor=50, ftol=1e-15, gtol=1e-5))
        if res.success and np.isfinite(res.fun):
            break
        if np.isfinite(res.fun):
            x = res.x
    if not (res.success and np.isfinite(res.fun)):
        warnings.warn(f"fit did not converge: {res.message}")
    x = res.x
    theta = x[sl["theta"]].copy()
    gamma = x[sl["gamma"]].reshape(P, U) * ou
    beta = x[sl["beta"]].copy()
    a = np.exp(x[sl["loga"]])[a_group] if A else np.ones(U)
    raw = {"policies": list(data.policies), "tests": list(data.tests), "units": units, "theta": theta.copy(),
           "gamma": gamma.copy(), "beta": beta.copy(), "a": a.copy(), "mu_theta": float(x[sl["mu"]][0]),
           "mu_g": dict(zip(group_names, x[sl["mu_g"]].tolist()))}
    f = UnifiedFit(data.policies, data.tests, units, uidx, observed_unit, informative_units(data, uidx, U),
                   theta, gamma, beta, a, a_group, a_by,
                   extras={"grad_max": float(np.abs(res.jac).max()), "objective": float(res.fun),
                           "converged": bool(res.success and np.isfinite(res.fun)), "n_eval": int(res.nfev),
                           "raw": raw})
    return normalise(f, panel)


def anchor_mask(f: UnifiedFit, j, panel=PANEL):
    """(P,) anchors of setting j (module docstring, convention 1)."""
    in_panel = np.isin(f.policies, panel)
    fam_of = np.array([FAMILY[u] for u in f.units])
    n_families = np.array([len(set(fam_of[row])) for row in f.observed_unit])
    on, inf = f.observed_unit[:, j], f.informative[:, j]
    for b in (inf & in_panel, inf & (n_families >= 2), inf):
        if b.sum() >= MIN_ANCHORS:
            return b
    return on


def normalise(f: UnifiedFit, panel=PANEL) -> UnifiedFit:
    """Apply the three conventions of the module docstring (in place) and return f. No prediction changes."""
    in_panel = np.isin(f.policies, panel)
    anchors = {}
    for j, u in enumerate(f.units):
        b = anchor_mask(f, j, panel)
        c = f.gamma[b, j].mean()
        f.gamma[:, j] -= c
        f.beta[f.unit_of_test == j] -= c
        anchors[u] = list(f.policies[b])
    k = np.exp(np.log(f.a[np.unique(f.a_group, return_index=True)[1]]).mean())
    f.a = f.a / k
    f.theta, f.gamma, f.beta = f.theta * k, f.gamma * k, f.beta * k
    shift = np.nanmean(f.overall()[in_panel]) if in_panel.any() else 0.0
    f.theta, f.beta = f.theta - shift, f.beta - shift
    f.extras["anchors"] = anchors
    return f

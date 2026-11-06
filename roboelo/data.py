"""Load RoboDojo per-test-case results into policy x test-case count matrices."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "robodojo_sim_per_task.csv"
EXTRA_CSV = Path(__file__).resolve().parents[1] / "data" / "extra_per_task.csv"  # RoboTwin 2.0 and audit rows
DIM_ORDER = ["generalization", "precision", "longHorizon", "memory", "open"]
PROTOCOL_SEEDS = 3  # RoboDojo: 3 training seeds x 50 (or 25+25) episodes per task


def _infer_seed_multiplier(g: pd.DataFrame) -> dict[int, int]:
    """Smallest m such that SR * base_n * m is an integer for every cell of that type.

    The leaderboard reports SR averaged over seeds, so SR granularity reveals the
    real episode count (e.g. 0.67% = 1/150 -> 3 seeds x 50 episodes). Types with
    too few informative cells fall back to the other type's m, else the protocol's 3.
    """
    out = {}
    for base_n, sub in g.groupby("n_episodes"):
        sr = sub.success_rate.to_numpy() / 100
        informative = ((sr > 0) & (sr < 1)).sum()
        m_found = None
        if informative >= 3:
            for m in range(1, 5):
                k = sr * base_n * m
                if np.all(np.abs(k - np.round(k)) < 0.02):
                    m_found = m
                    break
        out[base_n] = m_found
    known = [m for m in out.values() if m]
    for base_n, m in out.items():
        if m is None:
            out[base_n] = known[0] if known else PROTOCOL_SEEDS
    return out


@dataclass
class RatingData:
    policies: np.ndarray          # (P,)
    tests: np.ndarray             # (T,)
    dimension: np.ndarray         # (T,) capability dimension of each test case
    variant: np.ndarray           # (T,) "standard" | "random"
    n: np.ndarray                 # (P,T) episodes, 0 where unobserved
    k_sr: np.ndarray              # (P,T) successes
    k_score: np.ndarray           # (P,T) score/100 * n  (fractional "pseudo-successes")
    benchmark: np.ndarray | None = None   # (T,) source benchmark of each test case (None = single benchmark)
    offset_unit: np.ndarray | None = None  # (T,) unit that gets a per-policy competency offset (default: benchmark)

    @property
    def P(self):
        return len(self.policies)

    @property
    def T(self):
        return len(self.tests)

    @property
    def observed(self):
        return self.n > 0

    def k(self, target: str) -> np.ndarray:
        return {"sr": self.k_sr, "score": self.k_score}[target]

    def rate(self, target: str) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.observed, self.k(target) / self.n, np.nan)

    @property
    def group(self) -> np.ndarray:
        """Test-case group index for hierarchical difficulty priors: (dimension, variant)."""
        labels = [f"{d}/{v}" for d, v in zip(self.dimension, self.variant)]
        order = lambda s: (DIM_ORDER.index(s.split("/")[0]) if s.split("/")[0] in DIM_ORDER else len(DIM_ORDER), s)
        self._group_names = sorted(set(labels), key=order)
        return np.array([self._group_names.index(l) for l in labels])

    @property
    def group_names(self):
        _ = self.group
        return self._group_names

    def take_tests(self, idx) -> "RatingData":
        """Copy restricted to (or resampled over) the test cases at integer positions idx."""
        idx = np.asarray(idx)
        return replace(self, tests=self.tests[idx], dimension=self.dimension[idx], variant=self.variant[idx],
                       n=self.n[:, idx], k_sr=self.k_sr[:, idx], k_score=self.k_score[:, idx],
                       benchmark=None if self.benchmark is None else self.benchmark[idx],
                       offset_unit=None if self.offset_unit is None else self.offset_unit[idx])

    def subset(self, policies=None, tests=None) -> "RatingData":
        pi = np.arange(self.P) if policies is None else np.array([list(self.policies).index(p) for p in policies])
        ti = np.arange(self.T) if tests is None else np.array([list(self.tests).index(t) for t in tests])
        sub = self.take_tests(ti)
        return replace(sub, policies=self.policies[pi], n=sub.n[pi], k_sr=sub.k_sr[pi], k_score=sub.k_score[pi])

    def mask_cells(self, cells: np.ndarray) -> "RatingData":
        """Copy with the given boolean (P,T) cells hidden (n=0) -- used for cross-validation."""
        n = self.n.copy()
        n[cells] = 0
        return replace(self, n=n, k_sr=np.where(cells, 0, self.k_sr), k_score=np.where(cells, 0, self.k_score))


def load(csv: Path = DATA_CSV) -> RatingData:
    df = pd.read_csv(csv)
    mult = {p: _infer_seed_multiplier(g) for p, g in df.groupby("policy")}
    df["n_eff"] = [r.n_episodes * mult[r.policy][r.n_episodes] for r in df.itertuples()]
    df["k_sr"] = np.round(df.success_rate / 100 * df.n_eff)
    df["k_score"] = df.score / 100 * df.n_eff

    tests_meta = (df[["test_case", "dimension", "variant", "task"]].drop_duplicates("test_case")
                  .assign(o=lambda d: d.dimension.map(DIM_ORDER.index))
                  .sort_values(["o", "task", "variant"]))
    policies = np.array(sorted(df.policy.unique(), key=str.lower))
    tests = tests_meta.test_case.to_numpy()
    pidx = {p: i for i, p in enumerate(policies)}
    tidx = {t: i for i, t in enumerate(tests)}
    P, T = len(policies), len(tests)
    n, ks, kc = np.zeros((P, T)), np.zeros((P, T)), np.zeros((P, T))
    for r in df.itertuples():
        i, j = pidx[r.policy], tidx[r.test_case]
        n[i, j], ks[i, j], kc[i, j] = r.n_eff, r.k_sr, r.k_score
    return RatingData(policies, tests, tests_meta.dimension.to_numpy(), tests_meta.variant.to_numpy(), n, ks, kc)

"""Fetch high-credibility per-task results from other simulation benchmarks and tidy them.

Sources (see data/BENCHMARK_SOURCES.md):
  robotwin2        RoboTwin 2.0 official leaderboard JSON: 23 policies x 50 tasks x {easy, hard}, 100 trials each
  libero           Manipulation Benchmark Audit (ripl), official LIBERO protocol: 3 policies x 40 tasks x 50 episodes
  simplerenv_audit same audit, SimplerEnv WidowX block-stacking stress test: 5 policies x 7 shifted conditions x 288

Policy names are mapped onto the RoboDojo ids where the same model family appears there.
Output: data/extra_per_task.csv  (benchmark, policy, policy_raw, setting, test_case, task, variant,
group, n_episodes, successes, success_rate, source)
"""
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repository root
RAW = ROOT / "data" / "raw" / "candidates"
OUT = ROOT / "data" / "extra_per_task.csv"

ROBOTWIN_URL = "https://robotwin-platform.github.io/data/robotwin_leaderboard.json"
AUDIT = "https://raw.githubusercontent.com/ripl/ManipulationBenchmarkAudit/main/creeping_overfitting/results"
LIBERO_URL = f"{AUDIT}/libero/official_calibration/selected_episode_outcomes.csv"
SIMPLER_URL = f"{AUDIT}/simplerenv/distribution_overfitting/per_episode_results_all.csv"

# other benchmark's model name -> RoboDojo policy id (same model family). Unlisted names are new policies.
TO_ROBODOJO = {
    # RoboTwin 2.0 leaderboard entry ids
    "Pi_05": "Pi_05", "X_WAM": "X_WAM", "X_VLA": "X_VLA", "Xiaomi_Robotics_0": "Xiaomi_Robotics_0",
    "EventVLA": "EventVLA", "AHA_WAM": "AHA_WAM", "FastWAM": "FastWAM", "Spatial_Forcing": "Pi_05_SF",
    "starVLA": "starVLA", "Abot_M0": "Abot_M0", "RDT": "RDT_1B",
    # XPolicyLab, which produces both leaderboards, runs GalaxeaVLA as the G0 line (checkpoint G0Plus_3B-base):
    # RoboDojo's "GalaxeaVLA (G0)". G0.5 is a separate policy there (G05).
    "GalaxeaVLA": "GalaxeaVLA",
    "π0": "Pi_0", "ACT": "ACT",
    # audit policy ids
    "pi05_lerobot": "Pi_05", "spatial_forcing": "Pi_05_SF", "xvla": "X_VLA",
    "simvla": "SimVLA", "cogact": "CogACT", "spatialvla": "SpatialVLA",
    "internvla_m1": "InternVLA_M1", "dexbotic": "DB_MemVLA",
}


def get(url: str, name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 RoboElo-research"})
    with urllib.request.urlopen(req, timeout=180) as r:
        path.write_bytes(r.read())
    return path


def snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def robotwin_rows():
    d = json.loads(get(ROBOTWIN_URL, "robotwin2_leaderboard.json").read_text(encoding="utf-8"))
    n = 100  # board.eval_protocol: "100 trials/task under demo_clean and demo_randomized"
    for e in d["entries"]:
        raw = e["id"].split(":", 1)[1]
        for variant in ("easy", "hard"):
            for task, sr in zip(d["tasks"], e[variant]):
                k = round(sr / 100 * n)
                yield dict(benchmark="robotwin2", policy=TO_ROBODOJO.get(raw, raw), policy_raw=e["name"],
                           setting=e["setting"], test_case=f"robotwin2/{snake(task)}/{variant}", task=snake(task),
                           variant=variant, group=f"robotwin2/{variant}", n_episodes=n, successes=k,
                           success_rate=k / n, source=ROBOTWIN_URL)


def libero_rows():
    d = pd.read_csv(get(LIBERO_URL, "audit_libero_official_episodes.csv"))
    g = d.groupby(["policy", "suite", "task_name"]).success.agg(["sum", "size"]).reset_index()
    for r in g.itertuples():
        yield dict(benchmark="libero", policy=TO_ROBODOJO[r.policy], policy_raw=r.policy, setting="official",
                   test_case=f"libero/{r.suite}/{r.task_name}", task=r.task_name, variant=r.suite,
                   group=f"libero/{r.suite}", n_episodes=int(r.size), successes=int(r.sum),
                   success_rate=r.sum / r.size, source=LIBERO_URL)


def simplerenv_rows():
    d = pd.read_csv(get(SIMPLER_URL, "audit_simplerenv_episodes.csv"))
    g = d.groupby(["policy", "condition"]).success.agg(["sum", "size"]).reset_index()
    for r in g.itertuples():
        yield dict(benchmark="simplerenv_audit", policy=TO_ROBODOJO[r.policy], policy_raw=r.policy,
                   setting="widowx_stack_block_shifted", test_case=f"simplerenv_audit/{r.condition}",
                   task="stack_block", variant=r.condition, group="simplerenv_audit/shifted",
                   n_episodes=int(r.size), successes=int(r.sum), success_rate=r.sum / r.size, source=SIMPLER_URL)


def main():
    df = pd.DataFrame([*robotwin_rows(), *libero_rows(), *simplerenv_rows()])
    df.to_csv(OUT, index=False)
    s = df.groupby("benchmark").agg(policies=("policy", "nunique"), test_cases=("test_case", "nunique"),
                                    cells=("test_case", "size"), episodes=("n_episodes", "sum"),
                                    mean_sr=("success_rate", "mean"))
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()

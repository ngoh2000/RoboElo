"""Our own rollouts (data/own_runs/) as per-test-case rows in the format of data/extra_per_task.csv.

LIBERO: one test case per task (40), named like the audit rows ("libero/<suite>/<bddl file stem>").
Runs that were partly redone under MuJoCo 3.3.2 (folders <model>_mj332_libero_50) take the re-run rows
for the tasks that were re-run; the original rows of those tasks are dropped.
SimplerEnv (the standard tasks of SimplerEnv's official scripts, not the audit's shifted variants):
Google Robot visual matching and variant aggregation, 7 test cases each (coke can by orientation, move
near, open drawer, close drawer, put in drawer), and the 4 WidowX tasks.
"""
import json
from pathlib import Path

import pandas as pd

OWN = Path(__file__).resolve().parents[1] / "data" / "own_runs"
PARTIAL = OWN.parent / "own_runs_partial"   # copies of runs that are still going (marked "own run (partial)")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# folder prefix -> policy name used in the rating data (RoboDojo / RoboTwin / release-date names)
LIBERO_RUNS = {
    "pi05": "Pi_05", "pi0_replan5": "Pi_0", "smolvla": "SmolVLA", "pi0fast": "Pi0_FAST", "molmoact2": "MolmoACT2",
    "lingbot_va": "LingBot_VA", "g05": "G05", "groot_n17": "GR00T_N17", "openvla": "OpenVLA",
    "openvla_oft": "OpenVLA_OFT", "univla": "UniVLA", "vla_adapter": "VLA_Adapter", "nora": "NORA",
    "vla_jepa": "VLA_JEPA", "evo1": "Evo1", "fastwam": "FastWAM", "xvla": "X_VLA", "molmoact": "MolmoAct",
    "internvla_m1": "InternVLA_M1", "dm05": "OpenDM05", "abot_m0": "Abot_M0", "xiaomi_r0": "Xiaomi_Robotics_0",
}
# Not listed: the three starVLA runs (starvla_*). RoboDojo's "starVLA" entry is StarVLA-alpha, and it is not
# known which variant RoboTwin's "starVLA" entry is, so the runs are not linked to either.
# alternative settings of the same checkpoint, kept out of the main data
LIBERO_VARIANTS = {"pi0_mj332": "Pi_0 (chunk 50)"}
# suites whose results do not reproduce the paper's numbers; left out
EXCLUDE = {("NORA", "libero_10")}
SIMPLER_RUNS = {"rt1x": "RT-1-X", "octo_base": "Octo", "octo_small": "Octo_Small", "openvla": "OpenVLA",
                "univla": "UniVLA", "spatialvla": "SpatialVLA", "cogact": "CogACT", "groot_n16": "GR00T_N16",
                "groot_n17": "GR00T_N17", "xvla": "X_VLA", "vla_jepa": "VLA_JEPA", "rt1": "RT_1",
                "molmoact": "MolmoAct"}


def _root(folder):
    """Folder of a finished run, else of a partial copy, else None."""
    for root in (OWN, PARTIAL):
        if (root / folder).exists():
            return root
    return None


def _read(f):
    return pd.read_csv(f, on_bad_lines="skip")  # a partial copy may end in a half-written line


def _libero_episodes(prefix):
    """Episodes of one LIBERO run, with the MuJoCo 3.3.2 re-runs merged in."""
    root = _root(f"{prefix}_libero_50")
    if root is None:
        return None
    base, redo = root / f"{prefix}_libero_50", OWN / f"{prefix}_mj332_libero_50"
    frames = []
    for suite in SUITES:
        f = base / suite / "episodes.csv"
        if not f.exists():
            continue
        e = _read(f)
        r = redo / suite / "episodes.csv"
        if r.exists() and prefix not in ("pi0_mj332",):
            e2 = _read(r)
            e = pd.concat([e[~e.task_id.isin(set(e2.task_id))], e2])
        frames.append(e)
    if not frames:
        return None
    e = pd.concat(frames, ignore_index=True)
    e = e.drop_duplicates(["suite", "task_id", "init_state_id"], keep="last")
    return e


def libero_rows(prefix, policy):
    e = _libero_episodes(prefix)
    if e is None:
        return pd.DataFrame()
    e = e[[(policy, s) not in EXCLUDE for s in e.suite]]
    e = e[e.success.isin([0, 1])]
    e = e.assign(test_case="libero/" + e.suite + "/" + e.bddl_file.str.replace(r"\.bddl$", "", regex=True))
    g = e.groupby(["suite", "test_case", "task_name"]).agg(n_episodes=("success", "size"), successes=("success", "sum"))
    g = g.reset_index()
    return pd.DataFrame({
        "benchmark": "libero", "policy": policy, "policy_raw": prefix,
        "setting": "own run" if _root(f"{prefix}_libero_50") == OWN else "own run (partial)",
        "test_case": g.test_case, "task": g.task_name, "variant": g.suite, "group": "libero/" + g.suite,
        "n_episodes": g.n_episodes, "successes": g.successes, "success_rate": g.successes / g.n_episodes,
        "source": f"data/own_runs/{prefix}_libero_50"})


def _simpler_case(r):
    if r.robot_family == "widowx":
        return "widowx", r.env_name.replace("InScene-v0", "").replace("Scene-v0", "")
    prot = "vm" if r.protocol == "visual_matching" else "va"
    if r.task == "pick_coke_can":
        kw = json.loads(r.additional_env_build_kwargs) if isinstance(r.additional_env_build_kwargs, str) else {}
        orient = next(k for k in ("lr_switch", "upright", "laid_vertically") if kw.get(k))
        name = {"lr_switch": "coke_horizontal", "upright": "coke_upright", "laid_vertically": "coke_vertical"}[orient]
    elif r.task == "drawer":
        name = "open_drawer" if r.env_name.startswith("Open") else "close_drawer"
    else:
        name = r.task
    return f"google_{prot}", name


def simpler_rows(prefix, policy):
    root = _root(f"{prefix}_simplerenv")
    if root is None:
        return pd.DataFrame()
    e = _read(root / f"{prefix}_simplerenv" / "episodes.csv")
    e = e[e.success.isin([0, 1])]
    keys = [_simpler_case(r) for r in e.itertuples()]
    e = e.assign(group="simplerenv/" + pd.Series([k[0] for k in keys], index=e.index),
                 task_key=[k[1] for k in keys])
    g = e.groupby(["group", "task_key"]).agg(n_episodes=("success", "size"), successes=("success", "sum")).reset_index()
    return pd.DataFrame({
        "benchmark": "simplerenv", "policy": policy, "policy_raw": prefix,
        "setting": "own run" if root == OWN else "own run (partial)",
        "test_case": g.group + "/" + g.task_key, "task": g.task_key, "variant": g.group.str.split("/").str[1],
        "group": g.group, "n_episodes": g.n_episodes, "successes": g.successes,
        "success_rate": g.successes / g.n_episodes, "source": f"data/own_runs/{prefix}_simplerenv"})


def own_rows(include_variants=False):
    parts = [libero_rows(p, n) for p, n in LIBERO_RUNS.items()]
    if include_variants:
        parts += [libero_rows(p, n) for p, n in LIBERO_VARIANTS.items()]
    parts += [simpler_rows(p, n) for p, n in SIMPLER_RUNS.items()]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def combined_extra(extra_csv, include_variants=False):
    """data/extra_per_task.csv plus our own runs. Where we ran a model on LIBERO ourselves, the audit's
    rows for that model (Pi_05) are replaced by ours."""
    ex = pd.read_csv(extra_csv)
    own = own_rows(include_variants)
    own_libero = set(own[own.benchmark == "libero"].policy)
    ex = ex[~((ex.benchmark == "libero") & ex.policy.isin(own_libero))]
    return pd.concat([ex, own], ignore_index=True)

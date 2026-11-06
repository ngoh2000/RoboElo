# Data

Everything the rating is computed from. Each policy × test case cell becomes a number of successes out of a number of trials.

| File | Contents | Source | Snapshot |
|---|---|---|---|
| `robodojo_sim_per_task.csv` | RoboDojo simulation leaderboard: success rate, partial-progress score and episodes per seed, for every policy × test case | [RoboDojo](https://robodojo-benchmark.com), via `scripts/fetch_robodojo_sim.py` | 2026-10-24 |
| `extra_per_task.csv` | RoboTwin 2.0 leaderboard (clean = `easy`, randomized = `hard`; 100 episodes per cell); per-episode LIBERO records of π0.5, π0.5-SF and SimVLA; SimplerEnv WidowX block stacking under 7 shifted conditions | [RoboTwin 2.0 leaderboard](https://robotwin-platform.github.io/leaderboard); [Manipulation Benchmark Audit](https://github.com/ripl/ManipulationBenchmarkAudit) (arXiv 2606.04233); via `scripts/fetch_extra_benchmarks.py` | 2026-10-24 |
| `own_runs/` | our own per-episode LIBERO and SimplerEnv results | `evaluation/` | — |
| `policy_release_dates.csv` | first public paper and weights date per policy, with source links | collected by hand | — |
| `raw/` | the downloaded files that the two fetch scripts turn into the CSVs above | — | 2026-10-24 |

## Policy names

A policy that appears on several benchmarks gets one name everywhere: RoboDojo's id, mapped in `TO_ROBODOJO` of `scripts/fetch_extra_benchmarks.py` and in `roboelo/own_runs.py`. RoboTwin's GalaxeaVLA entry does not name the Galaxea model. Both leaderboards are produced through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), whose `policy/GalaxeaVLA` is the G0 line (checkpoint G0Plus_3B-base) and whose G0.5 is a separate policy (`policy/G05`); RoboDojo lists the two as "GalaxeaVLA (G0)" and "GalaxeaVLA (G0.5)". The RoboTwin entry is therefore linked to RoboDojo's GalaxeaVLA (G0).

## RoboDojo: trials per cell

The leaderboard reports success rates averaged over training seeds: usually 3 seeds × 50 episodes, or 3 × 25. The episode count it shows is therefore per seed.

`roboelo/data.py` infers each policy's seed count from the granularity of its success rates. For example, 0.67 % = 1/150 means 3 × 50 episodes. Most policies come out at 3 seeds; a few at 1, 2 or 4.

## Our own runs (`own_runs/`)

One folder per policy and benchmark:

| Folder | Contents |
|---|---|
| `<policy>_libero_50/<suite>/episodes.csv` | one row per LIBERO rollout |
| `<policy>_mj332_libero_50/` | LIBERO re-runs under MuJoCo 3.3.2 (see below) |
| `<policy>_simplerenv/episodes.csv` | one row per SimplerEnv episode |
| `<policy>_libero_plus/<suite>/episodes.csv` | one row per LIBERO-Plus task (not used in the fit yet) |

Next to them:
- `run_config_*.json` and `policy_config.json` record the exact settings of each run. A policy with one checkpoint per suite (InternVLA-M1) has its `policy_config` in each suite folder.
- `errors.csv` lists episodes that raised an error and were run again; those are not counted as failures.

Paths inside the files (videos, checkpoints) refer to the evaluation server's workspace. Videos are not included.

**LIBERO protocol**
- 4 suites × 10 tasks × 50 fixed initial states (0–49), with each model's own step limits and seeds.
- MuJoCo 3.4 and later change the initial scene of some tasks, so those tasks were re-run under MuJoCo 3.3.2. `roboelo/own_runs.py` takes the re-run rows task by task.
- π0 uses openpi's LIBERO setting, which re-plans every 5 steps (`pi0_replan5`). The checkpoint's own chunk-50 setting (`pi0_mj332`) is kept as an alternative but is not in the main fit.
- NORA's libero_10 is left out, because it does not reproduce the paper's numbers. LingBot-VA has the Long suite only.

**SimplerEnv protocol**
- Every episode of SimplerEnv's official evaluation scripts: 2,856 Google Robot episodes (visual matching and variant aggregation) and 96 WidowX episodes per policy.
- Test cases (18 in total):
  - Google Robot, per protocol: pick coke can (split by orientation into 3 test cases), move near, open drawer, close drawer, put in drawer;
  - WidowX: the 4 tasks.
- UniVLA has WidowX results only.
- `LIBERO_RUNS` and `SIMPLER_RUNS` in `roboelo/own_runs.py` list the runs that enter the fit.

**LIBERO-Plus protocol**
- The 10,030 perturbed tasks of LIBERO-Plus (commit 4976dc3), one rollout each from initial state 0, with the same checkpoint, settings and step limits as the policy's LIBERO run.
- Every variant that does not perturb the language gets the instruction of its LIBERO base task (`libero_plus_prompt = base` in `run_config_*.json`). See footnote ⁵ of the main README.

# RoboElo

RoboElo rates robot policies (vision-language-action models and others) and the simulation tasks they attempt on **one shared scale in Elo points**, across several benchmarks at once. A policy's rating and a task's difficulty together give the policy's chance of succeeding at that task.

## Method

```
log( P / (1 − P) ) = a_s × ( μ_m + γ_m,s − β_t ) / 173.72
```

| Symbol | Meaning |
|---|---|
| P | chance that policy m succeeds on test case t |
| μ_m | general ability of the policy |
| γ_m,s | offset of the policy on setting s (e.g. RoboTwin clean vs randomized) |
| β_t | difficulty of the test case: the rating that gives a 50 % chance of success |
| a_s | conversion factor of the setting: how strongly a rating gap turns into success |
| 173.72 | 400 / ln 10, so that a 400-point advantage means 10 : 1 odds |

How the scores are computed:

- **One joint fit.** All results of the milestone policies are fitted at once: binomial likelihood plus priors, MAP estimate with L-BFGS-B. The result does not depend on the order of results.
- **Priors and conversion factors, chosen by a hidden-benchmark (linking) test:**
  - offsets: σ_γ = 2 logits;
  - task difficulty: hierarchical prior with σ_β = 8 logits;
  - one conversion factor per benchmark.
- **Scale conventions, set after the fit.** These label the scale and change no prediction:
  - on each setting, the mean offset of the informative reference policies is 0 (other anchors where fewer than three ran it);
  - the geometric mean of the conversion factors is 1;
  - the reference panel's mean overall score is 1500.
- **Reference panel:** π0, π0.5, GR00T N1.7, SmolVLA, OpenVLA-OFT, X-VLA, ACT.
- **Two overall scores:**
  - **overall score:** the mean of a policy's levels (μ + γ) over the benchmarks it ran;
  - **full-coverage score:** the mean over all four benchmarks, a benchmark the policy did not run filled with its predicted level, where the linking test shows that one of the policy's benchmarks predicts it.

**[docs/ELO_METHOD.md](docs/ELO_METHOD.md)** explains every step, the reasons for each choice and the evidence behind them ([Chinese version](docs/ELO_METHOD_CN.md)).

## Benchmarks and data

The fit uses only the milestone policies (next section) that have results: 32 of the 34. Our runs of ECoT and GR00T N1.5 have finished, but they are not in the fit yet.

| Benchmark | Source | Milestone policies in the fit | All policies in data/ | Test cases | Trials in the fit |
|---|---|---|---|---|---|
| RoboDojo (simulation) | official leaderboard | 18 | 61 | 54 (42 tasks, 12 also randomized) | 111,000 |
| RoboTwin 2.0 | official leaderboard | 10 | 30 | 100 (50 tasks × clean / randomized) | 100,000 |
| LIBERO | our runs (22 policies) + Manipulation Benchmark Audit (2) | 19 | 24 | 40 (4 suites × 10 tasks) | 36,500 |
| SimplerEnv | our runs, official episode lists | 12 | 13 | 18 (Google Robot 7 × 2 protocols, WidowX 4) | 32,376 |
| **Total** | | **32** | **95** | **212** | **279,876** |

Snapshots of the public leaderboards: 2026-10-24. Sources and file formats are described in [data/README.md](data/README.md). `ROBOELO_POLICIES=all` fits every policy in data/ instead.

## Milestone policies

Last updated: 2026-11-04.

The table lists the 34 milestone policies used to follow progress over time: influential policies with public results, or with checkpoints we can run ourselves. The list is in [data/milestone_policies.csv](data/milestone_policies.csv). The rating uses only these policies, on RoboDojo, RoboTwin 2.0, LIBERO and SimplerEnv (see [ratings/](ratings/README.md)); the other leaderboard entries in data/ are left out of the fit.

The benchmarks form two groups: group 1 is RoboDojo and RoboTwin 2.0 (bimanual tasks), group 2 is LIBERO and SimplerEnv (single-arm tasks). Across the groups, LIBERO and RoboDojo or RoboTwin predict each other only slightly better than the task average, and SimplerEnv shares fewer than five policies with group 1 ([method, section 8.2](docs/ELO_METHOD.md#82-linking-from-one-source-benchmark-sources)). A policy can be compared reliably across both groups only if it was measured in both.

LIBERO-Plus is being added, and RoboCasa365 is under consideration. Neither is in the fit yet. Once their results are in, the linking test decides which group each belongs to.

| Cell | Meaning |
|---|---|
| **Public** | public per-task results |
| **Means** | public aggregate scores only, no per-task results (for example LIBERO suite means, the 7 perturbation-dimension means of LIBERO-Plus, the 3 split means of RoboCasa365) |
| **Runnable** | the policy's own checkpoint and evaluation code are public; not run by us yet |
| a percentage | our run of LIBERO: overall success over the 4 suites; under LIBERO-Plus, over its 10,030 tasks |
| three numbers | our run of SimplerEnv: Google Robot visual matching / variant aggregation / WidowX, each the mean over its tasks |
| **In progress** | running or queued on our evaluation server |
| **—** | no usable checkpoint, or not applicable |

| Release | Policy | LIBERO | LIBERO-Plus ⁵ | SimplerEnv | RoboTwin 2.0 | RoboDojo | RoboCasa365 (under consideration) ⁶ |
|---|---|---|---|---|---|---|---|
| 2022-12 | RT-1 | — | — | 52.9 / 38.8 / — | — | — | — |
| 2023-03 | Diffusion Policy | — | — | — | Public | — | Means · Runnable |
| 2023-04 | ACT | — | — | — | Public | Public | — |
| 2023-10 | RT-1-X | — | — | 40.3 / 28.1 / 3.1 | — | — | — |
| 2024-05 | Octo | — | — | 11.3 / 1.0 / 13.5 | — | — | — |
| 2024-06 | OpenVLA | 75.3 % | Means · In progress | 25.1 / 26.9 / 1.0 | — | — | — |
| 2024-07 | ECoT | — | — | — / — / 3.1 | — | — | — |
| 2024-10 | RDT-1B | — | — | — | Public | Public | — |
| 2024-10 | π0 | 71.2 % ¹ | Means · In progress | — | Public | Public | Means · Runnable |
| 2024-11 | CogACT | — | — | 76.1 / 61.5 / 53.1 ² | — | — | — |
| 2025-01 | π0-FAST | 89.8 % | Means · In progress | — | — | — | — |
| 2025-01 | SpatialVLA | — | — | 53.4 / 49.8 / 30.2 ² | — | — | — |
| 2025-02 | OpenVLA-OFT | 96.7 % | Means · In progress | — | — | Public | — |
| 2025-04 | π0.5 | 96.7 % ² | Means · In progress | — | Public | Public | Means · Runnable |
| 2025-05 | UniVLA | 95.6 % | Means · In progress | — / — / 35.4 | — | — | — |
| 2025-05 | GR00T N1.5 | 87.4 % | In progress | 32.6 / 27.3 / 54.2 ⁷ | — | — | Means · Runnable |
| 2025-06 | SmolVLA | 64.6 % | In progress | — | In progress | Public | — |
| 2025-08 | MolmoAct | 85.3 % | Means · Runnable | 61.1 / 56.9 / — | — | — | — |
| 2025-10 | X-VLA | 98.7 % | Means · In progress | 68.2 / 61.9 / 45.8 ² | Public | Public | — |
| 2025-10 | InternVLA-M1 | 96.6 % | In progress | 78.8 / 74.9 / 62.5 ² | — | — | — |
| 2025-12 | GR00T N1.6 | — | — | 42.9 / 39.3 / 39.6 | — | — | Means · Runnable |
| 2026-01 | LingBot-VA | 96.2 % (Long suite) ³ | Runnable (Long suite) ³ | — | In progress | — | — |
| 2026-02 | Xiaomi-Robotics-0 | 98.4 % | 78.9 % | 83.2 / 72.7 / 63.5 | Public | Public | — |
| 2026-02 | ABot-M0 | 98.2 % | Means · In progress | — | Public | Public | — |
| 2026-02 | VLA-JEPA | 98.2 % | Means · In progress | 76.6 / 57.0 / 46.9 | — | — | — |
| 2026-03 | FastWAM | 96.1 % | In progress | — | Public | Public | — |
| 2026-03 | GR00T N1.7 | 95.8 % | In progress | 58.4 / 53.1 / 28.1 | — | Public | — |
| 2026-04 | starVLA (3 variants) | 97.4 / 97.5 / 96.6 % ⁴ | In progress | — | Public | Public | — |
| 2026-05 | MolmoAct2 | 97.8 % | In progress | — | — | Public | — |
| 2026-06 | G0.5 | 98.6 % | In progress | — | In progress | Public | — |
| 2026-06 | hy_vla | — | — | — | In progress | Public | — |
| 2026-07 | DM0.5 | 98.2 % ⁴ | In progress | — | — | Public | — |
| 2026-07 | InternVLA-A1.5 | 98.1 % | Means · In progress | — | — | Public | — |
| 2026-07 | Xiaomi-Robotics-1 | — | — | — | — | Public | Means · Runnable |

Measured in both groups: 13 policies now; 14 once LingBot-VA (RoboTwin) has run.

These runs are finished but not yet in the fit:
- SimplerEnv: ECoT, InternVLA-M1, Xiaomi-Robotics-0 and GR00T N1.5.
- LIBERO: InternVLA-A1.5 and GR00T N1.5.
- LIBERO-Plus: Xiaomi-Robotics-0.

starVLA's runs are not used (footnote ⁴).

Progress of the runs marked In progress, on 2026-11-04:
- RoboTwin 2.0, episodes done of 4,900: SmolVLA 4,219, G0.5 2,013, LingBot-VA 191, hy_vla 141.
- LIBERO-Plus, tasks done of 10,030: InternVLA-M1 10,019, X-VLA 3,716. The other LIBERO-Plus runs start after the RoboTwin runs.

¹ openpi's LIBERO setting (re-plan every 5 steps); 65.9 % with the checkpoint's own chunk-50 setting.
² The Manipulation Benchmark Audit also published per-episode results for this policy. For SimplerEnv these are shifted conditions, used only in a sensitivity check.
³ The released LIBERO checkpoint covers the Long suite only.
⁴ starVLA: Qwen3-VL-PI / Qwen3-VL-OFT / Qwen2.5-VL-GR00T. These runs are not in the fit. RoboDojo's "starVLA" entry is StarVLA-alpha, a different model (RoboDojo lists starVLA-PI_v3, starVLA-OFT and starVLA-GR00T as separate entries), and RoboTwin's "starVLA" entry does not name its variant. DM0.5: our run (lerobot/dm05_libero) is linked to RoboDojo's DM0.5 entry (OpenDM05), not to Dexbotic_DM0.
⁵ [LIBERO-Plus](https://arxiv.org/abs/2510.13626) re-runs LIBERO checkpoints zero-shot on 10,030 perturbed tasks, one episode each. **Runnable** means the LIBERO checkpoint from the LIBERO column can be re-run. Public means come from the LIBERO-Plus paper or from later papers, and these sometimes disagree (π0.5: 65.0–86.5 %). Results from third-party fine-tunes are not counted. Our runs use the same checkpoint, settings and step limits as our LIBERO run, from initial state 0 of each task. Every variant that does not perturb the language gets the instruction of its LIBERO base task: the released benchmark (commit 4976dc3) appends the variant id to the instruction, for example "… on the plate view 0 0 100 2 352 initstate 0" ([issues #64, #65](https://github.com/sylvestf/LIBERO-plus/issues/65)).
⁶ [RoboCasa365](https://robocasa.ai/leaderboard.html): 50 kitchen tasks × 50 episodes. Every model is trained on the benchmark's own Human300 data set. The leaderboard has 21 entries; 16 have a public checkpoint and code. Of the policies in this table, six are among them: Diffusion Policy, π0, π0.5, GR00T N1.5, GR00T N1.6 and Xiaomi-Robotics-1.
⁷ NVIDIA's Isaac-GR00T repository (n1.5-release, examples/SimplerEnv) publishes per-task results with success counts for its released GR00T N1.5 checkpoints (bridge and fractal post-train). They were obtained with NVIDIA's fork of SimplerEnv (300 trials per task under visual matching), not the official episode lists, and do not split the Google Robot tasks into the test cases used here. Other papers cite them ([MolmoAct](https://arxiv.org/abs/2508.07917)) or re-run the model with other results (WidowX, mean over the 4 standard tasks: 67.8 % in NVIDIA's README, 61.9 % in [StarVLA](https://arxiv.org/abs/2604.05014) with the official protocol). We therefore ran the released checkpoints with the official episode lists. Our results are lower than NVIDIA's:
- WidowX: 54.2 % over the 4 standard tasks, with 24 episodes per task.
- Google Robot, visual matching: pick coke can 52 % vs 69 %, open drawer 10 % vs 27 %, close drawer 34 % vs 45 %.

**Rules for the table:**
- **Public per-task results are used as published.** Policies with a public checkpoint but no public per-task results are run by us. RoboDojo uses official results only.
- **Only checkpoints released by the original authors** (or official LeRobot ports) are run.
- **SimplerEnv uses one protocol for every policy:** the official episode lists and step limits. Numbers can therefore differ from those reported with a policy's own protocol.

## Status

All parts of the method are implemented. What the checks show (details in [section 8 of the method](docs/ELO_METHOD.md#8-uncertainty-and-checks)):

- **Linking test.** With one policy's whole benchmark hidden, its results there are predicted better than by the task average of the other policies on all four benchmarks. On RoboDojo and SimplerEnv this holds for the pooled loss, but only for about half of the held-out policies one by one.
- **Which benchmarks predict which.** Every pair of benchmarks with at least five shared policies predicts better than the task average, including LIBERO with RoboDojo and with RoboTwin, but the gains are small. SimplerEnv shares fewer than five policies with RoboDojo and RoboTwin. Policies run on both groups would strengthen the links most.
- **The largest misses** are policies much stronger or weaker on one benchmark than on their others, such as OpenVLA-OFT's LIBERO checkpoint and π0 on RoboTwin.

**The ratings are in [ratings/](ratings/README.md)**, written by `python scripts/write_ratings.py` from the outputs of `run_unified_scale.py`.

## Repository layout

```
roboelo/        the rating package
  data.py         RoboDojo loader (seed-count inference) and the RatingData container
  own_runs.py     our LIBERO and SimplerEnv episodes -> per-test-case counts
  unified.py      the model, priors, scale conventions and overall scores
  names.py        display names of policies
scripts/
  run_unified_scale.py     linking | fit | sources | bootstrap | cv | recovery | report  -> results/unified/
  write_ratings.py         the published ratings -> ratings/
  make_unified_figures.py  Wright map, conversion factors, progress over time -> figures/
  fetch_robodojo_sim.py    download the RoboDojo leaderboard -> data/
  fetch_extra_benchmarks.py  download RoboTwin 2.0 and the audit data -> data/
  rating_stability.py      ratings refitted on growing shares of the trials -> results/stability/, figures/
  agielo_figures.py        calibration, consistency and competency-gap checks (AGI-Elo style) -> figures/
  plot_overall_ratings.py  task-difficulty curves with policy overall scores -> figures/
  plot_frontier_forecast.py  frontier trend and three-year forecast, overall and per benchmark -> figures/
  plot_mastery_gap.py      share of each setting's tasks by the best policy's predicted success -> figures/
  mastery_table.py         gap to 90 % average success and years on trend -> results/unified/
  check_own_runs.py        completeness check of data/own_runs/
ratings/        the ratings: table, conversion factors, task difficulties
tests/          checks of the model code
data/           inputs (see data/README.md)
evaluation/     the scripts behind our own runs (see evaluation/README.md)
docs/
  ELO_METHOD.md   how the Elo score is computed, and why
```

## Usage

Python 3.10 or later. The rating code needs only a CPU.

```bash
pip install -e .
python scripts/run_unified_scale.py linking    # linking test over the grid of priors and conversion-factor granularities
python scripts/run_unified_scale.py fit        # main fit -> results/unified/
python scripts/run_unified_scale.py sources    # linking test from one source benchmark at a time
python scripts/run_unified_scale.py bootstrap  # 1,000 replicates: test cases resampled within every setting
python scripts/run_unified_scale.py cv         # random-cell cross-validation of the model variants
python scripts/run_unified_scale.py recovery   # parameter recovery on data simulated from the main fit
python scripts/run_unified_scale.py report     # ratings with 95 % intervals
python scripts/write_ratings.py               # ratings table -> ratings/
python scripts/make_unified_figures.py         # figures -> figures/
python -m pytest tests                         # checks of the model code (pip install -e ".[test]")
```

Run `linking` first and set the configuration it selects in `roboelo/unified.py`, then `fit`; the other commands read their output. They run in parallel on `ROBOELO_WORKERS` processes (default: number of CPUs − 2); on a 20-thread CPU the whole sequence takes about 10 minutes, most of it `linking` and `bootstrap`. The scripts can be started from any folder. They fit the milestone policies; `ROBOELO_POLICIES=all` fits every policy in data/ and writes to `results/unified_all_policies/`.

### Figures in the final report

After `fit` (and `bootstrap` for the intervals), these commands draw the figures used in the final report:

```bash
python scripts/plot_overall_ratings.py --split-robotwin   # task difficulty and overall scores
python scripts/make_unified_figures.py                   # conversion factors, held-out test, rating over time
python scripts/agielo_figures.py                         # predicted vs observed success, consistency table
python scripts/rating_stability.py && python scripts/rating_stability.py --slide   # stability of ratings
python scripts/plot_frontier_forecast.py && python scripts/plot_frontier_forecast.py --by-benchmark   # frontier forecast
python scripts/plot_mastery_gap.py && python scripts/mastery_table.py   # gap to mastery
```

## References

- Sun et al., *AGI-Elo: How Far Are We from Mastering a Task?*, NeurIPS 2025 (arXiv:2505.12844): rating models and test cases on one Elo scale.
- Rasch (1960); Birnbaum (1968): one- and two-parameter item response models.
- Bradlow, Wainer & Wang (1999), *A Bayesian random effects model for testlets*, Psychometrika.
- Kolen & Brennan, *Test Equating, Scaling, and Linking*.
- Elo (1978), *The Rating of Chessplayers, Past and Present*.

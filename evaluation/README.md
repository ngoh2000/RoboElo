# Evaluation scripts

These scripts produced our own LIBERO, SimplerEnv and RoboTwin 2.0 results (`data/own_runs/`). They document exactly how each result was obtained; they are not a turnkey package.

**Where they run.** A Linux server with NVIDIA GPUs and a workspace at `~/roboelo`, which holds:
- `third_party/`: checkouts of each model's and benchmark's repository;
- `micromamba/` and `envs/`: one environment per model family;
- `checkpoints/`, `results/`, `logs/` and `state/`.

**Rules every run follows:**
- **Official code and weights.** Each model runs with its original authors' evaluation code and official checkpoint, or the official LeRobot port.
- **Fixed episodes, one record each.** Episodes are fixed in advance (LIBERO initial states 0–49; SimplerEnv's official episode list). Every rollout is written as one CSV row as soon as it ends, so a run can stop and resume at any point.
- **Errors are retried.** An episode that raises an error is run again and is not counted as a failure.

| Group | Files |
|---|---|
| LIBERO record format and resume | `roboelo_libero_common.py` |
| LIBERO runners | `run_lerobot_libero.py` and `run_lerobot_libero_supervisor.sh` (LeRobot checkpoints) · `run_pi05_libero_client.py`, `run_pi05_libero_supervisor.sh` (π0.5, openpi) · `run_groot_libero_client.py`, `run_groot_libero_supervisor.sh` (GR00T N1.7) · `run_g05_libero_client.py`, `run_g05_libero_supervisor.sh` (G0.5) · `run_molmoact2_libero_supervisor.sh` · `run_openvla_libero.py` · `run_openvla_oft_libero.py` · `run_univla_libero.py` · `run_vla_adapter_libero.py` · `run_nora_libero.py` · `run_molmoact_libero.py` · `run_evo1_libero.py` · `run_starvla_libero.py` (starVLA, InternVLA-M1, ABot-M0) · `run_xiaomi_libero.py` |
| SimplerEnv | `simpler_episode_grid.py` (the episode list of the official scripts) · `run_simpler_env.py` with `simpler_policies.py` (one policy factory per model) · `run_simpler_supervisor.sh` · `make_simpler_smoke_episodes.py` · `patch_simpler_absee.py` (absolute end-effector controllers for X-VLA) |
| RoboTwin 2.0 | `run_lerobot_robotwin.py` · `run_robotwin_supervisor.sh` |
| Policy servers | `lerobot_policy_server.py` · `molmoact_vllm_server.py` · `starvla_server_launcher.py` · `xiaomi_policy_server.py` · `evo1_server_launcher.py` |
| Job queue | `roboelo_gpu_queue.sh` · `run_queue_job_now.sh` · `smoke_queue_job.sh` |
| Environments | `setup_*.sh` · `fix_openvla_oft_env.sh` · `fix_univla_env.sh` · `make_simpler_overlay_env.sh` |
| Checkpoints | `download_hf_resume.py` · `download_hf_models.sh` · `download_*.sh` · `download_cogact_base_parts.py` · `convert_safetensors_bf16.py` · `prepare_*` |

**Supervisors.** Each supervisor (`run_*_supervisor.sh`) runs one model on one GPU with several worker processes:
- it starts a worker again if it dies;
- it stops after repeated deaths without a new episode.

**Job queue.** The queue reads `~/roboelo/state/queue/jobs.txt`, one job per line:

```
name | GPU memory in MiB | files that must exist before the job may start | command
```

- It starts at most one job per GPU.
- It leaves a GPU alone while another user has recently started a process on it.
- It records each job's state in `state/queue/<name>.state`.

**Pipeline tests.** A full job waits for `state/<name>_tested.done`. That file is created after a short pipeline test (`smoke_queue_job.sh` or the `*_smoke` jobs) has been checked.

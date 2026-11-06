#!/usr/bin/env python
"""Record keeping shared by the RoboElo LIBERO runners: one CSV row per rollout, resume, summary.

Only the standard library is used, so every model's own Python environment can import it.

  <out>/<suite>/episodes.csv   one row per finished rollout (appended under a file lock, flushed at once)
  <out>/<suite>/errors.csv     rollouts that raised an exception (never counted as failures)
  <out>/<suite>/videos/        replay videos of failed rollouts
  <out>/<suite>/summary.json   per-task and suite success rates

Run as a script to (re)write the summaries of a results folder:
  python roboelo_libero_common.py <out> [--num-trials 50]
"""
import argparse
import collections
import csv
import datetime
import fcntl
import json
import pathlib

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
FIELDS = ["model", "checkpoint", "benchmark", "suite", "task_id", "task_name", "bddl_file", "init_state_id", "seed",
          "success", "steps", "max_steps", "goal_total", "goal_satisfied_final", "goal_satisfied_max",
          "partial_final", "partial_max", "n_infer", "wall_time_s", "finished_at", "video"]
ERROR_FIELDS = ["suite", "task_id", "init_state_id", "attempt", "time", "error"]
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def goal_status(libero_env):
    """Truth value of each goal predicate of the task (success = all true).

    `libero_env` is LIBERO's OffScreenRenderEnv. LIBERO itself reports only success or failure, but
    every task goal is a conjunction of BDDL predicates, which gives a partial-progress measure.
    """
    inner = libero_env.env
    return [bool(inner._eval_predicate(state)) for state in inner.parsed_problem["goal_state"]]


def _append_row(path, fields, row):
    """Append one row; several client processes may write to the same file."""
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def read_rows(path):
    if not pathlib.Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class EpisodeLog:
    """Rollout records of one suite."""

    def __init__(self, out_root, suite, model, checkpoint):
        self.suite, self.model, self.checkpoint = suite, model, checkpoint
        self.dir = pathlib.Path(out_root) / suite
        (self.dir / "videos").mkdir(parents=True, exist_ok=True)
        self.ep_csv, self.err_csv = self.dir / "episodes.csv", self.dir / "errors.csv"
        self.done = {(int(r["task_id"]), int(r["init_state_id"])) for r in read_rows(self.ep_csv)}
        self.n_new = self.n_err = 0

    def todo(self, task_id, init_ids):
        return [i for i in init_ids if (task_id, i) not in self.done]

    def video_path(self, task_id, init_id, success):
        tag = "success" if success else "failure"
        return str(self.dir / "videos" / f"task{task_id:02d}_init{init_id:02d}_{tag}.mp4")

    def write(self, *, task_id, task_name, bddl_file, init_state_id, seed, success, steps, max_steps, goal_total,
              goal_satisfied_final, goal_satisfied_max, n_infer, wall_time_s, video=""):
        row = dict(
            model=self.model, checkpoint=self.checkpoint, benchmark="libero", suite=self.suite, task_id=task_id,
            task_name=task_name, bddl_file=bddl_file, init_state_id=init_state_id, seed=seed, success=int(success),
            steps=steps, max_steps=max_steps, goal_total=goal_total, goal_satisfied_final=goal_satisfied_final,
            goal_satisfied_max=goal_satisfied_max, partial_final=round(goal_satisfied_final / goal_total, 4),
            partial_max=round(goal_satisfied_max / goal_total, 4), n_infer=n_infer,
            wall_time_s=round(wall_time_s, 1), finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
            video=video)
        _append_row(self.ep_csv, FIELDS, row)
        self.done.add((task_id, init_state_id))
        self.n_new += 1
        print(f"[{self.suite}] task {task_id} init {init_state_id}: success={int(success)} steps={steps} "
              f"goal {goal_satisfied_final}/{goal_total} ({wall_time_s:.0f}s)", flush=True)

    def error(self, task_id, init_id, attempt, exc):
        """An exception is recorded separately and the rollout is run again."""
        self.n_err += 1
        _append_row(self.err_csv, ERROR_FIELDS, dict(
            suite=self.suite, task_id=task_id, init_state_id=init_id, attempt=attempt,
            time=datetime.datetime.now().isoformat(), error=f"{type(exc).__name__}: {exc}"[:500]))

    def missing(self, task_ids, init_ids):
        return [(t, i) for t in task_ids for i in init_ids if (t, i) not in self.done]


def summarize(out_root, num_trials=50, n_tasks=10, quiet=False):
    """Write <suite>/summary.json for every suite with records; returns (successes, rollouts, missing)."""
    out_root = pathlib.Path(out_root)
    tot_ok = tot_n = tot_missing = 0
    for suite in SUITES:
        rows = read_rows(out_root / suite / "episodes.csv")
        if not rows:
            if not quiet:
                print(f"{suite:16s} no rollouts recorded")
            tot_missing += n_tasks * num_trials
            continue
        per_task = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            per_task[int(r["task_id"])][0] += int(r["success"])
            per_task[int(r["task_id"])][1] += 1
        keys = {(int(r["task_id"]), int(r["init_state_id"])) for r in rows}
        missing = sum((t, i) not in keys for t in range(n_tasks) for i in range(num_trials))
        ok = sum(int(r["success"]) for r in rows)
        summary = dict(suite=suite, rollouts=len(rows), successes=ok, success_rate=round(ok / len(rows), 4),
                       expected_rollouts=n_tasks * num_trials, missing_rollouts=missing,
                       duplicate_rows=len(rows) - len(keys), exceptions=len(read_rows(out_root / suite / "errors.csv")),
                       per_task={k: dict(successes=v[0], rollouts=v[1]) for k, v in sorted(per_task.items())})
        (out_root / suite / "summary.json").write_text(json.dumps(summary, indent=1))
        if not quiet:
            print(f"{suite:16s} {ok:4d}/{len(rows):4d} = {ok / len(rows):.3f}   missing {missing}")
        tot_ok, tot_n, tot_missing = tot_ok + ok, tot_n + len(rows), tot_missing + missing
    if not quiet:
        print(f"{'all':16s} {tot_ok:4d}/{tot_n:4d} = {tot_ok / max(1, tot_n):.3f}   missing {tot_missing}")
    return tot_ok, tot_n, tot_missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--num-trials", type=int, default=50)
    args = ap.parse_args()
    summarize(args.out, args.num_trials)

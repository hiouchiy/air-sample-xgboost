"""XGBoost HPO — A10 fan-out orchestrator (control-plane). Optional scale-out alternative to the
single-GPU training in step 1: it searches hyperparameters and registers the best model.

Runs the hyperparameter search as **N parallel single-`GPU_1xA10` jobs**: it shards the trial grid
across N `air run` submissions (each a slice of the same seeded grid), waits for them, then promotes
the **global best** to the `@champion` alias.

Why this shape for classic ML? HPO is embarrassingly parallel, so the realistic, cost-appropriate
way to scale it is to spread trials across many **cheap** A10 nodes — not to reach for a
`GPU_8xH100` box that a seconds-long XGBoost fit can't justify. Fan-out buys wall-clock parallelism
at A10 cost. At tiny scale the per-job cold start can outweigh the savings, so measure before
choosing; this shines when trials are longer / more numerous. (Contrast with the BERT sample, where
8-GPU DDP is genuine distributed training — different workload, different scaling strategy.)

This is a **control-plane** script (it only shells out to the `air` and `databricks` CLIs) — it needs
**no GPU** and **no extra Python deps** (not even mlflow), and does NOT run on AI Runtime itself. Run
it locally with your profile, from the repo root:

    NUM_WORKERS=2 TRIAL_TOTAL=8 python 02_cli/fanout_hpo.py --profile <profile>

Each worker registers a model version and writes a small result JSON (version + best AUC) to the
`predictions` UC Volume; this script reads them, picks the global best, and sets `@champion` to that
version via `databricks registered-models set-alias`. 02_batch_inference.py then loads `@champion`.
"""

import argparse
import logging
import math
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Env vars forwarded from the orchestrator to every worker job (so the same catalog/tuning applies).
_PASSTHROUGH = ("UC_CATALOG", "UC_SCHEMA", "REGISTERED_MODEL_NAME", "MAX_SAMPLES", "N_ESTIMATORS",
                "TEST_SIZE", "RANDOM_STATE")


def _run_worker(worker_idx, trial_start, trial_count, trial_total, fanout_tag, profile):
    """Submit one single-A10 `air run` job for a trial shard and wait for it (--watch)."""
    passthrough = " ".join(f"{k}={os.environ[k]}" for k in _PASSTHROUGH if k in os.environ)
    command = (
        f"{passthrough} TRIAL_TOTAL={trial_total} TRIAL_START={trial_start} NUM_TRIALS={trial_count} "
        f"FANOUT_TAG={fanout_tag} REGISTER_MODEL=true "
        f"python $CODE_SOURCE_PATH/02_cli/hpo_worker.py"
    ).strip()
    argv = ["air", "run", "--file", "02_cli/hpo_worker.yaml", "--watch",
            "--override", f"command={command}"]
    if profile:
        argv += ["--profile", profile]
    env = {**os.environ, "COPYFILE_DISABLE": "1"}  # keep macOS ._* out of the snapshot
    print(f"[worker {worker_idx}] trials [{trial_start}:{trial_start + trial_count}) — submitting...")
    proc = subprocess.run(argv, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    ok = proc.returncode == 0 and "Job status: SUCCESS" in (proc.stdout + proc.stderr)
    print(f"[worker {worker_idx}] {'SUCCESS' if ok else 'FAILED'}")
    if not ok:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
    return ok


def _dbx(args, profile):
    argv = ["databricks", *args]
    if profile:
        argv += ["--profile", profile]
    return subprocess.run(argv, capture_output=True, text=True)


def _promote_best(fqn, fanout_tag, catalog, schema, profile):
    """Read each worker's result JSON from the UC Volume, pick the global best AUC, and set @champion
    on that version — all via the `databricks` CLI (no local mlflow needed)."""
    import json

    vol = f"dbfs:/Volumes/{catalog}/{schema}/predictions"
    prefix = f"_fanout__{fanout_tag}__"
    ls = _dbx(["fs", "ls", vol, "--output", "json"], profile)
    entries = json.loads(ls.stdout) if ls.stdout.strip() else []
    names = [e.get("name") or os.path.basename(e.get("path", "")) for e in entries]
    result_files = [n for n in names if n.startswith(prefix) and n.endswith(".json")]
    candidates = []
    for n in result_files:
        cat = _dbx(["fs", "cat", f"{vol}/{n}"], profile)
        rec = json.loads(cat.stdout)
        candidates.append((float(rec["auc"]), int(rec["version"])))
    if not candidates:
        raise RuntimeError(f"No fan-out result files ({prefix}*.json) found under {vol}.")
    best_auc, best_version = max(candidates)
    promote = _dbx(["registered-models", "set-alias", fqn, "champion", str(best_version)], profile)
    if promote.returncode != 0:
        raise RuntimeError(f"set-alias failed: {promote.stdout} {promote.stderr}")
    print(f"Best of {len(candidates)} fan-out workers: v{best_version} (auc={best_auc:.4f}) "
          f"-> set @champion on {fqn}.")
    for n in result_files:  # tidy up this batch's result files
        _dbx(["fs", "rm", f"{vol}/{n}"], profile)
    return best_version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=_env("PROFILE", ""), help="Databricks/air profile name")
    args = ap.parse_args()

    num_workers = int(_env("NUM_WORKERS", "2"))
    trial_total = int(_env("TRIAL_TOTAL", "8"))
    fqn = f'{_env("UC_CATALOG", "main")}.{_env("UC_SCHEMA", "air_samples")}.' \
          f'{_env("REGISTERED_MODEL_NAME", "xgboost_classification")}'
    fanout_tag = f"fanout-{uuid.uuid4().hex[:8]}"

    # Even split of the shared grid across workers (last worker takes any remainder).
    per = math.ceil(trial_total / num_workers)
    shards = []
    start = 0
    while start < trial_total:
        shards.append((start, min(per, trial_total - start)))
        start += per
    print(f"Fan-out: {trial_total} trials across {len(shards)} single-A10 jobs "
          f"(batch {fanout_tag}); model {fqn}")

    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        results = list(pool.map(
            lambda a: _run_worker(a[0], a[1][0], a[1][1], trial_total, fanout_tag, args.profile),
            list(enumerate(shards)),
        ))
    if not all(results):
        sys.exit(f"{results.count(False)}/{len(results)} fan-out workers failed — see logs above.")

    _promote_best(fqn, fanout_tag, _env("UC_CATALOG", "main"),
                  _env("UC_SCHEMA", "air_samples"), args.profile)
    print("Fan-out HPO complete.")


if __name__ == "__main__":
    main()

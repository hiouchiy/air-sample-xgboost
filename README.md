# AI Runtime sample: XGBoost GPU classification

End-to-end **XGBoost GPU training, batch inference** on **Databricks AI Runtime**
(serverless NVIDIA GPUs). It trains an **XGBoost classifier** on a large synthetic
binary classification dataset (500k–2M rows × 100 features) — demonstrating **GPU-accelerated
gradient boosting** (10–20× faster than CPU), tracks everything with **MLflow**, registers
the model to **Unity Catalog**, and runs **GPU batch inference** to predict on new data.

> **Repo name**: `air-sample-xgboost`. AIR = AI Runtime.

## What it demonstrates

| Step | File | AIR compute | Shows |
|------|------|-------------|-------|
| 1. Single-GPU training | [`src/01_train_singlegpu.py`](src/01_train_singlegpu.py) | `GPU_1xA10` | GPU-accelerated XGBoost, MLflow tracking, UC registration |
| 2. Multi-GPU HPO | [`src/02_train_multigpu.py`](src/02_train_multigpu.py) | `GPU_8xH100` | Parallel hyperparameter search — one trial per GPU across all 8 |
| 3. GPU batch inference | [`src/03_batch_inference.py`](src/03_batch_inference.py) | `GPU_1xA10` | Loading the UC model, batched GPU scoring, writing to UC |

## What lands in the Databricks platform (in both modes)

Beyond running on AI Runtime GPUs, every step is wired into the wider Databricks platform:

- **MLflow experiment tracking** — 01/03 log params, metrics and the model to an MLflow run; **02
  logs every HPO trial as its own nested run** so you can compare all candidates in the Experiments UI.
- **Unity Catalog Model Registry + versioning** — 01/02 register the model to
  `main.air_samples.xgboost_classification`, creating a new **version** each run and promoting it to
  the **`@champion`** alias (02 promotes the best trial). 03 loads `@champion`, so version promotion
  is explicit and governed — no manual step.
- **Unity Catalog Volumes** — 03 writes its prediction file to a UC Volume.

## Every script runs two ways, with no code changes

This is a hard requirement for these samples:

1. **As a Databricks notebook** — import the `.py` into the workspace (it carries
   `# Databricks notebook source` markers) and **Run All** on AI Runtime. The `# MAGIC %pip`
   cells install dependencies in the notebook.
2. **As an AI Runtime CLI job** — `air run --file air/<step>.yaml`. The `# MAGIC` lines are plain
   Python comments (ignored); dependencies come from the YAML `environment.dependencies`.

The same file works in both because logic lives in functions, the entry point is a single
`if __name__ == "__main__": main()` (Databricks notebooks also expose `__name__ == "__main__"`),
and all settings are environment variables with sensible defaults.

## Prerequisites

- A **Databricks workspace where AI Runtime is enabled.** AI Runtime is currently available in
  **US regions on AWS/Azure** (EU/APJ/GCP were not yet GA as of this writing) — confirm with your
  Databricks contact if unsure.
- Permission to **create a Unity Catalog schema and volume** in some catalog (ask your admin which
  catalog you can write to, or use one you own).
- macOS/Linux/WSL with a terminal. (This repo was validated on `e2-demo-field-eng`.)

## Setup — one time, ~10 minutes (no Databricks experience needed)

Run these on your laptop. Replace `<workspace-url>` and pick a profile name (here `air`).

```bash
# a) Install the Databricks CLI
brew install databricks            # macOS; else: curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
databricks --version               # need v0.230+

# b) Log in — this opens a browser and saves an auth "profile" named `air`
databricks auth login --host https://<workspace-url>.cloud.databricks.com --profile air
databricks current-user me --profile air     # should print your email

# c) Install the AI Runtime CLI (`air`); it reuses the Databricks profiles above
curl -LsSf https://astral.sh/uv/install.sh | sh      # installs `uv` if you don't have it
uv tool install --force databricks-air --python 3.12
air --version

# d) Create the Unity Catalog schema + volume this demo writes to (one time).
#    Pick a catalog you can write to (e.g. `main`, or your own).
export CATALOG=main                                   # <-- change to YOUR catalog
databricks schemas create air_samples $CATALOG --profile air
databricks volumes create $CATALOG air_samples predictions MANAGED --profile air
```

> Prefer one command? Run `CATALOG=main PROFILE=air ./setup.sh` (see [`setup.sh`](setup.sh)).

## Point the demo at your catalog

The scripts default to catalog **`main`**, schema **`air_samples`** — the same values the Setup
step creates. **If you ran Setup with `CATALOG=main`, everything runs unchanged.** To use a
different catalog, set `UC_CATALOG` either way:

- edit the `UC_CATALOG` / `UC_SCHEMA` default lines near the top of each `src/*.py`, or
- prefix the YAML `command:` line, e.g.
  `command: UC_CATALOG=mycat python $CODE_SOURCE_PATH/src/01_train_singlegpu.py`.

Use the **same profile name** you created (`air`) in every `air run --profile ...` below.

## Run it (CLI) — do the steps in order

Step 3 needs the model that step 1 (or 2) registers, so **run 01 first.**

```bash
# macOS: the COPYFILE_DISABLE=1 prefix is REQUIRED — it keeps macOS ._* files out of the
# uploaded code snapshot (otherwise the job dies immediately). Harmless on Linux.

# 1) Train on one A10 GPU (device="cuda") → MLflow → register to Unity Catalog  (~5 min incl. GPU wait)
COPYFILE_DISABLE=1 air run --file air/train_singlegpu.yaml --watch --profile air

# 2) Parallel hyperparameter search on 8× H100 — one trial per GPU; registers the best model
COPYFILE_DISABLE=1 air run --file air/train_multigpu.yaml --watch --profile air

# 3) GPU batch inference over the held-out test set → predictions CSV on the UC Volume
COPYFILE_DISABLE=1 air run --file air/batch_inference.yaml --watch --profile air
```

Each `air run` ends with `Job status: SUCCESS` on success. The first run waits a few minutes for a
GPU to be provisioned — that is normal. (Note: `air logs` sometimes prints "No logs available" even
for successful runs; trust `Job status` and the MLflow links.)

## Run it (notebook)

Import any `src/*.py` into your Databricks workspace (**Workspace → Import → File**), attach it to
**AI Runtime**, and **Run All**. The `%pip` cells install dependencies automatically. Start with
`01_train_singlegpu.py`, then `03_batch_inference.py`.

- `02_train_multigpu.py` parallelizes over **whatever GPUs are attached** (`torch.cuda.device_count()`
  with a thread pool), so attach a **`GPU_8xH100`** AI Runtime compute to get 8-way parallelism. It
  still runs on a 1-GPU compute (trials just run sequentially).
- The **first** run waits several minutes (~5 min on A10, ~7 min on 8×H100) for GPU capacity before
  any cell executes — that's normal cold start, not a hang.

## Configuration (env vars, with defaults)

| Variable | Default | Meaning |
|----------|---------|---------|
| `UC_CATALOG` / `UC_SCHEMA` | `main` / `air_samples` | Unity Catalog target (matches Setup; change only to use another catalog) |
| `REGISTERED_MODEL_NAME` | `xgboost_classification` | UC registered model name |
| `NUM_TRAIN_SAMPLES` / `NUM_TEST_SAMPLES` | `500000` / `100000` | Dataset size; reduce for quick smoke tests |
| `NUM_FEATURES` / `NUM_CLASSES` | `100` / `2` | Synthetic-data shape |
| `NUM_TRIALS` (02 only) | `16` | HPO trials; dispatched one-per-GPU across the node |
| `N_ESTIMATORS`, `MAX_DEPTH`, `LEARNING_RATE` | see scripts | XGBoost hyper-parameters (02 samples these per trial) |
| `TREE_METHOD` / `DEVICE` (01) | `hist` / `cuda` | XGBoost 2.x GPU switch is `device="cuda"`; auto-falls back to CPU |

> **Keep `NUM_*` sample sizes and `RANDOM_STATE` the same across 01 and 03** — step 3 regenerates the
> identical synthetic dataset and scores its held-out split, so mismatched sizes would score
> out-of-distribution data. Both default to the same values, so the defaults just work.

Override any of these per run by prefixing the YAML `command:` line, e.g.
`command: NUM_TRAIN_SAMPLES=100000 python $CODE_SOURCE_PATH/src/01_train_singlegpu.py`.

## Troubleshooting

| Symptom | Cause & fix |
|---------|-------------|
| Job dies in seconds, `cd: .../._xxx: Not a directory` | macOS AppleDouble files — always run with `COPYFILE_DISABLE=1` (see above). |
| `RESOURCE_DOES_NOT_EXIST` / schema or volume not found | Run the Setup step (d); make sure `UC_CATALOG`/`UC_SCHEMA` match what you created. |
| Step 3 accuracy looks random (~0.5) | 01 and 03 used different `NUM_TRAIN_SAMPLES`/`RANDOM_STATE` → different synthetic data. Keep them equal (defaults do). |
| Step 03 log shows `spark-class ... ClassNotFoundException` / `dbconnect` errors | Harmless if followed by `Wrote ... to UC Volume`. AI Runtime GPU nodes have no Spark, so 03 writes a CSV to the UC Volume. These lines come from the runtime's Spark probe during MLflow logging (not from the demo code) and are safe to ignore. |
| `air logs` says "No logs available" | Known quirk; the run may still have succeeded. Check `Job status` and the MLflow run link. |
| Long "waiting for GPU capacity" | Normal for H100; retry later or run only step 1 (A10). AI Runtime is US-region only for now. |

## Repo layout

```
air-sample-xgboost/
├── src/                       # notebook-and-CLI dual-mode Python scripts
│   ├── 01_train_singlegpu.py
│   ├── 02_train_multigpu.py
│   └── 03_batch_inference.py
├── air/                       # AI Runtime CLI workload specs (one per step)
│   ├── train_singlegpu.yaml
│   ├── train_multigpu.yaml
│   └── batch_inference.yaml
├── docs/                      # deeper docs (architecture, AIR notes)
├── setup.sh                   # one-time UC schema + volume creation
├── requirements.txt
└── README.md
```

See [`docs/`](docs/) for the architecture walkthrough and AI Runtime specifics.

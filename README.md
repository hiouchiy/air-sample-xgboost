# AI Runtime sample: XGBoost GPU classification

End-to-end **XGBoost GPU training, batch inference** on **Databricks AI Runtime**
(serverless NVIDIA GPUs). It trains an **XGBoost multi-class classifier** on the public **Forest
CoverType** dataset (581,012 rows × 54 features, 7 classes) — demonstrating **GPU-accelerated
gradient boosting**, tracks everything with **MLflow**, registers the model to **Unity Catalog**, and
runs **GPU batch inference** to predict on new data.

> AIR = AI Runtime.

## Two ways to run — one folder each

Each step ships in **two forms**, so you can pick whichever fits and neither needs editing:

- **`01_notebook/`** — Databricks notebooks (`# Databricks notebook source` `.py`). **Import and step
  through the cells** (top to bottom). Rich per-cell markdown; `%pip` cells install dependencies.
- **`02_cli/`** — plain Python scripts + AI Runtime CLI workload YAMLs. **Submit with `air run`.**
  No notebook markers; dependencies come from the YAML.

Both forms share the same logic and are driven by environment variables with sensible defaults.
`03_docs/` has the architecture write-up.

## What it demonstrates

| Step | Notebook / CLI script | AIR compute | Shows |
|------|-----------------------|-------------|-------|
| 1. Single-GPU training | `01_train_singlegpu.py` | `GPU_1xA10` | GPU-accelerated XGBoost, MLflow tracking, UC registration + `@champion` |
| 2. Multi-GPU HPO | `02_train_multigpu.py` | `GPU_8xH100` | Parallel hyperparameter search — one trial per GPU across all 8 |
| 3. GPU batch inference | `03_batch_inference.py` | `GPU_1xA10` | Loading the `@champion` UC model, batched GPU scoring, CSV to a UC Volume |

## What lands in the Databricks platform (both forms)

Beyond running on AI Runtime GPUs, every step is wired into the wider Databricks platform:

- **MLflow experiment tracking** — 01/03 log params, metrics and the model to an MLflow run; **02
  logs every HPO trial as its own nested run** so you can compare all candidates in the Experiments UI.
- **Unity Catalog Model Registry + versioning** — 01/02 register the model to
  `main.air_samples.xgboost_classification`, creating a new **version** each run and promoting it to
  the **`@champion`** alias (02 promotes the best trial). 03 loads `@champion`, so version promotion
  is explicit and governed — no manual step.
- **Unity Catalog Volumes** — 03 writes its prediction CSV to a UC Volume.

## Prerequisites

- A **Databricks workspace where AI Runtime is enabled** — see the
  [AI Runtime documentation](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/) for
  current cloud/region availability.
- Permission to **create a Unity Catalog schema and volume** in some catalog (ask your admin which
  catalog you can write to, or use one you own).
- macOS/Linux/WSL with a terminal. (Validated on a workspace with AI Runtime enabled.)

## Setup — one time, ~10 minutes

Run these on your laptop. Replace `<workspace-url>` and pick a profile name (here `air`).

```bash
# a) Install the Databricks CLI
brew install databricks            # macOS; else: curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
databricks --version               # need v0.230+

# b) Log in — this opens a browser and saves an auth "profile" named `air`
databricks auth login --host https://<workspace-url>.cloud.databricks.com --profile air
databricks current-user me --profile air     # should print your email

# c) (CLI users only — SKIP if you'll run the 01_notebook/ notebooks) Install the AI Runtime CLI (`air`)
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
step creates. **`main` works in many demo workspaces but is often locked down (no `CREATE`) in
governed ones** — pick a catalog where you can create schemas/volumes/models. Set it:

- **Notebook (recommended):** use the **`UC_CATALOG` / `UC_SCHEMA` widgets** at the top of the
  notebook — no code edit, and it runs before anything else. Or
- edit the `UC_CATALOG` / `UC_SCHEMA` default lines near the top of each script, or
- prefix the YAML `command:` line, e.g.
  `command: UC_CATALOG=mycat python $CODE_SOURCE_PATH/02_cli/01_train_singlegpu.py`.

Use the **same profile name** you created (`air`) in every `air run --profile ...` below.

## Run it as a notebook (`01_notebook/`)

Import a file from `01_notebook/` into your Databricks workspace (**Workspace → Import → File**),
then **attach a serverless AI Runtime GPU** — there is no cluster to create:

1. Open the **compute** drop-down at the top of the notebook → **Serverless GPU**.
2. Click the **environment** icon to open the **Environment** side panel.
3. Set **Accelerator** (`GPU_1xA10` for 01/03; **`GPU_8xH100`** for `02_train_multigpu.py`) and
   leave the default **Base environment**.
4. Click **Apply**, then **Confirm**.

Then **run the cells one at a time, top to bottom**, reviewing each step's output (Run All works too). The `%pip` cells install
dependencies automatically. Start with `01_train_singlegpu.py`, then `03_batch_inference.py`. See
[Connect to serverless GPU compute](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting#gpu-compute).

- `02_train_multigpu.py` parallelizes over **whatever GPUs are attached** (`torch.cuda.device_count()`
  with a thread pool), so attach a **`GPU_8xH100`** AI Runtime compute to get 8-way parallelism. It
  still runs on a 1-GPU compute (trials just run sequentially).
- The **first** run waits several minutes (~5 min on A10, ~7 min on 8×H100) for GPU capacity before
  any cell executes — that's normal cold start, not a hang.

## Run it via the CLI (`02_cli/`) — do the steps in order

Step 3 needs the model that step 1 (or 2) registers, so **run 01 first.**

```bash
# macOS: the COPYFILE_DISABLE=1 prefix is REQUIRED — it keeps macOS ._* files out of the
# uploaded code snapshot (otherwise the job dies immediately). Harmless on Linux.

# 1) Train on one A10 GPU (device="cuda") → MLflow → register to Unity Catalog  (~5 min incl. GPU wait)
COPYFILE_DISABLE=1 air run --file 02_cli/train_singlegpu.yaml --watch --profile air

# 2) Parallel hyperparameter search on 8× H100 — one trial per GPU; registers the best model
COPYFILE_DISABLE=1 air run --file 02_cli/train_multigpu.yaml --watch --profile air

# 3) GPU batch inference over the held-out test set → predictions CSV on the UC Volume
COPYFILE_DISABLE=1 air run --file 02_cli/batch_inference.yaml --watch --profile air
```

Each `air run` ends with `Job status: SUCCESS` on success. The first run waits a few minutes for a
GPU to be provisioned — that is normal. (Note: `air logs` sometimes prints "No logs available" even
for successful runs; trust `Job status` and the MLflow links.)

## Configuration (env vars, with defaults)

| Variable | Default | Meaning |
|----------|---------|---------|
| `UC_CATALOG` / `UC_SCHEMA` | `main` / `air_samples` | Unity Catalog target (matches Setup; change only to use another catalog) |
| `REGISTERED_MODEL_NAME` | `xgboost_classification` | UC registered model name |
| `TEST_SIZE` | `0.2` | Held-out test fraction of the CoverType dataset |
| `MAX_SAMPLES` | `-1` | `-1` = all 581k rows; set a small number for a quick smoke test |
| `RANDOM_STATE` | `42` | Split seed — must match between 01/02 and 03 |
| `NUM_TRIALS` (02 only) | `16` | HPO trials; dispatched one-per-GPU across the node |
| `N_ESTIMATORS`, `MAX_DEPTH`, `LEARNING_RATE` | see scripts | XGBoost hyper-parameters (02 samples these per trial) |
| `TREE_METHOD` / `DEVICE` (01) | `hist` / `cuda` | XGBoost 2.x GPU switch is `device="cuda"`; auto-falls back to CPU |

> **Keep `TEST_SIZE` and `RANDOM_STATE` the same across 01 and 03** — step 3 re-downloads Forest
> CoverType and regenerates the identical held-out split, so mismatched values would score
> out-of-distribution data. Both default to the same values, so the defaults just work.

Override per run by prefixing the YAML `command:` line, e.g.
`command: MAX_SAMPLES=50000 python $CODE_SOURCE_PATH/02_cli/01_train_singlegpu.py`.

## Troubleshooting

| Symptom | Cause & fix |
|---------|-------------|
| Job dies in seconds, `cd: .../._xxx: Not a directory` | macOS AppleDouble files — always run with `COPYFILE_DISABLE=1` (see above). |
| `RESOURCE_DOES_NOT_EXIST` / schema or volume not found | Run the Setup step (d); make sure `UC_CATALOG`/`UC_SCHEMA` match what you created. |
| Step 3 accuracy looks off | 01 and 03 used different `TEST_SIZE`/`RANDOM_STATE` → different held-out split. Keep them equal (defaults do). |
| Job fails downloading the dataset | The GPU node needs internet egress for `fetch_covtype`. In a locked-down workspace, pre-stage the data in a UC Volume and load from there. |
| Step 03 log shows `spark-class ... ClassNotFoundException` / `dbconnect` errors | Harmless. AI Runtime GPU nodes have no Spark; these lines come from the runtime's Spark probe during MLflow logging (not from the demo code) and are safe to ignore — 03 writes a CSV to the UC Volume. |
| `air logs` says "No logs available" | Known quirk; the run may still have succeeded. Check `Job status` and the MLflow run link. |
| Long "waiting for GPU capacity" | Normal for H100; retry later or run only step 1 (A10). |

## Repo layout

```
air-sample-xgboost/
├── 01_notebook/               # Databricks notebooks — import + step through
│   ├── 01_train_singlegpu.py
│   ├── 02_train_multigpu.py
│   └── 03_batch_inference.py
├── 02_cli/                    # AI Runtime CLI — plain scripts + workload YAMLs (air run)
│   ├── 01_train_singlegpu.py … 03_batch_inference.py
│   ├── train_singlegpu.yaml
│   ├── train_multigpu.yaml
│   └── batch_inference.yaml
├── 03_docs/                   # architecture walkthrough & AI Runtime notes
│   └── architecture.md
├── setup.sh                   # one-time UC schema + volume creation
├── requirements.txt
└── README.md
```

See [`03_docs/architecture.md`](03_docs/architecture.md) for the architecture walkthrough and AI Runtime specifics.

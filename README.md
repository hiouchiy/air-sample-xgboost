# AI Runtime sample: XGBoost GPU classification

End-to-end **XGBoost GPU training, batch inference** on **Databricks AI Runtime**
(serverless NVIDIA GPUs). It trains an **XGBoost multi-class classifier** on the public **Forest
CoverType** dataset (581,012 rows × 54 features, 7 classes) — demonstrating **GPU-accelerated
gradient boosting**, tracks everything with **MLflow**, registers the model to **Unity Catalog**,
runs **GPU batch inference**, and **optionally** deploys a **CPU Model Serving** endpoint (tabular
inference needs no GPU).

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
| 2. GPU batch inference | `02_batch_inference.py` | `GPU_1xA10` | Loading the `@champion` UC model, batched GPU scoring, CSV to a UC Volume |
| 3. Model Serving *(Optional, notebook only)* | `03_serve.py` | **CPU** serving | Real-time class predictions via **Databricks Model Serving** — on **CPU** (tabular XGBoost needs no GPU); a separate product from AI Runtime (control-plane) |
| *Scale-out (optional)* | `fanout_hpo.py` (+ `hpo_worker.py`) | `GPU_1xA10` × N | An **alternative to step 1**: scale the hyperparameter search across **N cheap single-A10 jobs** (one trial shard each), then promote the global best to `@champion`. **CLI / control-plane only** (no notebook) |

> **No 8×H100 step, by design.** A CoverType-scale XGBoost fit takes seconds on a single A10, so a
> large multi-GPU box isn't justified. The realistic, cost-appropriate way to scale classic-ML HPO
> is horizontal **fan-out across cheap A10 nodes** (`fanout_hpo.py`, above). Contrast the BERT sample,
> where 8-GPU DDP is genuine distributed training — match the scaling strategy to the workload.

## What lands in the Databricks platform (both forms)

Beyond running on AI Runtime GPUs, every step is wired into the wider Databricks platform:

- **MLflow experiment tracking** — 01/02 log params, metrics and the model to an MLflow run; each
  **HPO worker logs every trial as its own nested run** so you can compare all candidates in the
  Experiments UI.
- **Unity Catalog Model Registry + versioning** — single-GPU training (01) and every HPO worker
  register the model to `main.air_samples.xgboost_classification`, creating a new **version** each
  run; `01` promotes its model and the fan-out promotes the **global best** to the **`@champion`**
  alias. 02 loads `@champion`, so version promotion is explicit and governed — no manual step.
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

# macOS only: `air run` packages the code snapshot with **GNU tar**. macOS's built-in
# bsdtar rejects air's `--anchored` flag (error: "tar: Option --anchored is not
# supported"), so install GNU tar and put it first on PATH:
brew install gnu-tar
export PATH="$(brew --prefix gnu-tar)/libexec/gnubin:$PATH"   # add to ~/.zshrc to persist

# d) Create the Unity Catalog schema + volume this demo writes to (one time).
#    Pick a catalog you can write to (e.g. `main`, or your own).
export CATALOG=main                                   # <-- change to YOUR catalog
databricks schemas create air_samples $CATALOG --profile air
databricks volumes create $CATALOG air_samples predictions MANAGED --profile air
```

> Prefer one command? Run `CATALOG=main PROFILE=air ./setup.sh` (see [`setup.sh`](setup.sh)).

> **This Setup is optional** — on first run the notebooks and CLI jobs **auto-create** the
> schema + `predictions` volume if you have `CREATE` on the catalog. Run it (or grant CREATE)
> only if the auto-create step reports a permission error.


## Point the demo at your catalog

The scripts default to catalog **`main`**, schema **`air_samples`** — the same values the Setup
step creates. **`main` works in many demo workspaces but is often locked down (no `CREATE`) in
governed ones** — pick a catalog where you can create schemas/volumes/models. Set it:

- **Notebook (recommended):** use the **`UC_CATALOG` / `UC_SCHEMA` widgets** at the top of the
  notebook — no code edit, and it runs before anything else. Or
- **CLI (no file edit):** override the workload env var, e.g.
  `air run --file 02_cli/train_singlegpu.yaml --override env_variables.UC_CATALOG=mycat --profile air`
  (each `02_cli/*.yaml` declares `env_variables: UC_CATALOG/UC_SCHEMA`).

Use the **same profile name** you created (`air`) in every `air run --profile ...` below.

## Run it as a notebook (`01_notebook/`)

Import a file from `01_notebook/` into your Databricks workspace (**Workspace → Import → File**),
then **attach a serverless AI Runtime GPU** — there is no cluster to create:

1. Open the **compute** drop-down at the top of the notebook → **Serverless GPU**.
2. Click the **environment** icon to open the **Environment** side panel.
3. Set **Accelerator** to **`GPU_1xA10`** (all notebooks) and leave the default **Base environment**.
4. Click **Apply**, then **Confirm**.

Then **run the cells one at a time, top to bottom**, reviewing each step's output (Run All works too). The `%pip` cells install
dependencies automatically. Start with `01_train_singlegpu.py`, then `02_batch_inference.py`, then
(optionally) `03_serve.py` — which is control-plane and needs **no GPU** (any compute). See
[Connect to serverless GPU compute](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting#gpu-compute).

- **The HPO fan-out is CLI-only** — the orchestrator (`02_cli/fanout_hpo.py`) submits multiple
  `air run` jobs, which is a control-plane task with no notebook form. The notebooks cover
  single-GPU training (01), batch inference (02) and serving (03).
- The **first** run waits several minutes (~5 min on A10) for GPU capacity before any cell executes —
  that's normal cold start, not a hang.

## Run it via the CLI (`02_cli/`) — do the steps in order

Step 2 needs the model that step 1 (or the fan-out) registers, so **train first.**

```bash
# macOS: the COPYFILE_DISABLE=1 prefix is REQUIRED — it keeps macOS ._* files out of the
# uploaded code snapshot (otherwise the job dies immediately). Harmless on Linux.

# 1) Train on one A10 GPU (device="cuda") → MLflow → register to Unity Catalog  (~5 min incl. GPU wait)
COPYFILE_DISABLE=1 air run --file 02_cli/train_singlegpu.yaml --watch --profile air

#    OPTIONAL scale-out (alternative to step 1): search hyperparameters across N cheap single-A10
#    jobs (fan-out), then promote the global best to @champion. Control-plane orchestrator — runs
#    locally, needs no GPU and no mlflow; it submits N `air run` jobs of 02_cli/hpo_worker.py.
NUM_WORKERS=2 TRIAL_TOTAL=8 python 02_cli/fanout_hpo.py --profile air

# 2) GPU batch inference over the held-out test set → predictions CSV on the UC Volume
COPYFILE_DISABLE=1 air run --file 02_cli/batch_inference.yaml --watch --profile air
```

Each `air run` ends with `Job status: SUCCESS`; the fan-out prints the global-best version it
promoted to `@champion`. The first GPU job waits a few minutes for capacity — that is normal.
(Note: `air logs` sometimes prints "No logs available" even for successful runs; trust `Job status`
and the MLflow links.)

> **Why fan-out instead of one big multi-GPU job?** HPO is embarrassingly parallel, so spreading
> trials across N cheap single-A10 jobs gives wall-clock parallelism at A10 cost. At this dataset's
> scale a trial takes seconds while GPU startup takes minutes, so for a quick demo a single worker
> (`NUM_WORKERS=1`) is fine; fan-out pays off as trials get longer / more numerous. You can also run
> one worker directly: `COPYFILE_DISABLE=1 air run --file 02_cli/hpo_worker.yaml --watch --profile air`.

> **Model Serving (step 3) is notebook-only.** Deploying a real-time endpoint is a control-plane
> step (not an AI Runtime GPU job), so it ships only as the `01_notebook/03_serve.py` notebook —
> run that after training. There is no CLI serving script.

## Configuration (env vars, with defaults)

| Variable | Default | Meaning |
|----------|---------|---------|
| `UC_CATALOG` / `UC_SCHEMA` | `main` / `air_samples` | Unity Catalog target (matches Setup; change only to use another catalog) |
| `REGISTERED_MODEL_NAME` | `xgboost_classification` | UC registered model name |
| `TEST_SIZE` | `0.2` | Held-out test fraction of the CoverType dataset |
| `MAX_SAMPLES` | `-1` | `-1` = all 581k rows; set a small number for a quick smoke test |
| `RANDOM_STATE` | `42` | Split seed — must match between training (01 / HPO) and 02 (batch inference) |
| `NUM_TRIALS` / `TRIAL_TOTAL` / `NUM_WORKERS` (HPO only) | `16` / `8` / `2` | Trials per worker; total trials fanned out; number of single-A10 jobs |
| `N_ESTIMATORS`, `MAX_DEPTH`, `LEARNING_RATE` | see scripts | XGBoost hyper-parameters (the HPO worker samples these per trial) |
| `TREE_METHOD` / `DEVICE` (01) | `hist` / `cuda` | XGBoost 2.x GPU switch is `device="cuda"`; auto-falls back to CPU |

> **Keep `TEST_SIZE` and `RANDOM_STATE` the same between training (01 / the HPO worker) and 02** —
> step 2 re-downloads Forest CoverType and regenerates the identical held-out split, so mismatched
> values would score out-of-distribution data. Both default to the same values, so the defaults just work.

Override per run by prefixing the YAML `command:` line, e.g.
`command: MAX_SAMPLES=50000 python $CODE_SOURCE_PATH/02_cli/01_train_singlegpu.py`.

## Troubleshooting

| Symptom | Cause & fix |
|---------|-------------|
| Job dies in seconds, `cd: .../._xxx: Not a directory` | macOS AppleDouble files — always run with `COPYFILE_DISABLE=1` (see above). |
| `air run` → `tar: Option --anchored is not supported` (macOS) | Recent `air` versions package the snapshot with **GNU tar**, and macOS's built-in bsdtar lacks `--anchored`. Either `brew install gnu-tar` and prepend `$(brew --prefix gnu-tar)/libexec/gnubin` to `PATH` (Setup step c), **or run the CLI from Linux/WSL** (GNU tar is the default there). CLI-only — the notebook path is unaffected. |
| `RESOURCE_DOES_NOT_EXIST` / schema or volume not found | Run the Setup step (d); make sure `UC_CATALOG`/`UC_SCHEMA` match what you created. |
| Step 2 accuracy looks off | The training run and 02 used different `TEST_SIZE`/`RANDOM_STATE` → different held-out split. Keep them equal (defaults do). |
| Job fails downloading the dataset | The GPU node needs internet egress for `fetch_covtype`. In a locked-down workspace, pre-stage the data in a UC Volume and load from there. |
| Step 02 log shows `spark-class ... ClassNotFoundException` / `dbconnect` errors | Harmless. AI Runtime GPU nodes have no Spark; these lines come from the runtime's Spark probe during MLflow logging (not from the demo code) and are safe to ignore — 02 writes a CSV to the UC Volume. |
| `air logs` says "No logs available" | Known quirk; the run may still have succeeded. Check `Job status` and the MLflow run link. |
| Long "waiting for GPU capacity" | Occasional even for A10; retry later, or run the fan-out with fewer workers (`NUM_WORKERS=1`). |
| Fan-out reports `N/N workers failed` | Check the per-worker logs printed above; a common cause is a worker `air run` that itself failed (e.g. missing catalog CREATE) — fix that, then re-run `fanout_hpo.py`. |

## Repo layout

```
air-sample-xgboost/
├── 01_notebook/               # Databricks notebooks — import + step through
│   ├── 01_train_singlegpu.py
│   ├── 02_batch_inference.py
│   └── 03_serve.py            # (optional) real-time CPU Model Serving — notebook only
├── 02_cli/                    # AI Runtime CLI — plain scripts + workload YAMLs (air run)
│   ├── 01_train_singlegpu.py
│   ├── 02_batch_inference.py
│   ├── fanout_hpo.py          # optional scale-out: control-plane orchestrator — fan HPO across N single-A10 jobs
│   ├── hpo_worker.py          # the per-job HPO worker fanout_hpo.py submits (single A10)
│   ├── train_singlegpu.yaml
│   ├── hpo_worker.yaml
│   └── batch_inference.yaml
├── 03_docs/                   # architecture walkthrough & AI Runtime notes
│   └── architecture.md
├── setup.sh                   # one-time UC schema + volume creation
├── requirements.txt
└── README.md
```

See [`03_docs/architecture.md`](03_docs/architecture.md) for the architecture walkthrough and AI Runtime specifics.

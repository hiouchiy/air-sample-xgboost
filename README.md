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

- **Databricks CLI** authenticated to the workspace (this repo was validated on
  `e2-demo-field-eng`, profile `DEFAULT`).
- **AI Runtime CLI**: `uv tool install --force databricks-air --python 3.12` (reuses your Databricks profile).
- A **Unity Catalog schema** for the registered model and outputs. Default:
  `hiroshi.air_samples` — override via env vars (see below).

## Quickstart (CLI)

```bash
# macOS: COPYFILE_DISABLE=1 keeps AppleDouble (._*) files out of the code snapshot.

# 1) Train on one A10 with GPU acceleration, log to MLflow, register to Unity Catalog
COPYFILE_DISABLE=1 air run --file air/train_singlegpu.yaml --watch --profile DEFAULT

# 2) Parallel hyperparameter search across 8× H100 (one trial per GPU); registers the best model
COPYFILE_DISABLE=1 air run --file air/train_multigpu.yaml --watch --profile DEFAULT

# 3) GPU batch inference over the synthetic test set, written to a UC Delta table or Volume
COPYFILE_DISABLE=1 air run --file air/batch_inference.yaml --watch --profile DEFAULT
```

## Quickstart (notebook)

Import any `src/*.py` into the workspace, attach it to AI Runtime, and **Run All**. Start with
`01_train_singlegpu.py`, then `03_batch_inference.py`.

## Configuration (env vars, with defaults)

| Variable | Default | Meaning |
|----------|---------|---------|
| `NUM_TRAIN_SAMPLES` / `NUM_TEST_SAMPLES` | `500000` / `100000` (single-GPU); `2000000` / `500000` (multi-GPU) | Dataset size; reduce for smoke tests |
| `NUM_FEATURES` | `100` | Dimensionality of synthetic data |
| `NUM_CLASSES` | `2` | Binary or multi-class classification |
| `N_ESTIMATORS`, `MAX_DEPTH`, `LEARNING_RATE` | `100`/`300`, `7`, `0.1` | XGBoost hyperparameters (02 samples depth/lr/… per trial) |
| `NUM_TRIALS` (02 only) | `16` | Hyperparameter trials; dispatched one-per-GPU across the node |
| `UC_CATALOG` / `UC_SCHEMA` | `hiroshi` / `air_samples` | Unity Catalog target |
| `REGISTERED_MODEL_NAME` | `xgboost_classification` | UC registered model name |
| `TREE_METHOD` / `DEVICE` (01) | `hist` / `cuda` | XGBoost 2.x GPU switch is `device="cuda"`; auto-falls back to CPU |

Override from the CLI by editing the YAML `command:` line, e.g.
`command: NUM_TRAIN_SAMPLES=100000 python $CODE_SOURCE_PATH/src/01_train_singlegpu.py`.

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
├── requirements.txt
└── README.md
```

See [`docs/`](docs/) for the architecture walkthrough and AI Runtime specifics.

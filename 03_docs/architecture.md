# Architecture & AI Runtime notes

## End-to-end flow

```
   Forest CoverType ──► AI Runtime GPU node (serverless)
   (sklearn download)  │
                       │  01 single-GPU (A10)      02 multi-GPU (8×H100)
                       │  device="cuda" hist       parallel HPO: 1 trial per GPU
                       │        │                        │  pick best by AUC
                       │        ▼                        ▼
                       │  MLflow run (params, metrics, model + register)
                       └────────┼───────────────────────────────────────
                                ▼
                Unity Catalog registered model
                main.air_samples.xgboost_classification  (promoted to @champion)
                                │
                 03 GPU batch inference (AI Runtime GPU)
                 → predictions CSV on a UC Volume
```

## Model & data

- **XGBoost** — the mainstream gradient-boosted decision tree library for tabular data. On XGBoost
  2.x, GPU training is selected with `tree_method="hist"` + `device="cuda"` (the old `gpu_hist`
  tree method and `gpu_id` are deprecated).
- **Forest CoverType** (`sklearn.datasets.fetch_covtype`) — a public tabular dataset of 581,012 rows
  × 54 numeric features (10 continuous cartographic measures + 44 binary indicators) and **7
  forest-cover-type classes**. It downloads once (~11 MB, cached under `~/scikit-learn_data`) and
  needs **no feature engineering** — every column is already numeric. Source labels are 1..7; we
  shift them to 0..6 for XGBoost's multi-class objective. The train/test split is deterministic
  (fixed `TEST_SIZE` + `RANDOM_STATE`), so batch inference (`03`) regenerates the identical held-out
  split. Swap in your own UC table for a real workload.

## The two GPU modes for XGBoost — and which this repo uses

XGBoost has two distinct ways to use more than one GPU. They solve different problems:

1. **Task-parallel — parallel hyperparameter search (this repo's `02`).** Independent models are
   trained concurrently, **one trial per GPU**. XGBoost releases the GIL during `train`, so a plain
   `ThreadPoolExecutor` (one worker per GPU, each with `device="cuda:<i>"`) runs the trials in true
   parallel — no cluster framework. This is the most common real reason to point 8 GPUs at a
   classic-ML job, and it runs on the **stock AI Runtime environment**. Validated: 8 trials across
   `cuda:0…cuda:7` finished in ~5 s wall time (vs ~24 s summed) — clear parallel speedup.

2. **Data-parallel — one model, data split across GPUs (`xgboost.dask` + Dask-CUDA).** Used only
   when a dataset is **too large for one GPU's memory**. A single H100 has 80 GB and trains most
   tabular data faster without the per-round AllReduce overhead, so this is a niche. **Note for AI
   Runtime:** `dask_cuda.LocalCUDACluster` hangs at worker startup on the stock AIR environment; to
   use it you need a **custom RAPIDS Docker image** (`air register image`). Multi-node scale-out for
   XGBoost is typically done with the **Spark connector** (`xgboost.spark`) on a lakehouse instead.

> Honesty for the customer: XGBoost multi-GPU is not "always on" like DL data parallelism. Default
> to a single GPU; reach for parallel HPO to use many GPUs, or Dask/Spark data-parallel only when the
> data genuinely exceeds one GPU.

## Notebook and CLI forms

Each step exists as two files that share the same logic:

- **`01_notebook/*.py`** — Databricks notebook source (`# Databricks notebook source`,
  `# COMMAND ----------`, `# MAGIC %pip`/`%md`). Imported into the workspace these become real cells
  (the `%pip` cells install deps) and *Run All* executes it.
- **`02_cli/*.py`** — the same logic as a plain Python script (markers stripped), submitted with
  `air run`; deps come from the workload YAML. Each pairs with a `02_cli/*.yaml` spec.

All config is environment variables with defaults, so neither form needs editing. Note that **none of
`01`/`02`/`03` uses `torchrun` or `serverless_gpu`** — the parallel-HPO thread pool and the
single-GPU trainer both run in one process, so the notebook and CLI files differ only by the
notebook markers.

## AI Runtime operational notes

1. **Environment version 4 preinstalls** Python 3.12, torch 2.7.1+cu126, mlflow, scikit-learn, and
   serverless_gpu. It does **not** include `xgboost` — the YAML/`%pip` add it (pinned `>=2.1,<3`).
2. **Model logging + UC registration works with the standard API**:
   `mlflow.xgboost.log_model(..., registered_model_name="main.air_samples.xgboost_classification")`
   inside a run. A model **signature** (via `infer_signature`) is required for UC registration.
   01/02 also promote the new version to the `@champion` alias, which 03 loads.
   **Egress caveat:** logging uploads the model artifacts to the workspace artifact store
   (`*.storage.cloud.databricks.com`). On egress-restricted or cross-region-capacity workspaces the
   GPU node may not reach it — the upload fails with `Connection refused` (seen on some `GPU_8xH100`
   capacity while `GPU_1xA10` in the same workspace succeeded). Workaround: `mlflow.xgboost.save_model()`
   to a UC Volume the node can reach, then register from a control-plane context. Validate on your
   target workspace first.
3. **The dataset downloads at job start** via `fetch_covtype` — the GPU node needs egress to the
   scikit-learn data host. In a locked-down workspace, pre-stage the
   data in a UC Volume and point the loader at it instead.
4. **macOS submitters:** prefix `air run` with `COPYFILE_DISABLE=1` to keep AppleDouble `._*` files
   out of the code snapshot.
5. **`air logs` may report "No logs available"** even for successful runs; `air run --watch`
   streams execution logs live. For debugging, write to a UC Volume.
6. **No Spark on AI Runtime GPU nodes** — so batch inference (`03`) writes its predictions directly
   to a CSV on a UC Volume (`/Volumes/<catalog>/air_samples/predictions/`); it never invokes Spark.

## Files

| File | Role |
|------|------|
| `01_train_singlegpu.py` (+ `02_cli/train_singlegpu.yaml`) | Single-GPU (A10) training, `device="cuda"` → MLflow → UC |
| `02_train_multigpu.py` (+ `02_cli/train_multigpu.yaml`) | 8×H100 parallel hyperparameter search (1 trial/GPU) → best model → UC |
| `03_batch_inference.py` (+ `02_cli/batch_inference.yaml`) | GPU batch inference from the UC model → predictions CSV on a UC Volume |
| `04_serve.py` *(optional)* | Deploy the `@champion` model to a **CPU** Model Serving endpoint + query it (control-plane; no GPU) |
| `02_cli/fanout_hpo.py` *(optional)* | Control-plane orchestrator: parallel HPO across N single-A10 jobs → promotes the best to `@champion` |

Each of the above exists in both `01_notebook/` (Run All) and `02_cli/` (`air run`) form.

## References

- [XGBoost GPU support (`device`)](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [XGBoost Dask (data-parallel)](https://xgboost.readthedocs.io/en/stable/python/dask.html)
- [xgboost.spark (multi-node)](https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html)

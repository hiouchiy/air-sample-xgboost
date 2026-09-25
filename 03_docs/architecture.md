# Architecture & AI Runtime notes

## End-to-end flow

```
   Forest CoverType ──► AI Runtime GPU node(s) (serverless)
   (sklearn download)  │
                       │  01 single-GPU (A10)      02 fan-out HPO (N × A10)
                       │  device="cuda" hist       N single-A10 jobs, each a
                       │        │                  trial shard → pick global best
                       │        ▼                        ▼
                       │  MLflow run (params, metrics, model + register)
                       └────────┼───────────────────────────────────────
                                ▼
                Unity Catalog registered model
                main.air_samples.xgboost_classification  (promoted to @champion)
                                │
                 03 GPU batch inference (AI Runtime GPU)
                 → predictions CSV on a UC Volume

   (02 fan-out is a local control-plane orchestrator, fanout_hpo.py, that submits the N jobs.)
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

## How this repo scales XGBoost — and why not 8×H100

XGBoost on a CoverType-scale dataset trains in **seconds on a single A10**, so a large multi-GPU box
is the wrong tool. This repo therefore scales the way classic ML actually scales in practice:

1. **Single-GPU training (`01`).** One A10, `device="cuda"`. This is the baseline and, for many real
   tabular jobs, all you need.

2. **Horizontal HPO fan-out (`02` — `fanout_hpo.py`).** HPO is embarrassingly parallel, so the
   realistic, cost-appropriate way to speed it up is to spread trials across **N cheap single-A10
   jobs** (each runs a shard of the same seeded grid via `hpo_worker.py`), then promote the global
   best to `@champion`. This is a **control-plane** orchestrator (submits N `air run` jobs; no GPU,
   no mlflow locally) and runs on the **stock AI Runtime environment**. Validated: 2 workers → best
   model promoted to `@champion`.

**Why not the on-node multi-GPU options?**
- **On-node multi-GPU (`GPU_8xH100`).** AIR's only multi-GPU node is 8×H100. You *could* run one HPO
  trial per GPU on it (XGBoost releases the GIL, so a `ThreadPoolExecutor` parallelizes trials), but
  for a seconds-long fit that's expensive and hard to justify — the honest signal to a customer is
  "use cheap A10 fan-out, not an H100 box." (The BERT sample *does* use 8×H100, because transformer
  DDP is genuine distributed training — different workload, different strategy.)
- **Data-parallel (`xgboost.dask` + Dask-CUDA).** Only for datasets **too large for one GPU's
  memory** (a single H100 has 80 GB and trains most tabular data faster without per-round AllReduce).
  On AIR, `dask_cuda.LocalCUDACluster` hangs at worker startup on the stock env (needs a custom
  RAPIDS image); multi-node XGBoost is usually done with the **Spark connector** (`xgboost.spark`).

> Honesty for the customer: match the scaling strategy to the workload. For classic-ML HPO that's
> horizontal fan-out across cheap nodes — not reaching for the biggest GPU box.

## Notebook and CLI forms

Each step exists as two files that share the same logic:

- **`01_notebook/*.py`** — Databricks notebook source (`# Databricks notebook source`,
  `# COMMAND ----------`, `# MAGIC %pip`/`%md`). Imported into the workspace these become real cells
  (the `%pip` cells install deps) and *Run All* executes it.
- **`02_cli/*.py`** — the same logic as a plain Python script (markers stripped), submitted with
  `air run`; deps come from the workload YAML. Each pairs with a `02_cli/*.yaml` spec.

All config is environment variables with defaults, so neither form needs editing. Note that **none of
the GPU scripts uses `torchrun` or `serverless_gpu`** — the single-GPU trainer and each HPO worker
run in one process, so the notebook and CLI files differ only by the notebook markers. The HPO
fan-out is the exception: it's a control-plane orchestrator (`02_cli/fanout_hpo.py`) with **no
notebook form**, because scaling HPO across nodes means submitting multiple `air run` jobs.

## AI Runtime operational notes

1. **Environment version 4 preinstalls** Python 3.12, torch 2.7.1+cu126, mlflow, scikit-learn, and
   serverless_gpu. It does **not** include `xgboost` — the YAML/`%pip` add it (pinned `>=2.1,<3`).
2. **Model logging + UC registration works with the standard API**:
   `mlflow.xgboost.log_model(..., registered_model_name="main.air_samples.xgboost_classification")`
   inside a run. A model **signature** (via `infer_signature`) is required for UC registration.
   Single-GPU training (01) promotes its version to the `@champion` alias and the fan-out promotes
   the global-best version; batch inference (02) loads `@champion`.
   **Egress caveat:** logging uploads the model artifacts to the workspace artifact store
   (`*.storage.cloud.databricks.com`). On egress-restricted or cross-region-capacity workspaces the
   GPU node may not reach it — the upload fails with `Connection refused`. Workaround:
   `mlflow.xgboost.save_model()` to a UC Volume the node can reach, then register from a
   control-plane context. Validate on your target workspace first.
3. **The dataset downloads at job start** via `fetch_covtype` — the GPU node needs egress to the
   scikit-learn data host. In a locked-down workspace, pre-stage the
   data in a UC Volume and point the loader at it instead.
4. **macOS submitters:** prefix `air run` with `COPYFILE_DISABLE=1` to keep AppleDouble `._*` files
   out of the code snapshot.
5. **`air logs` may report "No logs available"** even for successful runs; `air run --watch`
   streams execution logs live. For debugging, write to a UC Volume.
6. **No Spark on AI Runtime GPU nodes** — so batch inference (`02`) writes its predictions directly
   to a CSV on a UC Volume (`/Volumes/<catalog>/air_samples/predictions/`); it never invokes Spark.

## Files

| File | Role |
|------|------|
| `01_train_singlegpu.py` (+ `02_cli/train_singlegpu.yaml`) | Single-GPU (A10) training, `device="cuda"` → MLflow → UC. Notebook + CLI. |
| `02_batch_inference.py` (+ `02_cli/batch_inference.yaml`) | GPU batch inference from the UC model → predictions CSV on a UC Volume. Notebook + CLI. |
| `03_serve.py` *(optional)* | Deploy the `@champion` model to a **CPU** Model Serving endpoint + query it (control-plane; no GPU). **Notebook-only.** |
| `02_cli/fanout_hpo.py` *(optional scale-out)* | Control-plane orchestrator: fan HPO across N single-A10 jobs → promote the global best to `@champion`. An alternative to step 1. **CLI-only** (no notebook). |
| `02_cli/hpo_worker.py` (+ `02_cli/hpo_worker.yaml`) | The per-job HPO worker `fanout_hpo.py` submits: a trial shard on one A10 → best-of-shard → UC. Also runnable standalone. CLI-only. |

Steps 1 and 2 ship in both `01_notebook/` (Run All) and `02_cli/` (`air run`) form; step 3 (serving)
is notebook-only and the HPO fan-out is CLI-only, as noted above.

## References

- [XGBoost GPU support (`device`)](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [XGBoost Dask (data-parallel)](https://xgboost.readthedocs.io/en/stable/python/dask.html)
- [xgboost.spark (multi-node)](https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html)

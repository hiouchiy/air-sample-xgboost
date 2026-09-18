# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost — Multi-GPU parallel hyperparameter search (8×H100)
# MAGIC
# MAGIC This example uses **all 8 H100 GPUs of a single AI Runtime node in parallel** to run a
# MAGIC **hyperparameter search**: many XGBoost models are trained concurrently, **one trial per GPU**,
# MAGIC and the best model (by validation AUC) is registered to Unity Catalog.
# MAGIC
# MAGIC ## Why this pattern for "multi-GPU classic ML"?
# MAGIC For gradient-boosted trees, a single H100 (80 GB) trains most tabular datasets quickly, so the
# MAGIC realistic way to put 8 GPUs to work is **task-parallel hyperparameter tuning** — an
# MAGIC embarrassingly parallel workload where each GPU trains an independent candidate. It needs no
# MAGIC cluster framework: XGBoost releases the GIL during training, so a simple thread pool dispatches
# MAGIC one training per GPU (`device="cuda:<i>"`) and they run truly concurrently.
# MAGIC
# MAGIC (Data-parallel single-model training across GPUs — `xgboost.dask` + Dask-CUDA — is the other
# MAGIC multi-GPU mode, reserved for datasets too large for one GPU; on AI Runtime it needs a custom
# MAGIC RAPIDS image, whereas this parallel-HPO pattern runs on the stock environment.)
# MAGIC
# MAGIC ## Runs two ways, without code changes
# MAGIC 1. **Notebook** — open in the workspace and *Run All* (attach to AI Runtime).
# MAGIC 2. **AI Runtime CLI** — `air run --file air/train_multigpu.yaml --watch --profile DEFAULT`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies (notebook only)

# COMMAND ----------

# MAGIC %pip install -U "xgboost>=2.1,<3" "scikit-learn>=1.3,<2"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    gpu_type: str = _env("GPU_TYPE", "H100")
    # Number of hyperparameter trials. Defaults to 2× the GPU count so every GPU does real work.
    num_trials: int = int(_env("NUM_TRIALS", "16"))

    # Dataset generation --------------------------------------------------
    num_train_samples: int = int(_env("NUM_TRAIN_SAMPLES", "500000"))
    num_test_samples: int = int(_env("NUM_TEST_SAMPLES", "100000"))
    num_features: int = int(_env("NUM_FEATURES", "100"))
    num_classes: int = int(_env("NUM_CLASSES", "2"))
    random_state: int = int(_env("RANDOM_STATE", "42"))

    n_estimators: int = int(_env("N_ESTIMATORS", "300"))

    # Unity Catalog (model registry) -----------------------------------
    uc_catalog: str = _env("UC_CATALOG", "main")
    uc_schema: str = _env("UC_SCHEMA", "air_samples")
    registered_model_name: str = _env("REGISTERED_MODEL_NAME", "xgboost_classification")
    register_model: bool = _env("REGISTER_MODEL", "true").lower() == "true"

    @property
    def uc_model_fqn(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.registered_model_name}"


CFG = Config()
print(CFG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate the (shared) dataset
# MAGIC One dataset is generated once and shared read-only across all trials.

# COMMAND ----------

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def generate_dataset(cfg: Config):
    print(f"Generating {cfg.num_train_samples + cfg.num_test_samples} synthetic samples...")
    X, y = make_classification(
        n_samples=cfg.num_train_samples + cfg.num_test_samples,
        n_features=cfg.num_features,
        n_informative=max(2, cfg.num_features // 2),
        n_redundant=max(0, cfg.num_features // 4),
        n_classes=cfg.num_classes,
        shuffle=False,  # deterministic column order so train/infer distributions match
        random_state=cfg.random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.num_test_samples / (cfg.num_train_samples + cfg.num_test_samples),
        random_state=cfg.random_state,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return (
        X_train.astype(np.float32), X_test.astype(np.float32),
        y_train.astype(np.float32), y_test.astype(np.float32),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sample the hyperparameter grid

# COMMAND ----------

def sample_hyperparameters(cfg: Config):
    """Random search space. Returns a list of `num_trials` parameter dicts."""
    rng = np.random.default_rng(cfg.random_state)
    space = {
        "max_depth": [4, 6, 8, 10, 12],
        "learning_rate": [0.02, 0.05, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    trials = []
    for _ in range(cfg.num_trials):
        trials.append({k: type(v[0])(rng.choice(v)) for k, v in space.items()})
    return trials

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train the trials in parallel — one per GPU
# MAGIC A `ThreadPoolExecutor` with one worker per GPU dispatches trainings. Each trial sets
# MAGIC `device="cuda:<gpu>"`; XGBoost releases the GIL during `train`, so the trials run concurrently
# MAGIC on distinct GPUs. Falls back to CPU when no GPU is present (threads still parallelize).

# COMMAND ----------

import time
from concurrent.futures import ThreadPoolExecutor

import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score


def _train_one_trial(trial_idx, params, gpu_id, cfg, data, use_gpu):
    X_train, X_test, y_train, y_test = data
    device = f"cuda:{gpu_id}" if use_gpu else "cpu"
    dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
    dtest = xgb.QuantileDMatrix(X_test, label=y_test, ref=dtrain)
    full_params = {
        "objective": "binary:logistic" if cfg.num_classes == 2 else "multi:softprob",
        "num_class": cfg.num_classes if cfg.num_classes > 2 else None,
        "eval_metric": "logloss" if cfg.num_classes == 2 else "mlogloss",
        "tree_method": "hist",
        "device": device,
        **params,
    }
    full_params = {k: v for k, v in full_params.items() if v is not None}
    t0 = time.time()
    booster = xgb.train(full_params, dtrain, num_boost_round=cfg.n_estimators,
                        evals=[(dtest, "test")], verbose_eval=False)
    train_time = time.time() - t0
    proba = booster.predict(dtest)
    if cfg.num_classes == 2:
        auc = float(roc_auc_score(y_test, proba))
        acc = float(accuracy_score(y_test, (proba > 0.5).astype(int)))
    else:
        auc = float("nan")
        acc = float(accuracy_score(y_test, np.argmax(proba, axis=-1)))
    print(f"[trial {trial_idx:02d} on {device}] auc={auc:.4f} acc={acc:.4f} "
          f"time={train_time:.1f}s params={params}")
    return {"trial": trial_idx, "gpu": gpu_id, "device": device, "params": params,
            "auc": auc, "accuracy": acc, "train_seconds": train_time, "booster": booster}


def run_hpo(cfg: Config, data):
    import torch

    n_gpu = torch.cuda.device_count()
    use_gpu = n_gpu > 0
    workers = n_gpu if use_gpu else min(cfg.num_trials, os.cpu_count() or 4)
    print(f"Running {cfg.num_trials} trials across {workers} "
          f"{'GPU' if use_gpu else 'CPU'} worker(s)...")
    trials = sample_hyperparameters(cfg)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_train_one_trial, i, p, (i % n_gpu if use_gpu else 0), cfg, data, use_gpu)
            for i, p in enumerate(trials)
        ]
        results = [f.result() for f in futures]
    wall = time.time() - t0
    print(f"HPO wall time for {cfg.num_trials} trials: {wall:.1f}s")
    results.sort(key=lambda r: (r["auc"] if not np.isnan(r["auc"]) else r["accuracy"]), reverse=True)
    return results, wall, n_gpu

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log all trials to MLflow & register the best model to Unity Catalog

# COMMAND ----------

import mlflow


def log_and_register(cfg: Config, results, wall, n_gpu):
    from mlflow.models.signature import infer_signature

    mlflow.set_registry_uri("databricks-uc")
    best = results[0]

    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name="xgboost-hpo-multigpu", nested=nested) as run:
        mlflow.log_params({
            "num_trials": cfg.num_trials,
            "num_gpus": n_gpu,
            "num_train_samples": cfg.num_train_samples,
            "n_estimators": cfg.n_estimators,
            "training_mode": f"parallel-hpo-{n_gpu}x{cfg.gpu_type}",
        })
        mlflow.log_metrics({
            "hpo_wall_seconds": wall,
            "best_auc": best["auc"],
            "best_accuracy": best["accuracy"],
        })
        # Log every trial as a child run for comparison in the MLflow UI.
        for r in results:
            with mlflow.start_run(run_name=f"trial-{r['trial']:02d}", nested=True):
                mlflow.log_params({**r["params"], "gpu": r["gpu"]})
                mlflow.log_metrics({"auc": r["auc"], "accuracy": r["accuracy"],
                                    "train_seconds": r["train_seconds"]})
        # Register the best booster.
        booster = best["booster"]
        booster.set_param({"device": "cpu"})  # portable artifact; serving/infer can re-set cuda.
        example = np.random.randn(2, cfg.num_features).astype(np.float32)
        output = booster.predict(xgb.DMatrix(example))
        signature = infer_signature(model_input=example, model_output=output)
        info = mlflow.xgboost.log_model(
            xgb_model=booster, artifact_path="model", signature=signature,
            input_example=example,
            registered_model_name=cfg.uc_model_fqn if cfg.register_model else None,
        )
        print("Best trial:", {k: best[k] for k in ("trial", "gpu", "auc", "accuracy", "params")})
        print("Logged model:", info.model_uri)
        if cfg.register_model:
            from mlflow.tracking import MlflowClient

            v = info.registered_model_version
            MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(
                cfg.uc_model_fqn, "champion", v)
            print(f"Registered {cfg.uc_model_fqn} version {v} and set alias @champion "
                  f"(this is the version 03 will load).")
        return run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Entry point

# COMMAND ----------

def main():
    import torch

    print(f"CUDA available: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
    data = generate_dataset(CFG)
    results, wall, n_gpu = run_hpo(CFG, data)
    run_id = log_and_register(CFG, results, wall, n_gpu)
    print(f"Done. Best AUC={results[0]['auc']:.4f}. MLflow run_id={run_id}")
    return results[0]


# COMMAND ----------

if __name__ == "__main__":
    main()

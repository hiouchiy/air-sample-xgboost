"""XGBoost multi-GPU parallel hyperparameter search (8xH100) — AI Runtime CLI script.

Runs one XGBoost trial per GPU concurrently (ThreadPoolExecutor; XGBoost releases the GIL), then
registers the best model. Uses whatever GPUs are attached to the node.
    COPYFILE_DISABLE=1 air run --file 02_cli/train_multigpu.yaml --watch --profile <your-profile>
The notebook equivalent is 01_notebook/02_train_multigpu.py.
"""

import os
import logging
from dataclasses import dataclass


def _logmodel_model_kw():
    """Cross-version: MLflow >= 3 takes name=, MLflow 2.x (AIR CLI env) requires artifact_path=."""
    import mlflow
    return {"name": "model"} if int(mlflow.__version__.split(".")[0]) >= 3 else {"artifact_path": "model"}

# Serverless/AI Runtime enforces a py4j method whitelist, so MLflow's optional run-context tag
# lookup logs a benign `Py4JSecurityException ... extraContext ... not whitelisted` warning during
# logging. It's harmless (MLflow skips a couple of optional tags and continues) — quiet just that
# logger so it doesn't look like a failure.
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)
# Serverless also emits benign pyspark-connect / py4j chatter during MLflow logging; quiet it too.
logging.getLogger("pyspark.sql.connect").setLevel(logging.ERROR)
logging.getLogger("py4j").setLevel(logging.ERROR)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    gpu_type: str = _env("GPU_TYPE", "H100")
    # Number of hyperparameter trials. Defaults to 2x the GPU count so every GPU does real work.
    num_trials: int = int(_env("NUM_TRIALS", "16"))

    # Dataset (Forest CoverType — see 01_train_singlegpu.py) ---------------
    test_size: float = float(_env("TEST_SIZE", "0.2"))
    max_samples: int = int(_env("MAX_SAMPLES", "-1"))  # -1 = use all rows
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


import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split


def load_dataset(cfg: Config):
    """Download the Forest CoverType dataset and split it (labels shifted 1..7 -> 0..6)."""
    print("Downloading Forest CoverType (~11 MB, cached under ~/scikit-learn_data)...")
    data = fetch_covtype()
    X = data.data.astype(np.float32)
    y = (data.target - 1).astype(np.int32)

    if 0 < cfg.max_samples < len(X):
        idx = np.sort(np.random.default_rng(cfg.random_state).permutation(len(X))[: cfg.max_samples])
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, classes: {int(y.max()) + 1}")
    return X_train, X_test, y_train, y_test


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


import time
from concurrent.futures import ThreadPoolExecutor

import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score


def _train_one_trial(trial_idx, params, gpu_id, cfg, data, num_class, use_gpu):
    X_train, X_test, y_train, y_test = data
    device = f"cuda:{gpu_id}" if use_gpu else "cpu"
    dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
    dtest = xgb.QuantileDMatrix(X_test, label=y_test, ref=dtrain)
    full_params = {
        "objective": "multi:softprob",
        "num_class": num_class,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": device,
        **params,
    }
    t0 = time.time()
    booster = xgb.train(full_params, dtrain, num_boost_round=cfg.n_estimators,
                        evals=[(dtest, "test")], verbose_eval=False)
    train_time = time.time() - t0
    proba = booster.predict(dtest)
    auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
    acc = float(accuracy_score(y_test, np.argmax(proba, axis=1)))
    print(f"[trial {trial_idx:02d} on {device}] auc={auc:.4f} acc={acc:.4f} "
          f"time={train_time:.1f}s params={params}")
    return {"trial": trial_idx, "gpu": gpu_id, "device": device, "params": params,
            "auc": auc, "accuracy": acc, "train_seconds": train_time, "booster": booster}


def run_hpo(cfg: Config, data, num_class):
    import torch

    n_gpu = torch.cuda.device_count()
    use_gpu = n_gpu > 0
    workers = n_gpu if use_gpu else min(cfg.num_trials, os.cpu_count() or 4)
    print(f"Running {cfg.num_trials} trials across {workers} "
          f"{'GPU' if use_gpu else 'CPU'} worker(s)...")
    trials = sample_hyperparameters(cfg)

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_train_one_trial, i, p, (i % n_gpu if use_gpu else 0),
                        cfg, data, num_class, use_gpu): i
            for i, p in enumerate(trials)
        }
        for fut, i in futures.items():
            try:
                results.append(fut.result())
            except Exception as exc:  # isolate trials: one bad trial must not discard the rest
                print(f"[trial {i:02d}] FAILED: {exc}")
    wall = time.time() - t0
    if not results:
        raise RuntimeError("All HPO trials failed - see the per-trial errors above.")
    print(f"HPO: {len(results)}/{cfg.num_trials} trials succeeded in {wall:.1f}s")
    results.sort(key=lambda r: r["auc"], reverse=True)
    return results, wall, n_gpu


import mlflow


def log_and_register(cfg: Config, results, wall, n_gpu, X_sample):
    from mlflow.models.signature import infer_signature

    mlflow.set_registry_uri("databricks-uc")
    best = results[0]

    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name="xgboost-hpo-multigpu", nested=nested) as run:
        mlflow.log_params({
            "dataset": "sklearn.fetch_covtype",
            "num_trials": cfg.num_trials,
            "num_gpus": n_gpu,
            "n_estimators": cfg.n_estimators,
            "training_mode": f"parallel-hpo-{n_gpu}gpu",
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
        # Use real rows (like 01) so the logged signature/input example match production inputs.
        example = X_sample.astype(np.float32)
        output = booster.predict(xgb.DMatrix(example))
        signature = infer_signature(model_input=example, model_output=output)
        info = mlflow.xgboost.log_model(
            xgb_model=booster, **_logmodel_model_kw(), signature=signature,
            input_example=example,
            registered_model_name=cfg.uc_model_fqn if cfg.register_model else None,
        )
        print("Best trial:", {k: best[k] for k in ("trial", "gpu", "auc", "accuracy", "params")})
        print("Logged model:", info.model_uri)
        if cfg.register_model:
            from mlflow.tracking import MlflowClient

            v = info.registered_model_version
            MlflowClient().set_registered_model_alias(
                cfg.uc_model_fqn, "champion", v)
            print(f"Registered {cfg.uc_model_fqn} version {v} and set alias @champion "
                  f"(this is the version 03 will load).")
        return run.info.run_id


def main():
    import torch

    print(f"CUDA available: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
    X_train, X_test, y_train, y_test = load_dataset(CFG)
    data = (X_train, X_test, y_train, y_test)
    num_class = int(y_train.max()) + 1
    results, wall, n_gpu = run_hpo(CFG, data, num_class)
    run_id = log_and_register(CFG, results, wall, n_gpu, X_train[:5])
    print(f"Done. Best AUC={results[0]['auc']:.4f}. MLflow run_id={run_id}")
    return results[0]


if __name__ == "__main__":
    main()

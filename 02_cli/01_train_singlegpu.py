"""XGBoost single-GPU training on Forest CoverType — AI Runtime CLI script.

Plain Python (no notebook markers). Submit as an AI Runtime job:
    COPYFILE_DISABLE=1 air run --file 02_cli/train_singlegpu.yaml --watch --profile <your-profile>
GPU training uses tree_method="hist" + device="cuda". Config is via env vars; deps come from the
YAML. The notebook-optimized equivalent is 01_notebook/01_train_singlegpu.py.
"""

import os
import logging
from dataclasses import dataclass

# Serverless/AI Runtime enforces a py4j method whitelist, so MLflow's optional run-context tag
# lookup logs a benign `Py4JSecurityException ... extraContext ... not whitelisted` warning during
# logging. It's harmless (MLflow skips a couple of optional tags and continues) — quiet just that
# logger so it doesn't look like a failure.
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)


def _env(name: str, default: str) -> str:
    """Read an env var, falling back to a default. Keeps notebook and CLI in sync."""
    return os.environ.get(name, default)


@dataclass
class Config:
    # Dataset --------------------------------------------------------------
    # Forest CoverType (sklearn): 581,012 rows x 54 numeric features, 7 forest-cover-type classes.
    # Downloaded on first use and cached under ~/scikit-learn_data. Every column is already numeric
    # (10 continuous cartographic measures + 44 binary indicators), so no feature engineering is
    # needed. Set MAX_SAMPLES to a small number for a quick smoke test.
    test_size: float = float(_env("TEST_SIZE", "0.2"))
    max_samples: int = int(_env("MAX_SAMPLES", "-1"))  # -1 = use all rows
    random_state: int = int(_env("RANDOM_STATE", "42"))

    # XGBoost hyperparameters ------------------------------------
    n_estimators: int = int(_env("N_ESTIMATORS", "200"))
    max_depth: int = int(_env("MAX_DEPTH", "8"))
    learning_rate: float = float(_env("LEARNING_RATE", "0.1"))
    subsample: float = float(_env("SUBSAMPLE", "0.8"))
    colsample_bytree: float = float(_env("COLSAMPLE_BYTREE", "0.8"))

    # GPU configuration -----------------------------------------------
    # XGBoost 2.x selects the GPU via device="cuda" (the old "gpu_hist" tree method and
    # "gpu_id" are deprecated). tree_method stays "hist"; the device does the GPU switch.
    # main() downgrades device to "cpu" automatically if no CUDA device is present.
    tree_method: str = _env("TREE_METHOD", "hist")
    device: str = _env("DEVICE", "cuda")

    # Unity Catalog (model registry) -----------------------------------
    uc_catalog: str = _env("UC_CATALOG", "main")
    uc_schema: str = _env("UC_SCHEMA", "air_samples")
    registered_model_name: str = _env("REGISTERED_MODEL_NAME", "xgboost_classification")
    register_model: bool = _env("REGISTER_MODEL", "true").lower() == "true"

    # Output ----------------------------------------------------------
    output_dir: str = _env("OUTPUT_DIR", "/tmp/xgboost_model")

    @property
    def uc_model_fqn(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.registered_model_name}"


CFG = Config()
print(CFG)


import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split


def load_dataset(cfg: Config):
    """Download the Forest CoverType dataset and split it into train/test.

    The source labels are 1..7; XGBoost multi-class expects 0..6, so we shift them by one. The
    split is deterministic (fixed test_size + random_state), so batch inference (03) regenerates
    the identical held-out split.
    """
    print("Downloading Forest CoverType (~11 MB, cached under ~/scikit-learn_data)...")
    data = fetch_covtype()
    X = data.data.astype(np.float32)
    y = (data.target - 1).astype(np.int32)  # 1..7 -> 0..6

    if 0 < cfg.max_samples < len(X):
        # Deterministic, class-covering subsample for smoke tests (same subset in 01 and 03).
        idx = np.sort(np.random.default_rng(cfg.random_state).permutation(len(X))[: cfg.max_samples])
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, classes: {int(y.max()) + 1}")
    return X_train, X_test, y_train, y_test


import xgboost as xgb


def train_xgboost(cfg: Config, X_train, X_test, y_train, y_test):
    """Train an XGBoost multi-class classifier on GPU (or CPU if no GPU is present)."""
    import time
    import torch

    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()} -> training on device={device}")

    num_class = int(y_train.max()) + 1

    # QuantileDMatrix is the memory-efficient structure for GPU "hist"; it builds the
    # histogram bins directly on the device.
    dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
    dtest = xgb.QuantileDMatrix(X_test, label=y_test, ref=dtrain)

    params = {
        "objective": "multi:softprob",  # returns a per-class probability matrix (n_rows, num_class)
        "num_class": num_class,
        "eval_metric": "mlogloss",
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "tree_method": cfg.tree_method,
        "device": device,  # "cuda" runs the whole boosting on the GPU (XGBoost 2.x API).
    }

    evals = [(dtrain, "train"), (dtest, "test")]
    evals_result = {}

    print(f"Training with params: {params}")
    t0 = time.time()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.n_estimators,
        evals=evals,
        evals_result=evals_result,
        verbose_eval=20,
    )
    train_time = time.time() - t0

    print(f"Training completed in {train_time:.1f}s")
    return booster, num_class, train_time


from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def evaluate_xgboost(booster, X_test, y_test):
    """Evaluate the trained multi-class model (accuracy, macro one-vs-rest AUC, log loss)."""
    proba = booster.predict(xgb.DMatrix(X_test))  # shape (n_rows, num_class)
    preds = np.argmax(proba, axis=1)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc_ovr_macro": float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro")),
        "log_loss": float(log_loss(y_test, proba)),
    }


import mlflow


def log_and_register(cfg: Config, booster, num_class, metrics, X_train):
    """Log the model to MLflow and register to Unity Catalog."""
    from mlflow.models.signature import infer_signature

    mlflow.set_registry_uri("databricks-uc")

    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name="xgboost-classification-singlegpu", nested=nested) as run:
        mlflow.log_params(
            {
                "dataset": "sklearn.fetch_covtype",
                "num_features": X_train.shape[1],
                "num_classes": num_class,
                "n_estimators": cfg.n_estimators,
                "max_depth": cfg.max_depth,
                "learning_rate": cfg.learning_rate,
                "tree_method": cfg.tree_method,
                "training_mode": "single-gpu",
            }
        )
        mlflow.log_metrics(metrics)

        # A real sample row makes the logged signature/input example match production inputs.
        input_example = X_train[:5]
        output_example = booster.predict(xgb.DMatrix(input_example))
        signature = infer_signature(model_input=input_example, model_output=output_example)

        model_info = mlflow.xgboost.log_model(
            xgb_model=booster,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=cfg.uc_model_fqn if cfg.register_model else None,
        )
        print("Logged model:", model_info.model_uri)
        if cfg.register_model:
            _promote_to_champion(cfg, model_info)
        return run.info.run_id


def _promote_to_champion(cfg: Config, model_info):
    """Tag the just-registered version with the @champion alias — this is what batch inference
    (03) loads by default, so version promotion is explicit and governed."""
    from mlflow.tracking import MlflowClient

    version = model_info.registered_model_version
    client = MlflowClient(registry_uri="databricks-uc")
    client.set_registered_model_alias(cfg.uc_model_fqn, "champion", version)
    print(f"Registered {cfg.uc_model_fqn} as version {version} and set alias @champion "
          f"(this is the version 03 will load).")


def main():
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    X_train, X_test, y_train, y_test = load_dataset(CFG)
    booster, num_class, train_time = train_xgboost(CFG, X_train, X_test, y_train, y_test)
    metrics = evaluate_xgboost(booster, X_test, y_test)
    metrics["train_seconds"] = train_time
    print(f"Evaluation metrics: {metrics}")
    run_id = log_and_register(CFG, booster, num_class, metrics, X_train)
    print(f"Done. MLflow run_id={run_id}")
    return metrics


if __name__ == "__main__":
    main()

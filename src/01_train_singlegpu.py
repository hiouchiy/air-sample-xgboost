# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost — Single-GPU training on synthetic classification (Databricks AI Runtime)
# MAGIC
# MAGIC This example trains an **XGBoost classifier** on a large synthetic classification dataset
# MAGIC using a **single GPU** on Databricks AI Runtime. It demonstrates **GPU-accelerated gradient
# MAGIC boosting** (up to 20× faster than CPU), tracks the run with **MLflow**, and registers the
# MAGIC trained model to **Unity Catalog** for batch inference and serving.
# MAGIC
# MAGIC ## Why XGBoost on GPU?
# MAGIC XGBoost on GPU (`tree_method="hist"` + `device="cuda"`, the XGBoost 2.x API) delivers a large
# MAGIC speedup over CPU for big datasets (100k+ rows). This example uses **500k rows × 100 features**
# MAGIC to make the GPU speedup tangible. The model is a production-ready gradient-boosted classifier;
# MAGIC step 02 shows genuine multi-GPU data-parallel training for datasets too large for one GPU.
# MAGIC
# MAGIC ## This notebook runs two ways, **without any code changes**
# MAGIC 1. **As a Databricks notebook** — open it in the workspace and *Run All*. The
# MAGIC    `# MAGIC %pip` cells below install dependencies in the notebook only.
# MAGIC 2. **As an AI Runtime CLI job** — `air run --file air/train_singlegpu.yaml`.
# MAGIC    The `# MAGIC` lines are plain Python comments and are ignored; dependencies come
# MAGIC    from the YAML `environment.dependencies` instead.
# MAGIC
# MAGIC All behaviour is controlled by environment variables (see the `Config` cell), so the
# MAGIC exact same file is portable across both execution modes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies (notebook only)
# MAGIC These `%pip`/`%restart_python` magics run **only** when the file is opened as a
# MAGIC notebook. Under the AI Runtime CLI the dependencies are declared in the workload YAML.

# COMMAND ----------

# MAGIC %pip install -U "xgboost>=2.1,<3" "scikit-learn>=1.3,<2"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC Every knob is an environment variable with a sensible default, so the notebook and the
# MAGIC CLI job behave identically. Override any value from the CLI with, e.g.
# MAGIC `air run --file air/train_singlegpu.yaml --override 'command=NUM_TRAIN_SAMPLES=1000000 python ...'`
# MAGIC or by editing `environment` variables in the YAML.

# COMMAND ----------

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    """Read an env var, falling back to a default. Keeps notebook and CLI in sync."""
    return os.environ.get(name, default)


@dataclass
class Config:
    # Dataset generation --------------------------------------------------
    # Synthetic dataset: 500k rows × 100 features × 2 classes (binary classification).
    # For smoke tests, reduce NUM_TRAIN_SAMPLES to 10000 and NUM_TEST_SAMPLES to 1000.
    num_train_samples: int = int(_env("NUM_TRAIN_SAMPLES", "500000"))
    num_test_samples: int = int(_env("NUM_TEST_SAMPLES", "100000"))
    num_features: int = int(_env("NUM_FEATURES", "100"))
    num_classes: int = int(_env("NUM_CLASSES", "2"))
    random_state: int = int(_env("RANDOM_STATE", "42"))

    # XGBoost hyperparameters ------------------------------------
    n_estimators: int = int(_env("N_ESTIMATORS", "100"))
    max_depth: int = int(_env("MAX_DEPTH", "7"))
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
    uc_catalog: str = _env("UC_CATALOG", "hiroshi")
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate synthetic dataset
# MAGIC We generate a large balanced binary classification dataset with sklearn. The dataset is
# MAGIC random but reproducible (via random_state). For a production model, replace this with
# MAGIC your own data loaded from a UC table or external source.

# COMMAND ----------

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def generate_dataset(cfg: Config):
    """Generate a synthetic classification dataset."""
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
        X, y, test_size=cfg.num_test_samples / (cfg.num_train_samples + cfg.num_test_samples),
        random_state=cfg.random_state,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train XGBoost on GPU
# MAGIC We train with `tree_method="hist"` + `device="cuda"` (XGBoost 2.x GPU switch), which
# MAGIC delivers a large speedup over CPU for big datasets. `main()` falls back to CPU if no GPU.

# COMMAND ----------

import xgboost as xgb


def train_xgboost(cfg: Config, X_train, X_test, y_train, y_test):
    """Train XGBoost classifier on GPU or CPU."""
    import time
    import torch

    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()} -> training on device={device}")

    # QuantileDMatrix is the memory-efficient structure for GPU "hist"; it builds the
    # histogram bins directly on the device.
    dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
    dtest = xgb.QuantileDMatrix(X_test, label=y_test, ref=dtrain)

    params = {
        "objective": "binary:logistic" if cfg.num_classes == 2 else "multi:softmax",
        "num_class": cfg.num_classes if cfg.num_classes > 2 else None,
        "eval_metric": "logloss" if cfg.num_classes == 2 else "mlogloss",
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "tree_method": cfg.tree_method,
        "device": device,  # "cuda" runs the whole boosting on the GPU (XGBoost 2.x API).
    }
    # Remove None values to avoid XGBoost warnings.
    params = {k: v for k, v in params.items() if v is not None}

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
    print(f"Evals result: {evals_result}")

    return booster, evals_result, train_time

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluate the model
# MAGIC We compute accuracy and AUC (for binary) or multiclass AUC.

# COMMAND ----------

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def evaluate_xgboost(cfg: Config, booster, X_test, y_test):
    """Evaluate the trained model."""
    dtest = xgb.DMatrix(X_test, label=y_test)
    preds_proba = booster.predict(dtest)

    # For binary classification, sklearn expects probabilities for the positive class.
    if cfg.num_classes == 2:
        preds_binary = (preds_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_test, preds_binary)
        auc = roc_auc_score(y_test, preds_proba)
        loss = log_loss(y_test, preds_proba)
        return {
            "accuracy": accuracy,
            "auc": auc,
            "log_loss": loss,
        }
    else:
        # Multi-class: preds_proba is (n_samples,) with class indices.
        preds = np.argmax(preds_proba, axis=-1) if len(preds_proba.shape) > 1 else preds_proba
        accuracy = accuracy_score(y_test, preds)
        return {
            "accuracy": accuracy,
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log to MLflow & register to Unity Catalog
# MAGIC We log the trained XGBoost model with the MLflow `xgboost` flavor and register it
# MAGIC directly to the **Unity Catalog Model Registry**, so it can be loaded for batch
# MAGIC inference and deployed to Model Serving with no extra packaging code. Parameters and
# MAGIC metrics are logged to the same MLflow run.

# COMMAND ----------

import mlflow


def log_and_register(cfg: Config, booster, metrics):
    """Log the model to MLflow and register to Unity Catalog."""
    import numpy as np
    from mlflow.models.signature import infer_signature

    mlflow.set_registry_uri("databricks-uc")

    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name="xgboost-classification-singlegpu", nested=nested) as run:
        mlflow.log_params(
            {
                "num_train_samples": cfg.num_train_samples,
                "num_features": cfg.num_features,
                "n_estimators": cfg.n_estimators,
                "max_depth": cfg.max_depth,
                "learning_rate": cfg.learning_rate,
                "tree_method": cfg.tree_method,
                "training_mode": "single-gpu",
            }
        )
        mlflow.log_metrics(metrics)

        # Create an input example for signature inference (required for UC registration).
        input_example = np.random.randn(1, cfg.num_features).astype(np.float32)
        # Infer signature from input and a sample prediction.
        dmatrix_example = xgb.DMatrix(input_example)
        output_example = booster.predict(dmatrix_example)

        signature = infer_signature(
            model_input=input_example,
            model_output=output_example,
        )

        model_info = mlflow.xgboost.log_model(
            xgb_model=booster,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=cfg.uc_model_fqn if cfg.register_model else None,
        )
        print("Logged model:", model_info.model_uri)
        if cfg.register_model:
            print("Registered to Unity Catalog:", cfg.uc_model_fqn)
        return run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Entry point
# MAGIC `main()` wires the steps together. The single `if __name__ == "__main__"` guard below
# MAGIC fires in **both** modes: Databricks notebooks expose `__name__ == "__main__"`, so
# MAGIC *Run All* triggers it, and the AI Runtime CLI runs the file as a script
# MAGIC (`python .../01_train_singlegpu.py`), which triggers it too — exactly once each way.

# COMMAND ----------

def main():
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    X_train, X_test, y_train, y_test = generate_dataset(CFG)
    booster, evals_result, train_time = train_xgboost(CFG, X_train, X_test, y_train, y_test)
    metrics = evaluate_xgboost(CFG, booster, X_test, y_test)
    metrics["train_seconds"] = train_time
    print(f"Evaluation metrics: {metrics}")
    run_id = log_and_register(CFG, booster, metrics)
    print(f"Done. MLflow run_id={run_id}")
    return metrics


# COMMAND ----------

if __name__ == "__main__":
    main()

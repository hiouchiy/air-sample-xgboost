"""XGBoost single-GPU training on synthetic data — AI Runtime CLI script.

Plain Python (no notebook markers). Submit as an AI Runtime job:
    COPYFILE_DISABLE=1 air run --file 02_cli/train_singlegpu.yaml --watch --profile <your-profile>
GPU training uses tree_method="hist" + device="cuda". Config is via env vars; deps come from the
YAML. The notebook-optimized equivalent is 01_notebook/01_train_singlegpu.py.
"""

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
    X_train, X_test, y_train, y_test = generate_dataset(CFG)
    booster, evals_result, train_time = train_xgboost(CFG, X_train, X_test, y_train, y_test)
    metrics = evaluate_xgboost(CFG, booster, X_test, y_test)
    metrics["train_seconds"] = train_time
    print(f"Evaluation metrics: {metrics}")
    run_id = log_and_register(CFG, booster, metrics)
    print(f"Done. MLflow run_id={run_id}")
    return metrics


if __name__ == "__main__":
    main()

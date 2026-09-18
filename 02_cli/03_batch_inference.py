"""XGBoost GPU batch inference on synthetic data — AI Runtime CLI script.

Loads the @champion model from Unity Catalog, scores the held-out test split on the GPU, and writes
predictions as a CSV to a UC Volume. Requires a model registered by 01/02 first.
    COPYFILE_DISABLE=1 air run --file 02_cli/batch_inference.yaml --watch --profile <your-profile>
The notebook equivalent is 01_notebook/03_batch_inference.py.
"""

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    uc_catalog: str = _env("UC_CATALOG", "main")
    uc_schema: str = _env("UC_SCHEMA", "air_samples")
    registered_model_name: str = _env("REGISTERED_MODEL_NAME", "xgboost_classification")
    # Empty -> use models:/<catalog>.<schema>.<name>@champion or latest version.
    model_uri: str = _env("MODEL_URI", "")

    # Test dataset generation ---
    # To score data drawn from the SAME distribution the model was trained on, we regenerate the
    # identical synthetic dataset (same n_samples total, shuffle=False, random_state) and take the
    # held-out test split — matching 01_train_singlegpu.py exactly. (make_classification's per-
    # cluster transforms depend on the RNG stream, so the total sample count must match training.)
    num_train_samples: int = int(_env("NUM_TRAIN_SAMPLES", "500000"))
    num_test_samples: int = int(_env("NUM_TEST_SAMPLES", "100000"))
    num_features: int = int(_env("NUM_FEATURES", "100"))
    num_classes: int = int(_env("NUM_CLASSES", "2"))
    batch_size: int = int(_env("BATCH_SIZE", "10000"))
    random_state: int = int(_env("RANDOM_STATE", "42"))

    output_name: str = _env("OUTPUT_NAME", "xgboost_classification_predictions")

    @property
    def uc_model_fqn(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.registered_model_name}"

    @property
    def resolved_model_uri(self) -> str:
        return self.model_uri or f"models:/{self.uc_model_fqn}@champion"


CFG = Config()
print(CFG)


import mlflow
import torch


def load_model(cfg: Config):
    """Load the registered model from UC onto GPU."""
    mlflow.set_registry_uri("databricks-uc")
    uri = cfg.resolved_model_uri
    try:
        booster = mlflow.xgboost.load_model(uri)
        print(f"Loaded model from {uri}")
    except Exception as exc:
        print(f"Could not load {uri} ({exc}); falling back to latest version.")
        from mlflow.tracking import MlflowClient

        client = MlflowClient(registry_uri="databricks-uc")
        versions = client.search_model_versions(f"name='{cfg.uc_model_fqn}'")
        latest = max(int(v.version) for v in versions)
        uri = f"models:/{cfg.uc_model_fqn}/{latest}"
        print(f"Loading {uri}")
        booster = mlflow.xgboost.load_model(uri)
    # Run prediction on the GPU when one is present (XGBoost 2.x device switch).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    booster.set_param({"device": device})
    print(f"Inference device: {device}")
    return booster, uri


import numpy as np
from sklearn.datasets import make_classification


def load_inputs(cfg: Config):
    """Regenerate the identical dataset used in training and return the held-out test split, so
    the model scores in-distribution data (see the Config note)."""
    from sklearn.model_selection import train_test_split

    total = cfg.num_train_samples + cfg.num_test_samples
    print(f"Regenerating {total} samples; scoring the {cfg.num_test_samples}-row test split...")
    X, y = make_classification(
        n_samples=total,
        n_features=cfg.num_features,
        n_informative=max(2, cfg.num_features // 2),
        n_redundant=max(0, cfg.num_features // 4),
        n_classes=cfg.num_classes,
        shuffle=False,  # deterministic column order so train/infer distributions match
        random_state=cfg.random_state,
    )
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg.num_test_samples / total, random_state=cfg.random_state,
    )
    return X_test.astype(np.float32), y_test


import time
import xgboost as xgb


def run_inference(cfg: Config, booster, X_test):
    """Run batch inference on GPU."""
    dtest = xgb.DMatrix(X_test)

    t0 = time.time()
    preds_proba = booster.predict(dtest)
    elapsed = time.time() - t0

    throughput = len(X_test) / elapsed if elapsed > 0 else float("nan")
    print(f"Scored {len(X_test)} rows in {elapsed:.1f}s ({throughput:.0f} rows/s)")

    return preds_proba, elapsed, throughput


def persist(cfg: Config, X_test, preds_proba, y_test):
    import pandas as pd

    rows = {"predicted_probability": preds_proba}
    if cfg.num_classes == 2:
        rows["predicted_class"] = (preds_proba > 0.5).astype(int)
    else:
        rows["predicted_class"] = np.argmax(preds_proba, axis=-1)
    if y_test is not None:
        rows["true_label"] = y_test
    pdf = pd.DataFrame(rows)

    out_dir = f"/Volumes/{cfg.uc_catalog}/{cfg.uc_schema}/predictions"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{cfg.output_name}.csv"
    pdf.to_csv(path, index=False)
    print(f"Wrote {len(pdf)} predictions to UC Volume: {path}")
    return path


def main():
    from sklearn.metrics import accuracy_score

    mlflow.set_registry_uri("databricks-uc")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    booster, uri = load_model(CFG)
    X_test, y_test = load_inputs(CFG)

    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name="xgboost-classification-batch-inference", nested=nested):
        preds_proba, elapsed, throughput = run_inference(CFG, booster, X_test)
        mlflow.log_params({"model_uri": uri, "batch_size": CFG.batch_size, "n_rows": len(X_test)})
        mlflow.log_metric("inference_seconds", elapsed)
        mlflow.log_metric("rows_per_second", throughput)

        if y_test is not None:
            if CFG.num_classes == 2:
                preds = (preds_proba > 0.5).astype(int)
            else:
                preds = np.argmax(preds_proba, axis=-1)
            acc = accuracy_score(y_test, preds)
            mlflow.log_metric("accuracy", acc)
            print(f"Batch inference accuracy: {acc:.4f}")

        target = persist(CFG, X_test, preds_proba, y_test)
        print(f"Output: {target}")


if __name__ == "__main__":
    main()

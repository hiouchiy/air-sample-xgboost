"""XGBoost GPU batch inference on Forest CoverType — AI Runtime CLI script.

Loads the @champion model from Unity Catalog, scores the held-out test split on the GPU, and writes
predictions as a CSV to a UC Volume. Requires a model registered by 01 (or the fan-out HPO) first.
    COPYFILE_DISABLE=1 air run --file 02_cli/batch_inference.yaml --watch --profile <your-profile>
The notebook equivalent is 01_notebook/02_batch_inference.py.
"""

import os
import logging
from dataclasses import dataclass

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


def _ensure_uc(catalog, schema, volume=None):
    """Create the UC schema (and optionally a MANAGED volume) if missing, so a fresh catalog runs
    top-to-bottom with no manual setup. Falls back to an actionable message without CREATE rights."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import VolumeType

    w = WorkspaceClient()

    def _create(fn, what):
        try:
            fn()
            print(f"Created {what}")
        except Exception as e:
            m = str(e).lower()
            if "already exists" in m:
                return
            if any(t in m for t in ("permission", "denied", "does not have", "unauthorized")):
                raise RuntimeError(
                    f"Cannot create {what}: {e}\nGrant CREATE on catalog '{catalog}', or pre-create "
                    f"it (see setup.sh / the README), or set UC_CATALOG/UC_SCHEMA to an existing one."
                ) from e
            raise

    _create(lambda: w.schemas.create(name=schema, catalog_name=catalog), f"schema {catalog}.{schema}")
    if volume:
        _create(lambda: w.volumes.create(catalog_name=catalog, schema_name=schema, name=volume,
                                         volume_type=VolumeType.MANAGED), f"volume {catalog}.{schema}.{volume}")


@dataclass
class Config:
    uc_catalog: str = _env("UC_CATALOG", "main")
    uc_schema: str = _env("UC_SCHEMA", "air_samples")
    registered_model_name: str = _env("REGISTERED_MODEL_NAME", "xgboost_classification")
    # Empty -> use models:/<catalog>.<schema>.<name>@champion (else falls back to latest version).
    model_uri: str = _env("MODEL_URI", "")

    # Test data ---------------------------------------------------------
    # We re-download Forest CoverType and take the SAME held-out test split as training. Because the
    # split is deterministic (identical TEST_SIZE + RANDOM_STATE, matching the training run), it scores
    # exactly the rows it did not train on — keep these values equal to the training run.
    test_size: float = float(_env("TEST_SIZE", "0.2"))
    max_samples: int = int(_env("MAX_SAMPLES", "-1"))
    random_state: int = int(_env("RANDOM_STATE", "42"))
    batch_size: int = int(_env("BATCH_SIZE", "10000"))

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
    except mlflow.exceptions.MlflowException as exc:
        print(f"Could not load {uri} ({exc}); falling back to latest version.")
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{cfg.uc_model_fqn}'")
        if not versions:
            raise RuntimeError(f"No versions for {cfg.uc_model_fqn}; run 01 (or the fan-out HPO) first.")
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
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_inputs(cfg: Config):
    """Re-download Forest CoverType and return the identical held-out test split used in training,
    so the model scores in-distribution data it never saw (see the Config note)."""
    print("Downloading Forest CoverType (~11 MB, cached under ~/scikit-learn_data)...")
    data = fetch_covtype()
    X = data.data.astype(np.float32)
    y = (data.target - 1).astype(np.int32)

    if 0 < cfg.max_samples < len(X):
        idx = np.sort(np.random.default_rng(cfg.random_state).permutation(len(X))[: cfg.max_samples])
        X, y = X[idx], y[idx]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y,
    )
    print(f"Scoring the {len(X_test)}-row held-out test split...")
    return X_test, y_test


import time
import xgboost as xgb


def run_inference(booster, X_test):
    """Run batch inference on GPU."""
    dtest = xgb.DMatrix(X_test)

    t0 = time.time()
    proba = booster.predict(dtest)  # shape (n_rows, num_class)
    elapsed = time.time() - t0

    throughput = len(X_test) / elapsed if elapsed > 0 else float("nan")
    print(f"Scored {len(X_test)} rows in {elapsed:.1f}s ({throughput:.0f} rows/s)")

    return proba, elapsed, throughput


def persist(cfg: Config, proba, y_test):
    import pandas as pd

    pdf = pd.DataFrame({
        "predicted_class": np.argmax(proba, axis=1),
        "predicted_probability": proba.max(axis=1),  # confidence of the predicted class
    })
    if y_test is not None:
        pdf["true_label"] = y_test

    # Auto-create the schema + predictions volume if missing (friendly fallback without CREATE).
    _ensure_uc(cfg.uc_catalog, cfg.uc_schema, "predictions")
    out_dir = f"/Volumes/{cfg.uc_catalog}/{cfg.uc_schema}/predictions"
    path = f"{out_dir}/{cfg.output_name}.csv"
    pdf.to_csv(path, index=False)
    print(f"Wrote {len(pdf)} predictions to UC Volume: {path}")
    return path


def main():

    mlflow.set_registry_uri("databricks-uc")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    booster, uri = load_model(CFG)
    X_test, y_test = load_inputs(CFG)

    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name="xgboost-classification-batch-inference", nested=nested):
        proba, elapsed, throughput = run_inference(booster, X_test)
        mlflow.log_params({"model_uri": uri, "batch_size": CFG.batch_size, "n_rows": len(X_test)})
        mlflow.log_metric("inference_seconds", elapsed)
        mlflow.log_metric("rows_per_second", throughput)

        if y_test is not None:
            acc = accuracy_score(y_test, np.argmax(proba, axis=1))
            mlflow.log_metric("accuracy", acc)
            print(f"Batch inference accuracy: {acc:.4f}")

        target = persist(CFG, proba, y_test)
        print(f"Output: {target}")


if __name__ == "__main__":
    main()

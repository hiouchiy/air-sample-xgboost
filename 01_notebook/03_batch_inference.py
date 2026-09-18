# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost — GPU batch inference on AI Runtime
# MAGIC
# MAGIC Loads the trained XGBoost model that `01` or `02` registered to **Unity Catalog**
# MAGIC (the `@champion` version) and runs **GPU batch inference** over the held-out test set on an
# MAGIC AI Runtime GPU. It reports accuracy and throughput and writes the scored rows to a CSV on a
# MAGIC **Unity Catalog Volume** (a plain file write — no Spark).
# MAGIC
# MAGIC ## How to run this notebook
# MAGIC Import it into the workspace, attach it to an **AI Runtime** compute (`GPU_1xA10` is enough),
# MAGIC and **Run All**. The `%pip` cell below installs the dependencies. Run `01` (or `02`) first — it
# MAGIC registers the model and sets the `@champion` alias this step loads.
# MAGIC
# MAGIC > Prefer submitting from a terminal? The CLI equivalent is `02_cli/03_batch_inference.py` — run
# MAGIC > it with `air run --file 02_cli/batch_inference.yaml --watch`.

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
# MAGIC By default we load the registered UC model's **`@champion`** alias and score the held-out
# MAGIC synthetic test split. Point `MODEL_URI` at a specific version or alias to score with a
# MAGIC different model.

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load the model from Unity Catalog onto the GPU
# MAGIC We first try the `@champion` alias; if it is not set we fall back to the latest version.

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load input rows (synthetic or from UC table)

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run GPU batch inference

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write predictions to a Unity Catalog Volume
# MAGIC A plain file write to the UC Volume — no Spark involved.

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Entry point

# COMMAND ----------

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


# COMMAND ----------

if __name__ == "__main__":
    main()

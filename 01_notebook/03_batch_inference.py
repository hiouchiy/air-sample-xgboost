# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost — GPU batch inference on AI Runtime
# MAGIC
# MAGIC Loads the trained XGBoost model that `01` or `02` registered to **Unity Catalog**
# MAGIC (the `@champion` version) and runs **GPU batch inference** over the held-out test set on an
# MAGIC AI Runtime GPU. It reports accuracy and throughput and writes the scored rows to a CSV on a
# MAGIC **Unity Catalog Volume** (a plain file write — no Spark).

# COMMAND ----------

# MAGIC %md
# MAGIC ## ▶ Before you start — attach a serverless GPU
# MAGIC AI Runtime GPUs are **serverless** — there is no cluster to create. This notebook needs a
# MAGIC **single-GPU `GPU_1xA10`**. Attach one from the notebook itself:
# MAGIC 1. Open the **compute** drop-down at the top of the notebook → **Serverless GPU**.
# MAGIC 2. Click the **environment** icon to open the **Environment** side panel.
# MAGIC 3. Set **Accelerator** to a **single A10** (`GPU_1xA10`); leave the default **Base environment**.
# MAGIC 4. Click **Apply**, then **Confirm**.
# MAGIC
# MAGIC Run `01` (or `02`) first — it registers the model and sets the `@champion` alias this step
# MAGIC loads — then **run the cells one at a time, top to bottom**, reviewing each step's output. (Run All works too, but stepping through is recommended for a sample you're evaluating.)
# MAGIC Docs: [Connect to serverless GPU compute](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting#gpu-compute).
# MAGIC
# MAGIC > Prefer submitting from a terminal? The CLI equivalent is `02_cli/03_batch_inference.py` — run
# MAGIC > it with `air run --file 02_cli/batch_inference.yaml --watch`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies
# MAGIC The `%pip` cell installs the dependencies. **`%restart_python`** (a Databricks magic) then
# MAGIC restarts the notebook's Python process so those freshly installed versions are the ones
# MAGIC imported below — run both once, at the top. (The CLI copy in `02_cli/` gets its dependencies
# MAGIC from the workload YAML instead.)

# COMMAND ----------

# MAGIC %pip install -U "xgboost>=2.1,<3" "scikit-learn>=1.3,<2"

# COMMAND ----------

# MAGIC # Restarts the Python interpreter so the versions just installed above are the ones imported
# MAGIC # below. Databricks-specific magic; it clears in-memory state, so continue from the next cell.
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC By default we load the registered UC model's **`@champion`** alias and score the held-out
# MAGIC test split. Keep `TEST_SIZE`/`RANDOM_STATE` equal to the training run so we regenerate the
# MAGIC identical split. Point `MODEL_URI` at a specific version or alias to score a different model.
# MAGIC
# MAGIC This step loads the **`@champion`** version of `<catalog>.<schema>.xgboost_classification`.
# MAGIC Both `01` (single-GPU) and `02` (multi-GPU HPO) set `@champion` to the model they just
# MAGIC trained, so **03 scores whichever you ran last**. To score a specific model: re-run `01` or
# MAGIC `02` (it re-promotes `@champion`), or set `MODEL_URI` to a version/alias, e.g.
# MAGIC `models:/<catalog>.<schema>.xgboost_classification/3` (empty `MODEL_URI` = `@champion`).

# COMMAND ----------

import os
import logging
from dataclasses import dataclass

# Serverless/AI Runtime enforces a py4j method whitelist, so MLflow's optional run-context tag
# lookup logs a benign `Py4JSecurityException ... extraContext ... not whitelisted` warning during
# logging. It's harmless (MLflow skips a couple of optional tags and continues) — quiet just that
# logger so it doesn't look like a failure.
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    uc_catalog: str = _env("UC_CATALOG", "main")
    uc_schema: str = _env("UC_SCHEMA", "air_samples")
    registered_model_name: str = _env("REGISTERED_MODEL_NAME", "xgboost_classification")
    # Empty -> use models:/<catalog>.<schema>.<name>@champion (else falls back to latest version).
    model_uri: str = _env("MODEL_URI", "")

    # Test data ---------------------------------------------------------
    # We re-download Forest CoverType and take the SAME held-out test split as training. Because the
    # split is deterministic (identical TEST_SIZE + RANDOM_STATE, matching 01/02), the model scores
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


# Run it: load the @champion model onto the GPU.
print(f"CUDA available: {torch.cuda.is_available()}")
booster, uri = load_model(CFG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load input rows — the held-out test split
# MAGIC We re-download Forest CoverType and take the identical held-out split the model never trained
# MAGIC on. For real scoring, replace this cell with your own rows loaded from a UC table.

# COMMAND ----------

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split


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


# Run it.
X_test, y_test = load_inputs(CFG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run GPU batch inference
# MAGIC Score the held-out rows on the GPU and log latency/throughput (and accuracy vs. the true
# MAGIC labels) to an MLflow run.

# COMMAND ----------

import time
import xgboost as xgb
from sklearn.metrics import accuracy_score


def run_inference(booster, X_test):
    """Run batch inference on GPU."""
    dtest = xgb.DMatrix(X_test)

    t0 = time.time()
    proba = booster.predict(dtest)  # shape (n_rows, num_class)
    elapsed = time.time() - t0

    throughput = len(X_test) / elapsed if elapsed > 0 else float("nan")
    print(f"Scored {len(X_test)} rows in {elapsed:.1f}s ({throughput:.0f} rows/s)")

    return proba, elapsed, throughput


# Run it: score inside an MLflow run and log metrics.
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write predictions to a Unity Catalog Volume
# MAGIC A plain file write to the UC Volume — no Spark involved.

# COMMAND ----------

def persist(cfg: Config, proba, y_test):
    import pandas as pd

    pdf = pd.DataFrame({
        "predicted_class": np.argmax(proba, axis=1),
        "predicted_probability": proba.max(axis=1),  # confidence of the predicted class
    })
    if y_test is not None:
        pdf["true_label"] = y_test

    out_dir = f"/Volumes/{cfg.uc_catalog}/{cfg.uc_schema}/predictions"
    # A UC Volume can't be created by mkdir on the /Volumes FUSE mount (that raises a cryptic
    # Errno 95). If the Volume is missing, tell the user exactly how to create it.
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(
            f"UC Volume {out_dir} not found. Create it once:\n"
            f"  databricks volumes create {cfg.uc_catalog} {cfg.uc_schema} predictions MANAGED\n"
            f"(or run setup.sh with CATALOG={cfg.uc_catalog}), or set UC_CATALOG/UC_SCHEMA to an "
            f"existing Volume."
        )
    path = f"{out_dir}/{cfg.output_name}.csv"
    pdf.to_csv(path, index=False)
    print(f"Wrote {len(pdf)} predictions to UC Volume: {path}")
    return path


# Run it.
target = persist(CFG, proba, y_test)
print("Output:", target)

# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost — Single-GPU training on Forest CoverType (Databricks AI Runtime)
# MAGIC
# MAGIC This example trains an **XGBoost multi-class classifier** on the public **Forest CoverType**
# MAGIC dataset (581,012 rows × 54 features, 7 forest-cover-type classes) using a **single GPU** on
# MAGIC Databricks AI Runtime. It demonstrates **GPU-accelerated gradient boosting**, tracks the run
# MAGIC with **MLflow**, and registers the trained model to **Unity Catalog** for batch inference.
# MAGIC
# MAGIC ## Why XGBoost on GPU?
# MAGIC XGBoost on GPU (`tree_method="hist"` + `device="cuda"`, the XGBoost 2.x/3.x API) speeds up
# MAGIC training on big datasets (100k+ rows). Rather than assert a number, this notebook **measures**
# MAGIC it on the data you're running: the *(optional)* benchmark cell after training reports the
# MAGIC GPU-vs-CPU ratio **on this dataset and this compute** (hardware- and size-dependent). Forest
# MAGIC CoverType's columns are already numeric, so no feature engineering is needed. Step 02 shows how
# MAGIC to put many GPUs to work for classic ML via parallel hyperparameter search.
# MAGIC
# MAGIC References: [XGBoost GPU support](https://xgboost.readthedocs.io/en/stable/gpu/) ·
# MAGIC [UCI Covertype dataset](https://archive.ics.uci.edu/dataset/31/covertype).

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
# MAGIC Then **run the cells one at a time, top to bottom**, reviewing each step's output. (Run All works too, but stepping through is recommended for a sample you're evaluating.)
# MAGIC Docs: [Connect to serverless GPU compute](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting#gpu-compute).
# MAGIC
# MAGIC > Prefer submitting from a terminal? The CLI equivalent is `02_cli/01_train_singlegpu.py` —
# MAGIC > run it with `air run --file 02_cli/train_singlegpu.yaml`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies
# MAGIC The `%pip` cell installs the dependencies. **`%restart_python`** (a Databricks magic) then
# MAGIC restarts the notebook's Python process so those freshly installed versions are the ones
# MAGIC imported below — run both once, at the top. (The CLI copy in `02_cli/` gets its dependencies
# MAGIC from the workload YAML instead.)

# COMMAND ----------

# MAGIC %pip install "xgboost>=2.1" "scikit-learn>=1.3"

# COMMAND ----------

# MAGIC # Restarts the Python interpreter so the versions just installed above are the ones imported
# MAGIC # below. Databricks-specific magic; it clears in-memory state, so continue from the next cell.
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC Every knob is an environment variable with a sensible default. In the notebook you can set
# MAGIC one before running, e.g. `import os; os.environ["MAX_SAMPLES"] = "50000"` for a quick smoke
# MAGIC test. (The CLI form sets them in the workload YAML — see `02_cli/train_singlegpu.yaml`.)

# COMMAND ----------

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


# Notebook widget for the UC catalog/schema — set a catalog where you can CREATE schemas/volumes/
# models (`main` is often locked down in governed workspaces). Runs before Config reads the env.
# Notebook-only; the CLI copy takes these from the workload YAML / env instead.
dbutils.widgets.text("UC_CATALOG", "main", "Unity Catalog (must have CREATE)")
dbutils.widgets.text("UC_SCHEMA", "air_samples", "Schema")
os.environ["UC_CATALOG"] = dbutils.widgets.get("UC_CATALOG")
os.environ["UC_SCHEMA"] = dbutils.widgets.get("UC_SCHEMA")


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
    # train_xgboost() downgrades device to "cpu" automatically if no CUDA device is present.
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Download the dataset
# MAGIC We download the public **Forest CoverType** dataset via scikit-learn (cached locally after
# MAGIC the first run) and split it into train/test. The source labels are 1..7; XGBoost multi-class
# MAGIC expects 0..6, so we shift them by one. The split is deterministic (fixed `TEST_SIZE` +
# MAGIC `RANDOM_STATE`), so batch inference (03) can regenerate the identical held-out split. For a
# MAGIC production model, swap this cell for your own data loaded from a UC table.

# COMMAND ----------

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split


def load_dataset(cfg: Config):
    """Download the Forest CoverType dataset and split it into train/test."""
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


# Run it: download CoverType and split into train/test.
X_train, X_test, y_train, y_test = load_dataset(CFG)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Peek at the data
# MAGIC A few rows (label + the 54 numeric features) and the class balance, so the task is concrete
# MAGIC before we train (CoverType is imbalanced across its 7 cover-type classes).

# COMMAND ----------

import pandas as pd

_prev = pd.DataFrame(X_train[:5], columns=[f"f{i}" for i in range(X_train.shape[1])])
_prev.insert(0, "label", y_train[:5])
display(_prev)  # display() renders a rich table on Databricks
_u, _c = np.unique(y_train, return_counts=True)
print("class distribution (0..6):", dict(zip(_u.tolist(), _c.tolist())))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train XGBoost on GPU
# MAGIC We train with `tree_method="hist"` + `device="cuda"` (the XGBoost 2.x GPU switch) using the
# MAGIC `multi:softprob` objective, which outputs a per-class probability matrix. It falls back to
# MAGIC CPU automatically if no GPU is attached.

# COMMAND ----------

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
        "objective": "multi:softprob",  # per-class probability matrix (n_rows, num_class)
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
    return booster, num_class, train_time, evals_result


# Run it: train on the GPU.
booster, num_class, train_time, evals_result = train_xgboost(CFG, X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Optional) Measured GPU vs CPU speedup
# MAGIC Trains the same config on GPU then CPU on a 100k subsample and prints the **measured** ratio —
# MAGIC concrete for this dataset/hardware, not an unsourced "Nx". Skip it to keep the run fast.

# COMMAND ----------

import time as _t

_Xd, _yd = X_train[:100_000], y_train[:100_000]
_base = dict(objective="multi:softprob", num_class=num_class, eval_metric="mlogloss",
             max_depth=CFG.max_depth, learning_rate=CFG.learning_rate, tree_method="hist")


def _bench(device):
    _d = xgb.QuantileDMatrix(_Xd, label=_yd)
    _s = _t.time()
    xgb.train({**_base, "device": device}, _d, num_boost_round=CFG.n_estimators)
    return _t.time() - _s


import torch

if torch.cuda.is_available():
    _g, _c = _bench("cuda"), _bench("cpu")
    print(f"GPU {_g:.1f}s vs CPU {_c:.1f}s -> {_c / _g:.1f}x faster "
          f"({len(_Xd)}x{X_train.shape[1]} rows, {CFG.n_estimators} rounds, on this compute)")
else:
    print("No GPU attached; skipping the GPU-vs-CPU benchmark.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluate the model
# MAGIC We report accuracy, macro one-vs-rest AUC, and log loss on the held-out test split.

# COMMAND ----------

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def evaluate_xgboost(booster, X_test, y_test):
    """Evaluate the trained multi-class model."""
    proba = booster.predict(xgb.DMatrix(X_test))  # shape (n_rows, num_class)
    preds = np.argmax(proba, axis=1)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc_ovr_macro": float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro")),
        "log_loss": float(log_loss(y_test, proba)),
    }


# Run it.
metrics = evaluate_xgboost(booster, X_test, y_test)
metrics["train_seconds"] = train_time
print("Evaluation metrics:", metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log to MLflow & register to Unity Catalog
# MAGIC We log the trained XGBoost model with the MLflow `xgboost` flavor and register it directly to
# MAGIC the **Unity Catalog Model Registry**. Parameters and metrics are logged to the same MLflow
# MAGIC run, and the new version is promoted to the **`@champion`** alias.

# COMMAND ----------

import mlflow


def log_and_register(cfg: Config, booster, num_class, metrics, X_train, evals_result):
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

        # Per-round boosting curves (train/test mlogloss) -> the MLflow experiment-tracking value.
        tr = evals_result.get("train", {}).get("mlogloss", [])
        te = evals_result.get("test", {}).get("mlogloss", [])
        for i, (a, b) in enumerate(zip(tr, te)):
            mlflow.log_metrics({"train_mlogloss": a, "test_mlogloss": b}, step=i)

        # A real sample row makes the logged signature/input example match production inputs.
        input_example = X_train[:5]
        output_example = booster.predict(xgb.DMatrix(input_example))
        signature = infer_signature(model_input=input_example, model_output=output_example)

        model_info = mlflow.xgboost.log_model(
            xgb_model=booster,
            **_logmodel_model_kw(),
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
    client = MlflowClient()
    client.set_registered_model_alias(cfg.uc_model_fqn, "champion", version)
    print(f"Registered {cfg.uc_model_fqn} as version {version} and set alias @champion "
          f"(this is the version 03 will load).")


# Run it: log params/metrics/model and promote to @champion.
run_id = log_and_register(CFG, booster, num_class, metrics, X_train, evals_result)
print("MLflow run_id:", run_id)

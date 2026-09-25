# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost — (Optional) Deploy to Databricks Model Serving (CPU)
# MAGIC
# MAGIC Creates (or updates) a **CPU Model Serving** endpoint that serves the XGBoost model registered
# MAGIC to Unity Catalog by `01`/`02` (the `@champion` version), waits until it is ready, and sends a
# MAGIC few Forest CoverType rows for real-time class predictions.
# MAGIC
# MAGIC ## Optional — and it's Model Serving, not AI Runtime
# MAGIC This repo's core value is **AI Runtime** (serverless-GPU training, parallel HPO, batch
# MAGIC inference). Real-time serving is a nice add-on, so this step is **optional**. It uses
# MAGIC **Databricks Model Serving** — a general-purpose serving product **independent of AI Runtime**.
# MAGIC
# MAGIC ## Classic-ML serving needs no GPU
# MAGIC Tabular XGBoost inference is light, so this serves on a **CPU** endpoint (`workload_type="CPU"`)
# MAGIC — cheaper than BERT's `GPU_SMALL` endpoint, and a useful cost message: you don't need a GPU to
# MAGIC serve a gradient-boosted tree model.
# MAGIC
# MAGIC ## How Model Serving works (mental model)
# MAGIC Register model in UC → create a **serving endpoint** for it → Databricks **builds a container
# MAGIC and provisions compute** (a few minutes, one-time) → the endpoint reaches **READY** → you send
# MAGIC **JSON requests** over HTTPS → it **scales to zero** when idle (the first call after idle pays a
# MAGIC cold start).
# MAGIC
# MAGIC ## Note on execution mode
# MAGIC This is a **control-plane** step (it calls the Serving REST API), not a GPU job — so it needs
# MAGIC **no GPU attach** (any compute works) and does **not** use the AI Runtime CLI. Just run the
# MAGIC cells top to bottom: the `%pip` cell installs the deps and auth is the notebook's own.
# MAGIC (Serving ships **as this notebook only** — there is no CLI counterpart, since deploying an
# MAGIC endpoint is a control-plane call rather than an `air run` GPU job.)
# MAGIC It uses the MLflow **Deployments** client (`get_deploy_client("databricks")`), whose dict config
# MAGIC is the stable REST shape — so it does not break across `databricks-sdk` versions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install dependencies
# MAGIC The `%pip` cell installs `mlflow` (Deployments client) and `scikit-learn` (to fetch a few
# MAGIC CoverType rows to score). **`%restart_python`** (a Databricks magic) then restarts the Python
# MAGIC process so the freshly installed versions are the ones imported below.

# COMMAND ----------

# MAGIC %pip install -U "mlflow>=2.15.0" "scikit-learn>=1.3"

# COMMAND ----------

# MAGIC # Restarts the Python interpreter so the versions just installed above are the ones imported below.
# MAGIC # Databricks-specific magic; it clears in-memory state, so continue from the next cell.
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC The serving knobs (one line each — these are the ones people trip on):
# MAGIC - **`endpoint_name`** — the REST endpoint's name (its URL path).
# MAGIC - **`model_version` / `MODEL_VERSION`** — which UC version to serve (empty = latest).
# MAGIC - **`workload_type`** — the serving compute tier. **`CPU`** here (tabular XGBoost doesn't need a
# MAGIC   GPU); contrast with BERT's `GPU_SMALL`. Note these names differ from AI Runtime's
# MAGIC   (`GPU_1xA10`/`GPU_8xH100`).
# MAGIC - **`workload_size`** — `Small`/`Medium`/`Large` = provisioned **concurrency** (simultaneous
# MAGIC   requests), **not** the hardware.
# MAGIC - **`scale_to_zero`** — scale to **0 replicas when idle (no cost)**; the next request pays a
# MAGIC   **cold start**. Trade cost for tail latency.

# COMMAND ----------

import os
import time
from dataclasses import dataclass

# Notebook widget for the UC catalog/schema — set a catalog where the model was registered
# (`main` is often locked down in governed workspaces). Runs before Config reads the env.
dbutils.widgets.text("UC_CATALOG", "main", "Unity Catalog (must have CREATE)")
dbutils.widgets.text("UC_SCHEMA", "air_samples", "Schema")
os.environ["UC_CATALOG"] = dbutils.widgets.get("UC_CATALOG")
os.environ["UC_SCHEMA"] = dbutils.widgets.get("UC_SCHEMA")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    uc_catalog: str = _env("UC_CATALOG", "main")
    uc_schema: str = _env("UC_SCHEMA", "air_samples")
    registered_model_name: str = _env("REGISTERED_MODEL_NAME", "xgboost_classification")
    endpoint_name: str = _env("ENDPOINT_NAME", "xgboost-covertype")
    model_version: str = _env("MODEL_VERSION", "")  # empty -> latest
    workload_type: str = _env("WORKLOAD_TYPE", "CPU")  # tabular XGBoost is light -> CPU serving
    workload_size: str = _env("WORKLOAD_SIZE", "Small")
    scale_to_zero: bool = _env("SCALE_TO_ZERO", "true").lower() == "true"

    @property
    def uc_model_fqn(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}.{self.registered_model_name}"


CFG = Config()
print(CFG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create (or update) the endpoint
# MAGIC The endpoint config has two parts that confuse people:
# MAGIC - **`served_entities`** — *which model/version* to serve (plus its compute tier + concurrency).
# MAGIC - **`traffic_config.routes`** — how to split traffic: send `traffic_percentage` to a
# MAGIC   `served_model_name`. With one model it's 100% to it; this is also how you'd do canary / A-B.
# MAGIC
# MAGIC **Create vs update is idempotent** — re-running updates the existing endpoint instead of
# MAGIC failing. `_wait_ready` polls until `state.ready == READY`; the **first** deploy takes a few
# MAGIC minutes while Databricks builds the serving container.

# COMMAND ----------

import mlflow
from mlflow.deployments import get_deploy_client


def latest_version(cfg: Config) -> str:
    mlflow.set_registry_uri("databricks-uc")
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{cfg.uc_model_fqn}'")
    if not versions:
        raise RuntimeError(f"No versions found for {cfg.uc_model_fqn}. Run 01 (or the fan-out HPO) first.")
    return str(max(int(v.version) for v in versions))


def _served_config(cfg: Config, version: str) -> dict:
    return {
        "served_entities": [
            {
                "entity_name": cfg.uc_model_fqn,
                "entity_version": version,
                "workload_type": cfg.workload_type,
                "workload_size": cfg.workload_size,
                "scale_to_zero_enabled": cfg.scale_to_zero,
            }
        ],
        "traffic_config": {
            "routes": [
                {"served_model_name": f"{cfg.registered_model_name}-{version}", "traffic_percentage": 100}
            ]
        },
    }


def _endpoint_exists(client, name: str) -> bool:
    # Membership check instead of a broad try/except (which could mask auth/network errors).
    return any(e.get("name") == name for e in (client.list_endpoints() or []))


def _wait_ready(client, name: str, timeout_s: int = 2400):
    """Poll until state.ready == READY and config_update == NOT_UPDATING."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = (client.get_endpoint(name) or {}).get("state", {})
        ready, updating = state.get("ready"), state.get("config_update")
        print(f"  endpoint state: ready={ready} config_update={updating}")
        if ready == "READY" and updating in ("NOT_UPDATING", None):
            return
        time.sleep(30)
    raise TimeoutError(f"Endpoint {name} not ready within {timeout_s}s")


def deploy(cfg: Config):
    client = get_deploy_client("databricks")
    version = cfg.model_version or latest_version(cfg)
    config = _served_config(cfg, version)
    print(f"Deploying {cfg.uc_model_fqn} v{version} to endpoint '{cfg.endpoint_name}' "
          f"(this provisions {cfg.workload_type} serving and may take several minutes).")

    if _endpoint_exists(client, cfg.endpoint_name):
        print("Endpoint exists — updating served entity.")
        client.update_endpoint(endpoint=cfg.endpoint_name, config=config)
    else:
        print("Creating endpoint.")
        client.create_endpoint(name=cfg.endpoint_name, config=config)

    _wait_ready(client, cfg.endpoint_name)
    print("Endpoint is ready.")
    return client, version


# Run it: create/update the endpoint and wait until it is ready (first deploy takes a few minutes).
client, version = deploy(CFG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Send a test request
# MAGIC The XGBoost model was logged with an array signature (54 numeric features), so the **request**
# MAGIC body is `{"inputs": [[f0, ..., f53], ...]}`. With the `multi:softprob` objective the **response**
# MAGIC is one **per-class probability vector** per row (`predictions`), and the predicted cover-type
# MAGIC class is its `argmax`. The first call after idle is a **cold start** (slower); later calls are fast.

# COMMAND ----------

import numpy as np
from sklearn.datasets import fetch_covtype


def query(client, cfg: Config):
    rows = fetch_covtype().data.astype(np.float32)[-3:].tolist()  # a few real CoverType rows
    response = client.predict(endpoint=cfg.endpoint_name, inputs={"inputs": rows})
    predictions = response["predictions"] if isinstance(response, dict) else response.predictions
    for i, proba in enumerate(predictions):
        pred_class = int(np.argmax(proba)) if isinstance(proba, list) else int(proba)
        print(f"row {i}: predicted cover-type class = {pred_class}")
    return predictions


# Run it: send a few real-time requests to the endpoint.
query(client, CFG)
print(f"Done. Endpoint '{CFG.endpoint_name}' serving {CFG.uc_model_fqn} v{version}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Call it from outside the notebook (plain REST / curl)
# MAGIC The endpoint is a standard HTTPS REST service — call it from anywhere with a bearer token:
# MAGIC ```bash
# MAGIC curl -s -X POST \
# MAGIC   -H "Authorization: Bearer $DATABRICKS_TOKEN" \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d '{"inputs": [[2596,51,3,258,0,510,221,232,148,6279,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}' \
# MAGIC   https://<workspace-host>/serving-endpoints/xgboost-covertype/invocations
# MAGIC ```
# MAGIC Replace `<workspace-host>` with your workspace URL and `xgboost-covertype` with `ENDPOINT_NAME`.
# MAGIC The response is the same `{"predictions": [[7 class probabilities], ...]}` shape as above.

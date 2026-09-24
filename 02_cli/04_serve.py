"""XGBoost -> Databricks Model Serving — control-plane script (not a GPU job).

Creates/updates a **CPU** serving endpoint for the @champion UC model and sends a test query.
Tabular XGBoost inference is light, so it serves on **CPU Model Serving** (no GPU) — cheaper than
BERT's GPU endpoint, and a nice reminder that classic-ML serving needs no GPU. Run it locally (it
only needs mlflow + scikit-learn to build a few sample rows), not via the AI Runtime CLI:
    pip install -r requirements.txt
    DATABRICKS_CONFIG_PROFILE=<your-profile> python 02_cli/04_serve.py
The notebook equivalent is 01_notebook/04_serve.py.
"""

import os
import time
from dataclasses import dataclass


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


import mlflow
from mlflow.deployments import get_deploy_client


def latest_version(cfg: Config) -> str:
    mlflow.set_registry_uri("databricks-uc")
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{cfg.uc_model_fqn}'")
    if not versions:
        raise RuntimeError(f"No versions found for {cfg.uc_model_fqn}. Run 01/02 first.")
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


def _sample_rows(n=3):
    """A few real Forest CoverType feature rows (54 numeric features) to score."""
    import numpy as np
    from sklearn.datasets import fetch_covtype

    X = fetch_covtype().data.astype(np.float32)
    return X[-n:].tolist()  # a handful of rows from the tail of the dataset


def query(client, cfg: Config):
    rows = _sample_rows()
    # XGBoost model logged with an array signature -> send {"inputs": [[54 floats], ...]}.
    response = client.predict(endpoint=cfg.endpoint_name, inputs={"inputs": rows})
    predictions = response["predictions"] if isinstance(response, dict) else response.predictions
    for i, proba in enumerate(predictions):
        # multi:softprob returns a per-class probability vector; the class is its argmax.
        pred_class = max(range(len(proba)), key=lambda k: proba[k]) if isinstance(proba, list) else proba
        print(f"row {i}: predicted cover-type class = {pred_class}")
    return predictions


def main():
    client, version = deploy(CFG)
    query(client, CFG)
    print(f"Done. Endpoint '{CFG.endpoint_name}' serving {CFG.uc_model_fqn} v{version}.")


if __name__ == "__main__":
    main()


import os
import mlflow
from mlflow.tracking import MlflowClient
from contextlib import contextmanager

CLASS_EXP = "EMI_Classification"
REG_EXP = "EMI_Regression"

def set_tracking_uri(uri: str = None):
    """Set MLflow tracking URI. If none, try env var MLFLOW_TRACKING_URI, else default file-based mlruns."""
    if uri is None:
        uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    else:
        # default: local mlruns folder
        mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))
    return mlflow.get_tracking_uri()

def _get_client():
    return MlflowClient(tracking_uri=mlflow.get_tracking_uri())

def create_experiment_if_not_exists(name: str):
    """Create experiment if not present and return experiment_id."""
    client = _get_client()
    try:
        exp = client.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id
        else:
            return client.create_experiment(name)
    except Exception:
        # If backend doesn't support API create (rare), fallback to mlflow API
        return mlflow.create_experiment(name)

@contextmanager
def start_run(experiment_type: str = 'classification', run_name: str = None, **start_kwargs):
    """
    Context manager wrapper for mlflow.start_run that ensures the right experiment exists.
    experiment_type: 'classification' or 'regression'
    """
    if experiment_type.lower().startswith('class'):
        exp_name = CLASS_EXP
    else:
        exp_name = REG_EXP

    exp_id = create_experiment_if_not_exists(exp_name)
    # set active experiment by id
    mlflow.set_experiment(exp_name)
    run = mlflow.start_run(run_name=run_name, **start_kwargs)
    try:
        yield run
    finally:
        mlflow.end_run()

def log_params(params: dict):
    for k, v in (params or {}).items():
        mlflow.log_param(k, v)

def log_metrics(metrics: dict, step: int = None):
    for k, v in (metrics or {}).items():
        if step is None:
            mlflow.log_metric(k, float(v))
        else:
            mlflow.log_metric(k, float(v), step=step)

def log_artifact(local_path: str, artifact_path: str = None):
    mlflow.log_artifact(local_path, artifact_path)

def log_model_sklearn(model, artifact_path: str, registered_model_name: str = None, conda_env: dict = None):
    """
    Log sklearn model using mlflow.sklearn and optionally register it.
    Returns model info or registered model version.
    """
    import mlflow.sklearn
    model_info = mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path, conda_env=conda_env)
    if registered_model_name:
        client = _get_client()
        model_uri = model_info.model_uri  # e.g., 'runs:/<run-id>/<artifact_path>'
        # register model (create registered model if necessary)
        try:
            client.get_registered_model(registered_model_name)
        except Exception:
            client.create_registered_model(registered_model_name)
        mv = client.create_model_version(name=registered_model_name, source=model_uri, run_id=mlflow.active_run().info.run_id)
        return mv
    return model_info

def register_model(model_uri: str, name: str):
    """
    Register an existing model (by model_uri) into the registry under `name`.
    Returns the ModelVersion object.
    """
    client = _get_client()
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name)
    mv = client.create_model_version(name=name, source=model_uri, run_id=mlflow.active_run().info.run_id)
    return mv

def transition_model_stage(name: str, version: int, stage: str = "Production"):
    """
    Move a registered model version into a stage (Production, Staging, Archived).
    """
    client = _get_client()
    client.transition_model_version_stage(name=name, version=version, stage=stage)

def list_experiments():
    client = _get_client()
    return client.list_experiments()

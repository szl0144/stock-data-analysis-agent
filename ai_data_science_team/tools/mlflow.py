

from typing import Optional, Dict, Any, Union, List
from langchain.tools import tool

_LOADED_DATA = None

@tool(response_format='content_and_artifact')
def mlflow_search_experiments(
    filter_string: Optional[str] = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None
) -> str:
    """
    Search and list existing MLflow experiments.
    
    Parameters
    ----------
    filter_string : str, optional
        Filter query string (e.g., "name = 'my_experiment'"), defaults to
        searching for all experiments. 
        
    tracking_uri: str, optional
        Address of local or remote tracking server. 
        If not provided, defaults
        to the service set by mlflow.tracking.set_tracking_uri. See Where Runs Get Recorded <../tracking.html#where-runs-get-recorded>_ for more info.
    registry_uri: str, optional
        Address of local or remote model registry
        server. If not provided,
        defaults to the service set by mlflow.tracking.set_registry_uri. If no such service was set, defaults to the tracking uri of the client.

    Returns
    -------
    tuple
        - JSON-serialized list of experiment metadata (ID, name, etc.).
        - DataFrame of experiment metadata.
    """
    print("    * Tool: mlflow_search_experiments")
    from mlflow.tracking import MlflowClient
    import pandas as pd

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    experiments = client.search_experiments(filter_string=filter_string)
    # Convert to a dictionary in a list
    experiments_data = [
        dict(e)
        for e in experiments
    ]
    # Convert to a DataFrame
    experiments_df = pd.DataFrame(experiments_data)
    # Convert timestamps to datetime objects
    experiments_df["last_update_time"] = pd.to_datetime(experiments_df["last_update_time"], unit="ms")
    experiments_df["creation_time"] = pd.to_datetime(experiments_df["creation_time"], unit="ms")
    
    return (experiments_df.to_dict(), experiments_df.to_dict())


@tool(response_format='content_and_artifact')
def mlflow_search_runs(
    experiment_ids: Optional[Union[List[str], List[int], str, int]] = None,
    filter_string: Optional[str] = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None
) -> str:
    """
    Search runs within one or more MLflow experiments, optionally filtering by a filter_string.

    Parameters
    ----------
    experiment_ids : list or str or int, optional
        One or more Experiment IDs.
    filter_string : str, optional
        MLflow filter expression, e.g. "metrics.rmse < 1.0".
    tracking_uri: str, optional
        Address of local or remote tracking server. 
        If not provided, defaults
        to the service set by mlflow.tracking.set_tracking_uri. See Where Runs Get Recorded <../tracking.html#where-runs-get-recorded>_ for more info.
    registry_uri: str, optional
        Address of local or remote model registry
        server. If not provided,
        defaults to the service set by mlflow.tracking.set_registry_uri. If no such service was set, defaults to the tracking uri of the client.

    Returns
    -------
    str
        JSON-formatted list of runs that match the query.
    """
    print("    * Tool: mlflow_search_runs")
    from mlflow.tracking import MlflowClient
    import pandas as pd
    
    client = MlflowClient(
        tracking_uri=tracking_uri, 
        registry_uri=registry_uri
    )
    
    if experiment_ids is None:
        experiment_ids = []
    if isinstance(experiment_ids, (str, int)):
        experiment_ids = [experiment_ids]
    
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string
    )
    
    # If no runs are found, return an empty DataFrame
    if not runs:
        return "No runs found.", pd.DataFrame()
    
    # Extract relevant information
    data = []
    for run in runs:
        run_info = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
            "end_time": pd.to_datetime(run.info.end_time, unit="ms"),
            "experiment_id": run.info.experiment_id,
            "user_id": run.info.user_id
        }

        # Flatten metrics, parameters, and tags
        run_info.update(run.data.metrics)
        run_info.update({f"param_{k}": v for k, v in run.data.params.items()})
        run_info.update({f"tag_{k}": v for k, v in run.data.tags.items()})

        data.append(run_info)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return (df.iloc[:,0:15].to_dict(), df.to_dict())



@tool(response_format='content')
def mlflow_create_experiment(experiment_name: str) -> str:
    """
    Create a new MLflow experiment by name.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment to create.

    Returns
    -------
    str
        The experiment ID or an error message if creation failed.
    """
    print("    * Tool: mlflow_create_experiment")
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp_id = client.create_experiment(experiment_name)
    return f"Experiment created with ID: {exp_id}, name: {experiment_name}"




@tool(response_format='content_and_artifact')
def mlflow_predict_from_run_id(run_id: str, tracking_uri: Optional[str] = None) -> tuple:
    """
    Predict using an MLflow model (PyFunc) directly from a given run ID.

    Parameters
    ----------
    run_id : str
        The ID of the MLflow run that logged the model.
    tracking_uri : str, optional
        The MLflow tracking server URI.

    Returns
    -------
    tuple
        JSON-formatted prediction results and DataFrame output.
    """
    print("    * Tool: mlflow_predict_from_run_id")
    from mlflow.tracking import MlflowClient
    import mlflow.pyfunc
    import pandas as pd
    import json
    
    client = MlflowClient(tracking_uri=tracking_uri)

    # Fetch model artifact path from run ID
    model_uri = f"runs:/{run_id}/model"

    try:
        # Load model dynamically
        model = mlflow.pyfunc.load_model(model_uri)

        # Convert input data to DataFrame
        if "_LOADED_DATA" not in globals():
            if _LOADED_DATA is None:
                return "No `_LOADED_DATA` in globals(). To fix, try running: \n\nglobal _LOADED_DATA\n\n_LOADED_DATA = df", pd.DataFrame()
        df = pd.DataFrame(_LOADED_DATA)

        # Make predictions
        preds = model.predict(df)
        
        if preds is pd.DataFrame:
            return f"The predictions have been returned, here are the first five: {preds[:5].to_json()}", preds.to_dict()

        # Convert output to JSON
        if hasattr(preds, "to_json"):
            return f"The predictions have been returned, here are the first five: {preds[:5].to_json()}", preds.to_dict()
        else:
            return json.dumps(preds.tolist()[:5]), dict(preds)

    except Exception as e:
        return f"Error loading model or making predictions: {str(e)}", pd.DataFrame()





# @tool
# def get_or_create_experiment(experiment_name):
#     """
#     Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

#     This function checks if an experiment with the given name exists within MLflow.
#     If it does, the function returns its ID. If not, it creates a new experiment
#     with the provided name and returns its ID.

#     Parameters:
#     - experiment_name (str): Name of the MLflow experiment.

#     Returns:
#     - str: ID of the existing or newly created MLflow experiment.
#     """
#     import mlflow
#     if experiment := mlflow.get_experiment_by_name(experiment_name):
#         return experiment.experiment_id
#     else:
#         return mlflow.create_experiment(experiment_name)



# @tool("mlflow_set_tracking_uri", return_direct=True)
# def mlflow_set_tracking_uri(tracking_uri: str) -> str:
#     """
#     Set or change the MLflow tracking URI.

#     Parameters
#     ----------
#     tracking_uri : str
#         The URI/path where MLflow logs & metrics are stored.

#     Returns
#     -------
#     str
#         Confirmation message.
#     """
#     import mlflow
#     mlflow.set_tracking_uri(tracking_uri)
#     return f"MLflow tracking URI set to: {tracking_uri}"


# @tool("mlflow_list_experiments", return_direct=True)
# def mlflow_list_experiments() -> str:
#     """
#     List existing MLflow experiments.

#     Returns
#     -------
#     str
#         JSON-serialized list of experiment metadata (ID, name, etc.).
#     """
#     from mlflow.tracking import MlflowClient
#     import json

#     client = MlflowClient()
#     experiments = client.list_experiments()
#     # Convert to a JSON-like structure
#     experiments_data = [
#         dict(experiment_id=e.experiment_id, name=e.name, artifact_location=e.artifact_location)
#         for e in experiments
#     ]
    
#     return json.dumps(experiments_data)


# @tool("mlflow_create_experiment", return_direct=True)
# def mlflow_create_experiment(experiment_name: str) -> str:
#     """
#     Create a new MLflow experiment by name.

#     Parameters
#     ----------
#     experiment_name : str
#         The name of the experiment to create.

#     Returns
#     -------
#     str
#         The experiment ID or an error message if creation failed.
#     """
#     from mlflow.tracking import MlflowClient

#     client = MlflowClient()
#     exp_id = client.create_experiment(experiment_name)
#     return f"Experiment created with ID: {exp_id}"


# @tool("mlflow_set_experiment", return_direct=True)
# def mlflow_set_experiment(experiment_name: str) -> str:
#     """
#     Set or create an MLflow experiment for subsequent logging.

#     Parameters
#     ----------
#     experiment_name : str
#         The name of the experiment to set.

#     Returns
#     -------
#     str
#         Confirmation of the chosen experiment name.
#     """
#     import mlflow
#     mlflow.set_experiment(experiment_name)
#     return f"Active MLflow experiment set to: {experiment_name}"


# @tool("mlflow_start_run", return_direct=True)
# def mlflow_start_run(run_name: Optional[str] = None) -> str:
#     """
#     Start a new MLflow run under the current experiment.

#     Parameters
#     ----------
#     run_name : str, optional
#         Optional run name.

#     Returns
#     -------
#     str
#         The run_id of the newly started MLflow run.
#     """
#     import mlflow
#     with mlflow.start_run(run_name=run_name) as run:
#         run_id = run.info.run_id
#     return f"MLflow run started with run_id: {run_id}"


# @tool("mlflow_log_params", return_direct=True)
# def mlflow_log_params(params: Dict[str, Any]) -> str:
#     """
#     Log a batch of parameters to the current MLflow run.

#     Parameters
#     ----------
#     params : dict
#         A dictionary of parameter name -> parameter value.

#     Returns
#     -------
#     str
#         Confirmation message.
#     """
#     import mlflow
#     mlflow.log_params(params)
#     return f"Logged parameters: {params}"


# @tool("mlflow_log_metrics", return_direct=True)
# def mlflow_log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> str:
#     """
#     Log a dictionary of metrics to the current MLflow run.

#     Parameters
#     ----------
#     metrics : dict
#         Metric name -> numeric value.
#     step : int, optional
#         The training step or iteration number.

#     Returns
#     -------
#     str
#         Confirmation message.
#     """
#     import mlflow
#     mlflow.log_metrics(metrics, step=step)
#     return f"Logged metrics: {metrics} at step {step}"


# @tool("mlflow_log_artifact", return_direct=True)
# def mlflow_log_artifact(artifact_path: str, artifact_folder_name: Optional[str] = None) -> str:
#     """
#     Log a local file or directory as an MLflow artifact.

#     Parameters
#     ----------
#     artifact_path : str
#         The local path to the file/directory to be logged.
#     artifact_folder_name : str, optional
#         Subfolder within the run's artifact directory.

#     Returns
#     -------
#     str
#         Confirmation message.
#     """
#     import mlflow
#     if artifact_folder_name:
#         mlflow.log_artifact(artifact_path, artifact_folder_name)
#         return f"Artifact logged from {artifact_path} into folder '{artifact_folder_name}'"
#     else:
#         mlflow.log_artifact(artifact_path)
#         return f"Artifact logged from {artifact_path}"


# @tool("mlflow_log_model", return_direct=True)
# def mlflow_log_model(model_path: str, registered_model_name: Optional[str] = None) -> str:
#     """
#     Log a model artifact (e.g., an H2O-saved model directory) to MLflow.

#     Parameters
#     ----------
#     model_path : str
#         The local filesystem path containing the model artifacts.
#     registered_model_name : str, optional
#         If provided, will also attempt to register the model under this name.

#     Returns
#     -------
#     str
#         Confirmation message with any relevant registration details.
#     """
#     import mlflow
#     if registered_model_name:
#         mlflow.pyfunc.log_model(
#             artifact_path="model",
#             python_model=None,  # if you have a pyfunc wrapper, specify it
#             registered_model_name=registered_model_name,
#             code_path=None,
#             conda_env=None,
#             model_path=model_path  # for certain model flavors, or use flavors
#         )
#         return f"Model logged and registered under '{registered_model_name}' from path {model_path}"
#     else:
#         # Simple log as generic artifact
#         mlflow.pyfunc.log_model(
#             artifact_path="model",
#             python_model=None,
#             code_path=None,
#             conda_env=None,
#             model_path=model_path
#         )
#         return f"Model logged (no registration) from path {model_path}"


# @tool("mlflow_end_run", return_direct=True)
# def mlflow_end_run() -> str:
#     """
#     End the current MLflow run (if one is active).

#     Returns
#     -------
#     str
#         Confirmation message.
#     """
#     import mlflow
#     mlflow.end_run()
#     return "MLflow run ended."


# @tool("mlflow_search_runs", return_direct=True)
# def mlflow_search_runs(
#     experiment_names_or_ids: Optional[Union[List[str], List[int], str, int]] = None,
#     filter_string: Optional[str] = None
# ) -> str:
#     """
#     Search runs within one or more MLflow experiments, optionally filtering by a filter_string.

#     Parameters
#     ----------
#     experiment_names_or_ids : list or str or int, optional
#         Experiment IDs or names.
#     filter_string : str, optional
#         MLflow filter expression, e.g. "metrics.rmse < 1.0".

#     Returns
#     -------
#     str
#         JSON-formatted list of runs that match the query.
#     """
#     import mlflow
#     import json
#     if experiment_names_or_ids is None:
#         experiment_names_or_ids = []
#     if isinstance(experiment_names_or_ids, (str, int)):
#         experiment_names_or_ids = [experiment_names_or_ids]

#     df = mlflow.search_runs(
#         experiment_names=experiment_names_or_ids if all(isinstance(e, str) for e in experiment_names_or_ids) else None,
#         experiment_ids=experiment_names_or_ids if all(isinstance(e, int) for e in experiment_names_or_ids) else None,
#         filter_string=filter_string
#     )
#     return df.to_json(orient="records")


# @tool("mlflow_get_run", return_direct=True)
# def mlflow_get_run(run_id: str) -> str:
#     """
#     Retrieve details (params, metrics, etc.) for a specific MLflow run by ID.

#     Parameters
#     ----------
#     run_id : str
#         The ID of the MLflow run to retrieve.

#     Returns
#     -------
#     str
#         JSON-formatted data containing run info, params, and metrics.
#     """
#     from mlflow.tracking import MlflowClient
#     import json

#     client = MlflowClient()
#     run = client.get_run(run_id)
#     data = {
#         "run_id": run.info.run_id,
#         "experiment_id": run.info.experiment_id,
#         "status": run.info.status,
#         "start_time": run.info.start_time,
#         "end_time": run.info.end_time,
#         "artifact_uri": run.info.artifact_uri,
#         "params": run.data.params,
#         "metrics": run.data.metrics,
#         "tags": run.data.tags
#     }
#     return json.dumps(data)


# @tool("mlflow_load_model", return_direct=True)
# def mlflow_load_model(model_uri: str) -> str:
#     """
#     Load an MLflow-model (PyFunc flavor or other) into memory, returning a handle reference.
#     For demonstration, we store the loaded model globally in a registry dict.

#     Parameters
#     ----------
#     model_uri : str
#         The URI of the model to load, e.g. "runs:/<RUN_ID>/model" or "models:/MyModel/Production".

#     Returns
#     -------
#     str
#         A reference key identifying the loaded model (for subsequent predictions), 
#         or a direct message if you prefer to store it differently.
#     """
#     import mlflow.pyfunc
#     from uuid import uuid4

#     # For demonstration, create a global registry:
#     global _LOADED_MODELS
#     if "_LOADED_MODELS" not in globals():
#         _LOADED_MODELS = {}

#     loaded_model = mlflow.pyfunc.load_model(model_uri)
#     model_key = f"model_{uuid4().hex}"
#     _LOADED_MODELS[model_key] = loaded_model

#     return f"Model loaded with reference key: {model_key}"


# @tool("mlflow_predict", return_direct=True)
# def mlflow_predict(model_key: str, data: List[Dict[str, Any]]) -> str:
#     """
#     Predict using a previously loaded MLflow model (PyFunc), identified by its reference key.

#     Parameters
#     ----------
#     model_key : str
#         The reference key for the loaded model (returned by mlflow_load_model).
#     data : List[Dict[str, Any]]
#         The data rows for which predictions should be made.

#     Returns
#     -------
#     str
#         JSON-formatted prediction results.
#     """
#     import pandas as pd
#     import json

#     global _LOADED_MODELS
#     if model_key not in _LOADED_MODELS:
#         return f"No model found for key: {model_key}"

#     model = _LOADED_MODELS[model_key]
#     df = pd.DataFrame(data)
#     preds = model.predict(df)
#     # Convert to JSON (DataFrame or Series)
#     if hasattr(preds, "to_json"):
#         return preds.to_json(orient="records")
#     else:
#         # If preds is just a numpy array or list
#         return json.dumps(preds.tolist())


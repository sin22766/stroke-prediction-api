from pathlib import Path

import mlflow
from ruamel.yaml import YAML

mlflow.set_tracking_uri("https://mlflow.spikehub.win/")


def read_yaml(file: Path):
    """
    Read a YAML file and return its content as a dictionary.

    Parameters
    ----------
    file : Path
        The path to the YAML file to be read.

    Returns
    -------
    dict
        The content of the YAML file as a dictionary.
    """
    yaml = YAML(typ="safe")
    with open(file, "r") as f:
        return yaml.load(f)


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters
    ----------
    experiment_name : str
        The name of the MLflow experiment to retrieve or create.

    Returns
    -------
    str
        The ID of the MLflow experiment.
    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

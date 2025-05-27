from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import onnx
import pandas as pd
import typer
from catboost import CatBoostClassifier, Pool
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from stroke_prediction.util import read_yaml


class SupportedModels(str, Enum):
    CATBOOST = "catboost"
    MLP = "mlp"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"


DEFAULT_PARAMS = {
    "catboost": {
        "iterations": 1000,
        "colsample_bylevel": 0.09512938089563211,
        "depth": 8,
        "l2_leaf_reg": 8.115228714496485,
        "boosting_type": "Plain",
        "bootstrap_type": "MVS",
        "eval_metric": "F1",
        "auto_class_weights": "Balanced",
        "random_seed": 42,
    },
    "decision_tree": {
        "criterion": "gini",
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 3,
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 197,
        "criterion": "gini",
        "max_depth": 4,
        "min_samples_split": 6,
        "min_samples_leaf": 1,
        "random_state": 42,
    },
    "mlp": {
        "hidden_layer_sizes": 166,
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0015382308040279,
        "learning_rate_init": 0.0034877126245459306,
        "beta_1": 0.5296178606936361,
        "beta_2": 0.935488107125883,
        "epsilon": 2.907208890659844e-08,
        "max_iter": 300,
        "random_state": 42,
        "learning_rate": "adaptive",
    },
}


def create_model(model_name: SupportedModels, params: dict):
    if model_name == SupportedModels.CATBOOST:
        return CatBoostClassifier(**params)
    elif model_name == SupportedModels.DECISION_TREE:
        return DecisionTreeClassifier(**params)
    elif model_name == SupportedModels.RANDOM_FOREST:
        return RandomForestClassifier(**params)
    elif model_name == SupportedModels.MLP:
        return MLPClassifier(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


app = typer.Typer()


@app.command()
def train_model(
    train_data: Annotated[Path, typer.Argument(help="Path to the training data parquet file.")],
    val_data: Annotated[Path, typer.Argument(help="Path to the validation data parquet file.")],
    model_output: Annotated[
        Path, typer.Argument(help="Directory path to save the trained model .onnx file.")
    ],
    model_name: Annotated[
        SupportedModels, typer.Option("--model", help="Which model to train")
    ] = SupportedModels.DECISION_TREE,
    auto_replace: Annotated[
        bool, typer.Option("-y", help="Automatic overwrite the output")
    ] = False,
    params_path: Annotated[
        Optional[Path], typer.Option("--params", help="Path to the parameters file")
    ] = None,
):
    """
    Train a stroke prediction model using the provided training data.

    Parameters
    ----------
    train_data_path : Path
        Path to the training data parquet file.
    val_data_path : Path
        Path to the validation data parquet file.
    model_output : Path
        Directory path where the trained model will be saved.
    auto_replace : bool, optional
        Whether to automatically overwrite existing model files, by default False.
    params_path : Optional[Path], optional
        Path to a file containing additional parameters for the model, by default None.
    """
    if not train_data.exists() or not train_data.is_file():
        raise typer.Abort("Training data file does not exist or is not a file.")

    if not val_data.exists() or not val_data.is_file():
        raise typer.Abort("Validation data file does not exist or is not a file.")

    if params_path is not None and not params_path.exists():
        raise typer.Abort("Parameters file does not exist.")

    params = DEFAULT_PARAMS.get(model_name, {}).copy()

    if params_path is not None:
        params_file = read_yaml(params_path)

        if "model_type" in params_file:
            model_name = params_file["model_type"]
            if model_name not in DEFAULT_PARAMS:
                raise typer.Abort(f"Unsupported model type: {model_name}")

        if "random_seed" in params_file:
            if "random_seed" in params:
                params["random_seed"] = params_file["random_seed"]
            elif "random_state" in params:
                params["random_state"] = params_file["random_seed"]

        model_params = params_file.get(model_name, {})

        params.update(model_params)

    train_df = pd.read_parquet(train_data)
    if train_df.empty:
        raise typer.Abort("Training data is empty.")

    val_df = pd.read_parquet(val_data)
    if val_df.empty:
        raise typer.Abort("Validation data is empty.")

    X_train = train_df.drop("stroke", axis=1)
    y_train = train_df["stroke"]

    X_val = val_df.drop("stroke", axis=1)
    y_val = val_df["stroke"]

    model = create_model(model_name, params)

    if model_name == "catboost":
        train_pool = Pool(data=X_train, label=y_train)
        val_pool = Pool(data=X_val, label=y_val)

        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=0)
    else:
        model.fit(X_train, y_train)

    if model_output.exists() and not auto_replace:
        typer.confirm(
            f"Model output directory {model_output} already exists. Do you want to overwrite it?",
            abort=True,
        )

    if model_name == "catboost":
        model.save_model(model_output, format="onnx")
    else:
        initial_type = [("input", FloatTensorType([None, X_val.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        onnx.save_model(onnx_model, model_output)


if __name__ == "__main__":
    app()

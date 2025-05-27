from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from catboost import CatBoostClassifier, Pool

from stroke_prediction.util import read_yaml

app = typer.Typer()


@app.command()
def train_model(
    train_data: Annotated[Path, typer.Argument(help="Path to the training data parquet file.")],
    val_data: Annotated[Path, typer.Argument(help="Path to the validation data parquet file.")],
    model_output: Annotated[Path, typer.Argument(help="Directory path to save the trained model .cbm file.")],
    auto_replace: Annotated[
        bool, typer.Option("-y", help="Automatic overwrite the output")
    ] = False,
    params_file: Annotated[
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
    params_file : Optional[Path], optional
        Path to a file containing additional parameters for the model, by default None.
    """

    print("Starting model training...")

    if not train_data.exists() or not train_data.is_file():
        raise typer.Abort("Training data file does not exist or is not a file.")

    if not val_data.exists() or not val_data.is_file():
        raise typer.Abort("Validation data file does not exist or is not a file.")

    if params_file is not None and not params_file.exists():
        raise typer.Abort("Parameters file does not exist.")

    params = {
        "iterations": 1000,
        "colsample_bylevel": 0.09512938089563211,
        "depth": 8,
        "l2_leaf_reg": 8.115228714496485,
        "boosting_type": "Plain",
        "bootstrap_type": "MVS",
        "eval_metric": "F1",
        "auto_class_weights": "Balanced",
        "random_seed": 42,
    }

    if params_file is not None:
        params_content = read_yaml(params_file)

        if "model" in params_content:
            params["iterations"] = params_content["model"].get("iterations", params["iterations"])
            params["colsample_bylevel"] = params_content["model"].get("colsample_bylevel", params["colsample_bylevel"])
            params["depth"] = params_content["model"].get("depth", params["depth"])
            params["l2_leaf_reg"] = params_content["model"].get("l2_leaf_reg", params["l2_leaf_reg"])
            params["boosting_type"] = params_content["model"].get("boosting_type", params["boosting_type"])
            params["bootstrap_type"] = params_content["model"].get("bootstrap_type", params["bootstrap_type"])
        
        if "random_seed" in params_content:
            params["random_seed"] = params_content["random_seed"]

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

    train_pool = Pool(
        data=X_train,
        label=y_train,
    )

    val_pool = Pool(
        data=X_val,
        label=y_val,
    )

    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=100,
        verbose=0,
    )

    if model_output.exists() and not auto_replace:
        typer.confirm(
            f"Model output directory {model_output} already exists. Do you want to overwrite it?",
            abort=True,
        )

    model.save_model(model_output)

if __name__ == "__main__":
    app()

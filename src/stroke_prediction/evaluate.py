
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
import typer
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from stroke_prediction.plot import plot_feature_importance
from stroke_prediction.util import get_or_create_experiment

app = typer.Typer()

@app.command()
def evaluate_model(
    test_data: Annotated[Path, typer.Argument(help="Path to the test data parquet file.")],
    model_path: Annotated[Path, typer.Argument(help="Path to the trained model .cbm file.")],
):
    """
    Evaluate a stroke prediction model using the provided test data.

    Parameters
    ----------
    test_data : Path
        Path to the test data parquet file.
    model_path : Path
        Path to the trained model .cbm file.
    """

    if not test_data.exists() or not test_data.is_file():
        raise typer.Abort("Test data file does not exist or is not a file.")

    if not model_path.exists() or not model_path.is_file():
        raise typer.Abort("Model file does not exist or is not a file.")

    # Load the test data
    test_df = pd.read_parquet(test_data)

    X = test_df.drop(columns=["stroke"])
    y = test_df["stroke"]

    # Create a CatBoost Pool for the test data
    test_pool = Pool(data=X, label=y)

    # Load the trained model
    model = CatBoostClassifier()
    model.load_model(model_path)

    # Evaluate the model
    y_pred = model.predict(test_pool)

    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    experiment_id = get_or_create_experiment("Stroke Prediction Evaluation")

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log confusion matrix
        confuse_matrix = ConfusionMatrixDisplay.from_predictions(y, y_pred)
        confuse_matrix.figure_.tight_layout()
        feature_importance = plot_feature_importance(model, X)

        mlflow.log_figure(confuse_matrix.figure_, "confusion_matrix.png")
        mlflow.log_figure(feature_importance, "feature_importance.png")


if __name__ == "__main__":
    app()
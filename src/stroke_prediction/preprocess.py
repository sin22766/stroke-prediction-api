from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from stroke_prediction.util import read_yaml

DEFAULT_DROP_COLUMNS = ["id", "gender", "Residence_type"]
DEFAULT_ONE_HOT_COLUMNS = ["ever_married", "work_type", "smoking_status"]

def get_col_transformer(
    drop_columns: list[str] = DEFAULT_DROP_COLUMNS,
    one_hot_columns: list[str] = DEFAULT_ONE_HOT_COLUMNS,
) -> ColumnTransformer:
    """
    Create a ColumnTransformer for preprocessing data.

    Parameters
    ----------
    drop_columns : list of str, optional
        Columns to be dropped, by default DEFAULT_DROP_COLUMNS.
    one_hot_columns : list of str, optional
        Categorical columns to apply one-hot encoding to, by default DEFAULT_ONE_HOT_COLUMNS.

    Returns
    -------
    ColumnTransformer
        A configured ColumnTransformer for preprocessing.
    """
    return ColumnTransformer(
        transformers=[
            ("drop", "drop", drop_columns),
            (
                "one_hot",
                OneHotEncoder(sparse_output=False, drop="first"),
                one_hot_columns,
            ),
        ],
        remainder="passthrough",
        force_int_remainder_cols=False,
        verbose_feature_names_out=False,
    )


def get_train_pipeline(
    k_neighbors: int = 5,
    drop_columns: list[str] = DEFAULT_DROP_COLUMNS,
    one_hot_columns: list[str] = DEFAULT_ONE_HOT_COLUMNS,
) -> Pipeline:
    """
    Create a training preprocessing pipeline with column transformer and KNN imputer.

    Parameters
    ----------
    k_neighbors : int, optional
        Number of neighbors for KNN imputer, by default 5.
    drop_columns : list of str, optional
        Columns to be dropped, by default DEFAULT_DROP_COLUMNS.
    one_hot_columns : list of str, optional
        Categorical columns to apply one-hot encoding to, by default DEFAULT_ONE_HOT_COLUMNS.

    Returns
    -------
    Pipeline
        A scikit-learn pipeline for preprocessing training data.
    """
    return Pipeline(
        [
            ("column_transformer", get_col_transformer(drop_columns, one_hot_columns)),
            ("imputer", KNNImputer(n_neighbors=k_neighbors)),
        ]
    )


def get_prod_pipeline() -> Pipeline:
    """
    Create a production preprocessing pipeline without imputation.

    Returns
    -------
    Pipeline
        A scikit-learn pipeline suitable for production use where no imputation is required.
    """
    return Pipeline(
        [
            ("column_transformer", get_col_transformer()),
        ]
    )


app = typer.Typer()


@app.command()
def preprocess_data(
    input: Path,
    output: Path,
    auto_replace: Annotated[
        bool, typer.Option("-y", help="Automatic overwrite the output")
    ] = False,
    test_size: Annotated[
        float, typer.Option(help="Fraction of data for test set (e.g., 0.15)")
    ] = 0.15,
    val_size: Annotated[
        float, typer.Option(help="Fraction of data for validation set (e.g., 0.15)")
    ] = 0.15,
    params_file: Annotated[
        Optional[Path], typer.Option("--params", help="Path to the parameters file")
    ] = None,
    experiment_name: Annotated[
        Optional[str], typer.Option("--experiment", help="Name of the experiment")
    ] = "Data Preprocessing",
):
    """
    Preprocess input CSV data for stroke prediction. Applies transformations and splits into train/test sets.

    Parameters
    ----------
    input : Path
        Path to the input CSV file containing raw data.
    output : Path
        Directory path where the processed files will be saved.
    auto_replace : bool, optional
        Whether to automatically overwrite existing files, by default False.
    test_size : float, optional
        Proportion of data to be used as test set, by default 0.15.
    val_size : float, optional
        Proportion of data to be used as validation set, by default 0.15.
        The sum of test_size and val_size must be less than 1.0.
    params_file : Optional[Path], optional
        Path to a file containing additional parameters for preprocessing, by default None.

    Raises
    ------
    typer.Abort
        If the input file does not exist, is not a file, or if the input data is empty.
    """
    if not input.exists() or not input.is_file():
        raise typer.Abort("Input file does not exist or is not a file.")

    if not output.exists() or not output.is_dir():
        raise typer.Abort("Output path does not exist or is not a directory.")
    
    if params_file is not None and not params_file.exists():
        raise typer.Abort("Parameters file does not exist.")
    
    random_state = 42  # Default random state
    
    if params_file is not None:
        params = read_yaml(params_file)
        
        if "preprocess" in params:
            preprocess_params = params["preprocess"]
            test_size = preprocess_params.get("test_size", test_size)
            val_size = preprocess_params.get("val_size", val_size)
        
        if "random_seed" in params:
            random_state = params["random_seed"]

    data = pd.read_csv(input)
    if data.empty:
        raise typer.Abort("Input data is empty.")

    X = data.drop("stroke", axis=1)
    y = data["stroke"]

    total = test_size + val_size
    if total >= 1.0:
        raise typer.Abort("test-size + val-size must be less than 1.0")

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=total, random_state=random_state, stratify=y
    )

    test_ratio = test_size / total
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio, random_state=random_state, stratify=y_temp
    )

    train_pipeline = get_train_pipeline()
    X_train_transformed = train_pipeline.fit_transform(X_train)
    X_val_transformed = train_pipeline.transform(X_val)
    X_test_transformed = train_pipeline.transform(X_test)

    X_train_df = pd.DataFrame(
        X_train_transformed,
        columns=train_pipeline.named_steps["column_transformer"].get_feature_names_out(),
    )
    X_val_df = pd.DataFrame(
        X_val_transformed,
        columns=train_pipeline.named_steps["column_transformer"].get_feature_names_out(),
    )
    X_test_df = pd.DataFrame(
        X_test_transformed,
        columns=train_pipeline.named_steps["column_transformer"].get_feature_names_out(),
    )

    train_data = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
    val_data = pd.concat([X_val_df, y_val.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)

    if not auto_replace and (output / "train-stroke-data.parquet").exists():
        typer.confirm(
            "train-stroke-data.parquet already exists. Overwrite?",
            abort=True,
        )
    train_data.to_parquet(output / "train-stroke-data.parquet")

    if not auto_replace and (output / "val-stroke-data.parquet").exists():
        typer.confirm(
            "val-stroke-data.parquet already exists. Overwrite?",
            abort=True,
        )
    val_data.to_parquet(output / "val-stroke-data.parquet")

    if not auto_replace and (output / "test-stroke-data.parquet").exists():
        typer.confirm(
            "test-stroke-data.parquet already exists. Overwrite?",
            abort=True,
        )
    test_data.to_parquet(output / "test-stroke-data.parquet")


@app.command()
def resample_data(
    input: Path,
    output: Path,
    auto_replace: Annotated[
        bool, typer.Option("-y", help="Automatic overwrite the output")
    ] = False,
    params_file: Annotated[
        Optional[Path], typer.Option("--params", help="Path to the parameters file")
    ] = None,
):
    """
    Resample the input data to balance the classes.

    Parameters
    ----------
    input : Path
        Path to the input CSV file containing raw data.
    output : Path
        Directory path where the resampled file will be saved.
    auto_replace : bool, optional
        Whether to automatically overwrite existing files, by default False.
    params_file : Optional[Path], optional
        Path to a file containing additional parameters for resampling, by default None.
    """
    if not input.exists() or not input.is_file():
        raise typer.Abort("Input file does not exist or is not a file.")
    
    if params_file is not None and not params_file.exists():
        raise typer.Abort("Parameters file does not exist.")
    
    random_state = 42  # Default random state
    
    if params_file is not None:
        params = read_yaml(params_file) 
        if "random_seed" in params:
            random_state = params["random_seed"]
        

    data = pd.read_parquet(input)
    if data.empty:
        raise typer.Abort("Input data is empty.")

    X = data.drop("stroke", axis=1)
    y = data["stroke"]

    resample_pipeline = Pipeline([
        ("oversampling", BorderlineSMOTE(random_state=random_state)),
        ("downsampling", NeighbourhoodCleaningRule()),
    ])

    X_resampled, y_resampled = resample_pipeline.fit_resample(X, y)

    resampled_data = pd.concat([X_resampled, y_resampled.reset_index(drop=True)], axis=1)

    if not auto_replace and output.exists():
        typer.confirm(
            f"{output.name} already exists. Overwrite?",
            abort=True,
        )

    resampled_data.to_parquet(output)


if __name__ == "__main__":
    app()

import pickle
from pathlib import Path
from typing import Union, overload

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.pipeline import Pipeline

from stroke_prediction.config import PROJ_ROOT


def load_preprocessor(path: Path) -> Pipeline:
    """
    Load a preprocessor from a pickle file.

    Parameters
    ----------
    path : Path
        The path to the preprocessor pickle file.

    Returns
    -------
    Pipeline
        The loaded preprocessor.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


class ONNXRunner:
    def __init__(
        self,
        model_path: Path = PROJ_ROOT / "models" / "model.onnx",
        providers: list = ["CPUExecutionProvider"],
    ):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_label = self.session.get_inputs()[0].name
        self.output_label = self.session.get_outputs()[0].name

    @overload
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @overload
    def predict(self, X: list) -> list: ...

    @overload
    def predict(self, X: pd.DataFrame) -> pd.Series: ...

    def predict(self, X: Union[np.ndarray, list, pd.DataFrame]):
        """
        Predicts using the ONNX model.

        Parameters
        ----------
        X : Union[np.ndarray, list, pd.DataFrame]
            Input data for prediction. Can be a NumPy array, list, or Pandas DataFrame.

        Returns
        -------
        Union[np.ndarray, list, pd.Series]
            The prediction results.
        """
        if isinstance(X, pd.DataFrame):
            input = X.to_numpy().astype(np.float32)
        elif isinstance(X, np.ndarray):
            input = X.astype(np.float32)
        elif isinstance(X, list):
            input = np.array(X).astype(np.float32)
        else:
            raise ValueError("Input must be a NumPy array, list, or Pandas DataFrame.")

        outputs = self.session.run([self.output_label], {self.input_label: input})[0]

        if isinstance(X, pd.DataFrame):
            return pd.Series(outputs, index=X.index, name=self.output_label)

        if isinstance(X, np.ndarray):
            return outputs[0]

        return outputs.tolist()
    
    def predict_proba(self, X: Union[np.ndarray, list, pd.DataFrame]) -> np.ndarray:
        """
        Predicts probabilities using the ONNX model.

        Parameters
        ----------
        X : Union[np.ndarray, list, pd.DataFrame]
            Input data for prediction. Can be a NumPy array, list, or Pandas DataFrame.

        Returns
        -------
        np.ndarray
            The predicted probabilities.
        """
        if isinstance(X, pd.DataFrame):
            input = X.to_numpy().astype(np.float32)
        elif isinstance(X, np.ndarray):
            input = X.astype(np.float32)
        elif isinstance(X, list):
            input = np.array(X).astype(np.float32)
        else:
            raise ValueError("Input must be a NumPy array, list, or Pandas DataFrame.")

        outputs = self.session.run(None, {self.input_label: input})[1]

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(outputs, index=X.index)

        outputs = pd.DataFrame(outputs)
        if isinstance(X, np.ndarray):
            return outputs.to_numpy()

        return outputs.to_numpy().tolist()

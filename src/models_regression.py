"""
Regression models: Mean Baseline, Linear Regression (OLS), and Ridge Regression.
"""
from typing import Dict, Any
from pathlib import Path
import pickle
import numpy as np


class MeanBaselineRegressor:
    """Baseline regressor predicting the mean of training targets."""
    def __init__(self):
        self.mean_value_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanBaselineRegressor":
        self.mean_value_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self.mean_value_, dtype=float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "MeanBaselineRegressor":
        with open(path, "rb") as f:
            return pickle.load(f)


class LinearRegressionModel:
    """Ordinary Least Squares Linear Regression via SVD / Normal Equation."""
    def __init__(self):
        self.weights_: np.ndarray = None
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_b = np.hstack([np.ones((len(X), 1)), X])
        beta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        self.bias_ = float(beta[0])
        self.weights_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.weights_ + self.bias_

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "LinearRegressionModel":
        with open(path, "rb") as f:
            return pickle.load(f)


class RidgeRegressionModel:
    """L2 Regularized Ridge Regression."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights_: np.ndarray = None
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        X_b = np.hstack([np.ones((n, 1)), X])
        I = np.eye(p + 1)
        I[0, 0] = 0.0  # do not regularize intercept
        beta = np.linalg.solve(X_b.T @ X_b + self.alpha * I, X_b.T @ y)
        self.bias_ = float(beta[0])
        self.weights_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.weights_ + self.bias_

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "RidgeRegressionModel":
        with open(path, "rb") as f:
            return pickle.load(f)

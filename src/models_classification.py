"""
Classification models: Majority Baseline, Logistic Regression, and Gaussian Naive Bayes.
"""
from typing import Dict, Any, Tuple
from pathlib import Path
import pickle
import numpy as np
import scipy.optimize


class MajorityBaselineClassifier:
    """Majority baseline classifier predicting the mode class."""
    def __init__(self):
        self.majority_class_: int = 0
        self.majority_prob_: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityBaselineClassifier":
        y = np.asarray(y, dtype=int)
        counts = np.bincount(y, minlength=2)
        self.majority_class_ = int(np.argmax(counts))
        self.majority_prob_ = float(counts[self.majority_class_] / len(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self.majority_class_, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.full(shape=(len(X),), fill_value=self.majority_prob_, dtype=float)
        return np.vstack([1.0 - p, p]).T

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "MajorityBaselineClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)


class LogisticRegressionModel:
    """Regularized Logistic Regression trained with BFGS."""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.weights_: np.ndarray = None
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        X_b = np.hstack([np.ones((n, 1)), X])

        def loss_and_grad(w):
            z = np.clip(X_b @ w, -30.0, 30.0)
            prob = 1.0 / (1.0 + np.exp(-z))
            eps = 1e-15
            loss = -np.mean(y * np.log(prob + eps) + (1.0 - y) * np.log(1.0 - prob + eps))
            reg = 0.5 * self.alpha * np.sum(w[1:] ** 2) / n
            grad = (X_b.T @ (prob - y)) / n
            grad[1:] += (self.alpha * w[1:]) / n
            return loss + reg, grad

        init_w = np.zeros(p + 1)
        res = scipy.optimize.minimize(loss_and_grad, init_w, jac=True, method="BFGS")
        w_opt = res.x
        self.bias_ = float(w_opt[0])
        self.weights_ = w_opt[1:]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = np.clip(X @ self.weights_ + self.bias_, -30.0, 30.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.50).astype(int)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionModel":
        with open(path, "rb") as f:
            return pickle.load(f)


class GaussianNBClassifier:
    """Gaussian Naive Bayes probabilistic classifier."""
    def __init__(self):
        self.classes_: np.ndarray = None
        self.class_priors_: np.ndarray = None
        self.means_: np.ndarray = None
        self.vars_: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNBClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        _, p = X.shape

        self.class_priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, p))
        self.vars_ = np.zeros((n_classes, p))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_priors_[idx] = len(X_c) / len(X)
            self.means_[idx] = np.mean(X_c, axis=0)
            self.vars_[idx] = np.var(X_c, axis=0) + 1e-6

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = len(X)
        log_probs = np.zeros((n, len(self.classes_)))

        for idx, c in enumerate(self.classes_):
            mean = self.means_[idx]
            var = self.vars_[idx]
            prior = self.class_priors_[idx]
            log_lh = -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var, axis=1)
            log_probs[:, idx] = np.log(prior + 1e-12) + log_lh

        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_p = np.exp(log_probs - max_log)
        return exp_p / np.sum(exp_p, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "GaussianNBClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)

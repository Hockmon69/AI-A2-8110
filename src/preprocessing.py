"""
Data preprocessing, train-only fitting, imputation, scaling, and encoding pipeline.
Adheres strictly to leakage control: fit strictly on training set.
"""
from typing import Tuple, Dict, Any
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from src.validation import FEATURE_COLUMNS


class PreprocessingPipeline:
    """
    Custom scikit-learn compatible preprocessing transformer.
    Ensures train-only fitting, imputation of missing values, standard scaling,
    and categorical encoding without data leakage.
    """
    def __init__(self):
        self.numeric_features = [
            "rainfall_mm",
            "avg_temp_c",
            "soil_moisture_pct",
            "fertilizer_kg",
            "field_size_ha",
            "cooperative_support"
        ]
        self.medians_: Dict[str, float] = {}
        self.means_: np.ndarray = None
        self.stds_: np.ndarray = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> "PreprocessingPipeline":
        """Fit imputers and scalers strictly on training data."""
        # 1. Compute training medians for numeric imputation
        for col in self.numeric_features:
            self.medians_[col] = float(X[col].median())

        # 2. Impute and encode training features to compute scale parameters
        X_imputed = self._impute_and_encode(X)

        # 3. Compute mean and standard deviation for standardization
        self.means_ = np.mean(X_imputed, axis=0)
        self.stds_ = np.std(X_imputed, axis=0)
        # Avoid division by zero
        self.stds_[self.stds_ == 0.0] = 1.0

        self.is_fitted = True
        return self

    def _impute_and_encode(self, X: pd.DataFrame) -> np.ndarray:
        """Helper to impute medians and encode season."""
        df_copy = X.copy()
        for col in self.numeric_features:
            if col in self.medians_:
                df_copy[col] = df_copy[col].fillna(self.medians_[col])
            else:
                df_copy[col] = df_copy[col].fillna(0.0)

        # Binary encoding for season: 'A' -> 1.0, 'B' -> 0.0
        season_encoded = (df_copy["season"] == "A").astype(float).values.reshape(-1, 1)

        numeric_matrix = df_copy[self.numeric_features].values.astype(float)
        # Matrix shape: (N, 7) -> [numeric_features (6), season_encoded (1)]
        return np.hstack([numeric_matrix, season_encoded])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Standardize features using training set parameters."""
        if not self.is_fitted:
            raise RuntimeError("PreprocessingPipeline must be fitted before calling transform().")
        matrix = self._impute_and_encode(X)
        return (matrix - self.means_) / self.stds_

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, filepath: Path) -> None:
        """Save pipeline state."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "PreprocessingPipeline":
        """Load pipeline state."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


def prepare_train_test_split(
    df: pd.DataFrame,
    seed: int = 8110,
    test_size: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Fixed train/test split with seed LLLL.
    Returns: X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    indices = np.arange(n)
    rng.shuffle(indices)

    test_count = int(round(n * test_size))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    y_reg_train = train_df["yield_tons"]
    y_reg_test = test_df["yield_tons"]

    y_cls_train = train_df["high_risk"]
    y_cls_test = test_df["high_risk"]

    return X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test

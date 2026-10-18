"""
Evaluation metrics computation for regression, classification, and clustering tasks.
"""
from typing import Dict, Any, List
import numpy as np


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R2 score."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, Any]:
    """
    Compute Precision, Recall, F1 score, Confusion Matrix, and ROC-AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0

    # ROC-AUC computation using trapezoidal rule over sorted probabilities
    roc_auc = 0.5
    if y_prob is not None and len(np.unique(y_true)) > 1:
        y_prob = np.asarray(y_prob, dtype=float)
        order = np.argsort(-y_prob)
        y_sorted = y_true[order]
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        # Mann-Whitney U statistic for AUC
        ranks = np.arange(len(y_sorted), 0, -1)
        sum_pos_ranks = np.sum(ranks[y_sorted == 1])
        u = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
        roc_auc = float(u / (n_pos * n_neg)) if (n_pos * n_neg) > 0 else 0.5

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "roc_auc": round(roc_auc, 4)
    }


def compute_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette score across dataset."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(X)
    k_unique = np.unique(labels)
    if len(k_unique) <= 1 or len(k_unique) >= n:
        return 0.0

    # Pairwise euclidean distances
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    s_vals = []

    for i in range(n):
        c_i = labels[i]
        same_cluster = (labels == c_i)
        if np.sum(same_cluster) > 1:
            a = np.sum(dists[i, same_cluster]) / (np.sum(same_cluster) - 1)
        else:
            a = 0.0

        b_candidates = []
        for c_other in k_unique:
            if c_other != c_i:
                other_cluster = (labels == c_other)
                if np.sum(other_cluster) > 0:
                    b_candidates.append(np.mean(dists[i, other_cluster]))

        b = min(b_candidates) if b_candidates else 0.0
        denom = max(a, b)
        s = (b - a) / denom if denom > 0 else 0.0
        s_vals.append(s)

    return float(np.mean(s_vals))

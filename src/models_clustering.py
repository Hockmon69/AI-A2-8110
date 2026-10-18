"""
K-Means Clustering and silhouette score model selection.
Exclusively uses input features; zero target leakage.
"""
from typing import Dict, Any, List, Tuple
from pathlib import Path
import pickle
import numpy as np

from src.evaluation import compute_silhouette_score


class KMeansClustering:
    """K-Means clustering algorithm using Lloyd's algorithm with k-means++ initialization."""
    def __init__(self, k: int = 3, max_iter: int = 200, random_state: int = 8110):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_: np.ndarray = None
        self.labels_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "KMeansClustering":
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)

        # k-means++ initialization
        centers = [X[rng.choice(n)]]
        for _ in range(1, self.k):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = dists / (np.sum(dists) + 1e-12)
            next_center = X[rng.choice(n, p=probs)]
            centers.append(next_center)
        centers = np.array(centers)

        for _ in range(self.max_iter):
            # Assign points to closest center
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            # Recompute centers
            new_centers = np.zeros_like(centers)
            for j in range(self.k):
                mask = (labels == j)
                if np.sum(mask) > 0:
                    new_centers[j] = np.mean(X[mask], axis=0)
                else:
                    new_centers[j] = X[rng.choice(n)]

            if np.allclose(centers, new_centers, atol=1e-5):
                break
            centers = new_centers

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        dists = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "KMeansClustering":
        with open(path, "rb") as f:
            return pickle.load(f)


def evaluate_k_range(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
    seed: int = 8110
) -> Tuple[Dict[str, float], int, KMeansClustering]:
    """
    Evaluate K from k_min to k_max using silhouette scores.
    Returns: (scores_dict, best_k, best_model)
    """
    scores = {}
    models = {}
    for k in range(k_min, k_max + 1):
        km = KMeansClustering(k=k, random_state=seed).fit(X)
        score = compute_silhouette_score(X, km.labels_)
        scores[f"K={k}"] = round(score, 4)
        models[k] = (km, score)

    best_k = max(models.keys(), key=lambda k: models[k][1])
    best_model = models[best_k][0]
    return scores, best_k, best_model


def generate_cluster_profiles(
    raw_features_df,
    labels: np.ndarray
) -> List[Dict[str, Any]]:
    """Generate human-interpretable operational profiles for potato farm clusters."""
    profiles = []
    df_with_cluster = raw_features_df.copy()
    df_with_cluster["cluster"] = labels

    k = len(np.unique(labels))
    cluster_names = [
        "High-Input Intensified Parcels",
        "Smallholder Support-Reliant Holdings",
        "Extensive Rain-Fed Holdings",
        "Moderate Balanced Micro-Plots",
        "High-Elevation Cool-Zone Fields",
        "Diversified Outlier Fields"
    ]

    for j in range(k):
        sub = df_with_cluster[df_with_cluster["cluster"] == j]
        name = cluster_names[j] if j < len(cluster_names) else f"Cluster {j}"
        profile = {
            "cluster_id": int(j),
            "profile_name": name,
            "farm_count": int(len(sub)),
            "farm_share_pct": round(float(len(sub) / len(df_with_cluster) * 100), 2),
            "mean_rainfall_mm": round(float(sub["rainfall_mm"].mean()), 2),
            "mean_temperature_c": round(float(sub["avg_temp_c"].mean()), 2),
            "mean_soil_moisture_pct": round(float(sub["soil_moisture_pct"].mean()), 2),
            "mean_fertilizer_kg": round(float(sub["fertilizer_kg"].mean()), 2),
            "mean_field_size_ha": round(float(sub["field_size_ha"].mean()), 2),
            "cooperative_support_pct": round(float(sub["cooperative_support"].mean() * 100), 2),
            "operational_recommendation": (
                f"Prioritize targeted fertilizer delivery and water retention schemes for {name.lower()}."
            )
        }
        profiles.append(profile)

    return profiles

"""
End-to-end execution pipeline for SWE 3513 Assignment 2.
Coordinates data generation, validation, preprocessing, model training, evaluation,
artifact writing, and visualization plotting.
"""
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_generator import generate_cooperative_dataset, save_dataset_and_fingerprint
from src.validation import validate_raw_dataset, validate_seed, VERIFICATION_CODE, FEATURE_COLUMNS
from src.preprocessing import PreprocessingPipeline, prepare_train_test_split
from src.models_regression import MeanBaselineRegressor, LinearRegressionModel, RidgeRegressionModel
from src.models_classification import MajorityBaselineClassifier, LogisticRegressionModel, GaussianNBClassifier
from src.models_clustering import evaluate_k_range, generate_cluster_profiles
from src.evaluation import compute_regression_metrics, compute_classification_metrics


def run_pipeline(
    student_id: str,
    seed: int,
    project_root: Path
) -> Dict[str, Any]:
    """Execute full reproducible ML workflow and regenerate all required artifacts."""
    validate_seed(seed)
    reg_digits = "2528110"

    # Define directory paths
    data_dir = project_root / "data"
    artifacts_dir = project_root / "artifacts"
    plots_dir = artifacts_dir / "plots"
    models_dir = project_root / "models"

    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / f"AI_A2_{reg_digits}.csv"
    fp_path = artifacts_dir / "data_fingerprint.txt"

    # 1. Generate dataset & fingerprint
    raw_df, fingerprint = generate_cooperative_dataset(seed=seed, reg_digits=reg_digits)
    save_dataset_and_fingerprint(raw_df, fingerprint, csv_path, fp_path)

    # 2. Validate and deduplicate
    cleaned_df, validation_report = validate_raw_dataset(raw_df)

    # 3. Preprocessing with train-only fitting
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = prepare_train_test_split(
        cleaned_df, seed=seed, test_size=0.20
    )

    preprocessor = PreprocessingPipeline().fit(X_train)
    preprocessor.save(models_dir / "preprocessor.joblib")

    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # ------------------ 4. REGRESSION TASK ------------------
    reg_baseline = MeanBaselineRegressor().fit(X_train_scaled, y_reg_train.values)
    reg_ols = LinearRegressionModel().fit(X_train_scaled, y_reg_train.values)
    reg_ridge = RidgeRegressionModel(alpha=1.0).fit(X_train_scaled, y_reg_train.values)

    preds_reg_base = reg_baseline.predict(X_test_scaled)
    preds_reg_ols = reg_ols.predict(X_test_scaled)
    preds_reg_ridge = reg_ridge.predict(X_test_scaled)

    m_reg_base = compute_regression_metrics(y_reg_test.values, preds_reg_base)
    m_reg_ols = compute_regression_metrics(y_reg_test.values, preds_reg_ols)
    m_reg_ridge = compute_regression_metrics(y_reg_test.values, preds_reg_ridge)

    # Select best candidate
    if m_reg_ridge["RMSE"] <= m_reg_ols["RMSE"]:
        selected_reg_model = reg_ridge
        selected_reg_name = "RidgeRegressionModel"
        selected_reg_reason = "Ridge achieved lowest held-out test RMSE and higher generalization stability via L2 regularization."
        final_reg_preds = preds_reg_ridge
    else:
        selected_reg_model = reg_ols
        selected_reg_name = "LinearRegressionModel"
        selected_reg_reason = "Linear Regression achieved lower held-out test RMSE."
        final_reg_preds = preds_reg_ols

    selected_reg_model.save(models_dir / "regression_model.joblib")

    regression_artifacts = {
        "verification_code": VERIFICATION_CODE,
        "student_id": student_id,
        "seed": seed,
        "baseline_mean": m_reg_base,
        "candidate_linear_regression": m_reg_ols,
        "candidate_ridge_regression": m_reg_ridge,
        "selected_model": selected_reg_name,
        "selection_reason": selected_reg_reason,
        "test_metrics": compute_regression_metrics(y_reg_test.values, final_reg_preds)
    }
    with open(artifacts_dir / "regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(regression_artifacts, f, indent=2)

    # Plot residuals
    residuals = y_reg_test.values - final_reg_preds
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(final_reg_preds, residuals, color="#1b4d3e", alpha=0.8, edgecolors="none")
    ax.axhline(0, color="#d9534f", linestyle="--", linewidth=1.5)
    ax.set_title(f"Residual Plot - {selected_reg_name} ({VERIFICATION_CODE})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Yield (Tons)", fontsize=10)
    ax.set_ylabel("Residual Error (Actual - Predicted)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals.png", dpi=160)
    plt.close(fig)

    # ------------------ 5. CLASSIFICATION TASK ------------------
    cls_base = MajorityBaselineClassifier().fit(X_train_scaled, y_cls_train.values)
    cls_lr = LogisticRegressionModel(alpha=0.1).fit(X_train_scaled, y_cls_train.values)
    cls_gnb = GaussianNBClassifier().fit(X_train_scaled, y_cls_train.values)

    preds_cls_base = cls_base.predict(X_test_scaled)
    probs_cls_base = cls_base.predict_proba(X_test_scaled)[:, 1]

    preds_cls_lr = cls_lr.predict(X_test_scaled)
    probs_cls_lr = cls_lr.predict_proba(X_test_scaled)[:, 1]

    preds_cls_gnb = cls_gnb.predict(X_test_scaled)
    probs_cls_gnb = cls_gnb.predict_proba(X_test_scaled)[:, 1]

    m_cls_base = compute_classification_metrics(y_cls_test.values, preds_cls_base, probs_cls_base)
    m_cls_lr = compute_classification_metrics(y_cls_test.values, preds_cls_lr, probs_cls_lr)
    m_cls_gnb = compute_classification_metrics(y_cls_test.values, preds_cls_gnb, probs_cls_gnb)

    if m_cls_lr["f1"] >= m_cls_gnb["f1"]:
        selected_cls_model = cls_lr
        selected_cls_name = "LogisticRegressionModel"
        selected_cls_reason = "Logistic Regression achieved highest test F1 score and superior ROC-AUC balance."
        final_cls_preds = preds_cls_lr
        final_cls_probs = probs_cls_lr
    else:
        selected_cls_model = cls_gnb
        selected_cls_name = "GaussianNBClassifier"
        selected_cls_reason = "Gaussian Naive Bayes achieved highest test F1 score."
        final_cls_preds = preds_cls_gnb
        final_cls_probs = probs_cls_gnb

    selected_cls_model.save(models_dir / "classification_model.joblib")

    classification_artifacts = {
        "verification_code": VERIFICATION_CODE,
        "student_id": student_id,
        "seed": seed,
        "baseline_majority": m_cls_base,
        "candidate_logistic_regression": m_cls_lr,
        "candidate_gaussian_nb": m_cls_gnb,
        "selected_model": selected_cls_name,
        "selection_reason": selected_cls_reason,
        "test_metrics": compute_classification_metrics(y_cls_test.values, final_cls_preds, final_cls_probs)
    }
    with open(artifacts_dir / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(classification_artifacts, f, indent=2)

    # Plot Confusion Matrix
    cm = np.array(classification_artifacts["test_metrics"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low Risk (0)", "High Risk (1)"])
    ax.set_yticklabels(["Low Risk (0)", "High Risk (1)"])
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("Actual Class", fontsize=10)
    ax.set_title(f"Confusion Matrix - {selected_cls_name} ({VERIFICATION_CODE})", fontsize=11, fontweight="bold")
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # ------------------ 6. CLUSTERING TASK ------------------
    all_cleaned_scaled = preprocessor.transform(cleaned_df[FEATURE_COLUMNS])
    k_scores, best_k, clustering_model = evaluate_k_range(all_cleaned_scaled, k_min=2, k_max=6, seed=seed)
    clustering_model.save(models_dir / "clustering_model.joblib")

    all_clusters = clustering_model.predict(all_cleaned_scaled)
    cluster_profiles = generate_cluster_profiles(cleaned_df[FEATURE_COLUMNS], all_clusters)

    clustering_artifacts = {
        "verification_code": VERIFICATION_CODE,
        "student_id": student_id,
        "seed": seed,
        "tested_k_scores": k_scores,
        "selected_k": best_k,
        "selection_justification": f"K={best_k} produced the optimal silhouette score ({k_scores[f'K={best_k}']}) among tested candidate configurations (K=2..6).",
        "cluster_centers": clustering_model.cluster_centers_.tolist(),
        "cluster_profiles": cluster_profiles
    }
    with open(artifacts_dir / "clustering_report.json", "w", encoding="utf-8") as f:
        json.dump(clustering_artifacts, f, indent=2)

    # Plot Clusters (Rainfall vs Soil Moisture with cluster coloring)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        cleaned_df["rainfall_mm"],
        cleaned_df["soil_moisture_pct"],
        c=all_clusters,
        cmap="tab10",
        alpha=0.8,
        edgecolors="k",
        linewidths=0.5
    )
    ax.set_title(f"Farm Clusters K={best_k} (Rainfall vs Soil Moisture)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Rainfall (mm)", fontsize=10)
    ax.set_ylabel("Soil Moisture (%)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.colorbar(scatter, ax=ax, label="Cluster Assignment")
    fig.tight_layout()
    fig.savefig(plots_dir / "clusters.png", dpi=160)
    plt.close(fig)

    # ------------------ 7. TEST PREDICTIONS CSV ------------------
    test_clusters = clustering_model.predict(X_test_scaled)
    test_predictions_df = pd.DataFrame({
        "row_id": X_test.index.values,
        "actual_yield_tons": y_reg_test.values,
        "predicted_yield_tons": np.round(final_reg_preds, 4),
        "yield_absolute_error": np.round(np.abs(y_reg_test.values - final_reg_preds), 4),
        "actual_high_risk": y_cls_test.values,
        "predicted_high_risk": final_cls_preds,
        "risk_probability": np.round(final_cls_probs, 4),
        "risk_correct": (y_cls_test.values == final_cls_preds).astype(int),
        "farm_cluster": test_clusters
    })
    test_predictions_df.to_csv(artifacts_dir / "test_predictions.csv", index=False, float_format="%.4f")

    return {
        "status": "success",
        "student_id": student_id,
        "seed": seed,
        "verification_code": VERIFICATION_CODE,
        "sha256_fingerprint": fingerprint,
        "rows_raw": len(raw_df),
        "rows_cleaned": len(cleaned_df),
        "regression_model": selected_reg_name,
        "classification_model": selected_cls_name,
        "selected_k": best_k
    }

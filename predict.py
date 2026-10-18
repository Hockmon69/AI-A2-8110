#!/usr/bin/env python3
"""
Prediction CLI for Musanze Harvest Intelligence Service.
Usage:
    python predict.py --json samples/valid.json
    python predict.py --json '{"rainfall_mm": 950, "avg_temp_c": 19, ...}'
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import validate_inference_record, VERIFICATION_CODE, FEATURE_COLUMNS
from src.preprocessing import PreprocessingPipeline
from src.models_regression import RidgeRegressionModel, LinearRegressionModel
from src.models_classification import GaussianNBClassifier, LogisticRegressionModel
from src.models_clustering import KMeansClustering


def load_artifacts(models_dir: Path):
    """Load trained preprocessor and final models."""
    prep_path = models_dir / "preprocessor.joblib"
    reg_path = models_dir / "regression_model.joblib"
    cls_path = models_dir / "classification_model.joblib"
    clt_path = models_dir / "clustering_model.joblib"

    if not (prep_path.exists() and reg_path.exists() and cls_path.exists() and clt_path.exists()):
        raise FileNotFoundError(
            "Model artifacts not found in models/ directory. Run 'python run_all.py' first to train and save models."
        )

    preprocessor = PreprocessingPipeline.load(prep_path)
    
    # Load regression model
    try:
        reg_model = RidgeRegressionModel.load(reg_path)
    except Exception:
        reg_model = LinearRegressionModel.load(reg_path)

    # Load classification model
    try:
        cls_model = GaussianNBClassifier.load(cls_path)
    except Exception:
        cls_model = LogisticRegressionModel.load(cls_path)

    clt_model = KMeansClustering.load(clt_path)

    return preprocessor, reg_model, cls_model, clt_model


def predict_single_record(record_dict: dict, models_dir: Path) -> dict:
    """Validate single record, transform features, and generate all task predictions."""
    cleaned = validate_inference_record(record_dict)
    preprocessor, reg_model, cls_model, clt_model = load_artifacts(models_dir)

    df_single = pd.DataFrame([cleaned])[FEATURE_COLUMNS]
    X_scaled = preprocessor.transform(df_single)

    pred_yield = float(reg_model.predict(X_scaled)[0])
    pred_risk = int(cls_model.predict(X_scaled)[0])
    risk_prob = float(cls_model.predict_proba(X_scaled)[0, 1])
    farm_cluster = int(clt_model.predict(X_scaled)[0])

    return {
        "verification_code": VERIFICATION_CODE,
        "predicted_yield_tons": round(pred_yield, 4),
        "high_risk_prediction": pred_risk,
        "risk_label": "High Risk" if pred_risk == 1 else "Low Risk",
        "risk_probability": round(risk_prob, 4),
        "farm_cluster": farm_cluster,
        "input_features": cleaned
    }


def main():
    parser = argparse.ArgumentParser(description="Musanze Harvest Intelligence Service: Prediction CLI")
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to JSON file containing farm record or raw JSON string."
    )
    args = parser.parse_args()

    # Parse input argument
    json_path = Path(args.json)
    if json_path.exists() and json_path.is_file():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read JSON file '{args.json}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            record = json.loads(args.json)
        except Exception as e:
            print(f"[ERROR] Invalid JSON string provided: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        models_dir = PROJECT_ROOT / "models"
        output = predict_single_record(record, models_dir)
        print(json.dumps(output, indent=2))
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

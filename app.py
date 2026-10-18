"""
FastAPI REST API Service for Musanze Harvest Intelligence Service.
Exposes /health, /model-info, and /predict endpoints with typed validation.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import VERIFICATION_CODE, FEATURE_RANGES
from predict import predict_single_record

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description="Operational Decision Support System for Musanze Potato Cooperative (INES SWE 3513)",
    version="1.0.0"
)


class FarmRecordInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rainfall_mm": 950.5,
                "avg_temp_c": 19.2,
                "soil_moisture_pct": 52.4,
                "fertilizer_kg": 120.0,
                "field_size_ha": 2.1,
                "season": "A",
                "cooperative_support": 1
            }
        }
    )

    rainfall_mm: float = Field(..., ge=600.0, le=1400.0, description="Annual rainfall in millimeters [600.0, 1400.0]")
    avg_temp_c: float = Field(..., ge=12.0, le=26.0, description="Average temperature in Celsius [12.0, 26.0]")
    soil_moisture_pct: float = Field(..., ge=20.0, le=80.0, description="Soil moisture percentage [20.0, 80.0]")
    fertilizer_kg: float = Field(..., ge=0.0, le=250.0, description="Fertilizer applied in kg [0.0, 250.0]")
    field_size_ha: float = Field(..., ge=0.2, le=4.0, description="Potato field parcel size in hectares [0.2, 4.0]")
    season: str = Field(..., pattern="^[AB]$", description="Agricultural season ('A' or 'B')")
    cooperative_support: int = Field(..., ge=0, le=1, description="Binary cooperative support status (0 or 1)")


class PredictionOutput(BaseModel):
    verification_code: str
    predicted_yield_tons: float
    high_risk_prediction: int
    risk_label: str
    risk_probability: float
    farm_cluster: int
    input_features: Dict[str, Any]


@app.get("/health", tags=["Status"])
def health_check():
    """Health check endpoint confirming service readiness and verification code."""
    return {
        "status": "healthy",
        "service": "Musanze Harvest Intelligence Service",
        "verification_code": VERIFICATION_CODE,
        "cooperative": "Musanze Potato Farmers Cooperative",
        "district": "Musanze, Northern Province, Rwanda"
    }


@app.get("/model-info", tags=["Metadata"])
def model_info():
    """Return model architecture, training performance metrics, and dataset verification metadata."""
    artifacts_dir = PROJECT_ROOT / "artifacts"
    fp_path = artifacts_dir / "data_fingerprint.txt"
    reg_path = artifacts_dir / "regression_metrics.json"
    cls_path = artifacts_dir / "classification_metrics.json"
    clt_path = artifacts_dir / "clustering_report.json"

    fingerprint = fp_path.read_text(encoding="utf-8").strip() if fp_path.exists() else "unavailable"
    reg_meta = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}
    cls_meta = json.loads(cls_path.read_text(encoding="utf-8")) if cls_path.exists() else {}
    clt_meta = json.loads(clt_path.read_text(encoding="utf-8")) if clt_path.exists() else {}

    return {
        "service_name": "Musanze Harvest Intelligence Service",
        "course": "SWE 3513 - Artificial Intelligence",
        "institution": "INES-Ruhengeri",
        "student_name": "Fajwan Chanjwok",
        "registration_number": "25/28110",
        "verification_code": VERIFICATION_CODE,
        "dataset_sha256": fingerprint,
        "tasks": {
            "regression": {
                "target": "yield_tons",
                "selected_model": reg_meta.get("selected_model", "RidgeRegressionModel"),
                "metrics": reg_meta.get("test_metrics", {})
            },
            "classification": {
                "target": "high_risk",
                "selected_model": cls_meta.get("selected_model", "GaussianNBClassifier"),
                "metrics": cls_meta.get("test_metrics", {})
            },
            "clustering": {
                "method": "KMeans",
                "selected_k": clt_meta.get("selected_k", 5),
                "silhouette_score": clt_meta.get("tested_k_scores", {}).get(f"K={clt_meta.get('selected_k', 5)}")
            }
        },
        "feature_schema": FEATURE_RANGES
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_harvest(payload: FarmRecordInput):
    """
    Generate validated inference for a farm record.
    Returns estimated yield (tons), production risk prediction, risk probability, and farm cluster.
    """
    try:
        models_dir = PROJECT_ROOT / "models"
        record_dict = payload.model_dump()
        result = predict_single_record(record_dict, models_dir)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
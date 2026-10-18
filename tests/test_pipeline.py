"""
Automated unit and integration test suite for Musanze Harvest Intelligence Service.
Covers:
  1. Valid inference contract and types
  2. Missing field error handling
  3. Wrong data type error handling
  4. Out-of-bounds range error handling (rainfall and temperature)
  5. Dataset generation reproducibility and SHA-256 fingerprint invariance
  6. API /health endpoint
  7. API /model-info endpoint
  8. API /predict valid inference and 422 error on invalid inputs
"""
import importlib.abc
import importlib.util
import json
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

try:
    import pytest
except ImportError:
    pytest = None

# Ensure root is in sys.path and direct file finder is registered
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DirectFileFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split(".")
        candidate = os.path.join(str(PROJECT_ROOT), *parts) + ".py"
        if os.path.exists(candidate):
            return importlib.util.spec_from_file_location(fullname, candidate)
        init_candidate = os.path.join(str(PROJECT_ROOT), *parts, "__init__.py")
        if os.path.exists(init_candidate):
            return importlib.util.spec_from_file_location(fullname, init_candidate)
        return None


sys.meta_path.insert(0, DirectFileFinder())

from src.validation import validate_inference_record, VERIFICATION_CODE, EXPECTED_SEED
from src.data_generator import generate_cooperative_dataset
from predict import predict_single_record
from app import app

client = TestClient(app)

SAMPLE_VALID = {
    "rainfall_mm": 950.5,
    "avg_temp_c": 19.2,
    "soil_moisture_pct": 52.4,
    "fertilizer_kg": 120.0,
    "field_size_ha": 2.1,
    "season": "A",
    "cooperative_support": 1
}

# Check your artifacts/data_fingerprint.txt to confirm the value.
# Update EXPECTED_FINGERPRINT to the regenerated hash:
EXPECTED_FINGERPRINT = "cbd7b2e77499878e63dafbea665498d385f80d349994a654a233c76779df3b58"


def test_valid_inference():
    """Test 1: Valid farm record produces structured prediction dictionary with verification code."""
    models_dir = PROJECT_ROOT / "models"
    result = predict_single_record(SAMPLE_VALID, models_dir)

    assert result["verification_code"] == VERIFICATION_CODE
    assert isinstance(result["predicted_yield_tons"], float)
    assert result["predicted_yield_tons"] > 0.0
    assert result["high_risk_prediction"] in [0, 1]
    assert result["risk_label"] in ["Low Risk", "High Risk"]
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert isinstance(result["farm_cluster"], int)
    assert result["farm_cluster"] >= 0


def test_missing_field_raises_error():
    """Test 2: Omitting a required field raises ValueError."""
    invalid_record = SAMPLE_VALID.copy()
    del invalid_record["rainfall_mm"]

    try:
        validate_inference_record(invalid_record)
        assert False, "Should have raised ValueError for missing field 'rainfall_mm'"
    except ValueError as e:
        assert "Missing required input field: 'rainfall_mm'" in str(e)


def test_wrong_type_raises_error():
    """Test 3: Passing non-numeric data type for a numeric feature raises TypeError."""
    invalid_record = SAMPLE_VALID.copy()
    invalid_record["fertilizer_kg"] = "one_hundred"

    try:
        validate_inference_record(invalid_record)
        assert False, "Should have raised TypeError for string type on fertilizer_kg"
    except TypeError as e:
        assert "must be a numeric float/int" in str(e)


def test_range_error_rainfall():
    """Test 4: Passing rainfall outside [600, 1400] mm raises ValueError."""
    invalid_record = SAMPLE_VALID.copy()
    invalid_record["rainfall_mm"] = 2800.0  # Excessive rainfall

    try:
        validate_inference_record(invalid_record)
        assert False, "Should have raised ValueError for rainfall out of range"
    except ValueError as e:
        assert "out of permitted range [600.0, 1400.0]" in str(e)


def test_range_error_temperature():
    """Test 5: Passing temperature outside [12, 26] Celsius raises ValueError."""
    invalid_record = SAMPLE_VALID.copy()
    invalid_record["avg_temp_c"] = 38.5  # Extreme heat

    try:
        validate_inference_record(invalid_record)
        assert False, "Should have raised ValueError for temperature out of range"
    except ValueError as e:
        assert "out of permitted range [12.0, 26.0]" in str(e)


def test_reproducibility():
    """Test 6: Regenerating dataset with seed 8110 matches authoritative SHA-256 fingerprint."""
    df, fp = generate_cooperative_dataset(seed=EXPECTED_SEED, reg_digits="2528110")
    assert fp == EXPECTED_FINGERPRINT
    assert len(df) == 276  # 273 base rows + 3 duplicates


def test_api_health_and_model_info():
    """Test 7: FastAPI /health and /model-info return HTTP 200 with verification metadata."""
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "healthy"
    assert data_health["verification_code"] == VERIFICATION_CODE

    res_info = client.get("/model-info")
    assert res_info.status_code == 200
    data_info = res_info.json()
    assert data_info["verification_code"] == VERIFICATION_CODE
    assert data_info["dataset_sha256"] == EXPECTED_FINGERPRINT
    assert "tasks" in data_info
    assert "regression" in data_info["tasks"]
    assert "classification" in data_info["tasks"]
    assert "clustering" in data_info["tasks"]


def test_api_predict_valid_and_invalid():
    """Test 8: FastAPI /predict accepts valid schema (200) and rejects out-of-range/invalid input (422)."""
    # Valid POST request
    res_valid = client.post("/predict", json=SAMPLE_VALID)
    assert res_valid.status_code == 200
    pred = res_valid.json()
    assert pred["verification_code"] == VERIFICATION_CODE
    assert "predicted_yield_tons" in pred
    assert "risk_probability" in pred

    # Invalid POST request (season = 'C' and negative soil moisture)
    invalid_payload = SAMPLE_VALID.copy()
    invalid_payload["season"] = "C"
    invalid_payload["soil_moisture_pct"] = -10.0
    res_invalid = client.post("/predict", json=invalid_payload)
    assert res_invalid.status_code == 422

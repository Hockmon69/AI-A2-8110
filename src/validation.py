"""
Data validation and schema integrity enforcement.
Validates inputs, ranges, missing values, duplicates, and checks against target leakage.
"""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

EXPECTED_SEED = 8110
VERIFICATION_CODE = "AI-A2-8110"

FEATURE_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support"
]

TARGET_COLUMNS = ["yield_tons", "high_risk"]
ALL_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS

FEATURE_RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
    "cooperative_support": (0, 1)
}


def validate_seed(seed: int) -> None:
    """Ensure submitted seed matches the required LLLL verification seed."""
    if seed != EXPECTED_SEED:
        raise ValueError(
            f"Invalid seed {seed}. This assignment requires personal registration seed {EXPECTED_SEED} (Verification: {VERIFICATION_CODE})."
        )


def validate_raw_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate dataset schema, inspect missing values, detect and remove duplicate rows,
    and verify absence of target leakage.
    """
    report = {}

    # 1. Column presence
    missing_cols = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    report["columns_checked"] = True

    # 2. Missing value audit
    missing_counts = df[ALL_COLUMNS].isna().sum().to_dict()
    report["missing_values"] = {k: int(v) for k, v in missing_counts.items() if v > 0}

    # 3. Duplicate row detection and cleaning
    initial_rows = len(df)
    duplicate_count = int(df.duplicated().sum())
    report["duplicate_rows_detected"] = duplicate_count
    cleaned_df = df.drop_duplicates().copy().reset_index(drop=True)
    report["rows_after_deduplication"] = len(cleaned_df)
    report["rows_removed"] = initial_rows - len(cleaned_df)

    # 4. Range validation on non-null numeric values
    for col, (low, high) in FEATURE_RANGES.items():
        valid_vals = cleaned_df[col].dropna()
        out_of_bounds = ((valid_vals < low - 1e-4) | (valid_vals > high + 1e-4)).sum()
        if out_of_bounds > 0:
            raise ValueError(f"Feature '{col}' contains {out_of_bounds} values outside expected range [{low}, {high}].")

    # Season categorical validation
    invalid_seasons = (~cleaned_df["season"].isin(["A", "B"])).sum()
    if invalid_seasons > 0:
        raise ValueError(f"Found {invalid_seasons} invalid season entries outside ['A', 'B'].")

    # Target sanity check
    if (cleaned_df["yield_tons"] < 0.50 - 1e-4).any():
        raise ValueError("Target 'yield_tons' contains values below minimum clip threshold of 0.50 tons.")

    if not cleaned_df["high_risk"].isin([0, 1]).all():
        raise ValueError("Classification target 'high_risk' contains non-binary values.")

    # 5. Target leakage assertion
    verify_no_target_leakage(FEATURE_COLUMNS)
    report["target_leakage_prevented"] = True

    return cleaned_df, report


def verify_no_target_leakage(feature_names: List[str]) -> None:
    """Verify that no target or target-derived variables exist in input features."""
    forbidden = ["yield_tons", "yield_per_ha", "yield_per_hectare", "high_risk", "target"]
    for f in feature_names:
        if f.lower() in forbidden:
            raise ValueError(f"Target leakage detected: '{f}' must never be used as an input feature!")


def validate_inference_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strict validation of single JSON prediction input record.
    Returns cleaned typed dictionary or raises descriptive ValueError.
    """
    for col in FEATURE_COLUMNS:
        if col not in record:
            raise ValueError(f"Missing required input field: '{col}'")

    cleaned = {}
    # Validate numeric ranges
    for col, (low, high) in FEATURE_RANGES.items():
        val = record[col]
        if val is None:
            raise ValueError(f"Field '{col}' cannot be null.")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"Field '{col}' must be a numeric float/int, got {type(val).__name__}.")
        if val < low or val > high:
            raise ValueError(f"Field '{col}' value {val} out of permitted range [{low}, {high}].")
        cleaned[col] = float(val) if col != "cooperative_support" else int(val)

    # Validate season
    season = record["season"]
    if season not in ["A", "B"]:
        raise ValueError(f"Field 'season' must be either 'A' or 'B', got '{season}'.")
    cleaned["season"] = season

    return cleaned

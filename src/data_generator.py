"""
Data generation module for Musanze potato cooperative farm records.
Reproducible dataset generator adhering to student registration number seed rules.
"""
import hashlib
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


def generate_cooperative_dataset(seed: int = 8110, reg_digits: str = "2528110") -> Tuple[pd.DataFrame, str]:
    """
    Generate synthetic farm records according to INES SWE 3513 Assignment 2 specification.

    Parameters:
        seed: Random seed LLLL (last 4 digits of registration number, e.g. 8110)
        reg_digits: Clean registration digits string (e.g. '2528110')

    Returns:
        tuple of (pd.DataFrame, sha256_fingerprint)
    """
    rng = np.random.default_rng(seed)
    base_rows = 240 + (seed % 41)

    # 1. Generate base features
    rainfall = rng.uniform(600.0, 1400.0, size=base_rows)
    avg_temp = rng.uniform(12.0, 26.0, size=base_rows)
    soil_moisture = rng.uniform(20.0, 80.0, size=base_rows)
    fertilizer = rng.uniform(0.0, 250.0, size=base_rows)
    field_size = rng.uniform(0.2, 4.0, size=base_rows)
    season = rng.choice(["A", "B"], size=base_rows)
    cooperative_support = rng.choice([0, 1], size=base_rows)

    # 2. Yield calculation
    season_effect = np.where(season == "A", 0.50, -0.20)
    noise = rng.normal(loc=0.0, scale=0.80, size=base_rows)

    yield_tons = (
        0.004 * rainfall
        + 0.06 * soil_moisture
        + 0.02 * fertilizer
        + 1.30 * field_size
        - 0.25 * np.abs(avg_temp - 18.0)
        + 0.70 * cooperative_support
        + season_effect
        + noise
    )
    yield_tons = np.clip(yield_tons, 0.50, None)

    # 3. High risk classification target (yield per hectare below median)
    yield_per_ha = yield_tons / field_size
    median_yield_per_ha = float(np.median(yield_per_ha))
    high_risk = (yield_per_ha < median_yield_per_ha).astype(int)

    df = pd.DataFrame({
        "rainfall_mm": rainfall,
        "avg_temp_c": avg_temp,
        "soil_moisture_pct": soil_moisture,
        "fertilizer_kg": fertilizer,
        "field_size_ha": field_size,
        "season": season,
        "cooperative_support": cooperative_support,
        "yield_tons": yield_tons,
        "high_risk": high_risk
    })

    # 4. Deliberate data problems: missing values
    m = int(round(0.05 * base_rows))
    idx_soil_moisture = rng.choice(base_rows, size=m, replace=False)
    idx_fertilizer = rng.choice(base_rows, size=m, replace=False)

    df.loc[idx_soil_moisture, "soil_moisture_pct"] = np.nan
    df.loc[idx_fertilizer, "fertilizer_kg"] = np.nan

    # 5. Deliberate data problems: append three exact duplicate rows
    duplicate_rows = df.iloc[:3].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # 6. Save formatted CSV bytes and compute SHA-256
    csv_text = df.to_csv(index=False, float_format="%.4f")
    csv_bytes = csv_text.encode("utf-8")
    fingerprint = hashlib.sha256(csv_bytes).hexdigest()

    return df, fingerprint


def save_dataset_and_fingerprint(
    df: pd.DataFrame,
    fingerprint: str,
    csv_path: Path,
    fingerprint_path: Path
) -> None:
    """Save dataset CSV with 4 decimal places and fingerprint text file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False, float_format="%.4f")
    fingerprint_path.write_text(fingerprint.strip() + "\n", encoding="utf-8")

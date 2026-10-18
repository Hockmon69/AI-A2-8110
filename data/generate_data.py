#!/usr/bin/env python3
"""
Stand-alone generator script for INES Musanze potato cooperative dataset.
Run: python data/generate_data.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import generate_cooperative_dataset, save_dataset_and_fingerprint

if __name__ == "__main__":
    seed = 8110
    reg_digits = "2528110"
    csv_file = PROJECT_ROOT / "data" / f"AI_A2_{reg_digits}.csv"
    fp_file = PROJECT_ROOT / "artifacts" / "data_fingerprint.txt"

    print(f"Generating individual dataset for seed {seed} (AI-A2-{seed})...")
    df, fp = generate_cooperative_dataset(seed=seed, reg_digits=reg_digits)
    save_dataset_and_fingerprint(df, fp, csv_file, fp_file)
    print(f"Successfully generated {len(df)} rows (base 273 + 3 duplicates).")
    print(f"Dataset saved to: {csv_file}")
    print(f"SHA-256 Fingerprint: {fp}")
    print(f"Fingerprint saved to: {fp_file}")

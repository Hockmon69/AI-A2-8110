#!/usr/bin/env python3
"""
Reproducible command-line entry point for SWE 3513 Assignment 2.
Usage:
    python run_all.py --student-id ug2528110@ines.ac.rw --seed 8110
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline
from src.validation import VERIFICATION_CODE, EXPECTED_SEED


def main():
    parser = argparse.ArgumentParser(
        description="Musanze Harvest Intelligence Service: Reproducible Training Pipeline"
    )
    parser.add_argument(
        "--student-id",
        type=str,
        required=True,
        help="Institutional username / student ID (e.g. ug2528110@ines.ac.rw)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Personal integer seed formed by last 4 digits of registration number (e.g. 8110)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  INES-Ruhengeri - Artificial Intelligence (SWE 3513)")
    print("  Musanze Harvest Intelligence Service Pipeline")
    print(f"  Verification Code : {VERIFICATION_CODE}")
    print(f"  Student ID        : {args.student_id}")
    print(f"  Execution Seed    : {args.seed}")
    print("=" * 70)

    if args.seed != EXPECTED_SEED:
        print(f"\n[ERROR] Seed {args.seed} is rejected!")
        print(f"Expected registration seed: {EXPECTED_SEED} (Verification: {VERIFICATION_CODE})")
        sys.exit(1)

    try:
        results = run_pipeline(student_id=args.student_id, seed=args.seed, project_root=PROJECT_ROOT)
        print("\n[SUCCESS] Pipeline executed successfully and regenerated all artifacts:")
        print(f"  - Dataset CSV         : data/AI_A2_2528110.csv ({results['rows_raw']} rows raw, {results['rows_cleaned']} cleaned)")
        print(f"  - SHA-256 Fingerprint : {results['sha256_fingerprint']}")
        print(f"  - Regression Model    : {results['regression_model']} (saved to models/regression_model.joblib)")
        print(f"  - Classification Model: {results['classification_model']} (saved to models/classification_model.joblib)")
        print(f"  - Farm Clustering     : KMeans with K={results['selected_k']} (saved to models/clustering_model.joblib)")
        print("  - Generated Artifacts :")
        print("      * artifacts/data_fingerprint.txt")
        print("      * artifacts/regression_metrics.json")
        print("      * artifacts/classification_metrics.json")
        print("      * artifacts/clustering_report.json")
        print("      * artifacts/test_predictions.csv")
        print("      * artifacts/plots/residuals.png")
        print("      * artifacts/plots/confusion_matrix.png")
        print("      * artifacts/plots/clusters.png")
        print("=" * 70)
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

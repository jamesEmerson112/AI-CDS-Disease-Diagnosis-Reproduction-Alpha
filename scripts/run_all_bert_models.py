#!/usr/bin/env python3
"""
Run all 3 BERT models sequentially for disease diagnosis comparison.

Models:
  1. Bio_ClinicalBERT  - trained on MIMIC-III clinical notes
  2. BiomedBERT        - trained on PubMed abstracts
  3. BlueBERT          - trained on PubMed + MIMIC-III (hybrid)

Each model produces a separate Prediction_Output_{ModelName}_{timestamp}/ directory.
Estimated runtime: ~15-30 min total on M1 Mac.

Usage:
    python scripts/run_all_bert_models.py
"""

import os
import sys
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    from src.models.bert_models import run_analysis, MODELS

    print("=" * 60)
    print("SEQUENTIAL 3-MODEL BERT COMPARISON")
    print("=" * 60)
    for mid, info in MODELS.items():
        print(f"  [{mid}] {info['name']}: {info['description']}")
    print("=" * 60)

    overall_start = time.time()
    output_dirs = []

    for model_id in ["1", "2", "3"]:
        model_name = MODELS[model_id]['name']
        print(f"\n{'#' * 60}")
        print(f"# [{model_id}/3] {model_name}")
        print(f"{'#' * 60}\n")

        start = time.time()
        output_dir = run_analysis(model_id)
        elapsed = time.time() - start
        output_dirs.append((model_name, output_dir, elapsed))

    overall_elapsed = time.time() - overall_start

    print(f"\n{'=' * 60}")
    print("ALL 3 MODELS COMPLETE")
    print(f"{'=' * 60}")
    for name, path, elapsed in output_dirs:
        mins = elapsed / 60
        print(f"  {name:20s}  {mins:5.1f} min  {path}")
    print(f"  {'TOTAL':20s}  {overall_elapsed/60:5.1f} min")
    print(f"{'=' * 60}")
    print("Compare PerformanceIndex.txt in each directory against baseline:")
    print("  TOP-10 @ 0.6: F1=0.489, P=0.621, R=0.412")
    print("  TOP-20 @ 0.6: F1=0.512, P=0.598, R=0.448")
    print("  TOP-30 @ 0.6: F1=0.521, P=0.587, R=0.467")
    print(f"{'=' * 60}")

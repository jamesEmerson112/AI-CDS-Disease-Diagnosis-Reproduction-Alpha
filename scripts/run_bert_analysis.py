#!/usr/bin/env python3
"""
Entry point for running the BERT-based disease diagnosis analysis.
Run from project root: python scripts/run_bert_analysis.py [--model 1|2|3|all]
"""

import argparse
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT Disease Diagnosis Analysis")
    parser.add_argument(
        "--model",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Model to run: 1=Bio_ClinicalBERT, 2=BiomedBERT, 3=BlueBERT, all=run all 3 sequentially (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AI-CDS Disease Diagnosis - BERT Models")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print("")

    from src.models.bert_models import run_analysis, MODELS

    if args.model == "all":
        output_dirs = []
        for model_id in ["1", "2", "3"]:
            print(f"\n{'#' * 60}")
            print(f"# Running model {model_id}: {MODELS[model_id]['name']}")
            print(f"{'#' * 60}\n")
            output_dir = run_analysis(model_id)
            output_dirs.append((MODELS[model_id]['name'], output_dir))

        print(f"\n{'=' * 60}")
        print("ALL MODELS COMPLETE")
        print(f"{'=' * 60}")
        for name, path in output_dirs:
            print(f"  {name}: {path}")
        print(f"{'=' * 60}")
    else:
        run_analysis(args.model)

#!/usr/bin/env python3
"""
Entry point for running the BERT-based disease diagnosis analysis.
Run from project root: python scripts/run_bert_analysis.py [--model 1|2|3|all]
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Imported before the parser is built so --pipeline's choices come from the
    # registry itself. A hand-kept list here would silently drift from
    # config._BY_NAME, and the failure mode is a config that exists but cannot
    # be selected -- which is exactly the bug 31bea66 had to fix.
    from aicds.config import PIPELINE_HELP, PIPELINE_NAMES, from_env, from_name

    parser = argparse.ArgumentParser(description="BERT Disease Diagnosis Analysis")
    parser.add_argument(
        "--model",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Model to run: 1=Bio_ClinicalBERT, 2=BiomedBERT, 3=BlueBERT, all=run all 3 sequentially (default: all)",
    )
    parser.add_argument(
        "--pipeline",
        choices=PIPELINE_NAMES,
        default=None,
        help=PIPELINE_HELP,
    )
    args = parser.parse_args()

    config = from_name(args.pipeline) if args.pipeline else from_env()

    print("=" * 60)
    print("AI-CDS Disease Diagnosis - BERT Models")
    print("=" * 60)
    print(f"Project root: {project_root}")
    # Every axis of the config goes in the banner, the grader included. Omitting
    # it is how an operator ends up with cosine numbers filed as DRG ones: the
    # run succeeds, the banner looks right, and nothing downstream records which
    # ruler was used.
    print(f"Pipeline:     {config.preprocess_version}  (folds: {config.fold_dir})")
    print(f"Grader:       {config.grader}")
    if config.grader == "drg-exact":
        print("              exact DRG label match; ceiling is 76/129 = 58.9% on")
        print("              folds_grouped, so 1.0 is unreachable by construction")
    print("")

    from aicds.models.bert_models import run_analysis, MODELS

    if args.model == "all":
        output_dirs = []
        for model_id in ["1", "2", "3"]:
            print(f"\n{'#' * 60}")
            print(f"# Running model {model_id}: {MODELS[model_id]['name']}")
            print(f"{'#' * 60}\n")
            output_dir = run_analysis(model_id, config=config)
            output_dirs.append((MODELS[model_id]['name'], output_dir))

        print(f"\n{'=' * 60}")
        print("ALL MODELS COMPLETE")
        print(f"{'=' * 60}")
        for name, path in output_dirs:
            print(f"  {name}: {path}")
        print(f"{'=' * 60}")
    else:
        run_analysis(args.model, config=config)

#!/usr/bin/env python3
"""
Entry point for running the baseline Sent2Vec disease diagnosis analysis.
Run from project root: python scripts/run_baseline.py [--pipeline NAME]
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

    parser = argparse.ArgumentParser(description="Baseline (Sent2Vec) Disease Diagnosis Analysis")
    parser.add_argument(
        "--pipeline",
        choices=PIPELINE_NAMES,
        default=None,
        help=PIPELINE_HELP,
    )
    parser.add_argument(
        "--out",
        metavar="ROOT",
        default=None,
        help="Output root. Default writes Prediction_Output_* into the current "
        "directory (the layout the golden pins). --out ROOT writes "
        "ROOT/<model>/<timestamp>/ -- the results*/ layout "
        "compare_models.py --results-dir reads directly.",
    )
    args = parser.parse_args()

    # DUA guard, for the same reason the config is resolved early: fail in
    # milliseconds, not after the 21GB model load. --out can aim HADM_ID-named
    # per-case files at a repo-internal path no .gitignore rule covers, where the
    # pre-commit hook is blind because those files are empty. out=None is not
    # checked -- that layout is covered by .gitignore and pinned by the golden.
    from aicds import runs

    try:
        runs.check_out_root(args.out)
    except runs.UnignoredOutRoot as exc:
        sys.exit("ERROR: %s" % exc)

    # Resolved here, before the module import, so a bad $AICDS_PIPELINE raises
    # from from_env in under a second rather than after the 21GB model load.
    config = from_name(args.pipeline) if args.pipeline else from_env()

    print("=" * 60)
    print("AI-CDS Disease Diagnosis - Baseline (Sent2Vec)")
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

    from aicds.models.baseline_sent2vec import run_analysis

    run_analysis(config=config, out=args.out)

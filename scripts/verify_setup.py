#!/usr/bin/env python3
"""
Quick smoke test to verify reorganization.
Run: python scripts/verify_setup.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 50)
    print("Project Reorganization Verification")
    print("=" * 50)

    checks = []
    failed = False

    # Check 1: Directory structure
    print("\n[1] Checking directory structure...")
    # "output" is deliberately NOT here. No code writes it -- runs go to
    # Prediction_Output_*/ in the cwd, or to --out ROOT/<key>/<stamp>/ -- so on a
    # clean clone this check failed for a directory nothing needs, and it passed
    # here only because a November 2025 run happened to leave one behind. A smoke
    # test that fails on a correct checkout teaches people to ignore it.
    required_dirs = [
        "src", "src/aicds", "src/aicds/models", "src/aicds/entity",
        "src/aicds/utils",
        "scripts", "tests", "data", "data/folds", "data/raw",
        "docs", "config", "archive"
    ]
    for d in required_dirs:
        path = os.path.join(project_root, d)
        if os.path.isdir(path):
            print(f"    OK: {d}/")
        else:
            print(f"    MISSING: {d}/")
            failed = True

    # Check 2: Import tests
    print("\n[2] Checking imports...")
    try:
        from aicds.entity.SymptomsDiagnosis import SymptomsDiagnosis
        print("    OK: src.entity.SymptomsDiagnosis")
    except ImportError as e:
        print(f"    FAIL: src.entity.SymptomsDiagnosis - {e}")
        failed = True

    try:
        from aicds.utils.Constants import CH_DIR, K_FOLD
        print(f"    OK: src.utils.Constants (CH_DIR={CH_DIR})")
    except ImportError as e:
        print(f"    FAIL: src.utils.Constants - {e}")
        failed = True

    try:
        # The run-directory contract, both halves: run_dirs() is what every arm
        # writes through and discover() is what every reporting script reads
        # through. It imports with base dependencies alone -- no torch, no
        # sent2vec, no matplotlib -- so a failure here is a broken checkout
        # rather than a missing optional dependency, which is what makes it worth
        # checking in a smoke test.
        from aicds.runs import discover, run_dirs
        print("    OK: aicds.runs (run_dirs writer + discover reader)")
    except ImportError as e:
        print(f"    FAIL: aicds.runs - {e}")
        failed = True

    try:
        from aicds.utils import cython_utils
        print("    OK: src.utils.cython_utils")
    except ImportError as e:
        if "nltk" in str(e) or "gensim" in str(e) or "sent2vec" in str(e):
            print(f"    SKIP: src.utils.cython_utils - missing dependency: {e}")
            print("          (Install with: pip install -r config/requirements.txt)")
        else:
            print(f"    FAIL: src.utils.cython_utils - {e}")
            failed = True

    # Check 3: Data files
    print("\n[3] Checking data files...")
    for i in range(10):
        fold_path = os.path.join(project_root, f"data/folds/Fold{i}")
        if os.path.isdir(fold_path):
            train = os.path.join(fold_path, "TrainingSet.txt")
            test = os.path.join(fold_path, "TestSet.txt")
            if os.path.isfile(train) and os.path.isfile(test):
                print(f"    OK: Fold{i}/ (TrainingSet.txt, TestSet.txt)")
            else:
                print(f"    PARTIAL: Fold{i}/ missing files")
                failed = True
        else:
            print(f"    MISSING: Fold{i}/")
            failed = True

    # Check 4: Config files
    print("\n[4] Checking config files...")
    config_files = [
        "config/requirements.txt",
        "config/requirements_bert.txt",
        "config/environment.yml"
    ]
    for f in config_files:
        path = os.path.join(project_root, f)
        if os.path.isfile(path):
            print(f"    OK: {f}")
        else:
            print(f"    MISSING: {f}")
            failed = True

    # Check 5: CH_DIR resolution
    print("\n[5] Checking Constants.CH_DIR resolution...")
    try:
        from aicds.utils.Constants import CH_DIR
        if os.path.isdir(CH_DIR):
            print(f"    OK: CH_DIR exists")
            if os.path.isfile(os.path.join(CH_DIR, "pyproject.toml")):
                print(f"    OK: CH_DIR is project root")
            else:
                print(f"    WARN: CH_DIR may not be project root")
        else:
            print(f"    FAIL: CH_DIR does not exist: {CH_DIR}")
            failed = True
    except Exception as e:
        print(f"    FAIL: {e}")
        failed = True

    # Summary
    print("\n" + "=" * 50)
    if not failed:
        print("SUCCESS: All checks passed!")
        return 0
    else:
        print("FAILED: Some checks did not pass.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

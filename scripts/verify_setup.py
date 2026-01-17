#!/usr/bin/env python3
"""
Quick smoke test to verify reorganization.
Run: python scripts/verify_setup.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    print("=" * 50)
    print("Project Reorganization Verification")
    print("=" * 50)

    failed = False

    # Check 1: Directory structure
    print("\n[1] Checking directory structure...")
    required_dirs = [
        "src", "src/biosentvec", "src/bert", "src/shared", "src/shared/entity",
        "scripts", "tests", "data", "data/folds", "data/raw",
        "output", "docs", "config", "archive"
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

    # Check shared.entity
    try:
        from src.shared.entity.SymptomsDiagnosis import SymptomsDiagnosis
        print("    OK: src.shared.entity.SymptomsDiagnosis")
    except ImportError as e:
        print(f"    FAIL: src.shared.entity.SymptomsDiagnosis - {e}")
        failed = True

    # Check shared.constants
    try:
        from src.shared.constants import CH_DIR, K_FOLD
        print(f"    OK: src.shared.constants (CH_DIR={CH_DIR})")
    except ImportError as e:
        print(f"    FAIL: src.shared.constants - {e}")
        failed = True

    # Check shared.preprocessing (requires nltk)
    try:
        from src.shared.preprocessing import preprocess_sentence
        print("    OK: src.shared.preprocessing")
    except ImportError as e:
        if "nltk" in str(e):
            print(f"    SKIP: src.shared.preprocessing - missing nltk")
        else:
            print(f"    FAIL: src.shared.preprocessing - {e}")
            failed = True

    # Check shared.similarity
    try:
        from src.shared.similarity import cosine_similarity
        print("    OK: src.shared.similarity")
    except ImportError as e:
        if "nltk" in str(e):
            print(f"    SKIP: src.shared.similarity - missing nltk")
        else:
            print(f"    FAIL: src.shared.similarity - {e}")
            failed = True

    # Check biosentvec module (requires sent2vec)
    try:
        from src.biosentvec import embeddings
        print("    OK: src.biosentvec.embeddings")
    except ImportError as e:
        if "sent2vec" in str(e) or "nltk" in str(e):
            print(f"    SKIP: src.biosentvec.embeddings - missing dependency: {e}")
            print("          (Install with: pip install -r config/requirements.txt)")
        else:
            print(f"    FAIL: src.biosentvec.embeddings - {e}")
            failed = True

    # Check bert module
    try:
        from src.bert import models
        print("    OK: src.bert.models (imports only)")
    except ImportError as e:
        if "sentence_transformers" in str(e) or "torch" in str(e):
            print(f"    SKIP: src.bert.models - missing dependency: {e}")
            print("          (Install with: pip install -r config/requirements_bert.txt)")
        else:
            print(f"    FAIL: src.bert.models - {e}")
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
        from src.shared.constants import CH_DIR
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

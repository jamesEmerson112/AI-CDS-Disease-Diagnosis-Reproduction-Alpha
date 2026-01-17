#!/usr/bin/env python3
"""
Entry point for running the baseline Sent2Vec disease diagnosis analysis.
Run from project root: python scripts/run_baseline.py
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import and run the baseline module
if __name__ == "__main__":
    print("=" * 60)
    print("AI-CDS Disease Diagnosis - Baseline (Sent2Vec)")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print("")

    # Import the main module - this will execute the analysis
    from src.biosentvec import model

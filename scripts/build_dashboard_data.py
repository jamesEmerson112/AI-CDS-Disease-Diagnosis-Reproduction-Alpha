#!/usr/bin/env python3
"""
Build dashboard-data.json from experiment outputs.

Reads PerformanceIndex.txt (3 models) and score_distribution_summary.txt,
outputs a single JSON file consumed by the React dashboard.

Usage:
    python scripts/build_dashboard_data.py
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Tuple


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# PerformanceIndex.txt parser (adapted from build_readme_plots.py)
# ---------------------------------------------------------------------------

def parse_metric_row(line: str) -> Tuple[float, Dict[str, float]] | None:
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    try:
        return float(parts[0]), {
            "TP": float(parts[1]),
            "FP": float(parts[2]),
            "P": float(parts[3]),
            "R": float(parts[4]),
            "F1": float(parts[5]),
            "PR": float(parts[6]),
        }
    except ValueError:
        return None


def parse_performance_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    model_name = None
    metrics: Dict[str, Dict[float, Dict[str, float]]] = {}
    timing: Dict[str, float] = {}

    sec_pattern = re.compile(r"^([^:]+):\s+([0-9]+(?:\.[0-9]+)?)\s+seconds")
    model_pattern = re.compile(r"^MODEL:\s+([^(]+)")

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        section = re.match(r"^10-FOLD PERFORMANCE INDEX of (.+?) SIMILARITY by MAX$", line.strip())
        if section:
            method = section.group(1).strip()
            metrics[method] = {}
            i += 1
            while i < len(lines):
                parsed = parse_metric_row(lines[i])
                if parsed is None:
                    if metrics.get(method):
                        break
                    i += 1
                    continue
                threshold, row = parsed
                metrics[method][threshold] = row
                i += 1
            continue

        sec_match = sec_pattern.match(line.strip())
        if sec_match:
            timing[sec_match.group(1).strip()] = float(sec_match.group(2))

        model_match = model_pattern.match(line.strip())
        if model_match:
            model_name = model_match.group(1).strip()

        i += 1

    if not model_name:
        model_name = os.path.basename(os.path.dirname(path)).replace("Prediction_Output_", "")

    return {"model": model_name, "metrics": metrics, "timing": timing}


# ---------------------------------------------------------------------------
# score_distribution_summary.txt parser
# ---------------------------------------------------------------------------

def parse_score_distribution(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip() for l in f.readlines()]

    result: Dict[str, Any] = {"pairwise": {}, "perPatientMax": {}, "diagnosisCount": {}}
    current_section = None
    current_model = None
    current_stats: Dict[str, Any] = {}

    for line in lines:
        stripped = line.strip()
        if "SECTION 1:" in stripped:
            current_section = "pairwise"
            continue
        if "SECTION 2:" in stripped:
            # Save last model from section 1
            if current_model and current_stats:
                result["pairwise"][current_model] = current_stats
            current_section = "perPatientMax"
            current_model = None
            current_stats = {}
            continue
        if "SECTION 3:" in stripped:
            if current_model and current_stats:
                result["perPatientMax"][current_model] = current_stats
            current_section = None
            continue

        if current_section in ("pairwise", "perPatientMax"):
            m = re.match(r"^Model:\s+(.+)$", stripped)
            if m:
                if current_model and current_stats:
                    result[current_section][current_model] = current_stats
                current_model = m.group(1).strip()
                current_stats = {"thresholdPct": {}}
                continue

            if current_model:
                for key, json_key in [("N =", "n"), ("Min", "min"), ("Max", "max"),
                                       ("Mean", "mean"), ("Median", "median"), ("Std", "std"),
                                       ("P5", "p5"), ("P25", "p25"), ("P75", "p75"), ("P95", "p95")]:
                    pat = re.match(rf"^\s*{re.escape(key)}\s*=\s*([0-9.]+)", stripped)
                    if pat:
                        val = float(pat.group(1))
                        current_stats[json_key] = int(val) if json_key == "n" else val
                        break

                pct = re.match(r"^%\s*>=\s*([0-9.]+)\s*=\s*([0-9.]+)%$", stripped)
                if pct:
                    current_stats["thresholdPct"][pct.group(1)] = float(pct.group(2))

        # Diagnosis count metadata (appears after Section 2 data)
        for key, json_key in [("Total unique diagnoses", "uniqueDiagnoses"),
                               ("Total patients", "patients"),
                               ("Total patient pairs", "totalPatientPairs")]:
            if stripped.startswith(key):
                m2 = re.match(rf"^{re.escape(key)}\s*=\s*(\d+)", stripped)
                if m2:
                    result["diagnosisCount"][json_key] = int(m2.group(1))

        m_mean = re.match(r"^\s*Mean\s*=\s*([0-9.]+)", stripped)
        if m_mean and "meanDiagnosesPerPatient" not in result["diagnosisCount"] and current_section is None:
            pass  # handled above in section context

    return result


# ---------------------------------------------------------------------------
# Build the combined JSON
# ---------------------------------------------------------------------------

def build_dashboard_data() -> Dict[str, Any]:
    root = project_root()

    # Find performance files
    pattern = os.path.join(root, "docs", "Prediction_Output_*", "PerformanceIndex.txt")
    perf_files = sorted(glob.glob(pattern))
    if not perf_files:
        raise FileNotFoundError(f"No PerformanceIndex.txt found matching {pattern}")

    model_order = ["Bio_ClinicalBERT", "BiomedBERT", "BlueBERT"]
    parsed = [parse_performance_file(p) for p in perf_files]
    parsed.sort(key=lambda r: model_order.index(r["model"]) if r["model"] in model_order else 999)

    # Parse score distributions
    dist_path = os.path.join(root, "docs", "score_distribution_analysis", "score_distribution_summary.txt")
    dist = parse_score_distribution(dist_path)

    # Build performance dict: model -> method -> threshold_str -> metrics
    performance: Dict[str, Any] = {}
    for r in parsed:
        model = r["model"]
        performance[model] = {}
        for method, thresholds in r["metrics"].items():
            performance[model][method] = {}
            for t, row in thresholds.items():
                performance[model][method][str(t)] = row

    # Build timing dict
    timing: Dict[str, Any] = {}
    for r in parsed:
        model = r["model"]
        t = r["timing"]
        timing[model] = {
            "modelLoading": t.get("Model Loading", 0),
            "symptomEmbeddings": t.get("Symptom Embeddings", 0),
            "diagnosisEmbeddings": t.get("Diagnosis Embeddings", 0),
            "embeddingsTotal": t.get("Embeddings Total", 0),
            "foldsProcessing": t.get("Total Folds Processing", 0),
            "totalExecution": t.get("TOTAL EXECUTION TIME", 0),
        }

    # Build saturation (alias of perPatientMax thresholdPct)
    saturation: Dict[str, Any] = {}
    for model, stats in dist["perPatientMax"].items():
        saturation[model] = stats.get("thresholdPct", {})

    models = [r["model"] for r in parsed]
    methods = ["MAX", "TOP-10", "TOP-20", "TOP-30", "TOP-40", "TOP-50"]
    thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]

    return {
        "meta": {
            "patients": dist["diagnosisCount"].get("patients", 129),
            "uniqueDiagnoses": dist["diagnosisCount"].get("uniqueDiagnoses", 145),
            "folds": 10,
            "meanDiagnosesPerPatient": 1.74,
            "totalPatientPairs": dist["diagnosisCount"].get("totalPatientPairs", 16512),
            "models": models,
            "methods": methods,
            "thresholds": thresholds,
            "modelColors": {
                "Bio_ClinicalBERT": "#1f77b4",
                "BiomedBERT": "#ff7f0e",
                "BlueBERT": "#2ca02c",
            },
        },
        "performance": performance,
        "timing": timing,
        "scoreDistribution": {
            "pairwise": dist["pairwise"],
            "perPatientMax": dist["perPatientMax"],
        },
        "saturation": saturation,
    }


def main() -> None:
    root = project_root()
    data = build_dashboard_data()

    out_dir = os.path.join(root, "dashboard", "public", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dashboard-data.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {out_path}")
    print(f"  Models: {data['meta']['models']}")
    print(f"  Methods: {len(data['meta']['methods'])}")
    print(f"  Performance entries: {sum(len(m) for m in data['performance'].values())}")


if __name__ == "__main__":
    main()

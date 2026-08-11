#!/usr/bin/env python3
"""
Build dashboard-data.json from experiment outputs.

Reads each run's PerformanceIndex.txt plus score_distribution_summary.txt and
writes the single JSON file the React dashboard fetches.

Usage:
    python scripts/build_dashboard_data.py
    python scripts/build_dashboard_data.py --runs-dir results_drg

Reading is delegated: ``aicds.analysis.performance_index`` parses the file and
``aicds.runs.discover`` finds the runs. The private parser that used to live here
was a copy of ``build_readme_plots``' one, and it "recovered" a model name by
stripping ``Prediction_Output_`` from a directory name -- which for the nested
``<key>/<stamp>/`` layout returned the *timestamp* as the model name, silently.
Discovery carries the label now, so the fallback names a model instead of a clock.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict

from aicds import runs
from aicds.analysis.performance_index import MetricRow, read


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# The one deliberate rename at the JSON boundary
# ---------------------------------------------------------------------------

# Column order as the file writes it, and as every previously generated
# dashboard-data.json carries it. Iterating this tuple (rather than the dict from
# as_dict()) is what keeps the emitted key order byte-identical.
_JSON_ROW_KEYS = ("TP", "FP", "P", "R", "FS", "PR")


def row_for_json(row: MetricRow) -> Dict[str, float]:
    """``MetricRow`` -> the dashboard's six-key row, with ``FS`` renamed ``F1``.

    The name is wrong and stays anyway. It is not an F1: across all 12,600
    committed BERT rows precision == recall == F-score, so the column is accuracy
    (docs/findings/04-metric-degeneracy.md). But ``F1VsThreshold.tsx`` reads
    ``?.F1 ?? 0``, so renaming the key here does not break a chart -- it silently
    zeroes every one of them, which is worse. Fixing the vocabulary is a dashboard
    decision and belongs to P38; until then the rename lives on this one line
    where a reader can see it, instead of being spread through a private parser.
    """
    columns = row.as_dict()
    return {("F1" if key == "FS" else key): columns[key] for key in _JSON_ROW_KEYS}


def load_run(run: runs.Run) -> Dict[str, Any]:
    """One run's model name, aggregate metrics and timing trailer."""
    index = read(run.performance_index)
    return {
        # The MODEL: line when the run wrote a trailer; otherwise the label
        # discovery derived from the layout. Both name a model.
        "model": index.model or run.label,
        "metrics": {
            strategy: {t: row_for_json(row) for t, row in rows.items()}
            for strategy, rows in index.aggregate.items()
        },
        "timing": index.timing,
    }


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

def build_dashboard_data(runs_dir: str | None = None) -> Dict[str, Any]:
    root = project_root()
    if runs_dir is None:
        runs_dir = os.path.join(root, "docs")

    discovered = runs.discover(runs_dir)
    if not discovered:
        raise FileNotFoundError(f"No run with a PerformanceIndex.txt found under {runs_dir}")

    model_order = ["Bio_ClinicalBERT", "BiomedBERT", "BlueBERT"]
    parsed = [load_run(run) for run in discovered]
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--runs-dir", default=None,
        help="directory to discover runs under. Default: docs/, which holds the "
             "three committed regression-oracle runs the dashboard ships with. "
             "Point it at results_drg/ (etc.) to rebuild from a newer pipeline.",
    )
    args = parser.parse_args()

    root = project_root()
    data = build_dashboard_data(args.runs_dir)

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

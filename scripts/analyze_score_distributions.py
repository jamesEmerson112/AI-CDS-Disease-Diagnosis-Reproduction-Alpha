#!/usr/bin/env python3
"""
Analyze diagnosis relevance score distributions, under a NAMED pipeline config.

Investigates why the three BERT models reach F1 = 1.000 at threshold 0.6 by
examining:
  1. All-pairwise cosine similarity across the unique diagnosis descriptions
  2. Per-patient-pair relevance -- what actually determines TP/FP
  3. Diagnosis count per patient (Cartesian-product amplification)

WHAT CHANGED, 2026-08-12 (TODO P37)
-----------------------------------
This script used to SIMULATE the grader: it built its own numpy cosine matrix --
a fourth cosine implementation in this repository -- and re-derived the
MAX-over-Cartesian-product aggregation and the ``prefix:description`` slicing by
hand, against no ``PipelineConfig`` at all. Its only guard,
``spot_check_against_cython``, compared the *kernel* on five random pairs, so it
passed while the aggregator and the prefix handling were unverified, and every
statistic it had ever produced was legacy-measured whatever the rest of the
project was running.

Section 2 now calls ``cython_utils.get_diagnosis_relevance(..., config)`` -- the
same single dispatch point both arms grade with -- and Section 1 uses
``cython_utils.cosine_similarity``. There is no second implementation of either
left in this file, and the output names the config that produced it.

WHICH CONFIG FIELDS THIS ANALYSIS ACTUALLY READS
------------------------------------------------
``preprocess_version``  YES -- decides which text is handed to the encoder
                        (``bert_models``' FIX 4), so it moves every number below.
``grader``              YES -- decides Section 2 entirely. Under ``drg-exact``
                        relevance is 0/1 and there is no distribution to
                        saturate.
``fold_dir``            NO -- and this is why the header says so out loud. Every
                        ordered patient pair is scored here, not fold test
                        cases, so the leakage fix is invisible to this script.
                        ``folds-only`` therefore measures identically to
                        ``legacy``, and ``preprocess-only`` identically to
                        ``corrected``. Do not read a corrected-vs-legacy delta
                        here as containing any part of the leakage effect.

Outputs (into --out, default docs/score_distribution_analysis/):
  score_distributions.png
  per_patient_max_distributions.png
  score_distribution_summary.txt

Usage:
    python scripts/analyze_score_distributions.py --pipeline corrected
    python scripts/analyze_score_distributions.py --pipeline drg --out /tmp/sd_drg
"""

import argparse
import os
import random
import time

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer

from aicds.config import (
    LEGACY,
    PIPELINE_HELP,
    PIPELINE_NAMES,
    from_env,
    from_name,
    name_of,
    require_supported_grader,
)
from aicds.entity.SymptomsDiagnosis import SymptomsDiagnosis
from aicds.utils.Constants import CH_DIR
from aicds.utils import cython_utils as util_cy
from aicds.utils.cython_utils import use_corrected_preprocessing
from aicds.utils.runtime import ensure_nltk_data

ensure_nltk_data()

# Matplotlib setup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model configs (same as bert_models.py)
MODELS = {
    'Bio_ClinicalBERT': 'emilyalsentzer/Bio_ClinicalBERT',
    'BiomedBERT': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
    'BlueBERT': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
}

MODEL_COLORS = {
    'Bio_ClinicalBERT': '#1f77b4',
    'BiomedBERT': '#ff7f0e',
    'BlueBERT': '#2ca02c',
}

THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 1.0]


def load_dataset():
    """Load admissions from Symptoms-Diagnosis.txt (same logic as bert_models.py)."""
    file_name = os.path.join(CH_DIR, "data", "raw", "Symptoms-Diagnosis.txt")
    lines = open(file_name, "r").readlines()

    admissions = {}
    for line in lines:
        line = line.replace("\n", "")
        attributes = line.split(';')
        a = SymptomsDiagnosis(
            attributes[SymptomsDiagnosis.CONST_HADM_ID],
            attributes[SymptomsDiagnosis.CONST_SUBJECT_ID],
            attributes[SymptomsDiagnosis.CONST_ADMITTIME],
            attributes[SymptomsDiagnosis.CONST_DISCHTIME],
            attributes[SymptomsDiagnosis.CONST_SYMPTOMS],
            util_cy.preprocess_diagnosis(attributes[SymptomsDiagnosis.CONST_DIAGNOSIS])
        )
        admissions[attributes[SymptomsDiagnosis.CONST_HADM_ID]] = a

    return admissions


def description_of(diagnosis_label):
    """Text after the first ':'. Mirrors the slice both graders do internally."""
    if ':' in diagnosis_label:
        return diagnosis_label[diagnosis_label.index(':') + 1:]
    return diagnosis_label


def extract_unique_diagnoses(admissions):
    """Unique diagnosis descriptions across all admissions, sorted for determinism."""
    unique = set()
    for admission in admissions.values():
        for diag in admission.diagnosis:
            unique.add(description_of(diag))
    return sorted(unique)


def build_embeddings(model, unique_diagnoses, config):
    """Embedding dict in the pipeline's own shape: raw-description keys, ``[vector]`` values.

    Reproduces ``bert_models.compute_bert_diagnosis_embeddings`` exactly,
    including FIX 4: under corrected preprocessing the ENCODED text is
    preprocessed while the KEYS stay raw, because ``get_diagnosis_relevance``
    looks up by the raw description sliced out of the diagnosis label.
    """
    if use_corrected_preprocessing(config):
        texts_to_encode = [util_cy.preprocess_sentence(d, config) for d in unique_diagnoses]
        print("  FIX 4 active: encoding preprocessed diagnosis text")
    else:
        texts_to_encode = list(unique_diagnoses)

    vectors = model.encode(
        texts_to_encode,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False  # Match the pipeline
    )

    return {desc: [vec] for desc, vec in zip(unique_diagnoses, vectors)}, vectors


def compute_pairwise_scores(embeddings, unique_diagnoses):
    """Upper-triangle cosine over the unique descriptions, via the SHIPPED kernel.

    ``util_cy.cosine_similarity`` rather than a numpy matrix product: this file
    used to carry its own, which is half of what made it P37.
    """
    scores = []
    n = len(unique_diagnoses)
    for i in range(n):
        u = embeddings[unique_diagnoses[i]][0]
        for j in range(i + 1, n):
            v = embeddings[unique_diagnoses[j]][0]
            scores.append(util_cy.cosine_similarity(u, v))
    return scores


def kernel_cross_check(vectors, n_checks=5):
    """Diagnostic ONLY -- produces no statistic in this report.

    Kept from the pre-P37 version and deliberately demoted. It compares a numpy
    normalise-and-dot against ``util_cy.cosine_similarity`` on a few random
    pairs, which is a check on the kernel and NOTHING else. It used to sit
    beneath a report whose aggregation and prefix handling were also
    hand-rolled, where it read as validation of the whole thing; those are the
    shipped functions now, so this is what it always actually was.
    """
    n = len(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = vectors / norms
    sim_matrix = normalized @ normalized.T

    random.seed(42)
    pairs = [(random.randint(0, n - 1), random.randint(0, n - 1)) for _ in range(n_checks)]

    max_diff = 0.0
    for i, j in pairs:
        diff = abs(sim_matrix[i, j] - util_cy.cosine_similarity(vectors[i], vectors[j]))
        max_diff = max(max_diff, diff)

    return max_diff


def compute_per_patient_relevance(admissions, embeddings, config):
    """Score every ordered patient pair with the SHIPPED grader.

    For each (A, B), A != B, this is exactly the call the fold loop makes:
    ``get_diagnosis_relevance(embeddings, gt_labels_A, predicted_labels_B,
    config)``. Under ``grader="cosine"`` that is MAX over the Cartesian product
    of descriptions; under ``"drg-exact"`` it is 1.0 on any shared DRG label and
    consults no embedding at all -- which is why the three models return
    identical vectors there, and why there is no distribution to saturate.
    """
    patient_ids = list(admissions.keys())
    scores = []

    for i, gt_id in enumerate(patient_ids):
        gt_diagnosis = admissions[gt_id].diagnosis
        if not gt_diagnosis:
            continue
        for j, pred_id in enumerate(patient_ids):
            if i == j:
                continue
            predicted_diagnosis = admissions[pred_id].diagnosis
            if not predicted_diagnosis:
                continue
            scores.append(
                util_cy.get_diagnosis_relevance(embeddings, gt_diagnosis, predicted_diagnosis, config)
            )

    return scores


def compute_diagnosis_counts(admissions):
    """Per-patient diagnosis counts."""
    return [len(admission.diagnosis) for admission in admissions.values()]


def compute_stats(values, label):
    """Descriptive statistics for a list of values."""
    arr = np.array(values)
    stats = {
        'label': label,
        'n': len(arr),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'p5': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
    }
    for t in THRESHOLDS:
        stats[f'pct_above_{t}'] = float(np.mean(arr >= t) * 100)
    return stats


def format_stats(stats):
    """Format a stats dict as a readable block.

    The exact spelling of these lines is a CONTRACT: ``build_dashboard_data.py``
    and ``build_readme_plots.py`` both parse this file. Do not reflow them
    without updating those two parsers.
    """
    lines = []
    lines.append(f"  N = {stats['n']}")
    lines.append(f"  Min    = {stats['min']:.4f}")
    lines.append(f"  Max    = {stats['max']:.4f}")
    lines.append(f"  Mean   = {stats['mean']:.4f}")
    lines.append(f"  Median = {stats['median']:.4f}")
    lines.append(f"  Std    = {stats['std']:.4f}")
    lines.append(f"  P5     = {stats['p5']:.4f}")
    lines.append(f"  P25    = {stats['p25']:.4f}")
    lines.append(f"  P75    = {stats['p75']:.4f}")
    lines.append(f"  P95    = {stats['p95']:.4f}")
    for t in THRESHOLDS:
        lines.append(f"  % >= {t:.1f} = {stats[f'pct_above_{t}']:.2f}%")
    return '\n'.join(lines)


def pipeline_banner(config):
    """Human-readable identification of the config, for stdout and the report."""
    name = name_of(config)
    return "%s (fold_dir=%s, preprocess_version=%s, grader=%s)" % (
        name if name else "<unregistered>",
        config.fold_dir,
        config.preprocess_version,
        config.grader,
    )


def plot_score_distributions(all_pairwise_data, output_dir, config):
    """Generate score_distributions.png with histogram and CDF panels."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    suffix = f" [pipeline: {name_of(config) or 'unregistered'}]"

    # Top: overlaid histograms
    for model_name, scores in all_pairwise_data.items():
        ax1.hist(scores, bins=100, alpha=0.5, label=model_name,
                 color=MODEL_COLORS[model_name], density=True)
    for t in THRESHOLDS:
        ax1.axvline(x=t, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(t, ax1.get_ylim()[1] * 0.95, f'{t}', ha='center', fontsize=8, color='red')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Density')
    ax1.set_title('All-Pairwise Diagnosis Similarity Distributions (excluding self-pairs)' + suffix)
    ax1.legend()
    ax1.set_xlim(-0.1, 1.1)

    # Bottom: CDF curves
    for model_name, scores in all_pairwise_data.items():
        sorted_scores = np.sort(scores)
        cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax2.plot(sorted_scores, cdf, label=model_name, color=MODEL_COLORS[model_name])
    for t in THRESHOLDS:
        ax2.axvline(x=t, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('CDF of All-Pairwise Diagnosis Similarities' + suffix)
    ax2.legend()
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, 'score_distributions.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


def plot_per_patient_max(per_patient_data, output_dir, config):
    """Generate per_patient_max_distributions.png with histogram and threshold sensitivity."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    suffix = f" [pipeline: {name_of(config) or 'unregistered'}, grader: {config.grader}]"

    # Top: overlaid histograms of per-patient-pair relevance
    for model_name, scores in per_patient_data.items():
        ax1.hist(scores, bins=100, alpha=0.5, label=model_name,
                 color=MODEL_COLORS[model_name], density=True)
    for t in THRESHOLDS:
        ax1.axvline(x=t, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(t, ax1.get_ylim()[1] * 0.95, f'{t}', ha='center', fontsize=8, color='red')
    ax1.set_xlabel('Relevance score (per ordered patient pair)')
    ax1.set_ylabel('Density')
    ax1.set_title('Per-Patient Relevance Distributions' + suffix)
    ax1.legend()
    ax1.set_xlim(-0.1, 1.1)

    # Bottom: threshold sensitivity curve
    threshold_range = np.linspace(0.0, 1.0, 200)
    for model_name, scores in per_patient_data.items():
        arr = np.array(scores)
        fractions = [float(np.mean(arr >= t)) for t in threshold_range]
        ax2.plot(threshold_range, fractions, label=model_name, color=MODEL_COLORS[model_name])
    for t in THRESHOLDS:
        ax2.axvline(x=t, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Fraction of Patient Pairs Above Threshold')
    ax2.set_title('Threshold Sensitivity: Fraction of Patient Pairs Counted as TP' + suffix)
    ax2.legend()
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, 'per_patient_max_distributions.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


def write_summary(all_pairwise_stats, per_patient_stats, diag_count_stats,
                  cross_check_diffs, config, output_dir):
    """Write score_distribution_summary.txt.

    Two parsers read this file -- ``build_dashboard_data.parse_score_distribution``
    and ``build_readme_plots.parse_saturation_summary``. Both key off the
    ``SECTION n:`` prefixes, the ``Model:`` lines and the ``% >= t = v%`` lines,
    all of which are preserved verbatim; the header below sits before SECTION 1,
    where neither parser looks.
    """
    path = os.path.join(output_dir, 'score_distribution_summary.txt')
    grader = config.grader
    pipeline_name = name_of(config) or "<unregistered>"

    with open(path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DIAGNOSIS SIMILARITY SCORE DISTRIBUTION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # ------------------------------------------------------------------
        # Provenance header (TODO P37). Before P37 this file named no config at
        # all, so every figure in it was legacy-measured by default and nothing
        # said so.
        # ------------------------------------------------------------------
        f.write("PIPELINE: %s\n" % pipeline_name)
        f.write("  fold_dir           = %s   (NOT READ HERE -- see below)\n" % config.fold_dir)
        f.write("  preprocess_version = %s\n" % config.preprocess_version)
        f.write("  grader             = %s\n" % grader)
        f.write("\n")
        f.write("Section 2 is produced by cython_utils.get_diagnosis_relevance(..., config),\n")
        f.write("the same dispatch point both arms grade with. Section 1 uses\n")
        f.write("cython_utils.cosine_similarity. Neither is simulated here any more (P37).\n")
        f.write("\n")
        f.write("fold_dir is not read: this report scores EVERY ordered patient pair, not\n")
        f.write("fold test cases, so the patient-leakage fix is invisible to it. 'folds-only'\n")
        f.write("therefore measures identically to 'legacy', and 'preprocess-only'\n")
        f.write("identically to 'corrected'. A corrected-vs-legacy delta in this file is a\n")
        f.write("PREPROCESSING delta and contains no part of the leakage effect.\n")
        f.write("\n")

        # Section 1: All-pairwise
        f.write("SECTION 1: ALL-PAIRWISE DIAGNOSIS SIMILARITIES\n")
        f.write("-" * 60 + "\n")
        f.write("Each model embeds all unique diagnosis descriptions, then\n")
        f.write("computes cosine similarity for every pair (excluding self-pairs).\n")
        f.write("This is a property of the EMBEDDING SPACE and of preprocess_version;\n")
        f.write("it does not depend on the grader.\n\n")
        for model_name, stats in all_pairwise_stats.items():
            f.write(f"Model: {model_name}\n")
            f.write(format_stats(stats) + "\n\n")

        # Section 2: Per-patient relevance
        f.write("\nSECTION 2: PER-PATIENT MAX SIMILARITIES (grader = %s)\n" % grader)
        f.write("-" * 60 + "\n")
        f.write("For each ordered pair of patients (A, B), the score that\n")
        f.write("get_diagnosis_relevance() returns -- what determines TP/FP at each\n")
        f.write("threshold.\n")
        if grader == "cosine":
            f.write("Under 'cosine' that is the MAX over the Cartesian product of\n")
            f.write("(gt_diag_A, pred_diag_B) description embeddings.\n\n")
        elif grader == "drg-exact":
            f.write("Under 'drg-exact' it is 1.0 when A and B share any DRG label and 0.0\n")
            f.write("otherwise. It consults NO embedding, so the three models below are\n")
            f.write("identical by construction -- that identity is the measurement, not a\n")
            f.write("bug. The percentile rows are degenerate for the same reason.\n\n")
        else:
            f.write("\n")
        for model_name, stats in per_patient_stats.items():
            f.write(f"Model: {model_name}\n")
            f.write(format_stats(stats) + "\n\n")

        # Diagnosis count stats
        f.write("\nDiagnosis Count Per Patient:\n")
        f.write(f"  Min  = {diag_count_stats['min']}\n")
        f.write(f"  Max  = {diag_count_stats['max']}\n")
        f.write(f"  Mean = {diag_count_stats['mean']:.2f}\n")
        f.write(f"  Total unique diagnoses = {diag_count_stats['n_unique']}\n")
        f.write(f"  Total patients = {diag_count_stats['n_patients']}\n")
        f.write(f"  Total patient pairs = {diag_count_stats['n_patients'] * (diag_count_stats['n_patients'] - 1)}\n\n")

        # Kernel cross-check -- diagnostic only
        f.write("\nKernel cross-check (diagnostic only -- produces no figure above):\n")
        f.write("  numpy normalise-and-dot vs cython_utils.cosine_similarity, 5 random pairs.\n")
        for model_name, diff in cross_check_diffs.items():
            f.write(f"  {model_name}: max absolute difference = {diff:.2e}\n")

        # Section 3: Interpretation
        f.write("\n\nSECTION 3: INTERPRETATION\n")
        f.write("-" * 60 + "\n\n")
        f.write("Key Findings:\n\n")

        f.write("1. EMBEDDING SPACE COMPACTNESS\n")
        for model_name, stats in all_pairwise_stats.items():
            f.write(f"   {model_name}: mean pairwise similarity = {stats['mean']:.4f}, ")
            f.write(f"std = {stats['std']:.4f}\n")
        f.write("   Biomedical BERT models embed medical diagnosis text into a\n")
        f.write("   relatively narrow region of the embedding space, producing\n")
        f.write("   high baseline similarities even between unrelated diagnoses.\n\n")

        if grader == "cosine":
            f.write("2. MAX OPERATOR AMPLIFICATION\n")
            for model_name in all_pairwise_stats:
                pw_mean = all_pairwise_stats[model_name]['mean']
                pm_mean = per_patient_stats[model_name]['mean']
                f.write(f"   {model_name}: pairwise mean = {pw_mean:.4f} -> ")
                f.write(f"per-patient MAX mean = {pm_mean:.4f}\n")
            f.write(f"   With {diag_count_stats['mean']:.1f} diagnoses per patient on average,\n")
            f.write(f"   the Cartesian product contains ~{diag_count_stats['mean']**2:.0f} pairs.\n")
            f.write("   Taking the MAX over this product dramatically inflates the\n")
            f.write("   effective similarity, pushing nearly all pairs above 0.6.\n\n")

            f.write("3. THRESHOLD SATURATION\n")
            for model_name, stats in per_patient_stats.items():
                f.write(f"   {model_name}:\n")
                for t in THRESHOLDS:
                    f.write(f"     >= {t:.1f}: {stats[f'pct_above_{t}']:.2f}%\n")
            f.write("\n   This explains the perfect F1 at threshold 0.6:\n")
            f.write("   virtually every patient pair has MAX similarity >= 0.6.\n")
            f.write("   The evaluation metric is saturated at that threshold and cannot\n")
            f.write("   discriminate between models.\n\n")

            f.write("4. IMPLICATIONS\n")
            f.write("   - The cosine grader (MAX over the Cartesian product of diagnosis\n")
            f.write("     descriptions) is too lenient for these encoders at 0.6.\n")
            f.write("   - The project's answer is --pipeline drg: an encoder-independent\n")
            f.write("     grader with no threshold knob at all. Re-run this script with it\n")
            f.write("     to see Section 2 collapse to 0/1.\n")
        else:
            f.write("2. NO MAX AMPLIFICATION TO REPORT\n")
            f.write("   The grader is %r. It takes no maximum over the Cartesian product\n" % grader)
            f.write("   and consults no embedding, so Section 1's compactness cannot\n")
            f.write("   propagate into Section 2 at all.\n\n")

            f.write("3. NO SATURATION TO REPORT\n")
            for model_name, stats in per_patient_stats.items():
                f.write(f"   {model_name}: relevance is 1.0 for {stats['pct_above_1.0']:.2f}% of ordered\n")
                f.write("     patient pairs and 0.0 for the rest -- identical across models.\n")
            f.write("\n   Every threshold row above 0 and up to 1.0 returns that same\n")
            f.write("   percentage, which is what 'the threshold knob is gone' means\n")
            f.write("   numerically. The saturation this report was written to document is\n")
            f.write("   a property of the cosine grader, not of the encoders.\n\n")

            f.write("4. IMPLICATIONS\n")
            f.write("   - A ceiling still applies and is not 1.0: only 76 of 129 test cases\n")
            f.write("     have their correct label anywhere in their own fold's training\n")
            f.write("     pool under folds_grouped (58.9%). Never quote a drg number\n")
            f.write("     without that denominator.\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"[SAVED] {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnosis relevance score distributions, under a named pipeline config."
    )
    parser.add_argument("--pipeline", choices=PIPELINE_NAMES, default=None, help=PIPELINE_HELP)
    parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: docs/score_distribution_analysis/). "
             "Pass a scratch path for an exploratory run so the committed "
             "artifacts are not overwritten by a config nobody quotes.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="write only the summary text. Useful for a comparison run whose "
             "PNGs would not be committed.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = from_name(args.pipeline) if args.pipeline else from_env()
    require_supported_grader(config)

    print("=" * 80)
    print("DIAGNOSIS SIMILARITY SCORE DISTRIBUTION ANALYSIS")
    print("PIPELINE: " + pipeline_banner(config))
    print("=" * 80)

    if config.fold_dir != LEGACY.fold_dir:
        print("[NOTE] fold_dir is not read by this analysis -- every ordered patient")
        print("       pair is scored, not fold test cases. Only preprocess_version and")
        print("       grader can move a number below.")

    output_dir = args.out or os.path.join(project_root, 'docs', 'score_distribution_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load dataset
    print("\n[1/4] Loading dataset...")
    t0 = time.time()
    admissions = load_dataset()
    print(f"  Loaded {len(admissions)} admissions in {time.time() - t0:.1f}s")

    unique_diagnoses = extract_unique_diagnoses(admissions)
    print(f"  Found {len(unique_diagnoses)} unique diagnosis descriptions")

    diag_counts = compute_diagnosis_counts(admissions)
    diag_count_stats = {
        'min': int(np.min(diag_counts)),
        'max': int(np.max(diag_counts)),
        'mean': float(np.mean(diag_counts)),
        'n_unique': len(unique_diagnoses),
        'n_patients': len(admissions),
    }
    print(f"  Diagnoses per patient: min={diag_count_stats['min']}, "
          f"max={diag_count_stats['max']}, mean={diag_count_stats['mean']:.1f}")

    # Step 2: Process each model
    all_pairwise_data = {}
    per_patient_data = {}
    all_pairwise_stats = {}
    per_patient_stats = {}
    cross_check_diffs = {}

    for model_name, model_path in MODELS.items():
        print(f"\n[2/4] Processing {model_name}...")

        t0 = time.time()
        print(f"  Loading model: {model_path}")
        model = SentenceTransformer(model_path)
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        t0 = time.time()
        embeddings, vectors = build_embeddings(model, unique_diagnoses, config)
        print(f"  Encoded {len(unique_diagnoses)} diagnoses in {time.time() - t0:.1f}s")
        print(f"  Embedding shape: {vectors.shape}")

        t0 = time.time()
        pairwise_scores = compute_pairwise_scores(embeddings, unique_diagnoses)
        print(f"  All-pairwise: {len(pairwise_scores)} unique pairs in {time.time() - t0:.1f}s")

        cross_check_diffs[model_name] = kernel_cross_check(vectors)
        print(f"  Kernel cross-check max diff: {cross_check_diffs[model_name]:.2e}")

        t0 = time.time()
        relevance_scores = compute_per_patient_relevance(admissions, embeddings, config)
        print(f"  Per-patient relevance: {len(relevance_scores)} ordered pairs "
              f"in {time.time() - t0:.1f}s")

        all_pairwise_data[model_name] = pairwise_scores
        per_patient_data[model_name] = relevance_scores
        all_pairwise_stats[model_name] = compute_stats(pairwise_scores, f"{model_name} (all-pairwise)")
        per_patient_stats[model_name] = compute_stats(relevance_scores, f"{model_name} (per-patient)")

        pw = all_pairwise_stats[model_name]
        pm = per_patient_stats[model_name]
        print(f"  All-pairwise: mean={pw['mean']:.4f}, std={pw['std']:.4f}")
        print(f"  Per-patient: mean={pm['mean']:.4f}, % >= 0.6: {pm['pct_above_0.6']:.2f}%, "
              f"% >= 1.0: {pm['pct_above_1.0']:.2f}%")

        del model

    # Step 3: Visualizations
    if args.no_plots:
        print("\n[3/4] Skipping plots (--no-plots).")
    else:
        print("\n[3/4] Generating visualizations...")
        plot_score_distributions(all_pairwise_data, output_dir, config)
        plot_per_patient_max(per_patient_data, output_dir, config)

    # Step 4: Summary
    print("\n[4/4] Writing summary...")
    write_summary(all_pairwise_stats, per_patient_stats, diag_count_stats,
                  cross_check_diffs, config, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE -- pipeline: " + pipeline_banner(config))
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

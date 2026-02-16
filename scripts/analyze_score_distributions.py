#!/usr/bin/env python3
"""
Analyze BERT diagnosis similarity score distributions.

Investigates why all 3 BERT models achieve perfect F1 = 1.000 at threshold 0.6
by examining:
  1. All-pairwise cosine similarity distributions across 145 unique diagnoses
  2. Per-patient MAX similarity distributions (what actually determines TP/FP)
  3. Diagnosis count per patient (Cartesian product amplification)

Outputs:
  docs/score_distribution_analysis/score_distributions.png
  docs/score_distribution_analysis/per_patient_max_distributions.png
  docs/score_distribution_analysis/score_distribution_summary.txt

Usage:
    python scripts/analyze_score_distributions.py
"""

import os
import sys
import time
import random

import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sentence_transformers import SentenceTransformer

from src.entity.SymptomsDiagnosis import SymptomsDiagnosis
from src.utils.Constants import CH_DIR
from src.utils import cython_utils as util_cy

# Initialize NLTK data (required by cython_utils)
import nltk
from nltk.corpus import stopwords
from string import punctuation

for pkg in ['stopwords', 'punkt_tab', 'punkt']:
    try:
        nltk.data.find(f'corpora/{pkg}' if pkg == 'stopwords' else f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

util_cy.stop_words = set(stopwords.words('english'))

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
    """Load admissions from Symptoms-Diagnosis.txt (same logic as bert_models.py:375-386)."""
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


def extract_unique_diagnoses(admissions):
    """Extract unique diagnosis descriptions (text after ':') across all admissions."""
    unique = set()
    for admission in admissions.values():
        for diag in admission.diagnosis:
            if ':' in diag:
                desc = diag[diag.index(':') + 1:]
            else:
                desc = diag
            unique.add(desc)
    return sorted(unique)


def get_patient_diagnosis_descriptions(admission):
    """Get list of diagnosis description texts for a single patient."""
    descs = []
    for diag in admission.diagnosis:
        if ':' in diag:
            descs.append(diag[diag.index(':') + 1:])
        else:
            descs.append(diag)
    return descs


def compute_pairwise_matrix(embeddings_matrix):
    """Compute all-pairwise cosine similarity matrix using numpy.

    Args:
        embeddings_matrix: (N, D) numpy array of embeddings

    Returns:
        (N, N) cosine similarity matrix
    """
    # Compute norms
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normalized = embeddings_matrix / norms
    return normalized @ normalized.T


def spot_check_against_cython(embeddings_matrix, diag_texts, n_checks=5):
    """Verify numpy pairwise results against loop-based cython_utils.cosine_similarity()."""
    n = len(diag_texts)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings_matrix / norms
    sim_matrix = normalized @ normalized.T

    random.seed(42)
    pairs = [(random.randint(0, n - 1), random.randint(0, n - 1)) for _ in range(n_checks)]

    max_diff = 0.0
    for i, j in pairs:
        numpy_sim = sim_matrix[i, j]
        cython_sim = util_cy.cosine_similarity(embeddings_matrix[i], embeddings_matrix[j])
        diff = abs(numpy_sim - cython_sim)
        max_diff = max(max_diff, diff)

    return max_diff


def compute_per_patient_max_scores(admissions, diag_to_idx, sim_matrix):
    """Simulate get_diagnosis_similarity_by_description_max() for all patient pairs.

    For each (patient_A, patient_B) pair where A != B, compute the MAX cosine
    similarity across all (gt_diag, pred_diag) pairs from the Cartesian product.

    Returns:
        list of floats: MAX similarity scores for all patient pairs
    """
    patient_ids = list(admissions.keys())
    n_patients = len(patient_ids)
    max_scores = []

    for i in range(n_patients):
        gt_descs = get_patient_diagnosis_descriptions(admissions[patient_ids[i]])
        gt_indices = [diag_to_idx[d] for d in gt_descs if d in diag_to_idx]

        for j in range(n_patients):
            if i == j:
                continue
            pred_descs = get_patient_diagnosis_descriptions(admissions[patient_ids[j]])
            pred_indices = [diag_to_idx[d] for d in pred_descs if d in diag_to_idx]

            if not gt_indices or not pred_indices:
                continue

            # MAX over Cartesian product of (gt_diag, pred_diag) pairs
            max_sim = -1.0
            for gi in gt_indices:
                for pi in pred_indices:
                    s = sim_matrix[gi, pi]
                    if s > max_sim:
                        max_sim = s
            max_scores.append(max_sim)

    return max_scores


def compute_diagnosis_counts(admissions):
    """Compute per-patient diagnosis counts."""
    counts = []
    for admission in admissions.values():
        descs = get_patient_diagnosis_descriptions(admission)
        counts.append(len(descs))
    return counts


def compute_stats(values, label):
    """Compute descriptive statistics for a list of values."""
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
    """Format stats dict as a readable string block."""
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


def plot_score_distributions(all_pairwise_data, output_dir):
    """Generate score_distributions.png with histogram and CDF panels."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top: overlaid histograms
    for model_name, scores in all_pairwise_data.items():
        ax1.hist(scores, bins=100, alpha=0.5, label=model_name,
                 color=MODEL_COLORS[model_name], density=True)
    for t in THRESHOLDS:
        ax1.axvline(x=t, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(t, ax1.get_ylim()[1] * 0.95, f'{t}', ha='center', fontsize=8, color='red')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Density')
    ax1.set_title('All-Pairwise Diagnosis Similarity Distributions (excluding self-pairs)')
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
    ax2.set_title('CDF of All-Pairwise Diagnosis Similarities')
    ax2.legend()
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, 'score_distributions.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


def plot_per_patient_max(per_patient_data, output_dir):
    """Generate per_patient_max_distributions.png with histogram and threshold sensitivity."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top: overlaid histograms of per-patient MAX scores
    for model_name, scores in per_patient_data.items():
        ax1.hist(scores, bins=100, alpha=0.5, label=model_name,
                 color=MODEL_COLORS[model_name], density=True)
    for t in THRESHOLDS:
        ax1.axvline(x=t, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(t, ax1.get_ylim()[1] * 0.95, f'{t}', ha='center', fontsize=8, color='red')
    ax1.set_xlabel('MAX Cosine Similarity (per patient pair)')
    ax1.set_ylabel('Density')
    ax1.set_title('Per-Patient MAX Diagnosis Similarity Distributions')
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
    ax2.set_title('Threshold Sensitivity: Fraction of Patient Pairs Classified as TP')
    ax2.legend()
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, 'per_patient_max_distributions.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


def write_summary(all_pairwise_stats, per_patient_stats, diag_count_stats,
                   spot_check_diffs, output_dir):
    """Write score_distribution_summary.txt."""
    path = os.path.join(output_dir, 'score_distribution_summary.txt')
    with open(path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BERT DIAGNOSIS SIMILARITY SCORE DISTRIBUTION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Section 1: All-pairwise
        f.write("SECTION 1: ALL-PAIRWISE DIAGNOSIS SIMILARITIES\n")
        f.write("-" * 60 + "\n")
        f.write("Each model embeds all unique diagnosis descriptions, then\n")
        f.write("computes cosine similarity for every pair (excluding self-pairs).\n\n")
        for model_name, stats in all_pairwise_stats.items():
            f.write(f"Model: {model_name}\n")
            f.write(format_stats(stats) + "\n\n")

        # Section 2: Per-patient MAX
        f.write("\nSECTION 2: PER-PATIENT MAX SIMILARITIES\n")
        f.write("-" * 60 + "\n")
        f.write("For each pair of patients (A, B), compute the MAX cosine\n")
        f.write("similarity across all (gt_diag_A, pred_diag_B) pairs.\n")
        f.write("This is what get_diagnosis_similarity_by_description_max() returns\n")
        f.write("and what determines TP/FP at each threshold.\n\n")
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

        # Spot-check verification
        f.write("\nVerification: numpy vs cython_utils.cosine_similarity()\n")
        for model_name, diff in spot_check_diffs.items():
            f.write(f"  {model_name}: max absolute difference = {diff:.2e}\n")

        # Section 3: Interpretation
        f.write("\n\nSECTION 3: INTERPRETATION\n")
        f.write("-" * 60 + "\n\n")

        # Dynamically generate interpretation based on actual data
        f.write("Key Findings:\n\n")

        f.write("1. EMBEDDING SPACE COMPACTNESS\n")
        for model_name, stats in all_pairwise_stats.items():
            f.write(f"   {model_name}: mean pairwise similarity = {stats['mean']:.4f}, ")
            f.write(f"std = {stats['std']:.4f}\n")
        f.write("   Biomedical BERT models embed medical diagnosis text into a\n")
        f.write("   relatively narrow region of the embedding space, producing\n")
        f.write("   high baseline similarities even between unrelated diagnoses.\n\n")

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
        f.write("\n   This explains why all models achieve perfect F1 at 0.6:\n")
        f.write("   virtually every patient pair has MAX similarity >= 0.6.\n")
        f.write("   The evaluation metric is saturated and cannot discriminate\n")
        f.write("   between models or meaningfully compare against BioSentVec.\n\n")

        f.write("4. IMPLICATIONS\n")
        f.write("   - The current evaluation methodology (MAX over Cartesian product\n")
        f.write("     of diagnosis descriptions) is too lenient for BERT models.\n")
        f.write("   - Consider alternative evaluation strategies:\n")
        f.write("     a) Use MEAN instead of MAX over diagnosis pairs\n")
        f.write("     b) Use exact/partial DRG code matching\n")
        f.write("     c) Raise thresholds significantly (e.g., 0.95+)\n")
        f.write("     d) Use a different similarity metric that better\n")
        f.write("        discriminates in compact embedding spaces\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"[SAVED] {path}")


def main():
    print("=" * 80)
    print("BERT DIAGNOSIS SIMILARITY SCORE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Create output directory
    output_dir = os.path.join(project_root, 'docs', 'score_distribution_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load dataset
    print("\n[1/4] Loading dataset...")
    t0 = time.time()
    admissions = load_dataset()
    print(f"  Loaded {len(admissions)} admissions in {time.time() - t0:.1f}s")

    # Extract unique diagnoses
    unique_diagnoses = extract_unique_diagnoses(admissions)
    diag_to_idx = {d: i for i, d in enumerate(unique_diagnoses)}
    print(f"  Found {len(unique_diagnoses)} unique diagnosis descriptions")

    # Diagnosis count stats
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
    spot_check_diffs = {}

    for model_name, model_path in MODELS.items():
        print(f"\n[2/4] Processing {model_name}...")

        # Load model
        t0 = time.time()
        print(f"  Loading model: {model_path}")
        model = SentenceTransformer(model_path)
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        # Encode all unique diagnoses
        t0 = time.time()
        embeddings = model.encode(
            unique_diagnoses,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False  # Match current pipeline
        )
        print(f"  Encoded {len(unique_diagnoses)} diagnoses in {time.time() - t0:.1f}s")
        print(f"  Embedding shape: {embeddings.shape}")

        # Compute all-pairwise similarity matrix
        sim_matrix = compute_pairwise_matrix(embeddings)

        # Extract upper triangle (excluding diagonal) for all-pairwise scores
        n = len(unique_diagnoses)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_scores = sim_matrix[upper_tri_indices].tolist()
        n_pairs = len(pairwise_scores)
        print(f"  All-pairwise: {n_pairs} unique pairs")

        # Spot-check numpy vs cython_utils
        max_diff = spot_check_against_cython(embeddings, unique_diagnoses)
        spot_check_diffs[model_name] = max_diff
        print(f"  Spot-check max diff (numpy vs cython): {max_diff:.2e}")

        # Compute per-patient MAX scores
        t0 = time.time()
        max_scores = compute_per_patient_max_scores(admissions, diag_to_idx, sim_matrix)
        print(f"  Per-patient MAX: {len(max_scores)} patient pairs in {time.time() - t0:.1f}s")

        # Store data and compute stats
        all_pairwise_data[model_name] = pairwise_scores
        per_patient_data[model_name] = max_scores
        all_pairwise_stats[model_name] = compute_stats(pairwise_scores, f"{model_name} (all-pairwise)")
        per_patient_stats[model_name] = compute_stats(max_scores, f"{model_name} (per-patient MAX)")

        # Print quick summary
        pw = all_pairwise_stats[model_name]
        pm = per_patient_stats[model_name]
        print(f"  All-pairwise: mean={pw['mean']:.4f}, std={pw['std']:.4f}")
        print(f"  Per-patient MAX: mean={pm['mean']:.4f}, % >= 0.6: {pm['pct_above_0.6']:.1f}%, "
              f"% >= 0.9: {pm['pct_above_0.9']:.1f}%")

        # Free model memory
        del model

    # Step 3: Generate visualizations
    print("\n[3/4] Generating visualizations...")
    plot_score_distributions(all_pairwise_data, output_dir)
    plot_per_patient_max(per_patient_data, output_dir)

    # Step 4: Write summary
    print("\n[4/4] Writing summary...")
    write_summary(all_pairwise_stats, per_patient_stats, diag_count_stats,
                  spot_check_diffs, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

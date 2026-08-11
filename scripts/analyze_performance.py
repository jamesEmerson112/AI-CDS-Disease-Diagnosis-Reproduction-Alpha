"""
Performance Analysis and Visualization Script for Disease Diagnosis System

Reads one run's PerformanceIndex.txt and writes performance_analysis.pdf beside it.

Usage:
    python scripts/analyze_performance.py                  # most recent run in the cwd
    python scripts/analyze_performance.py <run_directory>  # one specific run

Two things used to live here that no longer do.

The **parser** is now ``aicds.analysis.performance_index``. The copy that was here
was the only one of the four that read per-fold blocks, and it read them wrongly:
its header test was a substring check that the *per-case* header also satisfies, so
129 single-trial rows landed on top of each fold's aggregate. On the real files the
aggregate is written last and overwrites them, which is why nobody noticed. The
shared parser recognises per-case blocks, counts them and drops their values.

The **module-level ``fold_data`` / ``aggregate_data`` defaultdicts** are gone. They
were created once at import, so two files parsed in one interpreter merged into each
other -- and because they were ``defaultdict``s, a missing strategy materialised as
an empty entry instead of raising. Everything is passed down as arguments now, which
is also why the plot helpers take their data explicitly.

``validate`` is deliberately *not* called: a run that died in fold 6 is exactly the
file you want to plot, and this script is a diagnostic, not a gate.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from aicds import runs
from aicds.analysis.performance_index import read

METHODS = ['MAX', 'TOP-10', 'TOP-20', 'TOP-30', 'TOP-40', 'TOP-50']
THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 1.0]


def find_latest_output_directory():
    """The most recently written run sitting *directly in* the cwd, or None.

    ``depth=1`` -- flat ``Prediction_Output_<Model>_<stamp>/`` runs only -- is
    load-bearing, not a leftover from the old glob. The default depth-2 walk
    reaches subdirectories, and ``discover`` refuses rather than skips a
    ``PerformanceIndex.txt`` in neither layout; the repository root always has
    three of those, because the committed ``docs/Prediction_Output_*`` oracle runs
    are flat runs at depth 2. So ``discover(os.getcwd())`` from the root -- the
    directory CLAUDE.md tells everyone to run from -- raised every time, even with
    a perfectly good fresh run beside them, and pointed the user at ``docs/``.

    Bounding the walk is the fix rather than softening ``discover``: skipping an
    unclassifiable run is the failure finding 10 documents, and this is the one
    caller that genuinely wants a narrower question. Auto-detect means "whatever an
    arm just wrote here", and an arm with no ``--out`` writes flat into the cwd
    (``runs.run_dirs``, frozen by the golden). A ``results*/`` tree is a *nested*
    root; name it on the command line and it is read in full.

    ``discover`` already keeps one run per arm (the newest, announcing the choice
    when there is more than one); "most recent" across arms is then the newest of
    those. mtime, not the directory name: ``DDMMYYYY`` does not sort
    chronologically.
    """
    try:
        candidates = runs.discover(os.getcwd(), depth=1)
    except runs.RunLayoutError as error:
        print(f"[ERROR] {error}")
        sys.exit(1)
    if not candidates:
        return None
    return max(candidates, key=lambda run: os.path.getmtime(run.path)).path


def resolve_run_directory(run_dir):
    """Validate an explicit run directory, or auto-detect the most recent one."""
    if run_dir is not None:
        perf_file = os.path.join(run_dir, "PerformanceIndex.txt")

        if not os.path.exists(run_dir):
            print(f"[ERROR] Directory not found: {run_dir}")
            sys.exit(1)

        if not os.path.exists(perf_file):
            print(f"[ERROR] PerformanceIndex.txt not found in: {run_dir}")
            sys.exit(1)

        print(f"[INFO] Using specified directory: {run_dir}")
        return run_dir

    directory = find_latest_output_directory()

    if directory is None:
        print("[ERROR] No Prediction_Output_* run directory found in the cwd")
        print("[INFO] Run scripts/run_bert_analysis.py (or scripts/run_baseline.py) first,")
        print("       or pass a run directory explicitly. Auto-detect only sees runs")
        print("       written straight into the cwd; a run harvested into")
        print("       results*/<model>/<timestamp>/ has to be named on the command line.")
        sys.exit(1)

    print(f"[INFO] Auto-detected most recent output: {directory}")
    return directory


def load_performance_index(path):
    """Parse one PerformanceIndex.txt, reporting what came out of it."""
    print(f"[INFO] Parsing {path}...")
    index = read(path)
    print(f"[SUCCESS] Parsed data for {len(index.folds)} folds")
    print(f"[SUCCESS] Found {len(index.aggregate)} aggregate method results")
    return index

def plot_aggregate_performance(aggregate, ax):
    """Plot aggregate performance across different TOP-K methods."""
    # Extract F-Score data for threshold 1.0
    fscores = []
    precisions = []
    recalls = []

    for method in METHODS:
        if method in aggregate and 1.0 in aggregate[method]:
            row = aggregate[method][1.0]
            fscores.append(row.f_score)
            precisions.append(row.precision)
            recalls.append(row.recall)
        else:
            fscores.append(0)
            precisions.append(0)
            recalls.append(0)

    x = np.arange(len(METHODS))
    width = 0.25

    ax.bar(x - width, precisions, width, label='Precision', color='#4472C4')
    ax.bar(x, recalls, width, label='Recall', color='#ED7D31')
    ax.bar(x + width, fscores, width, label='F-Score', color='#A5A5A5')

    ax.set_xlabel('Similarity Method', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Aggregate Performance Comparison (Threshold=1.0)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

def plot_threshold_sensitivity(aggregate, ax):
    """Plot how performance changes with different thresholds."""
    methods = ['MAX', 'TOP-10', 'TOP-30', 'TOP-50']
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']

    for method, color in zip(methods, colors):
        if method in aggregate:
            rows = aggregate[method]
            recalls = [rows[t].recall if t in rows else 0 for t in THRESHOLDS]
            ax.plot(THRESHOLDS, recalls, marker='o', label=method, color=color, linewidth=2)

    ax.set_xlabel('Threshold', fontsize=10)
    ax.set_ylabel('Recall', fontsize=10)
    ax.set_title('Threshold Sensitivity Analysis', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.55, 1.05)

def plot_precision_recall_curve(aggregate, ax):
    """Plot Precision-Recall curves for different methods."""
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47']

    for method, color in zip(METHODS, colors):
        if method in aggregate:
            # Get all thresholds for this method
            thresholds = sorted(aggregate[method].keys())
            precisions = [aggregate[method][t].precision for t in thresholds]
            recalls = [aggregate[method][t].recall for t in thresholds]

            ax.plot(recalls, precisions, marker='o', label=method, color=color, linewidth=2)

    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_title('Precision-Recall Curves', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def plot_fold_variance(folds, ax):
    """Plot variance in F-Score across folds for TOP-10 method."""
    method = 'TOP-10'
    threshold = 1.0

    fold_indices = sorted(folds.keys())
    fscores = []

    for fold in fold_indices:
        if method in folds[fold] and threshold in folds[fold][method]:
            fscores.append(folds[fold][method][threshold].f_score)
        else:
            fscores.append(0)

    bars = ax.bar(fold_indices, fscores, color='#5B9BD5', edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

    # Add mean line
    mean_score = np.mean(fscores)
    ax.axhline(y=mean_score, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_score:.3f}')

    ax.set_xlabel('Fold', fontsize=10)
    ax.set_ylabel('F-Score', fontsize=10)
    ax.set_title('F-Score Variance Across Folds (TOP-10, Threshold=1.0)',
                 fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

def plot_heatmap_fscores(aggregate, ax):
    """Create heatmap of F-Scores for different methods and thresholds."""
    # Create matrix of F-Scores
    matrix = np.zeros((len(METHODS), len(THRESHOLDS)))

    for i, method in enumerate(METHODS):
        if method in aggregate:
            for j, threshold in enumerate(THRESHOLDS):
                if threshold in aggregate[method]:
                    matrix[i, j] = aggregate[method][threshold].f_score

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)

    # Set ticks
    ax.set_xticks(np.arange(len(THRESHOLDS)))
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_xticklabels(THRESHOLDS)
    ax.set_yticklabels(METHODS)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('F-Score', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(METHODS)):
        for j in range(len(THRESHOLDS)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('F-Score Heatmap: Methods vs Thresholds', fontsize=11, fontweight='bold')
    ax.set_xlabel('Threshold', fontsize=10)
    ax.set_ylabel('Similarity Method', fontsize=10)

def plot_topk_comparison(aggregate, ax):
    """Compare performance improvement from MAX to TOP-K."""
    threshold = 1.0

    recalls = []
    fscores = []

    for method in METHODS:
        if method in aggregate and threshold in aggregate[method]:
            recalls.append(aggregate[method][threshold].recall)
            fscores.append(aggregate[method][threshold].f_score)

    x = np.arange(len(METHODS))
    width = 0.35

    ax.bar(x - width/2, recalls, width, label='Recall', color='#ED7D31')
    ax.bar(x + width/2, fscores, width, label='F-Score', color='#5B9BD5')

    ax.set_xlabel('Method', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Performance Improvement with TOP-K (Threshold=1.0)',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.7)

def generate_analysis_report(folds, aggregate, output_pdf):
    """Generate comprehensive PDF report with visualizations."""
    print(f"[INFO] Generating analysis report: {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        # Page 1: Overview and Aggregate Performance
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Disease Diagnosis Performance Analysis', fontsize=16, fontweight='bold')

        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

        # Summary statistics text
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')

        # Find best performing configuration. Ties keep the FIRST seen, so the
        # iteration order matters: it is the file's write order (MAX then TOP-K,
        # thresholds 0.9, 1, 0.6, 0.8, 0.7 from the writers' set literal), which
        # the parser preserves as insertion order.
        best_fscore = 0
        best_method = ''
        best_threshold = 0

        for method in aggregate:
            for threshold in aggregate[method]:
                fs = aggregate[method][threshold].f_score
                if fs > best_fscore:
                    best_fscore = fs
                    best_method = method
                    best_threshold = threshold

        best_row = aggregate[best_method][best_threshold]
        summary_text = [
            ['Metric', 'Value'],
            ['Total Folds Analyzed', str(len(folds))],
            ['Methods Compared', '6 (MAX, TOP-10 through TOP-50)'],
            ['Thresholds Tested', '5 (0.6, 0.7, 0.8, 0.9, 1.0)'],
            ['', ''],
            ['Best Configuration:', ''],
            ['  Method', best_method],
            ['  Threshold', str(best_threshold)],
            ['  F-Score', f'{best_fscore:.4f}'],
            ['  Recall', f"{best_row.recall:.4f}"],
            ['  Precision', f"{best_row.precision:.4f}"],
        ]

        table = ax1.table(cellText=summary_text, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax1.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

        # Aggregate performance bar chart
        ax2 = fig.add_subplot(gs[1, 0])
        plot_aggregate_performance(aggregate, ax2)

        # Precision-Recall curve
        ax3 = fig.add_subplot(gs[1, 1])
        plot_precision_recall_curve(aggregate, ax3)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Detailed Analysis
        fig2 = plt.figure(figsize=(11, 8.5))
        fig2.suptitle('Detailed Performance Analysis', fontsize=16, fontweight='bold')

        gs2 = fig2.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

        # Threshold sensitivity
        ax4 = fig2.add_subplot(gs2[0, 0])
        plot_threshold_sensitivity(aggregate, ax4)

        # TOP-K comparison
        ax5 = fig2.add_subplot(gs2[0, 1])
        plot_topk_comparison(aggregate, ax5)

        # Fold variance
        ax6 = fig2.add_subplot(gs2[1, 0])
        plot_fold_variance(folds, ax6)

        # Heatmap
        ax7 = fig2.add_subplot(gs2[1, 1])
        plot_heatmap_fscores(aggregate, ax7)

        pdf.savefig(fig2, bbox_inches='tight')
        plt.close()

    print(f"[SUCCESS] Analysis report generated: {output_pdf}")

def print_summary_statistics(aggregate):
    """Print summary statistics to console."""
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS SUMMARY (Threshold=1.0)")
    print("="*80)

    # Get configurations at threshold 1.0
    threshold = 1.0
    configs = []
    for method in METHODS:
        if method in aggregate and threshold in aggregate[method]:
            row = aggregate[method][threshold]
            configs.append((method, threshold, row.f_score, row.recall, row.precision))

    print(f"\n{'Method':<12} {'Threshold':<12} {'F-Score':<12} {'Recall':<12} {'Precision':<12}")
    print("-"*80)
    for method, thresh, fscore, recall, precision in configs:
        print(f"{method:<12} {thresh:<12.1f} {fscore:<12.4f} {recall:<12.4f} {precision:<12.4f}")

    if len(configs) > 1:
        print("\nKey Findings:")
        if len(configs) > 1:
            print(f"  • TOP-10 achieves ~{configs[1][3]:.1%} recall vs {configs[0][3]:.1%} for MAX")
        print(f"  • Threshold 1.0 represents perfect match requirement (highest precision)")
        if len(configs) > 5:
            print(f"  • TOP-50 recall: {configs[5][3]:.1%} (maximum achievable at this threshold)")
    print("="*80 + "\n")

def main(argv=None):
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Plot one run's PerformanceIndex.txt into performance_analysis.pdf "
                    "beside it."
    )
    parser.add_argument(
        "run_dir", nargs="?", default=None,
        help="a run directory containing PerformanceIndex.txt. "
             "Default: the most recently written run found under the cwd.",
    )
    args = parser.parse_args(argv)

    print("\n" + "="*80)
    print("Performance Analysis Script - Disease Diagnosis System")
    print("="*80 + "\n")

    try:
        run_dir = resolve_run_directory(args.run_dir)
        input_file = os.path.join(run_dir, "PerformanceIndex.txt")
        output_pdf = os.path.join(run_dir, "performance_analysis.pdf")

        index = load_performance_index(input_file)

        if not index.aggregate:
            print("[ERROR] No aggregate data found. Check file format.")
            sys.exit(1)

        # Print summary statistics
        print_summary_statistics(index.aggregate)

        # Generate visualization report
        generate_analysis_report(index.folds, index.aggregate, output_pdf)

        print("\n[SUCCESS] Analysis complete!")
        print(f"[INFO] PDF report saved to: {output_pdf}")

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        print("[INFO] Please ensure the PerformanceIndex.txt file exists in the correct location.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

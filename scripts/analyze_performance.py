"""
Performance Analysis and Visualization Script for Disease Diagnosis System
This script parses PerformanceIndex.txt and generates comprehensive visualizations.

Usage:
    python analyze_performance.py                    # Auto-detect most recent output
    python analyze_performance.py <directory_name>   # Analyze specific directory
"""

import re
import sys
import os
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from collections import defaultdict

# Data structures
fold_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
aggregate_data = defaultdict(lambda: defaultdict(dict))

def find_latest_output_directory():
    """Find the most recent Prediction_Output_* directory."""
    pattern = "Prediction_Output_*"
    directories = glob.glob(pattern)
    
    # Filter to only directories that contain PerformanceIndex.txt
    valid_dirs = []
    for dir_path in directories:
        perf_file = os.path.join(dir_path, "PerformanceIndex.txt")
        if os.path.isdir(dir_path) and os.path.exists(perf_file):
            valid_dirs.append(dir_path)
    
    if not valid_dirs:
        return None
    
    # Sort by modification time (most recent first)
    valid_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return valid_dirs[0]

def get_output_directory():
    """Get the output directory from command line or auto-detect."""
    if len(sys.argv) > 1:
        # User specified directory
        directory = sys.argv[1]
        perf_file = os.path.join(directory, "PerformanceIndex.txt")
        
        if not os.path.exists(directory):
            print(f"[ERROR] Directory not found: {directory}")
            sys.exit(1)
        
        if not os.path.exists(perf_file):
            print(f"[ERROR] PerformanceIndex.txt not found in: {directory}")
            sys.exit(1)
        
        print(f"[INFO] Using specified directory: {directory}")
        return directory
    else:
        # Auto-detect most recent
        directory = find_latest_output_directory()
        
        if directory is None:
            print("[ERROR] No Prediction_Output_* directories found with PerformanceIndex.txt")
            print("[INFO] Please ensure CS2V.py has been run to generate output.")
            sys.exit(1)
        
        print(f"[INFO] Auto-detected most recent output: {directory}")
        return directory

def parse_performance_index(filename):
    """Parse the PerformanceIndex.txt file to extract all metrics."""
    print(f"[INFO] Parsing {filename}...")
    
    current_fold = None
    current_method = None
    parsing_aggregate = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect fold header (but not 10-FOLD)
            fold_match = re.match(r'^FOLD (\d+):', line)
            if fold_match:
                current_fold = int(fold_match.group(1))
                parsing_aggregate = False
                continue
            
            # Detect aggregate section and method type in same line
            if '10-FOLD PERFORMANCE INDEX' in line:
                parsing_aggregate = True
                # Also extract method from this line
                if 'MAX SIMILARITY by MAX' in line and 'TOP-' not in line:
                    current_method = 'MAX'
                elif 'TOP-' in line:
                    match = re.search(r'TOP-(\d+)', line)
                    if match:
                        current_method = f'TOP-{match.group(1)}'
                continue
            
            # Detect method type for non-aggregate sections
            if not parsing_aggregate:
                if 'PERFORMANCE INDEX of MAX SIMILARITY by MAX' in line and 'TOP-' not in line:
                    current_method = 'MAX'
                    continue
                elif 'PERFORMANCE INDEX of TOP-' in line:
                    match = re.search(r'TOP-(\d+)', line)
                    if match:
                        current_method = f'TOP-{match.group(1)}'
                    continue
            
            # Parse data lines (threshold, TP, FP, P, R, FS, PR)
            if line and not line.startswith('TP') and not line.startswith('*') and not line.startswith('='):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        threshold = float(parts[0])
                        tp = float(parts[1])
                        fp = float(parts[2])
                        precision = float(parts[3])
                        recall = float(parts[4])
                        fscore = float(parts[5])
                        
                        if parsing_aggregate and current_method:
                            aggregate_data[current_method][threshold] = {
                                'TP': tp, 'FP': fp, 'P': precision,
                                'R': recall, 'FS': fscore
                            }
                        elif current_fold is not None and current_method:
                            fold_data[current_fold][current_method][threshold] = {
                                'TP': tp, 'FP': fp, 'P': precision,
                                'R': recall, 'FS': fscore
                            }
                    except (ValueError, IndexError):
                        continue
    
    print(f"[SUCCESS] Parsed data for {len(fold_data)} folds")
    print(f"[SUCCESS] Found {len(aggregate_data)} aggregate method results")
    return fold_data, aggregate_data

def plot_aggregate_performance(aggregate_data, ax):
    """Plot aggregate performance across different TOP-K methods."""
    methods = ['MAX', 'TOP-10', 'TOP-20', 'TOP-30', 'TOP-40', 'TOP-50']
    thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Extract F-Score data for threshold 1.0
    fscores = []
    precisions = []
    recalls = []
    
    for method in methods:
        if method in aggregate_data and 1.0 in aggregate_data[method]:
            fscores.append(aggregate_data[method][1.0]['FS'])
            precisions.append(aggregate_data[method][1.0]['P'])
            recalls.append(aggregate_data[method][1.0]['R'])
        else:
            fscores.append(0)
            precisions.append(0)
            recalls.append(0)
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax.bar(x - width, precisions, width, label='Precision', color='#4472C4')
    ax.bar(x, recalls, width, label='Recall', color='#ED7D31')
    ax.bar(x + width, fscores, width, label='F-Score', color='#A5A5A5')
    
    ax.set_xlabel('Similarity Method', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Aggregate Performance Comparison (Threshold=1.0)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

def plot_threshold_sensitivity(aggregate_data, ax):
    """Plot how performance changes with different thresholds."""
    thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]
    methods = ['MAX', 'TOP-10', 'TOP-30', 'TOP-50']
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
    
    for method, color in zip(methods, colors):
        if method in aggregate_data:
            recalls = [aggregate_data[method].get(t, {}).get('R', 0) for t in thresholds]
            ax.plot(thresholds, recalls, marker='o', label=method, color=color, linewidth=2)
    
    ax.set_xlabel('Threshold', fontsize=10)
    ax.set_ylabel('Recall', fontsize=10)
    ax.set_title('Threshold Sensitivity Analysis', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.55, 1.05)

def plot_precision_recall_curve(aggregate_data, ax):
    """Plot Precision-Recall curves for different methods."""
    methods = ['MAX', 'TOP-10', 'TOP-20', 'TOP-30', 'TOP-40', 'TOP-50']
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47']
    
    for method, color in zip(methods, colors):
        if method in aggregate_data:
            # Get all thresholds for this method
            thresholds = sorted(aggregate_data[method].keys())
            precisions = [aggregate_data[method][t]['P'] for t in thresholds]
            recalls = [aggregate_data[method][t]['R'] for t in thresholds]
            
            ax.plot(recalls, precisions, marker='o', label=method, color=color, linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_title('Precision-Recall Curves', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def plot_fold_variance(fold_data, ax):
    """Plot variance in F-Score across folds for TOP-10 method."""
    method = 'TOP-10'
    threshold = 1.0
    
    folds = sorted(fold_data.keys())
    fscores = []
    
    for fold in folds:
        if method in fold_data[fold] and threshold in fold_data[fold][method]:
            fscores.append(fold_data[fold][method][threshold]['FS'])
        else:
            fscores.append(0)
    
    bars = ax.bar(folds, fscores, color='#5B9BD5', edgecolor='black', linewidth=0.5)
    
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

def plot_heatmap_fscores(aggregate_data, ax):
    """Create heatmap of F-Scores for different methods and thresholds."""
    methods = ['MAX', 'TOP-10', 'TOP-20', 'TOP-30', 'TOP-40', 'TOP-50']
    thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Create matrix of F-Scores
    matrix = np.zeros((len(methods), len(thresholds)))
    
    for i, method in enumerate(methods):
        if method in aggregate_data:
            for j, threshold in enumerate(thresholds):
                if threshold in aggregate_data[method]:
                    matrix[i, j] = aggregate_data[method][threshold]['FS']
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.6)
    
    # Set ticks
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(thresholds)
    ax.set_yticklabels(methods)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('F-Score', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(thresholds)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('F-Score Heatmap: Methods vs Thresholds', fontsize=11, fontweight='bold')
    ax.set_xlabel('Threshold', fontsize=10)
    ax.set_ylabel('Similarity Method', fontsize=10)

def plot_topk_comparison(aggregate_data, ax):
    """Compare performance improvement from MAX to TOP-K."""
    threshold = 1.0
    methods = ['MAX', 'TOP-10', 'TOP-20', 'TOP-30', 'TOP-40', 'TOP-50']
    
    recalls = []
    fscores = []
    
    for method in methods:
        if method in aggregate_data and threshold in aggregate_data[method]:
            recalls.append(aggregate_data[method][threshold]['R'])
            fscores.append(aggregate_data[method][threshold]['FS'])
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax.bar(x - width/2, recalls, width, label='Recall', color='#ED7D31')
    ax.bar(x + width/2, fscores, width, label='F-Score', color='#5B9BD5')
    
    ax.set_xlabel('Method', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Performance Improvement with TOP-K (Threshold=1.0)', 
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.7)

def generate_analysis_report(fold_data, aggregate_data, output_pdf):
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
        
        # Find best performing configuration
        best_fscore = 0
        best_method = ''
        best_threshold = 0
        
        for method in aggregate_data:
            for threshold in aggregate_data[method]:
                fs = aggregate_data[method][threshold]['FS']
                if fs > best_fscore:
                    best_fscore = fs
                    best_method = method
                    best_threshold = threshold
        
        summary_text = [
            ['Metric', 'Value'],
            ['Total Folds Analyzed', str(len(fold_data))],
            ['Methods Compared', '6 (MAX, TOP-10 through TOP-50)'],
            ['Thresholds Tested', '5 (0.6, 0.7, 0.8, 0.9, 1.0)'],
            ['', ''],
            ['Best Configuration:', ''],
            ['  Method', best_method],
            ['  Threshold', str(best_threshold)],
            ['  F-Score', f'{best_fscore:.4f}'],
            ['  Recall', f"{aggregate_data[best_method][best_threshold]['R']:.4f}"],
            ['  Precision', f"{aggregate_data[best_method][best_threshold]['P']:.4f}"],
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
        plot_aggregate_performance(aggregate_data, ax2)
        
        # Precision-Recall curve
        ax3 = fig.add_subplot(gs[1, 1])
        plot_precision_recall_curve(aggregate_data, ax3)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Detailed Analysis
        fig2 = plt.figure(figsize=(11, 8.5))
        fig2.suptitle('Detailed Performance Analysis', fontsize=16, fontweight='bold')
        
        gs2 = fig2.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        
        # Threshold sensitivity
        ax4 = fig2.add_subplot(gs2[0, 0])
        plot_threshold_sensitivity(aggregate_data, ax4)
        
        # TOP-K comparison
        ax5 = fig2.add_subplot(gs2[0, 1])
        plot_topk_comparison(aggregate_data, ax5)
        
        # Fold variance
        ax6 = fig2.add_subplot(gs2[1, 0])
        plot_fold_variance(fold_data, ax6)
        
        # Heatmap
        ax7 = fig2.add_subplot(gs2[1, 1])
        plot_heatmap_fscores(aggregate_data, ax7)
        
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close()
    
    print(f"[SUCCESS] Analysis report generated: {output_pdf}")

def print_summary_statistics(aggregate_data):
    """Print summary statistics to console."""
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS SUMMARY (Threshold=1.0)")
    print("="*80)
    
    # Get configurations at threshold 1.0
    threshold = 1.0
    configs = []
    for method in ['MAX', 'TOP-10', 'TOP-20', 'TOP-30', 'TOP-40', 'TOP-50']:
        if method in aggregate_data and threshold in aggregate_data[method]:
            data = aggregate_data[method][threshold]
            configs.append((method, threshold, data['FS'], data['R'], data['P']))
    
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

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("Performance Analysis Script - Disease Diagnosis System")
    print("="*80 + "\n")
    
    try:
        # Get the output directory
        output_dir = get_output_directory()
        input_file = os.path.join(output_dir, "PerformanceIndex.txt")
        output_pdf = os.path.join(output_dir, "performance_analysis.pdf")
        
        # Parse the performance index file
        fold_data, aggregate_data = parse_performance_index(input_file)
        
        if not aggregate_data:
            print("[ERROR] No aggregate data found. Check file format.")
            sys.exit(1)
        
        # Print summary statistics
        print_summary_statistics(aggregate_data)
        
        # Generate visualization report
        generate_analysis_report(fold_data, aggregate_data, output_pdf)
        
        print("\n[SUCCESS] Analysis complete!")
        print(f"[INFO] PDF report saved to: {output_pdf}")
        
    except FileNotFoundError:
        print(f"[ERROR] File not found: {input_file}")
        print("[INFO] Please ensure the PerformanceIndex.txt file exists in the correct location.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

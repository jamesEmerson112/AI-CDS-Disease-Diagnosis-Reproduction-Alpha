import os
import shutil
import sys
import time

# ``import sent2vec`` used to sit here and it was pure dead weight: the only
# Sent2vecModel in the repo is constructed inside cython_utils.load_model, which
# imports sent2vec LOCALLY. Deleting it did not, on its own, make this module
# importable on Windows -- it only moved the failure one frame deeper, because
# the whole pipeline still ran at import time and reached that local ``import
# sent2vec`` anyway. Moving the pipeline into run_analysis() finished the job:
# importing this module now loads no model and runs no fold, which is what
# tests/test_baseline_import_safety.py pins.
import numpy
import numpy as np

from aicds.entity.SymptomsDiagnosis import SymptomsDiagnosis
from aicds.utils.Constants import *

from aicds.utils import cython_utils as util_cy
from aicds.config import LEGACY, from_env, require_supported_grader
from aicds.analysis.rank_report import RankAccumulator

# Debug mode - set to False to disable verbose logging
DEBUG_MODE = False
DEBUG_CASE_LIMIT = 3  # Only debug first N cases per fold

# Ensure NLTK data is available before first use.
# ``import nltk`` used to sit here; its last consumer in this module was the
# stop_words monkeypatch deleted in C1. NLTK is still reached, indirectly,
# through cython_utils' preprocessing -- which is why the data check stays.
from aicds.utils.runtime import ensure_nltk_data, format_time

# Call before any NLTK usage
ensure_nltk_data()

from pathlib import Path

# Import matplotlib for PDF generation
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Timing utilities live in aicds.utils.runtime, imported above.

def generate_timing_pdf(timing_data, output_path):
    """Generate a PDF report with timing visualizations"""
    print("[INFO] Generating PDF timing report with graphs...")

    # Create PDF with multiple pages
    with PdfPages(output_path) as pdf:
        # Page 1: Summary text and phase breakdown bar chart
        fig = plt.figure(figsize=(11, 8.5))

        # Title
        fig.suptitle('Disease Diagnosis System - Timing Report', fontsize=16, fontweight='bold')

        # Create subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Subplot 1: Summary text table (top left)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')

        summary_text = [
            ['Phase', 'Time', 'Minutes'],
            ['Dataset Loading', f"{timing_data['dataset_loading']:.2f}s", f"{timing_data['dataset_loading']/60:.2f}m"],
            ['Model Loading', f"{timing_data['model_loading']:.2f}s", f"{timing_data['model_loading']/60:.2f}m"],
            ['Symptom Embeddings', f"{timing_data['symptom_embeddings']:.2f}s", f"{timing_data['symptom_embeddings']/60:.2f}m"],
            ['Diagnosis Embeddings', f"{timing_data['diagnosis_embeddings']:.2f}s", f"{timing_data['diagnosis_embeddings']/60:.2f}m"],
            ['Total Folds', f"{timing_data['folds_total']:.2f}s", f"{timing_data['folds_total']/60:.2f}m"],
            ['', '', ''],
            ['TOTAL EXECUTION', f"{timing_data['total_execution']:.2f}s", f"{timing_data['total_execution']/60:.2f}m"],
        ]

        table = ax1.table(cellText=summary_text, cellLoc='left', loc='center',
                         colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style total row
        for i in range(3):
            table[(7, i)].set_facecolor('#E7E6E6')
            table[(7, i)].set_text_props(weight='bold')

        ax1.set_title('Execution Time Summary', fontsize=12, fontweight='bold', pad=20)

        # Subplot 2: Phase breakdown bar chart (bottom left)
        ax2 = fig.add_subplot(gs[1, 0])

        phases = ['Dataset\nLoading', 'Model\nLoading', 'Symptom\nEmbeddings',
                 'Diagnosis\nEmbeddings', 'Folds\nProcessing']
        times_minutes = [
            timing_data['dataset_loading'] / 60,
            timing_data['model_loading'] / 60,
            timing_data['symptom_embeddings'] / 60,
            timing_data['diagnosis_embeddings'] / 60,
            timing_data['folds_total'] / 60
        ]

        colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5']
        bars = ax2.bar(phases, times_minutes, color=colors)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}m',
                    ha='center', va='bottom', fontsize=9)

        ax2.set_ylabel('Time (minutes)', fontsize=10)
        ax2.set_title('Phase Execution Times', fontsize=11, fontweight='bold')
        ax2.tick_params(axis='x', labelsize=8)
        ax2.grid(axis='y', alpha=0.3)

        # Subplot 3: Time distribution pie chart (bottom right)
        ax3 = fig.add_subplot(gs[1, 1])

        pie_labels = ['Dataset', 'Model', 'Embeddings', 'Folds']
        pie_values = [
            timing_data['dataset_loading'],
            timing_data['model_loading'],
            timing_data['embeddings_total'],
            timing_data['folds_total']
        ]

        pie_colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#5B9BD5']
        wedges, texts, autotexts = ax3.pie(pie_values, labels=pie_labels, autopct='%1.1f%%',
                                            colors=pie_colors, startangle=90)

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_weight('bold')

        ax3.set_title('Time Distribution', fontsize=11, fontweight='bold')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Individual fold times
        fig2 = plt.figure(figsize=(11, 8.5))
        fig2.suptitle('Individual Fold Execution Times', fontsize=16, fontweight='bold')

        ax4 = fig2.add_subplot(111)

        fold_labels = [f'Fold {i}' for i in range(len(timing_data['fold_times']))]
        fold_times_minutes = [t / 60 for t in timing_data['fold_times']]

        bars = ax4.bar(fold_labels, fold_times_minutes, color='#5B9BD5', edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}m',
                    ha='center', va='bottom', fontsize=9)

        # Add statistics
        avg_time = np.mean(fold_times_minutes)
        min_time = np.min(fold_times_minutes)
        max_time = np.max(fold_times_minutes)

        ax4.axhline(y=avg_time, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_time:.2f}m')

        ax4.set_xlabel('Fold', fontsize=11)
        ax4.set_ylabel('Time (minutes)', fontsize=11)
        ax4.set_title(f'Statistics: Min={min_time:.2f}m, Max={max_time:.2f}m, Avg={avg_time:.2f}m',
                     fontsize=10, pad=10)
        ax4.legend(loc='upper right')
        ax4.grid(axis='y', alpha=0.3)

        pdf.savefig(fig2, bbox_inches='tight')
        plt.close()

    print(f"[SUCCESS] PDF report generated: {output_path}")


def run_analysis(encoder=None, config=LEGACY):
    """
    Run the full BioSentVec baseline disease diagnosis analysis pipeline.

    Args:
        encoder: optional pre-built encoder to use instead of loading the 21GB
                 BioSentVec binary. Must expose ``embed_sentence(text)`` -- the
                 only method cython_utils' embedding helpers call on it, and a
                 different contract from the BERT arm's ``.encode``/``.device``/
                 ``.max_seq_length``. Only the model construction is bypassed;
                 every downstream step is unchanged. Leave as None for normal
                 runs.
        config: PipelineConfig selecting which variant of each pipeline stage
                to run. Defaults to aicds.config.LEGACY, which reproduces the
                published pipeline bit-for-bit. Pass aicds.config.CORRECTED (or
                any other registry entry) for a pipeline that deliberately moves
                the numbers.

    Returns:
        str: Path to the output directory containing results.
    """
    # Fail before the 21GB model load and the embeddings, not 12 minutes into
    # the fold loop -- an unhonoured grader would otherwise emit cosine numbers
    # under another grader's name, which nothing downstream could detect.
    require_supported_grader(config)

    # Announce every axis of the honoured config, the grader included. This
    # print belongs in the function, not at module scope: only the function
    # knows which config is actually being run, so a module-level banner could
    # report the environment's and still be wrong the moment a caller passes
    # config= explicitly. It matters here because this arm writes its output to
    # a directory named only by timestamp -- no model name, no pipeline -- so
    # the run log is the ONLY place the configuration is recorded until
    # run_metadata.json lands. Without it a corrected or DRG-graded baseline run
    # is indistinguishable from a legacy one after the fact.
    print("=" * 60)
    print("AI-CDS Disease Diagnosis - BioSentVec baseline")
    print("=" * 60)
    print("Pipeline:     %s  (folds: %s)" % (config.preprocess_version,
                                             config.fold_dir))
    print("Grader:       %s" % config.grader)
    if config.grader == "drg-exact":
        print("              exact DRG label match; ceiling is 76/129 = 58.9% on")
        print("              folds_grouped, so 1.0 is unreachable by construction")
    print("=" * 60)

    # Timing storage
    timing_data = {}
    script_start_time = time.perf_counter()

    ################################################################################################################
    #READ DATASET
    ################################################################################################################
    # Use proper path joining instead of changing directory
    dataset_start = time.perf_counter()
    file_name = os.path.join(CH_DIR, "data", "raw", "Symptoms-Diagnosis.txt")
    f = open(file_name, "r").readlines()

    admissions = dict()
    for line in f:
        # Was a bare `line.replace(...)` whose result was thrown away. Assigning it is
        # provably inert here -- the diagnosis is the last ';'-delimited field, so it
        # is the only one carrying the newline, and preprocess_diagnosis rstrips its
        # input (verified: identical output with and without a trailing newline). Kept
        # as a real assignment because a discarded no-op reads as a live bug.
        line = line.replace("\n", "")
        attributes = line.split(';')
        a = SymptomsDiagnosis(attributes[SymptomsDiagnosis.CONST_HADM_ID], attributes[SymptomsDiagnosis.CONST_SUBJECT_ID], attributes[SymptomsDiagnosis.CONST_ADMITTIME],
                                                       attributes[SymptomsDiagnosis.CONST_DISCHTIME], attributes[SymptomsDiagnosis.CONST_SYMPTOMS],
                                                       util_cy.preprocess_diagnosis(attributes[SymptomsDiagnosis.CONST_DIAGNOSIS]))
        admissions.update({attributes[SymptomsDiagnosis.CONST_HADM_ID]:a})

    dataset_time = time.perf_counter() - dataset_start
    timing_data['dataset_loading'] = dataset_time
    print(f"[INFO] Dataset loaded: {len(admissions)} admissions")
    print(f"[INFO] Dataset file: {file_name}")
    print(f"[TIMING] Dataset loading: {format_time(dataset_time)}")

    ################################################################################################################
    #LOAD MODEL
    ################################################################################################################
    model_start = time.perf_counter()
    print("[INFO] Loading BioSentVec model (21GB - this may take 5-10 minutes)...")
    print(f"[INFO] Model path: {os.path.join(CH_DIR, 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin')}")
    injected = encoder is not None
    try:
        model = encoder if injected else util_cy.load_model()
        model_time = time.perf_counter() - model_start
        timing_data['model_loading'] = model_time
        print("[SUCCESS] Model loaded successfully!")
        print(f"[TIMING] Model loading: {format_time(model_time)}")
    except Exception as e:
        # An injected encoder that raises is a bug in the caller's stub, not an
        # absent 21GB download: sys.exit(1) would swallow the traceback the
        # caller needs. Same carve-out as bert_models.run_analysis.
        if injected:
            raise
        print(f"[ERROR] Loading model failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    ################################################################################################################
    #COMPUTE EMBENDINGS
    ################################################################################################################
    embeddings_start = time.perf_counter()
    print("[INFO] Computing symptom embeddings...")
    symptom_emb_start = time.perf_counter()
    embendings_symptoms = util_cy.embending_symptoms(model,admissions,config)
    symptom_emb_time = time.perf_counter() - symptom_emb_start
    timing_data['symptom_embeddings'] = symptom_emb_time
    print(f"[SUCCESS] Symptom embeddings computed: {len(embendings_symptoms)} items")
    print(f"[TIMING] Symptom embeddings: {format_time(symptom_emb_time)}")

    print("[INFO] Computing diagnosis embeddings...")
    diagnosis_emb_start = time.perf_counter()
    embendings_diagnosis = util_cy.embending_diagnosis(model,admissions,config)
    diagnosis_emb_time = time.perf_counter() - diagnosis_emb_start
    timing_data['diagnosis_embeddings'] = diagnosis_emb_time
    print(f"[SUCCESS] Diagnosis embeddings computed: {len(embendings_diagnosis)} items")
    print(f"[TIMING] Diagnosis embeddings: {format_time(diagnosis_emb_time)}")

    embeddings_total_time = time.perf_counter() - embeddings_start
    timing_data['embeddings_total'] = embeddings_total_time

    ################################################################################################################
    #PERFORMANCE MATRIX
    ################################################################################################################
    performance_matrix_max = util_cy.init_performance_matrix()
    performance_matrix_topK_max_dict = dict()
    for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
        performance_matrix_topK_max_dict.update({topk: util_cy.init_performance_matrix()})

    ################################################################################################################
    #OUTPUT DIRECTORIES
    ################################################################################################################
    # Fix: Replace colons with dashes for Windows compatibility
    timestamp = util_cy.current_time().replace('/','').replace(':', '-')
    directory_prediction_root = os.getcwd() + '/Prediction Output_' + timestamp + '/'
    directory_prediction_details_root = os.getcwd() + '/Prediction Symptom Details_' + timestamp + '/'
    shutil.rmtree(directory_prediction_root, ignore_errors=True)
    shutil.rmtree(directory_prediction_details_root, ignore_errors=True)
    Path(directory_prediction_root).mkdir(parents=True, exist_ok=True)
    Path(directory_prediction_details_root).mkdir(parents=True, exist_ok=True)
    # ONE handle, replacing the 'w' handle that was never closed plus the second
    # 'a' handle that reopened the same path for the trailer. That pair emitted
    # the same bytes this does, and the reason is worth recording rather than
    # re-deriving: rebinding performance_out_file to the 'a' handle dropped the
    # last reference to the 'w' object, so CPython flushed and closed it right
    # there -- before the trailer's first write, which O_APPEND then placed at
    # the true end of file. The emitted sequence was
    # [everything written through 'w'][trailer], which is exactly what one
    # with-block produces. Same path, same mode-'w' truncation, same default
    # encoding and newline translation, same write ORDER with the trailer still
    # last, and the trailer is still written when generate_timing_pdf raises
    # because its try/except is inside this block and unchanged.
    with open(directory_prediction_root + '/PerformanceIndex.txt', 'w') as performance_out_file:

        ################################################################################################################
        # WORK ON SINGLE FOLD
        ################################################################################################################
        folds_start = time.perf_counter()
        fold_times = []
        # P5: ONE accumulator per run, written once after the fold loop -- it
        # spans folds, so it cannot live inside the loop, and it takes the
        # config PARAMETER rather than a module global.
        rank_accumulator = RankAccumulator(config)
        for nFold in range(0,K_FOLD):
            fold_start = time.perf_counter()
            directory_prediction = directory_prediction_root + 'Fold' + str(nFold) + "/"
            directory_prediction_details = directory_prediction_details_root + 'Fold' + str(nFold) + "/"
            shutil.rmtree(directory_prediction, ignore_errors=True)
            shutil.rmtree(directory_prediction_details, ignore_errors=True)
            Path(directory_prediction).mkdir(parents=True, exist_ok=True)
            Path(directory_prediction_details).mkdir(parents=True, exist_ok=True)
            ##########################################################################################################
            # LOAD TRAIN AND TEST SET
            ##########################################################################################################
            x_test = util_cy.load_dataset(nFold,TEST,config)
            x_train = util_cy.load_dataset(nFold,TRAIN,config)
            performance_out_file.write('\n FOLD %s: LEN train: %s, LEN test: %s \n' % (nFold, len(x_train), len(x_test)))
            print('FOLD %s: LEN train: %s, LEN test: %s' % (nFold, len(x_train), len(x_test)))
            ##########################################################################################################
            # COMPUTE PREDICTION
            ##########################################################################################################
            nrow = len(x_test)
            # P5: declare the fold's nrow so the all-cases denominator provably matches
            # legacy's `recall = tp / nrow`. This arm has no `continue` in its per-case
            # loop today, unlike the BERT arm -- declaring it anyway means adding one
            # later fails loudly instead of shrinking a denominator by one case.
            rank_accumulator.begin_fold(nFold, nrow)
            ncol = len(x_train)
            similarity_matrix = numpy.zeros(shape=(nrow, ncol))
            confusion_matrix_max = util_cy.init_confusion_matrix()
            confusion_matrix_Top_K_max_dict = dict()
            for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
                confusion_matrix_Top_K_max_dict.update({topk:util_cy.init_confusion_matrix()})
            # WORK ON SINGLE PREDICTION
            for i in range(0, nrow):
                # TEST SYMPTHOMS
                index = list(x_test[i].keys())[0]
                test_symptoms = list(x_test[i].values())[0]
                # TEST ADMISSION
                test_admission = admissions.get(index)

                # DEBUG: Log first N cases per fold
                if DEBUG_MODE and i < DEBUG_CASE_LIMIT:
                    print(f"\n[DEBUG BASELINE] === Test Case {i} (Patient {index}) ===")
                    print(f"[DEBUG BASELINE] True Diagnosis Type: {type(test_admission.diagnosis)}")
                    print(f"[DEBUG BASELINE] True Diagnosis: {test_admission.diagnosis}")
                    print(f"[DEBUG BASELINE] Calling util_cy.predictS2V (C function)...")

                #######################################################
                # EXECUTE PREDICTION
                #######################################################
                top_similarities_max = util_cy.predictS2V(i, index, test_admission, test_symptoms, x_train, nrow, ncol, embendings_symptoms, embendings_diagnosis,
                        admissions, similarity_matrix, None, confusion_matrix_max, None, confusion_matrix_Top_K_max_dict,
                        directory_prediction, directory_prediction_details, performance_out_file, config)

                # P5: the same rank-ordered, pruned vector the confusion matrices scored.
                # predictS2V returns it rather than taking a 19th positional argument.
                rank_accumulator.add(nFold, index, top_similarities_max)

                # DEBUG: Log after prediction (confusion matrix updated by C code)
                if DEBUG_MODE and i < DEBUG_CASE_LIMIT:
                    print(f"[DEBUG BASELINE] util_cy.predictS2V completed")
                    print(f"[DEBUG BASELINE] Confusion Matrix (after case {i}):")
                    print(f"[DEBUG BASELINE]   TP={confusion_matrix_max[0]}, FP={confusion_matrix_max[1]}, TN={confusion_matrix_max[2]}, FN={confusion_matrix_max[3]}")
            ##########################################################################################################
            # COMPUTE PERFORMANCE INDEX
            ##########################################################################################################
            performance_out_file.write("PERFORMANCE INDEX of MAX SIMILARITY by MAX" + "\n")
            util_cy.compute_aggregated_performance_index(confusion_matrix_max, performance_matrix_max,nrow,performance_out_file)
            for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
                confusion_matrix_Top_K_max = confusion_matrix_Top_K_max_dict.get(topk)
                performance_matrix_topK_max = performance_matrix_topK_max_dict.get(topk)
                performance_out_file.write("\n PERFORMANCE INDEX of TOP-" + str(topk) + " SIMILARITY by MAX" + "\n")
                util_cy.compute_aggregated_performance_index(confusion_matrix_Top_K_max, performance_matrix_topK_max,nrow,performance_out_file)

            # Record fold time
            fold_time = time.perf_counter() - fold_start
            fold_times.append(fold_time)
            print(f"[TIMING] Fold {nFold} completed: {format_time(fold_time)}")
        #END BLOCK FOLD

        folds_total_time = time.perf_counter() - folds_start
        timing_data['folds_total'] = folds_total_time
        timing_data['fold_times'] = fold_times

        ##########################################################################################################
        # COMPUTE MEAN PERFORMANCE INDEX FOR ALL FOLDS
        ##########################################################################################################
        performance_out_file.write("************************************************************************************************************" + "\n")
        performance_out_file.write("\n" + str(K_FOLD) + "-FOLD PERFORMANCE INDEX of MAX SIMILARITY by MAX" + "\n")
        util_cy.print_performance_index(performance_matrix_max, performance_out_file)
        performance_out_file.write("************************************************************************************************************" + "\n")
        for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
            performance_matrix_topK_max = performance_matrix_topK_max_dict.get(topk)
            performance_out_file.write("\n" + str(K_FOLD) + "-FOLD PERFORMANCE INDEX of TOP-" + str(topk) + " SIMILARITY by MAX" + "\n")
            util_cy.print_performance_index(performance_matrix_topK_max, performance_out_file)
        performance_out_file.write("************************************************************************************************************" + "\n")

        ##########################################################################################################
        # GENERATE TIMING REPORT
        ##########################################################################################################
        total_execution_time = time.perf_counter() - script_start_time
        timing_data['total_execution'] = total_execution_time

        # Create timing report content
        timing_report = []
        timing_report.append("=" * 80)
        timing_report.append("TIMING REPORT - Disease Diagnosis System")
        timing_report.append("=" * 80)
        timing_report.append(f"Dataset Loading:           {format_time(timing_data['dataset_loading'])}")
        timing_report.append(f"Model Loading:             {format_time(timing_data['model_loading'])}")
        timing_report.append(f"Symptom Embeddings:        {format_time(timing_data['symptom_embeddings'])}")
        timing_report.append(f"Diagnosis Embeddings:      {format_time(timing_data['diagnosis_embeddings'])}")
        timing_report.append(f"Embeddings Total:          {format_time(timing_data['embeddings_total'])}")
        timing_report.append("-" * 80)

        # Individual fold times
        for i, fold_time in enumerate(timing_data['fold_times']):
            timing_report.append(f"Fold {i}:                    {format_time(fold_time)}")

        timing_report.append("-" * 80)
        timing_report.append(f"Total Folds Processing:    {format_time(timing_data['folds_total'])}")
        timing_report.append("-" * 80)
        timing_report.append(f"TOTAL EXECUTION TIME:      {format_time(total_execution_time)}")
        timing_report.append("=" * 80)

        # Print to console
        print("\n")
        for line in timing_report:
            print(line)

        # Write to separate timing report file
        timing_file_path = directory_prediction_root + 'timing_report.txt'
        with open(timing_file_path, 'w') as timing_file:
            for line in timing_report:
                timing_file.write(line + "\n")

        # Generate PDF report with visualizations
        timing_pdf_path = directory_prediction_root + 'timing_report.pdf'
        try:
            generate_timing_pdf(timing_data, timing_pdf_path)
        except Exception as e:
            print(f"[WARNING] PDF generation failed: {e}")
            print("[INFO] Text report still available in timing_report.txt")

        # Append the timing summary to PerformanceIndex.txt -- the trailer, and
        # the last thing written through this handle.
        performance_out_file.write("\n\n")
        for line in timing_report:
            performance_out_file.write(line + "\n")

    ##########################################################################################################
    # P5: RANK METRICS -- a SIBLING file, never a column in PerformanceIndex.txt
    ##########################################################################################################
    # Deliberately OUTSIDE the with-block above. PerformanceIndex.txt must be
    # complete and closed before anything optional runs: an exception raised while
    # writing rank metrics would otherwise leave the file without its timing trailer,
    # and test_golden.strip_trailer raises "No timing trailer found" -- surfacing as a
    # GOLDEN FAILURE THAT LOOKS LIKE THE PIPELINE CHANGED rather than as the local
    # error it actually is.
    rank_metrics_path = rank_accumulator.write(directory_prediction_root)
    print("[SUCCESS] Rank metrics written to " + os.path.basename(rank_metrics_path))

    print(f"\n[SUCCESS] Timing report saved to: {timing_file_path}")
    print(f"[SUCCESS] Timing summary appended to: {directory_prediction_root}/PerformanceIndex.txt")

    return directory_prediction_root


if __name__ == "__main__":
    # ``config=from_env()`` and not the bare default: before the pipeline moved
    # into run_analysis() this module resolved the env var at import time, so
    # ``AICDS_PIPELINE=corrected python -m aicds.models.baseline_sent2vec``
    # honoured it. A bare ``run_analysis()`` defaults to LEGACY and would have
    # silently dropped the variable on the one entry point that has no
    # ``--pipeline`` flag to fall back to. from_env() returns LEGACY when the
    # variable is unset, so the default path stays bit-identical and the golden
    # is unaffected.
    run_analysis(config=from_env())

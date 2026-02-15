import os
import shutil
import sys
import time
from math import floor

# Bio_ClinicalBERT imports
from sentence_transformers import SentenceTransformer
import torch

import numpy
import numpy as np
import sklearn
from scipy import spatial
from sklearn.model_selection import train_test_split, KFold

from src.entity.SymptomsDiagnosis import SymptomsDiagnosis
from src.utils.Constants import *

from src.utils import cython_utils as util_cy
from src.utils.cython_utils import cosine_similarity  # Direct import to avoid module caching issues

# Debug mode - set to False to disable verbose logging
DEBUG_MODE = False
DEBUG_CASE_LIMIT = 1  # Only debug first N cases per fold

# Debug log file will be created in output directory
debug_log_file = None

# Model configuration dictionary
MODELS = {
    '1': {
        'name': 'Bio_ClinicalBERT',
        'path': 'emilyalsentzer/Bio_ClinicalBERT',
        'description': 'MIMIC-III clinical notes',
        'best_for': 'Clinical terminology, diagnosis codes'
    },
    '2': {
        'name': 'BiomedBERT',
        'path': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
        'description': 'PubMed abstracts (biomedical literature)',
        'best_for': 'Biomedical concepts, disease descriptions'
    },
    '3': {
        'name': 'BlueBERT',
        'path': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        'description': 'PubMed + MIMIC-III (hybrid corpus)',
        'best_for': 'Balanced clinical and biomedical knowledge'
    }
}

def print_model_menu():
    """Display interactive menu for model selection"""
    print("\n" + "=" * 80)
    print("BERT MODEL SELECTION FOR DISEASE DIAGNOSIS")
    print("=" * 80)
    print("\nAvailable Biomedical BERT Models:\n")
    
    for key, model in MODELS.items():
        print(f"[{key}] {model['name']}")
        print(f"    Model: {model['path']}")
        print(f"    Training: {model['description']}")
        print(f"    Best for: {model['best_for']}\n")
    
    print("[Q] Quit")
    print("=" * 80)

def select_model(model_id=None):
    """Model selection - programmatic if model_id provided, interactive otherwise"""
    if model_id and model_id in MODELS:
        selected = MODELS[model_id]
        print(f"\n[INFO] Selected: {selected['name']}")
        print(f"[INFO] Model path: {selected['path']}")
        return selected

    # Interactive mode
    print_model_menu()

    while True:
        choice = input("\nSelect model [1-3, Q]: ").strip()

        if choice.upper() == 'Q':
            print("[INFO] Exiting...")
            sys.exit(0)
        elif choice in MODELS:
            selected = MODELS[choice]
            print(f"\n[INFO] Selected: {selected['name']}")
            print(f"[INFO] Model path: {selected['path']}")
            return selected
        else:
            print(f"[ERROR] Invalid choice '{choice}'. Please enter 1, 2, 3, or Q")

# Pure Python prediction functions to replace util_cy.predictS2V (Hugging Face compatible)
def compute_patient_similarity_pairwise(test_symptoms, train_symptoms, embeddings_symptoms):
    """
    Pairwise symptom-level similarity matching baseline cython_utils.py lines 59-88.

    For each test symptom, find the max cosine similarity across all train symptoms.
    Average the sum of max similarities by max(len_test, len_train).

    Args:
        test_symptoms: list of preprocessed symptom strings for test patient
        train_symptoms: list of preprocessed symptom strings for train patient
        embeddings_symptoms: dict keyed by symptom text -> [embedding]

    Returns:
        float: Average pairwise max similarity
    """
    max_symptoms_similarity = {}

    for x in test_symptoms:
        test_emb = embeddings_symptoms.get(x)
        if test_emb is None:
            continue
        max_similarity = MIN_SIMILARITY
        max_symptom = None

        for y in train_symptoms:
            train_emb = embeddings_symptoms.get(y)
            if train_emb is None:
                continue
            similarity = cosine_similarity(test_emb[0], train_emb[0])
            if similarity > max_similarity:
                max_similarity = similarity
                max_symptom = y

        if max_symptom is not None:
            max_symptoms_similarity[max_symptom + " for " + x] = max_similarity
        else:
            max_symptoms_similarity["No Similar symptom for " + x] = max_similarity

    # Denominator = max(len_test, len_train) — matching baseline lines 78-82
    max_den = max(len(test_symptoms), len(train_symptoms))
    if max_den == 0:
        return 0.0

    mean = sum(max_symptoms_similarity.values()) / max_den
    return mean

def predict_topk_diagnoses_pure(test_admission, test_symptoms, x_train,
                               embeddings_symptoms, embeddings_diagnosis,
                               admissions, k=None):
    """
    Pure Python TOP-K prediction replacing util_cy.predictS2V.

    Computes pairwise symptom-level similarity for each training case,
    applies PRUNING_SIMILARITY >= 0.5 threshold, then returns:
    - k=None (MAX): single best match above threshold
    - k=N (TOP-K): top N matches above threshold, sorted descending
    """
    # Compute similarities with all training patients using pairwise method
    similarities = []
    for train_dict in x_train:
        train_id = list(train_dict.keys())[0]
        train_symptoms = list(train_dict.values())[0]

        similarity = compute_patient_similarity_pairwise(
            test_symptoms, train_symptoms, embeddings_symptoms
        )

        train_admission = admissions.get(train_id)
        if train_admission:
            similarities.append({
                'patient_id': train_id,
                'similarity': similarity,
                'diagnosis': train_admission.diagnosis
            })

    # Sort by similarity descending
    similarities.sort(key=lambda x: x['similarity'], reverse=True)

    # Apply PRUNING_SIMILARITY threshold (matching baseline line 97, 128)
    filtered = [s for s in similarities if s['similarity'] >= PRUNING_SIMILARITY]

    if k is None:
        # MAX: single best match above threshold
        top_matches = filtered[:1] if filtered else []
    else:
        # TOP-K: top k matches above threshold
        top_matches = filtered[:k]

    predicted_diagnoses = [item['diagnosis'] for item in top_matches]
    similarity_scores = [item['similarity'] for item in top_matches]
    patient_ids = [item['patient_id'] for item in top_matches]

    return predicted_diagnoses, similarity_scores, patient_ids

# REMOVED: Now using util_cy.get_diagnosis_similarity_by_description_max() directly

def containGreaterOrEqualsValue(topk, similarity_list, threshold):
    """
    Check if ANY similarity in top-k list is >= threshold
    This matches util_cy.containGreaterOrEqualsValue()
    """
    for sim in similarity_list[:topk]:
        if sim >= threshold:
            return True
    return False

# Ensure NLTK data is available before first use
import nltk

def ensure_nltk_data():
    """
    Automatically download required NLTK data packages if they don't exist.
    This eliminates the need for manual downloads before running the script.
    """
    required_data = {
        'stopwords': 'corpora/stopwords',
        'punkt_tab': 'tokenizers/punkt_tab',
        'punkt': 'tokenizers/punkt'  # Fallback for older NLTK versions
    }

    missing = []
    for name, path in required_data.items():
        try:
            nltk.data.find(path)
        except LookupError:
            missing.append(name)

    if missing:
        print("[INFO] First-time setup: Downloading required NLTK data packages...")
        print(f"[INFO] Missing packages: {', '.join(missing)}")
        for name in missing:
            print(f"[INFO] Downloading '{name}'... ", end='', flush=True)
            try:
                nltk.download(name, quiet=True)
                print("OK")
            except Exception as e:
                print(f"ERROR (Error: {e})")
        print("[SUCCESS] NLTK data download complete!")
        print("")

# Call before any NLTK usage
ensure_nltk_data()

# FIX: Initialize missing stop_words global variable in util_cy module
# The compiled C code is missing this initialization
from nltk.corpus import stopwords
util_cy.stop_words = set(stopwords.words('english'))

from pathlib import Path

# Import matplotlib for PDF generation
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Timing utilities
def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{seconds:.2f} seconds ({minutes:.2f} minutes)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{seconds:.2f} seconds ({hours:.2f} hours, {minutes:.2f} minutes)"

def compute_bert_symptom_embeddings(model, admissions):
    """
    Compute BERT embeddings for individual symptoms, keyed by preprocessed symptom text.

    Matches baseline util_cy.embending_symptoms() which:
    1. Splits admission.symptoms by comma
    2. Preprocesses each symptom individually
    3. Keys the dict by preprocessed symptom text (not HADM_ID)
    """
    embeddings = {}

    print("[INFO] Preparing individual symptom texts for BERT encoding...")

    # Collect all unique preprocessed symptoms across all admissions
    unique_symptoms = set()
    for admission_key, admission in admissions.items():
        symptoms_list = admission.symptoms.split(',')
        for s in symptoms_list:
            preprocessed = util_cy.preprocess_sentence(s)
            unique_symptoms.add(preprocessed)

    unique_symptoms_list = list(unique_symptoms)
    print(f"[INFO] Found {len(unique_symptoms_list)} unique symptoms across {len(admissions)} admissions")
    print(f"[INFO] Batch size: 32, Model device: {model.device}")

    # Batch encode all unique symptoms
    bert_embeddings = model.encode(
        unique_symptoms_list,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    # Key by preprocessed symptom text, wrap in list for cosine_similarity(emb[0], ...) usage
    for symptom_text, embedding in zip(unique_symptoms_list, bert_embeddings):
        embeddings[symptom_text] = [embedding]

    print(f"[SUCCESS] BERT symptom embeddings computed: {len(embeddings)} unique symptoms")
    print(f"[INFO] Embedding dimension: {bert_embeddings.shape[1]}")
    print(f"[INFO] Format: Keyed by symptom text, wrapped in arrays for util_cy compatibility")

    return embeddings

def compute_bert_diagnosis_embeddings(model, admissions):
    """
    Compute BERT embeddings KEYED BY DIAGNOSIS TEXT (not HADM_ID)
    This matches util_cy.embending_diagnosis() which keys by diagnosis description
    """
    embeddings = {}
    
    print("[INFO] Preparing diagnosis texts for BERT encoding...")
    print("[INFO] CRITICAL FIX: Storing embeddings by DIAGNOSIS TEXT (like baseline), not patient ID")
    
    # Collect all unique diagnosis descriptions (like baseline util_cy.c line 251-257)
    for admission_key, admission in admissions.items():
        # admission.diagnosis is preprocessed list like ["apr,hcfa:liver disease", "ms:diabetes"]
        for diagnosis_with_drg in admission.diagnosis:
            # Extract description after ':' (baseline line 255)
            if ':' in diagnosis_with_drg:
                diagnosis_description = diagnosis_with_drg[diagnosis_with_drg.index(':')+1:]
            else:
                diagnosis_description = diagnosis_with_drg
            
            # Only compute if not already stored (avoid duplicates)
            if diagnosis_description not in embeddings:
                embeddings[diagnosis_description] = None

    # Batch encode all unique diagnosis descriptions
    unique_diagnoses = list(embeddings.keys())
    
    print(f"[INFO] Computing BERT embeddings for {len(unique_diagnoses)} unique diagnosis texts...")
    print(f"[INFO] Batch size: 32")

    # TESTING: normalize_embeddings=False to see if scores spread out more naturally
    bert_embeddings = model.encode(
        unique_diagnoses,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False  # Testing without L2 norm to spread similarity scores
    )

    # CRITICAL: Wrap in array format to match util_cy expectations (embedding[0] indexing)
    for diagnosis_text, embedding in zip(unique_diagnoses, bert_embeddings):
        embeddings[diagnosis_text] = [embedding]  # Wrap in list for util_cy compatibility

    print(f"[SUCCESS] BERT diagnosis embeddings: {len(embeddings)} unique texts")
    print(f"[INFO] Format: Wrapped in arrays for util_cy compatibility")
    return embeddings

def run_analysis(model_id=None):
    """
    Run the full BERT disease diagnosis analysis pipeline.

    Args:
        model_id: '1' (Bio_ClinicalBERT), '2' (BiomedBERT), '3' (BlueBERT),
                  or None for interactive selection.

    Returns:
        str: Path to the output directory containing results.
    """
    global debug_log_file

    timing_data = {}
    script_start_time = time.perf_counter()

    ################################################################################################################
    #READ DATASET
    ################################################################################################################
    # Use proper path joining instead of changing directory
    dataset_start = time.perf_counter()
    file_name = os.path.join(CH_DIR, "data", "raw", "Symptoms-Diagnosis.txt")
    f = open(file_name, "r").readlines()
    orig_stdout = sys.stdout

    admissions = dict()
    for line in f:
        line.replace("\n", "")
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
    #SELECT AND LOAD BERT MODEL
    ################################################################################################################
    model_config = select_model(model_id)
    model_name = model_config['name']
    model_path = model_config['path']

    model_start = time.perf_counter()
    print(f"\n[INFO] Loading {model_name} model...")
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Training: {model_config['description']}")
    print("[INFO] This replaces the 21GB BioSentVec model with contextualized BERT embeddings")
    print("[INFO] USING PURE PYTHON PREDICTION - No util_cy.predictS2V dependencies")
    try:
        model = SentenceTransformer(model_path)
        print(f"[INFO] Model device: {model.device}")
        print(f"[INFO] Model max sequence length: {model.max_seq_length}")
        model_time = time.perf_counter() - model_start
        timing_data['model_loading'] = model_time
        print(f"[SUCCESS] {model_name} loaded successfully!")
        print(f"[TIMING] Model loading: {format_time(model_time)}")
    except Exception as e:
        print(f"[ERROR] Loading {model_name} failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    ################################################################################################################
    #COMPUTE BERT EMBEDDINGS
    ################################################################################################################
    embeddings_start = time.perf_counter()
    print("[INFO] Computing BERT symptom embeddings...")
    symptom_emb_start = time.perf_counter()
    embendings_symptoms = compute_bert_symptom_embeddings(model, admissions)
    symptom_emb_time = time.perf_counter() - symptom_emb_start
    timing_data['symptom_embeddings'] = symptom_emb_time
    print(f"[TIMING] Symptom embeddings: {format_time(symptom_emb_time)}")

    print("[INFO] Computing BERT diagnosis embeddings...")
    diagnosis_emb_start = time.perf_counter()
    embendings_diagnosis = compute_bert_diagnosis_embeddings(model, admissions)
    diagnosis_emb_time = time.perf_counter() - diagnosis_emb_start
    timing_data['diagnosis_embeddings'] = diagnosis_emb_time
    print(f"[TIMING] Diagnosis embeddings: {format_time(diagnosis_emb_time)}")

    embeddings_total_time = time.perf_counter() - embeddings_start
    timing_data['embeddings_total'] = embeddings_total_time

    ################################################################################################################
    #OUTPUT DIRECTORIES
    ################################################################################################################
    # Create timestamped output directories with model name
    timestamp = time.strftime("%d%m%Y_%H-%M-%S")
    directory_prediction_root = os.getcwd() + f'/Prediction_Output_{model_name}_' + timestamp + '/'
    directory_prediction_details_root = os.getcwd() + f'/Prediction_Symptom_Details_{model_name}_' + timestamp + '/'
    shutil.rmtree(directory_prediction_root, ignore_errors=True)
    shutil.rmtree(directory_prediction_details_root, ignore_errors=True)
    Path(directory_prediction_root).mkdir(parents=True, exist_ok=True)
    Path(directory_prediction_details_root).mkdir(parents=True, exist_ok=True)
    performance_out_file = open(directory_prediction_root + '/PerformanceIndex.txt', 'w')

    # Open debug log file if DEBUG_MODE is enabled
    if DEBUG_MODE:
        debug_log_file = open(directory_prediction_root + '/debug_log.txt', 'w')
        debug_log_file.write("=" * 80 + "\n")
        debug_log_file.write("DEBUG LOG - Bio_ClinicalBERT Disease Diagnosis\n")
        debug_log_file.write("=" * 80 + "\n\n")

    ################################################################################################################
    # Initialize performance matrices - matching baseline structure
    ################################################################################################################
    # Thresholds to evaluate (same as util_cy)
    THRESHOLDS = [0.9, 1.0, 0.6, 0.8, 0.7]  # Same order as baseline

    # Performance matrix structure: {threshold: [TP, FP, P, R, FS, PR]}
    performance_matrix_max = util_cy.init_performance_matrix()
    performance_matrix_topK_max_dict = dict()
    for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
        performance_matrix_topK_max_dict.update({topk: util_cy.init_performance_matrix()})

    ################################################################################################################
    # WORK ON SINGLE FOLD - MATCHING BASELINE EVALUATION PROTOCOL
    ################################################################################################################
    folds_start = time.perf_counter()
    fold_times = []

    for nFold in range(0, K_FOLD):
        fold_start = time.perf_counter()
        print(f'\n[INFO] === FOLD {nFold} ===')

        directory_prediction = directory_prediction_root + 'Fold' + str(nFold) + "/"
        directory_prediction_details = directory_prediction_details_root + 'Fold' + str(nFold) + "/"
        shutil.rmtree(directory_prediction, ignore_errors=True)
        shutil.rmtree(directory_prediction_details, ignore_errors=True)
        Path(directory_prediction).mkdir(parents=True, exist_ok=True)
        Path(directory_prediction_details).mkdir(parents=True, exist_ok=True)

        ######################################################################################################
        # LOAD TRAIN AND TEST SET
        ######################################################################################################
        x_test = util_cy.load_dataset(nFold, TEST)
        x_train = util_cy.load_dataset(nFold, TRAIN)

        performance_out_file.write(f'\n FOLD {nFold}: LEN train: {len(x_train)}, LEN test: {len(x_test)} \n')
        print(f'FOLD {nFold}: LEN train: {len(x_train)}, LEN test: {len(x_test)}')

        ######################################################################################################
        # Initialize confusion matrices for this fold (one per threshold per strategy)
        ######################################################################################################
        nrow = len(x_test)

        # MAX similarity confusion matrix: {threshold: [TP, FP, TN, FN]}
        confusion_matrix_max = util_cy.init_confusion_matrix()

        # TOP-K confusion matrices: {topk: {threshold: [TP, FP, TN, FN]}}
        confusion_matrix_Top_K_max_dict = dict()
        for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
            confusion_matrix_Top_K_max_dict.update({topk: util_cy.init_confusion_matrix()})

        ######################################################################################################
        # PROCESS EACH TEST CASE - MATCHING BASELINE PREDICTSZV LOGIC
        ######################################################################################################
        for i in range(len(x_test)):
            # Get test case info
            index = list(x_test[i].keys())[0]
            test_symptoms = list(x_test[i].values())[0]
            test_admission = admissions.get(index)

            if not test_admission:
                print(f"[WARNING] Skipping test case {index} - missing data")
                continue

            gt_diagnosis = test_admission.diagnosis

            ##################################################################################################
            # COMPUTE FULL SIMILARITY ROW ONCE (optimization: reuse for MAX and all TOP-K)
            ##################################################################################################
            # Get the largest TOP-K worth of results in one call
            max_k = TOP_K_UPPER_BOUND - TOP_K_INCR  # 50
            all_diags, all_sims, all_pids = predict_topk_diagnoses_pure(
                test_admission, test_symptoms, x_train,
                embendings_symptoms, embendings_diagnosis,
                admissions, k=max_k
            )

            # Pre-compute diagnosis similarities for all returned predictions
            all_diag_sims = []
            for pred_diagnosis in all_diags:
                diag_sim = util_cy.get_diagnosis_similarity_by_description_max(
                    embendings_diagnosis, gt_diagnosis, pred_diagnosis, 'cosine'
                )
                all_diag_sims.append(diag_sim)

            ##################################################################################################
            # STRATEGY 1: MAX SIMILARITY (single most similar patient)
            ##################################################################################################
            performance_out_file.write(f"{i} - HADM_ID={index}: PERFORMANCE INDEX of MAX SIMILARITY by MAX\n")
            performance_out_file.write("         TP      FP       P      R       FS      PR\n")

            if len(all_diags) > 0:
                diagnosis_similarity_max = all_diag_sims[0]

                # DEBUG logging
                if DEBUG_MODE and i < DEBUG_CASE_LIMIT:
                    debug_msg = f"\n{'='*80}\n"
                    debug_msg += f"[DEBUG BERT] === Test Case {i} (Patient {index}) - FOLD {nFold} - MAX Strategy ===\n"
                    debug_msg += f"{'='*80}\n"
                    debug_msg += f"[DEBUG BERT] GT HADM_ID: {index}\n"
                    debug_msg += f"[DEBUG BERT] Predicted HADM_ID: {all_pids[0]}\n"
                    debug_msg += f"[DEBUG BERT] GT Diagnosis: {gt_diagnosis}\n"
                    debug_msg += f"[DEBUG BERT] Predicted Diagnosis: {all_diags[0]}\n"
                    debug_msg += f"[DEBUG BERT] Symptom Similarity: {all_sims[0]:.4f}\n"
                    debug_msg += f"[DEBUG BERT] **DIAGNOSIS BERT Similarity: {diagnosis_similarity_max:.4f}**\n"
                    debug_msg += f"[DEBUG BERT] Threshold pass status:\n"
                    for thresh in [0.6, 0.7, 0.8, 0.9, 1.0]:
                        status = "PASS (TP)" if diagnosis_similarity_max >= thresh else "FAIL (FP)"
                        debug_msg += f"[DEBUG BERT]   {thresh:.1f}: {status}\n"
                    debug_msg += f"{'='*80}\n"
                    print(debug_msg)
                    if debug_log_file:
                        debug_log_file.write(debug_msg + "\n")
                        debug_log_file.flush()

                for b in THRESHOLDS:
                    values = confusion_matrix_max.get(b)
                    if diagnosis_similarity_max >= b:
                        values[TP] += 1
                        tp, fp = 1, 0
                    else:
                        values[FP] += 1
                        tp, fp = 0, 1

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fp) if (tp + fp) > 0 else 0
                    fs = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    pr = 1.0 / nrow
                    performance_out_file.write(f"{b}     {tp}       {fp}       {precision}     {recall}     {fs}      {pr}\n")
            else:
                for b in THRESHOLDS:
                    performance_out_file.write(f"{b}     0       0       0.0     0.0     0       {1.0/nrow}\n")

            ##################################################################################################
            # STRATEGY 2-6: TOP-K SIMILARITY (10, 20, 30, 40, 50) — reuse precomputed data
            ##################################################################################################
            for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
                performance_out_file.write(f"{i} - HADM_ID={index}: PERFORMANCE INDEX of TOP-{topk} SIMILARITY by MAX\n")
                performance_out_file.write("         TP      FP       P      R       FS      PR\n")

                # Slice precomputed diagnosis similarities for this TOP-K
                top_similarities_max = all_diag_sims[:topk]

                confusion_matrix_Top_K_max = confusion_matrix_Top_K_max_dict.get(topk)

                for b in THRESHOLDS:
                    values = confusion_matrix_Top_K_max.get(b)

                    if len(top_similarities_max) > 0:
                        if containGreaterOrEqualsValue(topk, top_similarities_max, b):
                            values[TP] += 1
                            tp, fp = 1, 0
                        else:
                            values[FP] += 1
                            tp, fp = 0, 1
                    else:
                        tp, fp = 0, 0

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fp) if (tp + fp) > 0 else 0
                    fs = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    pr = 1.0 / nrow
                    performance_out_file.write(f"{b}     {tp}       {fp}       {precision}     {recall}     {fs}      {pr}\n")

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(x_test)} test cases...")

        ######################################################################################################
        # COMPUTE AGGREGATED PERFORMANCE FOR THIS FOLD
        ######################################################################################################
        performance_out_file.write("\nPERFORMANCE INDEX of MAX SIMILARITY by MAX" + "\n")
        util_cy.compute_aggregated_performance_index(confusion_matrix_max, performance_matrix_max, nrow, performance_out_file)

        for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
            confusion_matrix_Top_K_max = confusion_matrix_Top_K_max_dict.get(topk)
            performance_matrix_topK_max = performance_matrix_topK_max_dict.get(topk)
            performance_out_file.write("\n PERFORMANCE INDEX of TOP-" + str(topk) + " SIMILARITY by MAX" + "\n")
            util_cy.compute_aggregated_performance_index(confusion_matrix_Top_K_max, performance_matrix_topK_max, nrow, performance_out_file)

        # Record fold time
        fold_time = time.perf_counter() - fold_start
        fold_times.append(fold_time)
        print(f"[TIMING] Fold {nFold} completed: {format_time(fold_time)}")

    #END FOLD LOOP

    folds_total_time = time.perf_counter() - folds_start
    timing_data['folds_total'] = folds_total_time
    timing_data['fold_times'] = fold_times

    ################################################################################################################
    # COMPUTE MEAN PERFORMANCE INDEX ACROSS ALL FOLDS (matching baseline output)
    ################################################################################################################
    performance_out_file.write("************************************************************************************************************" + "\n")
    performance_out_file.write("\n" + str(K_FOLD) + "-FOLD PERFORMANCE INDEX of MAX SIMILARITY by MAX" + "\n")
    util_cy.print_performance_index(performance_matrix_max, performance_out_file)

    for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
        performance_matrix_topK_max = performance_matrix_topK_max_dict.get(topk)
        performance_out_file.write("\n" + str(K_FOLD) + "-FOLD PERFORMANCE INDEX of TOP-" + str(topk) + " SIMILARITY by MAX" + "\n")
        util_cy.print_performance_index(performance_matrix_topK_max, performance_out_file)

    performance_out_file.write("************************************************************************************************************" + "\n")

    ################################################################################################################
    # GENERATE TIMING REPORT
    ################################################################################################################
    total_execution_time = time.perf_counter() - script_start_time
    timing_data['total_execution'] = total_execution_time

    # Create timing report content
    timing_report = []
    timing_report.append("=" * 80)
    timing_report.append(f"TIMING REPORT - Disease Diagnosis System ({model_name})")
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
    timing_report.append(f"MODEL: {model_name} (BERT-768D)")
    timing_report.append(f"MODEL PATH: {model_path}")
    timing_report.append(f"TRAINING DATA: {model_config['description']}")
    timing_report.append("IMPLEMENTATION: Pure Python with continuous similarity scoring")
    timing_report.append("BASELINE COMPARISON: vs BioSentVec (700D)")
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

    # Append timing summary to PerformanceIndex.txt
    performance_out_file.write("\n\n")
    for line in timing_report:
        performance_out_file.write(line + "\n")
    performance_out_file.close()

    # Close debug log file if it was opened
    if debug_log_file:
        debug_log_file.write("\n" + "=" * 80 + "\n")
        debug_log_file.write("END DEBUG LOG\n")
        debug_log_file.write("=" * 80 + "\n")
        debug_log_file.close()
        debug_log_file = None
        print(f"[SUCCESS] Debug log saved to: {directory_prediction_root}debug_log.txt")

    print(f"\n[SUCCESS] {model_name} analysis complete!")
    print(f"[SUCCESS] Results directory: {directory_prediction_root}")
    print(f"[SUCCESS] Performance metrics written to PerformanceIndex.txt")
    print(f"[SUCCESS] Timing report saved to: {timing_file_path}")

    # Final comparison message
    print("\n" + "=" * 80)
    print(f"{model_name} INTEGRATION COMPLETE")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Training: {model_config['description']}")
    print("=" * 80)
    print("-> Multi-threshold evaluation (0.6, 0.7, 0.8, 0.9, 1.0)")
    print("-> Multi-strategy evaluation (MAX, TOP-10, TOP-20, TOP-30, TOP-40, TOP-50)")
    print("-> Per-patient performance metrics")
    print("-> Aggregated fold-level performance")
    print("-> Mean performance across all folds")
    print("-> Same output format as baseline CS2V.py")
    print("=" * 80)
    print("Results are now directly comparable with baseline!")
    print("=" * 80)

    return directory_prediction_root


# Run interactively when executed as a module directly (backwards compatible)
if __name__ == "__main__":
    run_analysis()

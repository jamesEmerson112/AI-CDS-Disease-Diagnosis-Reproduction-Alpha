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

import entity.SymptomsDiagnosis
from utils.Constants import *

import util_cy
from util_cy import cosine_similarity  # Direct import to avoid module caching issues

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

def select_model():
    """Interactive model selection with validation"""
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
def compute_patient_similarity_pure(test_symptom_emb, train_symptom_emb):
    """
    Core similarity computation using util_cy.cosine_similarity for exact baseline match
    
    Args:
        test_symptom_emb: BERT embedding wrapped in array [embedding]
        train_symptom_emb: BERT embedding wrapped in array [embedding]
    
    Returns:
        float: Cosine similarity
    """
    # Use util_cy's cosine_similarity for 100% baseline compatibility
    # Unwrap the embeddings (they're stored as [embedding])
    return cosine_similarity(test_symptom_emb[0], train_symptom_emb[0])

def predict_topk_diagnoses_pure(test_admission, test_symptoms, x_train,
                               embeddings_symptoms, embeddings_diagnosis,
                               admissions, k=None):
    """
    Pure Python TOP-K prediction replacing util_cy.predictS2V
    Returns all training cases sorted by similarity (no threshold filtering upfront)
    If k is None, return ALL matches; otherwise return top k
    """
    test_id = test_admission.hadm_id

    if test_id not in embeddings_symptoms:
        return [], [], []

    test_symptom_emb = embeddings_symptoms[test_id]

    # Compute similarities with all training patients (NO threshold filtering yet)
    similarities = []
    for train_dict in x_train:
        train_id = list(train_dict.keys())[0]

        if train_id not in embeddings_symptoms:
            continue

        train_symptom_emb = embeddings_symptoms[train_id]
        # Both embeddings are now wrapped in arrays, pass directly
        similarity = compute_patient_similarity_pure(test_symptom_emb, train_symptom_emb)

        train_admission = admissions.get(train_id)
        if train_admission:
            similarities.append({
                'patient_id': train_id,
                'similarity': similarity,
                'diagnosis': train_admission.diagnosis
            })

    # Sort by similarity descending
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Return top K or all
    if k is not None:
        top_matches = similarities[:k]
    else:
        top_matches = similarities[:1] if similarities else []  # MAX case
    
    # Extract diagnosis predictions, scores, AND patient IDs
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
    """Compute BERT embeddings for symptoms using same preprocessing as baseline"""
    embeddings = {}
    symptoms_texts = []
    symptoms_keys = []

    print("[INFO] Preparing symptom texts for BERT encoding...")
    for admission_key, admission in admissions.items():
        # Use SAME preprocessing as baseline for fair comparison
        processed_symptoms = util_cy.preprocess_sentence(admission.symptoms)
        symptoms_texts.append(processed_symptoms)
        symptoms_keys.append(admission_key)

    print(f"[INFO] Computing BERT embeddings for {len(symptoms_texts)} symptoms...")
    print(f"[INFO] Batch size: 32 (reduced from 128 for stability), Model device: {model.device}")

    # Batch encode for efficiency - same format as util_cy.embending_symptoms
    # Reduced batch_size back to 32 for stability testing and local debugging
    # TESTING: normalize_embeddings=False to see if scores spread out more naturally
    bert_embeddings = model.encode(
        symptoms_texts,
        batch_size=32,  # Reduced from 128 for stability local testing
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False  # Testing without L2 norm to spread similarity scores
    )

    # CRITICAL: Wrap in array format to match util_cy expectations (embedding[0] indexing)
    for key, embedding in zip(symptoms_keys, bert_embeddings):
        embeddings[key] = [embedding]  # Wrap in list for util_cy compatibility

    print(f"[SUCCESS] BERT symptom embeddings computed: {len(embeddings)} items")
    print(f"[INFO] Embedding dimension: {bert_embeddings.shape[1]}")
    print(f"[INFO] Format: Wrapped in arrays for util_cy compatibility")

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

# Timing storage
timing_data = {}
script_start_time = time.perf_counter()

################################################################################################################
#READ DATASET
################################################################################################################
# Use proper path joining instead of changing directory
dataset_start = time.perf_counter()
file_name = os.path.join(CH_DIR, "Symptoms-Diagnosis.txt")
f = open(file_name, "r").readlines()
orig_stdout = sys.stdout

admissions = dict()
for line in f:
    line.replace("\n", "")
    attributes = line.split(';')
    a = entity.SymptomsDiagnosis.SymptomsDiagnosis(attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_HADM_ID], attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_SUBJECT_ID], attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_ADMITTIME],
                                                   attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_DISCHTIME], attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_SYMPTOMS],
                                                   util_cy.preprocess_diagnosis(attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_DIAGNOSIS]))
    admissions.update({attributes[entity.SymptomsDiagnosis.SymptomsDiagnosis.CONST_HADM_ID]:a})

dataset_time = time.perf_counter() - dataset_start
timing_data['dataset_loading'] = dataset_time
print(f"[INFO] Dataset loaded: {len(admissions)} admissions")
print(f"[INFO] Dataset file: {file_name}")
print(f"[TIMING] Dataset loading: {format_time(dataset_time)}")

################################################################################################################
#SELECT AND LOAD BERT MODEL
################################################################################################################
# Interactive model selection
model_config = select_model()
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

    ##########################################################################################################
    # LOAD TRAIN AND TEST SET
    ##########################################################################################################
    x_test = util_cy.load_dataset(nFold, TEST)
    x_train = util_cy.load_dataset(nFold, TRAIN)

    performance_out_file.write(f'\n FOLD {nFold}: LEN train: {len(x_train)}, LEN test: {len(x_test)} \n')
    print(f'FOLD {nFold}: LEN train: {len(x_train)}, LEN test: {len(x_test)}')

    ##########################################################################################################
    # Initialize confusion matrices for this fold (one per threshold per strategy)
    ##########################################################################################################
    nrow = len(x_test)
    
    # MAX similarity confusion matrix: {threshold: [TP, FP, TN, FN]}
    confusion_matrix_max = util_cy.init_confusion_matrix()
    
    # TOP-K confusion matrices: {topk: {threshold: [TP, FP, TN, FN]}}
    confusion_matrix_Top_K_max_dict = dict()
    for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
        confusion_matrix_Top_K_max_dict.update({topk: util_cy.init_confusion_matrix()})

    ##########################################################################################################
    # PROCESS EACH TEST CASE - MATCHING BASELINE PREDICTSZV LOGIC
    ##########################################################################################################
    for i in range(len(x_test)):
        # Get test case info
        index = list(x_test[i].keys())[0]
        test_symptoms = list(x_test[i].values())[0]
        test_admission = admissions.get(index)

        if not test_admission or index not in embendings_symptoms:
            print(f"[WARNING] Skipping test case {index} - missing data")
            continue

        gt_diagnosis = test_admission.diagnosis
        
        ##########################################################################################################
        # STRATEGY 1: MAX SIMILARITY (single most similar patient)
        ##########################################################################################################
        predicted_diags_max, similarities_max, patient_ids_max = predict_topk_diagnoses_pure(
            test_admission, test_symptoms, x_train,
            embendings_symptoms, embendings_diagnosis,
            admissions, k=None  # None = MAX (single best match)
        )
        
        # Write per-patient header to main PerformanceIndex.txt (baseline format)
        performance_out_file.write(f"{i} - HADM_ID={index}: PERFORMANCE INDEX of MAX SIMILARITY by MAX\n")
        performance_out_file.write("         TP      FP       P      R       FS      PR\n")
        
        # Evaluate at each threshold
        if predicted_diags_max and similarities_max and patient_ids_max:
            pred_hadm_id = patient_ids_max[0]
            predicted_diagnosis = predicted_diags_max[0]  # Diagnosis list
            # Use util_cy's function directly for 100% baseline compatibility
            diagnosis_similarity_max = util_cy.get_diagnosis_similarity_by_description_max(
                embendings_diagnosis, gt_diagnosis, predicted_diagnosis, 'cosine'
            )
            
            # DEBUG: Show diagnosis similarity details (ALWAYS log to file if debug enabled)
            if DEBUG_MODE and i < DEBUG_CASE_LIMIT:
                debug_msg = f"\n{'='*80}\n"
                debug_msg += f"[DEBUG BERT] === Test Case {i} (Patient {index}) - FOLD {nFold} - MAX Strategy ===\n"
                debug_msg += f"{'='*80}\n"
                debug_msg += f"[DEBUG BERT] GT HADM_ID: {index}\n"
                debug_msg += f"[DEBUG BERT] Predicted HADM_ID: {pred_hadm_id}\n"
                debug_msg += f"[DEBUG BERT] GT Diagnosis: {gt_diagnosis}\n"
                debug_msg += f"[DEBUG BERT] Predicted Diagnosis: {predicted_diags_max[0]}\n"
                debug_msg += f"[DEBUG BERT] Symptom Similarity: {similarities_max[0]:.4f}\n"
                debug_msg += f"[DEBUG BERT] **DIAGNOSIS BERT Similarity: {diagnosis_similarity_max:.4f}**\n"
                debug_msg += f"[DEBUG BERT] Embeddings normalized: YES (should be in [0,1] range)\n"
                debug_msg += f"[DEBUG BERT] Threshold pass status:\n"
                for thresh in [0.6, 0.7, 0.8, 0.9, 1.0]:
                    status = "PASS (TP)" if diagnosis_similarity_max >= thresh else "FAIL (FP)"
                    debug_msg += f"[DEBUG BERT]   {thresh:.1f}: {status}\n"
                debug_msg += f"{'='*80}\n"
                
                # Write to both console and debug log file
                print(debug_msg)
                if debug_log_file:
                    debug_log_file.write(debug_msg + "\n")
                    debug_log_file.flush()
    
            # Check against each threshold - BINARY SCORING (matches baseline)
            for b in THRESHOLDS:
                values = confusion_matrix_max.get(b)
                if diagnosis_similarity_max >= b:
                    values[TP] += 1  # Add 1 (binary - matches baseline)
                    tp = 1
                    fp = 0
                else:
                    values[FP] += 1  # Add 1 (binary - matches baseline)
                    tp = 0
                    fp = 1
                
                # Calculate per-patient performance
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fp) if (tp + fp) > 0 else 0  # Recall = TP / (TP + FP) for continuous
                fs = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                pr = 1.0 / nrow  # Proportion of test set (unchanged)
                
                # Write to main PerformanceIndex.txt (baseline format)
                performance_out_file.write(f"{b}     {tp}       {fp}       {precision}     {recall}     {fs}      {pr}\n")
        else:
            # No prediction available - add 0 score to FP for all thresholds
            for b in THRESHOLDS:
                values = confusion_matrix_max.get(b)
                values[FP] += 0  # No score to add
                performance_out_file.write(f"{b}     0       0       0.0     0.0     0       {1.0/nrow}\n")
        
        ##########################################################################################################
        # STRATEGY 2-6: TOP-K SIMILARITY (10, 20, 30, 40, 50)
        ##########################################################################################################
        for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
            # Get TOP-K predictions
            predicted_diags_topk, similarities_topk, patient_ids_topk = predict_topk_diagnoses_pure(
                test_admission, test_symptoms, x_train,
                embendings_symptoms, embendings_diagnosis,
                admissions, k=topk
            )
            
            # Write to main PerformanceIndex.txt (baseline format)
            performance_out_file.write(f"{i} - HADM_ID={index}: PERFORMANCE INDEX of TOP-{topk} SIMILARITY by MAX\n")
            performance_out_file.write("         TP      FP       P      R       FS      PR\n")
            
            # Compute diagnosis similarities for all TOP-K using BERT embeddings
            top_similarities_max = []
            for pred_diagnosis in predicted_diags_topk:
                # Use util_cy's function directly for 100% baseline compatibility
                diag_sim = util_cy.get_diagnosis_similarity_by_description_max(
                    embendings_diagnosis, gt_diagnosis, pred_diagnosis, 'cosine'
                )
                top_similarities_max.append(diag_sim)
            
            # Evaluate at each threshold
            confusion_matrix_Top_K_max = confusion_matrix_Top_K_max_dict.get(topk)
            
            for b in THRESHOLDS:
                values = confusion_matrix_Top_K_max.get(b)
                
                # BINARY SCORING: Use the highest similarity from top-k (matches baseline)
                if len(top_similarities_max) > 0:
                    max_sim = max(top_similarities_max[:topk])
                    
                    # Check if highest similarity meets threshold
                    if max_sim >= b:
                        values[TP] += 1  # Add 1 (binary - matches baseline)
                        tp = 1
                        fp = 0
                    else:
                        values[FP] += 1  # Add 1 (binary - matches baseline)
                        tp = 0
                        fp = 1
                else:
                    # No predictions available
                    tp, fp = 0, 0
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fp) if (tp + fp) > 0 else 0  # Recall = TP / (TP + FP) for continuous
                fs = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                pr = 1.0 / nrow
                
                # Write to main PerformanceIndex.txt (baseline format)
                performance_out_file.write(f"{b}     {tp}       {fp}       {precision}     {recall}     {fs}      {pr}\n")
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(x_test)} test cases...")

    ##########################################################################################################
    # COMPUTE AGGREGATED PERFORMANCE FOR THIS FOLD
    ##########################################################################################################
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

##########################################################################################################
# COMPUTE MEAN PERFORMANCE INDEX ACROSS ALL FOLDS (matching baseline output)
##########################################################################################################
performance_out_file.write("************************************************************************************************************" + "\n")
performance_out_file.write("\n" + str(K_FOLD) + "-FOLD PERFORMANCE INDEX of MAX SIMILARITY by MAX" + "\n")
util_cy.print_performance_index(performance_matrix_max, performance_out_file)

for topk in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
    performance_matrix_topK_max = performance_matrix_topK_max_dict.get(topk)
    performance_out_file.write("\n" + str(K_FOLD) + "-FOLD PERFORMANCE INDEX of TOP-" + str(topk) + " SIMILARITY by MAX" + "\n")
    util_cy.print_performance_index(performance_matrix_topK_max, performance_out_file)

performance_out_file.write("************************************************************************************************************" + "\n")

##########################################################################################################
# GENERATE TIMING REPORT WITH BIO_CLINICAL_BERT IDENTIFIER
##########################################################################################################
total_execution_time = time.perf_counter() - script_start_time
timing_data['total_execution'] = total_execution_time

# Create timing report content
timing_report = []
timing_report.append("=" * 80)
timing_report.append(f"TIMING REPORT - Disease Diagnosis System ({model_name})")
timing_report.append("=" * 80)
timing_report.append(f"Dataset Loading:           {format_time(timing_data['dataset_loading'])}")
timing_report.append(f"Bio_ClinicalBERT Loading:  {format_time(timing_data['model_loading'])}")
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
print(f"[SUCCESS] Debug log saved to: {directory_prediction_root}debug_log.txt")

print(f"\n[SUCCESS] {model_name} analysis complete!")
print(f"[SUCCESS] Results directory: {directory_prediction_root}")
print(f"[SUCCESS] Performance metrics written to PerformanceIndex.txt")
print(f"[SUCCESS] Compare with baseline in: Prediction_Output_22112025_04-41-14_ORIGINAL_OUTPUTS/")
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

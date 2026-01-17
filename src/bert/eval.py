"""
Bio_ClinicalBERT Disease Diagnosis Evaluation System
====================================================
Modern implementation using HuggingFace transformers with Bio_ClinicalBERT embeddings.
Evaluates performance across multiple top-k strategies: MAX, TOP-10, TOP-20, TOP-30, TOP-40, TOP-50.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import shutil

# Import transformers for Bio_ClinicalBERT
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

# Import entity classes
import src.shared.entity.SymptomsDiagnosis as entity_module
from src.shared.entity.SymptomsDiagnosis import SymptomsDiagnosis
from src.shared.constants import *
from src.shared.preprocessing import preprocess_sentence, preprocess_diagnosis
from src.shared.similarity import cosine_similarity
from src.shared.data_loader import load_dataset, current_time, elapsed_time

# Alias for backward compatibility
class util_cy:
    """Compatibility shim for old util_cy references."""
    preprocess_sentence = staticmethod(preprocess_sentence)
    preprocess_diagnosis = staticmethod(preprocess_diagnosis)
    load_dataset = staticmethod(load_dataset)

# Ensure NLTK data is available
import nltk

def ensure_nltk_data():
    """Download required NLTK data packages if missing."""
    required_data = {
        'stopwords': 'corpora/stopwords',
        'punkt_tab': 'tokenizers/punkt_tab',
        'punkt': 'tokenizers/punkt'
    }
    
    missing = []
    for name, path in required_data.items():
        try:
            nltk.data.find(path)
        except LookupError:
            missing.append(name)
    
    if missing:
        print("[INFO] Downloading required NLTK data packages...")
        for name in missing:
            print(f"[INFO] Downloading '{name}'... ", end='', flush=True)
            try:
                nltk.download(name, quiet=True)
                print("✓")
            except Exception as e:
                print(f"✗ (Error: {e})")
        print("[SUCCESS] NLTK data download complete!\n")

ensure_nltk_data()

# Initialize stopwords for util_cy preprocessing
from nltk.corpus import stopwords
util_cy.stop_words = set(stopwords.words('english'))


class BioClinicalBERTEvaluator:
    """Bio_ClinicalBERT-based disease diagnosis evaluation system."""
    
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT'):
        """Initialize the evaluator with Bio_ClinicalBERT model."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.admissions = {}
        self.embeddings_symptoms = {}
        self.embeddings_diagnosis = {}
        
        # Timing data
        self.timing = {}
        
    def load_model(self):
        """Load Bio_ClinicalBERT model and tokenizer."""
        print(f"[INFO] Loading Bio_ClinicalBERT model: {self.model_name}")
        print(f"[INFO] Using device: {self.device}")
        
        start_time = time.perf_counter()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        load_time = time.perf_counter() - start_time
        self.timing['model_loading'] = load_time
        
        print(f"[SUCCESS] Model loaded successfully!")
        print(f"[TIMING] Model loading: {self.format_time(load_time)}\n")
        
    def load_dataset(self, filename='Symptoms-Diagnosis.txt'):
        """Load patient admission data from file."""
        print(f"[INFO] Loading dataset from: {filename}")
        
        start_time = time.perf_counter()
        
        file_path = os.path.join(CH_DIR, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.replace("\n", "")
            attributes = line.split(';')
            
            hadm_id = attributes[0]
            subject_id = attributes[1]
            admit_time = attributes[2]
            disch_time = attributes[3]
            symptoms = attributes[4]
            diagnosis = util_cy.preprocess_diagnosis(attributes[5])
            
            admission = entity.SymptomsDiagnosis.SymptomsDiagnosis(
                hadm_id, subject_id, admit_time, disch_time, symptoms, diagnosis
            )
            self.admissions[hadm_id] = admission
        
        load_time = time.perf_counter() - start_time
        self.timing['dataset_loading'] = load_time
        
        print(f"[SUCCESS] Dataset loaded: {len(self.admissions)} admissions")
        print(f"[TIMING] Dataset loading: {self.format_time(load_time)}\n")
        
    def get_bert_embedding(self, text: str) -> np.ndarray:
        """Get BERT embedding for a single text using mean pooling."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling over token embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        return embeddings.cpu().numpy()
    
    def compute_symptom_embeddings(self):
        """Compute BERT embeddings for all patient symptoms."""
        print("[INFO] Computing symptom embeddings with Bio_ClinicalBERT...")
        
        start_time = time.perf_counter()
        
        for hadm_id, admission in self.admissions.items():
            # Preprocess symptoms
            preprocessed_symptoms = util_cy.preprocess_sentence(admission.symptoms)
            
            # Get BERT embedding
            embedding = self.get_bert_embedding(preprocessed_symptoms)
            self.embeddings_symptoms[hadm_id] = embedding
        
        compute_time = time.perf_counter() - start_time
        self.timing['symptom_embeddings'] = compute_time
        
        print(f"[SUCCESS] Symptom embeddings computed: {len(self.embeddings_symptoms)} items")
        print(f"[TIMING] Symptom embeddings: {self.format_time(compute_time)}\n")
        
    def compute_diagnosis_embeddings(self):
        """Compute BERT embeddings for all unique diagnoses."""
        print("[INFO] Computing diagnosis embeddings with Bio_ClinicalBERT...")
        
        start_time = time.perf_counter()
        
        # Collect all unique diagnoses
        unique_diagnoses = set()
        for admission in self.admissions.values():
            for diag_with_drg in admission.diagnosis:
                # Extract diagnosis text (after colon)
                if ':' in diag_with_drg:
                    diagnosis_text = diag_with_drg[diag_with_drg.index(':') + 1:]
                else:
                    diagnosis_text = diag_with_drg
                unique_diagnoses.add(diagnosis_text)
        
        # Compute embeddings for each unique diagnosis
        for diagnosis_text in unique_diagnoses:
            embedding = self.get_bert_embedding(diagnosis_text)
            self.embeddings_diagnosis[diagnosis_text] = embedding
        
        compute_time = time.perf_counter() - start_time
        self.timing['diagnosis_embeddings'] = compute_time
        
        print(f"[SUCCESS] Diagnosis embeddings computed: {len(self.embeddings_diagnosis)} unique diagnoses")
        print(f"[TIMING] Diagnosis embeddings: {self.format_time(compute_time)}\n")
        
    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return 1.0 - cosine(vec1, vec2)
    
    def load_fold_data(self, fold_num: int, dataset_type: str) -> Dict[str, List[str]]:
        """
        Load test or training set for a specific fold.
        
        Args:
            fold_num: Fold number (0-9)
            dataset_type: 'TEST' or 'TRAIN'
            
        Returns:
            Dictionary mapping HADM_ID to list of diagnoses (lowercased to match embeddings)
        """
        if dataset_type == 'TEST':
            filename = f'data/folds/Fold{fold_num}/TestSet.txt'
        else:
            filename = f'data/folds/Fold{fold_num}/TrainingSet.txt'
        
        file_path = os.path.join(CH_DIR, filename)
        
        fold_data = {}
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Format: HADM_ID_diagnosis1,diagnosis2,...
                if '_' in line:
                    parts = line.split('_', 1)
                    hadm_id = parts[0]
                    diagnoses_str = parts[1] if len(parts) > 1 else ""
                    # CRITICAL FIX: Lowercase diagnoses to match embedding keys
                    diagnoses = [d.strip().lower() for d in diagnoses_str.split(',') if d.strip()]
                    fold_data[hadm_id] = diagnoses
        
        return fold_data
    
    def predict_topk_diagnoses(
        self,
        test_hadm_id: str,
        train_data: Dict[str, List[str]],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Predict top-k diagnoses for a test case based on symptom similarity.
        
        Args:
            test_hadm_id: Test patient admission ID
            train_data: Training set data
            k: Number of top similar cases to retrieve
            
        Returns:
            List of (diagnosis, similarity_score) tuples
        """
        # Get test symptom embedding
        test_embedding = self.embeddings_symptoms.get(test_hadm_id)
        if test_embedding is None:
            return []
        
        # Compute similarity with all training cases
        similarities = []
        for train_hadm_id in train_data.keys():
            train_embedding = self.embeddings_symptoms.get(train_hadm_id)
            if train_embedding is not None:
                sim = self.compute_cosine_similarity(test_embedding, train_embedding)
                similarities.append((train_hadm_id, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k cases
        top_k_cases = similarities[:k] if k > 0 else similarities[:1]
        
        # Collect diagnoses from top-k cases
        predicted_diagnoses = []
        for train_hadm_id, sim in top_k_cases:
            train_admission = self.admissions.get(train_hadm_id)
            if train_admission:
                for diag_with_drg in train_admission.diagnosis:
                    # Extract diagnosis text
                    if ':' in diag_with_drg:
                        diagnosis_text = diag_with_drg[diag_with_drg.index(':') + 1:]
                    else:
                        diagnosis_text = diag_with_drg
                    predicted_diagnoses.append((diagnosis_text, sim))
        
        return predicted_diagnoses
    
    def evaluate_prediction(
        self,
        ground_truth: List[str],
        predicted: List[Tuple[str, float]],
        threshold: float = 0.6
    ) -> Tuple[int, int]:
        """
        Evaluate prediction using diagnosis embedding similarity.
        
        Args:
            ground_truth: List of true diagnosis texts
            predicted: List of (diagnosis, similarity_score) tuples
            threshold: Similarity threshold for matching
            
        Returns:
            Tuple of (TP_count, FP_count)
        """
        if not ground_truth or not predicted:
            return 0, len(predicted) if predicted else 0
        
        # Extract predicted diagnosis texts (remove duplicates)
        pred_diagnoses = list(set([diag for diag, _ in predicted]))
        
        # Get embeddings for ground truth diagnoses
        gt_embeddings = []
        for gt_diag in ground_truth:
            if gt_diag in self.embeddings_diagnosis:
                gt_embeddings.append(self.embeddings_diagnosis[gt_diag])
        
        if not gt_embeddings:
            return 0, len(pred_diagnoses)
        
        # Evaluate each predicted diagnosis
        tp = 0
        fp = 0
        
        for pred_diag in pred_diagnoses:
            if pred_diag not in self.embeddings_diagnosis:
                fp += 1
                continue
            
            pred_embedding = self.embeddings_diagnosis[pred_diag]
            
            # Compute max similarity with ground truth diagnoses
            max_similarity = max([
                self.compute_cosine_similarity(pred_embedding, gt_emb)
                for gt_emb in gt_embeddings
            ])
            
            if max_similarity >= threshold:
                tp += 1
            else:
                fp += 1
        
        return tp, fp
    
    def compute_metrics(self, tp: int, fp: int, total_gt: int) -> Dict[str, float]:
        """Compute precision, recall, and F-score."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f_score': f_score
        }
    
    def run_evaluation(self, output_dir: str = None):
        """Run full 10-fold cross-validation evaluation."""
        print("=" * 80)
        print("Starting 10-Fold Cross-Validation Evaluation")
        print("=" * 80 + "\n")
        
        # Setup output directory
        if output_dir is None:
            timestamp = time.strftime('%d%m%Y_%H-%M-%S').replace(':', '-')
            output_dir = os.path.join(os.getcwd(), f'Prediction_Output_BioClinicalBERT_{timestamp}')
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Open performance output file
        perf_file_path = os.path.join(output_dir, 'PerformanceIndex.txt')
        perf_file = open(perf_file_path, 'w')
        
        # Initialize performance tracking
        strategies = ['MAX', 10, 20, 30, 40, 50]
        thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Performance matrices: strategy -> threshold -> [fold_metrics]
        performance_data = {
            strategy: {
                threshold: []
                for threshold in thresholds
            }
            for strategy in strategies
        }
        
        # Start fold processing
        folds_start = time.perf_counter()
        fold_times = []
        
        for fold_num in range(K_FOLD):
            fold_start = time.perf_counter()
            
            print(f"\n{'=' * 80}")
            print(f"Processing Fold {fold_num}")
            print(f"{'=' * 80}")
            
            # Load fold data
            test_data = self.load_fold_data(fold_num, 'TEST')
            train_data = self.load_fold_data(fold_num, 'TRAIN')
            
            print(f"[INFO] Fold {fold_num}: {len(train_data)} training, {len(test_data)} test cases")
            
            perf_file.write(f"\nFOLD {fold_num}: LEN train: {len(train_data)}, LEN test: {len(test_data)}\n")
            
            # Evaluate each strategy
            for strategy in strategies:
                k_value = 1 if strategy == 'MAX' else strategy
                strategy_name = f"{'MAX' if strategy == 'MAX' else f'TOP-{strategy}'}"
                
                for threshold in thresholds:
                    tp_total = 0
                    fp_total = 0
                    gt_total = 0
                    
                    # Evaluate each test case
                    for test_hadm_id in test_data.keys():
                        # Get ground truth diagnoses
                        gt_diagnoses = test_data[test_hadm_id]
                        gt_total += len(gt_diagnoses)
                        
                        # Predict diagnoses
                        predicted = self.predict_topk_diagnoses(test_hadm_id, train_data, k_value)
                        
                        # Evaluate
                        tp, fp = self.evaluate_prediction(gt_diagnoses, predicted, threshold)
                        tp_total += tp
                        fp_total += fp
                    
                    # Compute metrics
                    metrics = self.compute_metrics(tp_total, fp_total, gt_total)
                    
                    # Store results
                    performance_data[strategy][threshold].append({
                        'tp': tp_total,
                        'fp': fp_total,
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f_score': metrics['f_score']
                    })
            
            fold_time = time.perf_counter() - fold_start
            fold_times.append(fold_time)
            print(f"[TIMING] Fold {fold_num} completed: {self.format_time(fold_time)}")
        
        # Write aggregated results
        perf_file.write("\n" + "=" * 80 + "\n")
        perf_file.write(f"{K_FOLD}-FOLD AGGREGATED PERFORMANCE RESULTS\n")
        perf_file.write("=" * 80 + "\n\n")
        
        for strategy in strategies:
            strategy_name = 'MAX' if strategy == 'MAX' else f'TOP-{strategy}'
            perf_file.write(f"\nStrategy: {strategy_name}\n")
            perf_file.write("-" * 80 + "\n")
            perf_file.write(f"{'Threshold':<12} {'TP':<8} {'FP':<8} {'Precision':<12} {'Recall':<12} {'F-Score':<12}\n")
            perf_file.write("-" * 80 + "\n")
            
            for threshold in thresholds:
                fold_results = performance_data[strategy][threshold]
                
                # Compute means
                tp_mean = np.mean([r['tp'] for r in fold_results])
                fp_mean = np.mean([r['fp'] for r in fold_results])
                prec_mean = np.mean([r['precision'] for r in fold_results])
                rec_mean = np.mean([r['recall'] for r in fold_results])
                f_mean = np.mean([r['f_score'] for r in fold_results])
                
                perf_file.write(
                    f"{threshold:<12.2f} {tp_mean:<8.2f} {fp_mean:<8.2f} "
                    f"{prec_mean:<12.4f} {rec_mean:<12.4f} {f_mean:<12.4f}\n"
                )
        
        # Timing summary
        folds_total_time = time.perf_counter() - folds_start
        self.timing['folds_total'] = folds_total_time
        self.timing['fold_times'] = fold_times
        
        perf_file.write("\n" + "=" * 80 + "\n")
        perf_file.write("TIMING SUMMARY\n")
        perf_file.write("=" * 80 + "\n")
        perf_file.write(f"Dataset Loading:        {self.format_time(self.timing['dataset_loading'])}\n")
        perf_file.write(f"Model Loading:          {self.format_time(self.timing['model_loading'])}\n")
        perf_file.write(f"Symptom Embeddings:     {self.format_time(self.timing['symptom_embeddings'])}\n")
        perf_file.write(f"Diagnosis Embeddings:   {self.format_time(self.timing['diagnosis_embeddings'])}\n")
        perf_file.write(f"Fold Processing:        {self.format_time(folds_total_time)}\n")
        
        total_time = sum([
            self.timing.get('dataset_loading', 0),
            self.timing.get('model_loading', 0),
            self.timing.get('symptom_embeddings', 0),
            self.timing.get('diagnosis_embeddings', 0),
            folds_total_time
        ])
        perf_file.write(f"TOTAL EXECUTION:        {self.format_time(total_time)}\n")
        
        perf_file.close()
        
        print("\n" + "=" * 80)
        print("Evaluation Complete!")
        print("=" * 80)
        print(f"[SUCCESS] Results saved to: {perf_file_path}")
        print(f"[TIMING] Total execution: {self.format_time(total_time)}\n")
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{seconds:.2f} seconds ({minutes:.2f} minutes)"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{seconds:.2f} seconds ({hours:.2f} hours, {minutes:.2f} minutes)"


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("Bio_ClinicalBERT Disease Diagnosis Evaluation System")
    print("=" * 80 + "\n")
    
    # Initialize evaluator
    evaluator = BioClinicalBERTEvaluator()
    
    # Load model
    evaluator.load_model()
    
    # Load dataset
    evaluator.load_dataset()
    
    # Compute embeddings
    evaluator.compute_symptom_embeddings()
    evaluator.compute_diagnosis_embeddings()
    
    # Run evaluation
    evaluator.run_evaluation()
    
    print("\n[SUCCESS] All tasks completed successfully!")


if __name__ == "__main__":
    main()

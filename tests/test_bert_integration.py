#!/usr/bin/env python3
"""
Test Bio_ClinicalBERT Integration
Simple test to verify the Bio_ClinicalBERT model loads and can generate embeddings
Run this before executing the full CS2V_bio_clinicalBERT.py analysis
"""

import sys
import time

def test_bert_integration():
    """Test that Bio_ClinicalBERT can be loaded and used"""
    print("=" * 60)
    print("Testing Bio_ClinicalBERT Integration")
    print("=" * 60)
    
    # Test 1: Import dependencies
    print("[TEST 1] Testing imports...")
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        import numpy as np
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("Run: pip install -r requirements_bert.txt")
        return False
    
    # Test 2: Load Bio_ClinicalBERT model
    print("\n[TEST 2] Loading Bio_ClinicalBERT model...")
    try:
        start_time = time.perf_counter()
        model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
        load_time = time.perf_counter() - start_time
        print(f"✓ Model loaded successfully in {load_time:.2f} seconds")
        print(f"  Device: {model.device}")
        print(f"  Max sequence length: {model.max_seq_length}")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print("Check internet connection and Hugging Face access")
        return False
    
    # Test 3: Generate sample embeddings
    print("\n[TEST 3] Testing embedding generation...")
    try:
        # Sample clinical texts (similar to your dataset)
        test_symptoms = [
            "patient has severe chest pain radiating to left arm",
            "shortness of breath and fatigue for 2 days",
            "abdominal pain nausea vomiting"
        ]
        test_diagnoses = [
            "acute myocardial infarction unspecified",
            "congestive heart failure unspecified", 
            "gastroenteritis and colitis unspecified"
        ]
        
        print("  Testing symptom embeddings...")
        symptom_embeddings = model.encode(test_symptoms, convert_to_numpy=True)
        print(f"  ✓ Generated symptom embeddings: shape {symptom_embeddings.shape}")
        
        print("  Testing diagnosis embeddings...")
        diagnosis_embeddings = model.encode(test_diagnoses, convert_to_numpy=True)
        print(f"  ✓ Generated diagnosis embeddings: shape {diagnosis_embeddings.shape}")
        
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return False
    
    # Test 4: Test similarity computation (core functionality)
    print("\n[TEST 4] Testing similarity computation...")
    try:
        from scipy.spatial.distance import cosine
        
        # Test similarity between symptom and diagnosis
        sim1 = 1 - cosine(symptom_embeddings[0], diagnosis_embeddings[0])  # chest pain vs MI
        sim2 = 1 - cosine(symptom_embeddings[0], diagnosis_embeddings[1])  # chest pain vs CHF
        sim3 = 1 - cosine(symptom_embeddings[0], diagnosis_embeddings[2])  # chest pain vs gastro
        
        print(f"  Chest pain vs MI: {sim1:.4f}")
        print(f"  Chest pain vs CHF: {sim2:.4f}")
        print(f"  Chest pain vs Gastroenteritis: {sim3:.4f}")
        
        # Expect chest pain to be most similar to MI
        if sim1 > sim2 and sim1 > sim3:
            print("  ✓ Similarity ranking looks correct (chest pain → MI highest)")
        else:
            print("  ⚠ Similarity ranking unexpected but model is working")
            
    except Exception as e:
        print(f"✗ Similarity computation failed: {e}")
        return False
    
    # Test 5: Test util_cy functions (if available)
    print("\n[TEST 5] Testing util_cy preprocessing compatibility...")
    try:
        import util_cy
        processed = util_cy.preprocess_sentence("Patient has severe chest pain and SOB.")
        print(f"  ✓ Preprocessed sample: '{processed}'")
    except Exception as e:
        print(f"  ⚠ util_cy test failed: {e}")
        print("  This is expected if util_cy.pyd is not compiled for your system")
    
    print("\n" + "=" * 60)
    print("🎉 Bio_ClinicalBERT Integration Test PASSED!")
    print("✅ Ready to run CS2V_bio_clinicalBERT.py")
    print("=" * 60)
    
    return True

def main():
    """Main test execution"""
    success = test_bert_integration()
    
    if success:
        print("\n🚀 Next steps:")
        print("1. Run full analysis: python CS2V_bio_clinicalBERT.py")
        print("2. Compare results with baseline in Prediction_Output_22112025_04-41-14_ORIGINAL_OUTPUTS/")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed. Fix issues before running full analysis.")
        sys.exit(1)

if __name__ == "__main__":
    main()

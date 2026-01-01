# Pre-trained Models

This directory contains large pre-trained model files.

## BioSentVec Model

The baseline implementation requires the BioSentVec model file:
- **File**: `BioSentVec_PubMed_MIMICIII-bigram_d700.bin`
- **Size**: ~22.5 GB

### Download Instructions

1. Download from the official source: https://github.com/ncbi-nlp/BioSentVec
2. Place the `.bin` file in this directory
3. The model will be automatically loaded by `scripts/run_baseline.py`

## BERT Models

BERT models (Bio_ClinicalBERT, BiomedBERT, BlueBERT) are downloaded automatically from HuggingFace when first used.

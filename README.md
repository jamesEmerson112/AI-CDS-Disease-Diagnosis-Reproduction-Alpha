# AI-CDS Disease Diagnosis System

Clinical Decision Support System for disease diagnosis prediction using patient symptom similarity.

## Overview

This project reproduces and extends the clinical decision support system from *"AI-Driven Clinical Decision Support: Enhancing Disease Diagnosis Exploiting Patients Similarity"* (Comito et al., 2022). We first reproduce the original BioSentVec baseline, then replace it with three biomedical BERT models as an original extension. Our score distribution analysis reveals that the BERT evaluation metric saturates at the paper's threshold, making the F1 scores unsuitable for model comparison — this is the key finding and an open problem for future work.

## Baseline Reproduction

The original paper uses **BioSentVec** (700-dimensional sent2vec embeddings trained on PubMed + MIMIC-III) to compute symptom-level pairwise cosine similarities between patients. Diagnosis similarity is determined by taking the MAX similarity across the Cartesian product of ground-truth and predicted diagnosis descriptions, then applying a threshold to classify true/false positives.

**Baseline results at threshold = 0.6:**

| Method | F1 Score |
|--------|:--------:|
| TOP-10 | 0.489 |
| TOP-20 | 0.512 |
| TOP-30 | 0.521 |

```bash
python scripts/run_baseline.py
```

## BERT Extension (Original Contribution)

We replace BioSentVec with three biomedical BERT models that produce 768-dimensional embeddings:

| Model | HuggingFace Path | Training Data |
|-------|-------------------|---------------|
| Bio_ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | MIMIC-III clinical notes |
| BiomedBERT | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | PubMed abstracts |
| BlueBERT | `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` | PubMed + MIMIC-III |

**Results at threshold = 0.6 (baseline reference point):**

| Method | BioSentVec (Baseline) | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:---------------------:|:-----------------:|:----------:|:--------:|
| TOP-10 | 0.489 | 1.000 | 1.000 | 1.000 |
| TOP-20 | 0.512 | 1.000 | 1.000 | 1.000 |
| TOP-30 | 0.521 | 1.000 | 1.000 | 1.000 |

All three BERT models achieve perfect F1 = 1.000 at threshold 0.6. BiomedBERT maintains perfect scores through threshold 0.9. **However, these results are misleading** — see the score distribution analysis below.

```bash
python scripts/run_all_bert_models.py
```

See [docs/bert_model_comparison.md](docs/bert_model_comparison.md) for full results at all thresholds.

## Score Distribution Analysis (Key Finding)

The perfect F1 scores are an artifact of **embedding space compactness** combined with the **MAX-over-Cartesian-product** evaluation strategy, not genuine diagnostic accuracy.

**Why the metric saturates:**

1. **Compact embedding spaces** — Biomedical BERT models map diagnosis text into a narrow region. Even *unrelated* diagnoses have high cosine similarity:

   | Model | Mean Pairwise Sim | Min Pairwise Sim | Std |
   |-------|:-----------------:|:----------------:|:---:|
   | BiomedBERT | 0.93 | 0.72 | 0.03 |
   | Bio_ClinicalBERT | 0.83 | 0.65 | 0.05 |
   | BlueBERT | 0.72 | 0.48 | 0.07 |

2. **MAX operator amplification** — Taking the maximum similarity across all diagnosis pairs inflates scores further. Per-patient MAX similarity exceeds 0.6 for virtually all patient pairs:

   | Model | % of patient pairs with MAX >= 0.6 |
   |-------|:----------------------------------:|
   | Bio_ClinicalBERT | 100.00% |
   | BiomedBERT | 100.00% |
   | BlueBERT | 99.96% |

3. **Conclusion** — The evaluation metric is saturated at threshold 0.6 for BERT models. The F1 scores cannot discriminate between models or meaningfully compare against the baseline. Alternative evaluation strategies (MEAN instead of MAX, DRG code matching, higher thresholds) are needed.

Visualizations and full statistics are in [`docs/score_distribution_analysis/`](docs/score_distribution_analysis/).

```bash
python scripts/analyze_score_distributions.py
```

## Project Structure

```
src/                     # Source code
  models/                # Baseline (sent2vec) and BERT implementations
  entity/                # Data classes (Admission, Symptom, Drgcodes)
  utils/                 # Utilities, constants, cython similarity
  evaluation/            # Evaluation modules
scripts/                 # Entry point scripts
  run_baseline.py        # Run BioSentVec baseline
  run_all_bert_models.py # Run all 3 BERT models sequentially
  analyze_score_distributions.py  # Score distribution analysis
data/                    # Data files
  folds/                 # 10-fold cross-validation splits
  raw/                   # Raw data files
  models/                # Pre-trained model files
docs/                    # Documentation and analysis reports
config/                  # Environment and requirements files
tests/                   # Test files
```

## Setup

**Conda environment:**

```bash
conda env create -f config/environment.yml
conda activate disease-diagnosis
```

**Key dependencies:** sentence-transformers, torch, matplotlib, numpy

**For baseline only:** Also requires sent2vec and the BioSentVec pre-trained model (~21 GB). See [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for details.

## Citation

```bibtex
@article{comito2022ai,
  title={AI-Driven Clinical Decision Support: Enhancing Disease Diagnosis Exploiting Patients Similarity},
  author={Comito, Carmela and Falcone, Deborah and Forestiero, Agostino},
  journal={IEEE Access},
  volume={10},
  pages={6224--6234},
  year={2022},
  publisher={IEEE}
}
```

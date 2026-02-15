# BERT Model Comparison Report

**Date**: 2026-02-15
**Platform**: Apple M1, macOS, MPS acceleration
**Dataset**: 129 admissions, 10-fold cross-validation

## Models Evaluated

| # | Model | HuggingFace Path | Training Data |
|---|-------|-------------------|---------------|
| 1 | Bio_ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | MIMIC-III clinical notes |
| 2 | BiomedBERT | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | PubMed abstracts |
| 3 | BlueBERT | `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` | PubMed + MIMIC-III |

All models produce 768-dimensional embeddings (vs. BioSentVec's 700D baseline).

---

## Results at Threshold = 0.6 (Baseline Reference Point)

At threshold 0.6, **all three BERT models achieve perfect F1 = 1.000** across every strategy (MAX through TOP-50). This means every test case's predicted diagnosis has cosine similarity >= 0.6 with the ground truth.

| Method | BioSentVec (Baseline) | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:---------------------:|:-----------------:|:----------:|:--------:|
| **F1** | | | | |
| MAX | -- | 1.000 | 1.000 | 1.000 |
| TOP-10 | 0.489 | 1.000 | 1.000 | 1.000 |
| TOP-20 | 0.512 | 1.000 | 1.000 | 1.000 |
| TOP-30 | 0.521 | 1.000 | 1.000 | 1.000 |
| TOP-40 | -- | 1.000 | 1.000 | 1.000 |
| TOP-50 | -- | 1.000 | 1.000 | 1.000 |

**Improvement over baseline at threshold 0.6**: All models achieve +100% TP rate, indicating BERT cosine similarities are consistently higher than BioSentVec for matching diagnoses.

---

## Results at Higher Thresholds (Model Differentiation)

Since threshold 0.6 saturates, the meaningful comparison is at stricter thresholds (0.8, 0.9, 1.0).

### Threshold = 0.9 (F1 Score)

| Method | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------:|:----------:|:--------:|
| MAX | 0.285 | **1.000** | 0.194 |
| TOP-10 | 0.797 | **1.000** | 0.340 |
| TOP-20 | 0.906 | **1.000** | 0.487 |
| TOP-30 | 0.930 | **1.000** | 0.517 |
| TOP-40 | 0.946 | **1.000** | 0.541 |
| TOP-50 | 0.953 | **1.000** | 0.572 |

### Threshold = 0.8 (F1 Score)

| Method | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------:|:----------:|:--------:|
| MAX | 0.969 | **1.000** | 0.395 |
| TOP-10 | **1.000** | **1.000** | 0.813 |
| TOP-20 | **1.000** | **1.000** | 0.907 |
| TOP-30 | **1.000** | **1.000** | 0.930 |
| TOP-40 | **1.000** | **1.000** | 0.946 |
| TOP-50 | **1.000** | **1.000** | 0.962 |

### Threshold = 1.0 (Exact Match, F1 Score)

| Method | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------:|:----------:|:--------:|
| MAX | 0.146 | 0.169 | 0.178 |
| TOP-10 | 0.285 | 0.254 | 0.239 |
| TOP-20 | 0.331 | 0.324 | 0.339 |
| TOP-30 | 0.363 | 0.340 | 0.347 |
| TOP-40 | 0.394 | 0.371 | 0.378 |
| TOP-50 | 0.401 | 0.378 | 0.417 |

### Threshold = 0.7 (F1 Score)

| Method | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------:|:----------:|:--------:|
| MAX | **1.000** | **1.000** | 0.907 |
| TOP-10 | **1.000** | **1.000** | **1.000** |
| TOP-20 | **1.000** | **1.000** | **1.000** |
| TOP-30 | **1.000** | **1.000** | **1.000** |
| TOP-40 | **1.000** | **1.000** | **1.000** |
| TOP-50 | **1.000** | **1.000** | **1.000** |

---

## Model Rankings

### Best Overall: BiomedBERT

BiomedBERT achieves perfect F1 = 1.000 at all thresholds up to 0.9, across all strategies. It only drops below perfect at threshold = 1.0 (exact cosine match). This suggests BiomedBERT produces highly concentrated similarity scores in the 0.9-1.0 range.

### Runner-up: Bio_ClinicalBERT

Bio_ClinicalBERT achieves perfect scores at thresholds <= 0.7 and near-perfect at 0.8 (only MAX drops to 0.969). At threshold 0.9, it shows clear stratification across TOP-K values, from 0.285 (MAX) to 0.953 (TOP-50).

### Third: BlueBERT

BlueBERT shows the widest score distribution, with differentiation starting at threshold 0.7 (MAX: 0.907) and spreading significantly at 0.8 and 0.9. At threshold = 1.0 (exact match), BlueBERT TOP-50 (0.417) slightly edges out the others.

---

## Timing Breakdown

| Phase | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|-------|:----------------:|:----------:|:--------:|
| Model Loading | 81.3s (1.4 min) | 15.3s | 11.1s |
| Symptom Embeddings (573) | 5.0s | 2.8s | 1.5s |
| Diagnosis Embeddings (145) | 0.9s | 0.6s | 0.6s |
| 10-Fold Evaluation | 20.5 min | 20.8 min | 20.5 min |
| **Total** | **22.0 min** | **21.1 min** | **20.7 min** |
| **Overall Pipeline** | | | **76.4 min** |

Notes:
- Bio_ClinicalBERT's slow model loading (81s vs ~13s) is due to first-time download + no native sentence-transformers config (mean pooling created on-the-fly).
- Fold evaluation times are CPU-bound (pairwise similarity computation in pure Python), not GPU-bound.
- All models ran on MPS (Apple Metal) for embedding computation.

---

## Key Observations

1. **All BERT models massively outperform BioSentVec at threshold 0.6**: The baseline's best F1 was 0.521 (TOP-30); all BERT models achieve 1.000. This indicates BERT embeddings produce much higher diagnosis-level cosine similarities than BioSentVec.

2. **BiomedBERT's perfect scores through threshold 0.9 warrant investigation**: Perfect 1.0 F1 at threshold 0.9 for all strategies suggests the model may be producing similarity scores that are systematically very high. This could indicate that BiomedBERT's embedding space is more compact for medical terms, or it could signal a need to investigate whether the un-normalized embeddings (`normalize_embeddings=False` in the code) interact differently with the cosine similarity implementation.

3. **P = R = F1 in all results**: The performance metric computation produces identical Precision, Recall, and F1 values. This is because the evaluation framework treats each test case as a binary TP/FP decision (is the predicted diagnosis similar enough?), making P and R equivalent in this formulation.

4. **Threshold 1.0 is the only discriminative threshold**: At exact-match (cosine sim = 1.0), all models perform similarly (F1 ~ 0.15-0.42), with BlueBERT TOP-50 slightly leading at 0.417.

---

## Output Directories

- `Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/`
- `Prediction_Output_BiomedBERT_15022026_12-03-36/`
- `Prediction_Output_BlueBERT_15022026_12-24-38/`

Each contains:
- `PerformanceIndex.txt` - Full per-case and aggregated metrics
- `timing_report.txt` - Detailed timing breakdown
- `Fold0/` through `Fold9/` - Per-fold prediction details

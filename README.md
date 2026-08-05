# AI-CDS Disease Diagnosis System

Clinical Decision Support System for disease diagnosis prediction using patient symptom similarity.

This project reproduces the clinical decision support system of *"AI-Driven Clinical Decision
Support: Enhancing Disease Diagnosis Exploiting Patients Similarity"* (Comito et al., 2022) and then
swaps its 2019 BioSentVec encoder for three modern biomedical BERT models behind the **identical**
retrieval and scoring pipeline. The result is a null result, and the null result is the
contribution: under this evaluation the old encoder and the new ones cannot be told apart.

---

## Headline comparison — four encoders, one pipeline

All four arms were run on **one machine** (RunPod Linux, AMD Threadripper 7960X) from **one commit**
(`7da5901`) on 2026-08-05. The three transformer runs reproduce earlier Apple-silicon runs
**bit-for-bit to all 17 significant figures**, so hardware is not a confound.

### Threshold 1.0, TOP-10 — the only setting where no model is saturated

| Rank | Encoder | Dim | Precision | Recall | F1 | TP | FP | `TP+FP` | Pred. rate |
|:----:|---------|:---:|:---------:|:------:|:------:|:---:|:---:|:-------:|:----------:|
| 1 | Bio_ClinicalBERT | 768 | 0.2853 | 0.2853 | **0.2853** | 3.7 | 9.2 | 12.9 | 1.0000 |
| 2 | **BioSentVec — the 2019 baseline** | 700 | **0.3254** | 0.2474 | **0.2801** | 3.2 | 6.7 | **9.9** | **0.7679** |
| 3 | BiomedBERT | 768 | 0.2545 | 0.2545 | **0.2545** | 3.3 | 9.6 | 12.9 | 1.0000 |
| 4 | BlueBERT | 768 | 0.2391 | 0.2391 | **0.2391** | 3.1 | 9.8 | 12.9 | 1.0000 |

*TP and FP are means across the 10 folds (~12.9 test cases per fold). Pred. rate is the fraction of
test cases on which the system predicts at all.*

**Verdict — no encoder ranking is supported by this experiment.** The 700-dimensional,
non-contextual, 2019 baseline lands **second of four** and is statistically indistinguishable from
all three modern transformers. The total spread across the four arms is **0.046** (0.239 to 0.285);
the best transformer beats the baseline by **0.005**. Both are dwarfed by the patient-leakage
inflation of **+0.11 to +0.26** and are smaller than the per-fold standard deviation of
**0.071–0.124**. The differences under study are an order of magnitude below the noise and the known
contamination, so any claim that one of these encoders is better than another would be unfounded.

### The ranking inverts when you move the threshold

This is the sharpest evidence that the ranking is not a property of the encoders. Threshold 0.9 also
produces four distinct values — but a **completely different order**:

| Encoder | F1 @ 0.9 | rank | F1 @ 1.0 | rank | movement |
|---------|:--------:|:----:|:--------:|:----:|----------|
| BiomedBERT | **1.0000** *(saturated)* | 1 | 0.2545 | 3 | **1st → 3rd** |
| Bio_ClinicalBERT | 0.7974 | 2 | **0.2853** | 1 | 2nd → 1st |
| BlueBERT | 0.3404 | 3 | 0.2391 | 4 | 3rd → 4th |
| **BioSentVec (baseline)** | 0.2801 | 4 | **0.2801** | 2 | **4th → 2nd** |

Full F1-by-threshold grid at TOP-10:

| Encoder | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---------|:---:|:---:|:---:|:---:|:---:|
| Bio_ClinicalBERT | 1.0000 | 1.0000 | 1.0000 | 0.7974 | 0.2853 |
| BiomedBERT | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2545 |
| BlueBERT | 1.0000 | 1.0000 | 0.8135 | 0.3404 | 0.2391 |
| BioSentVec | 0.4824 | 0.3658 | 0.3222 | 0.2801 | 0.2801 |

**Whichever encoder "wins" is decided by an arbitrary threshold choice, not by diagnostic quality.**
BiomedBERT tops the table at 0.9 only because it is still pinned at the ceiling there — it is the
most compact of the three embedding spaces, so it is the last to fall off 1.000, and grading it with
its own cosine rewards exactly that compactness. Threshold 1.0 is the only setting where **no** model
is saturated, which is why the table above is the one to read; it is not, however, the only setting
where the four differ numerically.

**Read the `TP+FP` and prediction-rate columns together — they are the whole degeneracy story.**
Every BERT arm sums to exactly **12.9**, the mean fold test size, so `tp + fp == nrow`; precision
therefore reduces to `tp/nrow`, which *is* recall, and their harmonic mean is that same number. Every
BERT "F1" in this table is accuracy. The baseline sums to **9.9** because it abstains on 23.2% of
cases when nothing clears the pruning gate — which is also why it holds the **highest precision of
all four arms** (0.3254) while scoring second on F1.

### Threshold 0.6, TOP-10 — SATURATED. This is the metric, not the model.

| Encoder | Precision | Recall | F1 | Pred. rate | Reading |
|---------|:---------:|:------:|:------:|:----------:|---------|
| BioSentVec (baseline) | 0.5624 | 0.4256 | **0.4824** | 0.7679 | reproduces published 0.489 (Δ 0.007) |
| Bio_ClinicalBERT | 1.0000 | 1.0000 | **1.000** | 1.0000 | **saturated — not a result** |
| BiomedBERT | 1.0000 | 1.0000 | **1.000** | 1.0000 | **saturated — not a result** |
| BlueBERT | 1.0000 | 1.0000 | **1.000** | 1.0000 | **saturated — not a result** |

**The three 1.000s are an artifact and must not be read as an achievement.** At threshold 0.6 the
compactness of biomedical embedding space (mean pairwise cosine 0.72–0.93 *even between unrelated
diagnoses*) combined with the MAX-over-Cartesian-product aggregator puts ~100% of patient pairs above
the bar, so essentially everything counts as a match. 0.6 is the paper's threshold, which is why it
appears here at all; it cannot discriminate between models. The threshold-1.0 table above is the one
to read.

**These numbers carry three known defects** — metric saturation, metric degeneracy, and patient
leakage in the folds. Read [the caveat block immediately below](#three-caveats-that-apply-to-every-score-in-this-repo)
before quoting any figure from this README.

**Full six-page visual version:** `results/model_comparison.pdf`, generated by

```bash
python scripts/compare_models.py
```

It covers provenance, the summary table, threshold curves, TOP-K curves, the prediction-rate
(degeneracy) page, and the complete threshold × TOP-K grid.

**An inversion worth noting.** The baseline model file is **20.93 GiB**; the three transformers are
416 MB, 420 MB, and 420 MB — the 2019 baseline is roughly **17× larger than all three modern models
combined**, because sent2vec stores an explicit unigram + bigram embedding table while BERT computes
representations contextually from ~110M parameters. The larger, older, storage-heavy model matches
them.

---

## Three caveats that apply to every score in this repo

> **1. The metric saturates.** At threshold 0.6 nearly every diagnosis pair counts as a match, so
> F1 = 1.000 measures the metric, not the model
> ([details](docs/findings/03-metric-saturation.md)).
>
> **2. The metric is degenerate — every BERT "F1" here is accuracy.** Across all 12,600 rows of
> committed BERT results, precision, recall, and F-score are the *same number*, because every test
> case increments exactly one of TP or FP. **Confirmed 2026-08-05:** this is a property of the
> embedding space, not the code. The baseline arm finally ran and its precision and recall *do*
> diverge, because it abstains on ~30 of 129 cases (23.2%) when nothing clears the pruning gate.
> BERT's space is compact enough that it never abstains. Saturation and degeneracy are therefore
> **one root cause at two different gates** ([details](docs/findings/04-metric-degeneracy.md)).
>
> **3. The folds leak patients, by ~10× the effect under study.** Folds split on `HADM_ID`, but the
> 129 admissions come from only 100 patients — one holds 15. **41 of 129 test cases (31.8%)** have
> another admission from the same patient in their own retrieval pool. Inflation at threshold 1.0 is
> **+0.11 to +0.26**, against encoder differences of **0.015–0.046**. This affects **both arms**, so
> the published 0.489/0.512/0.521 carry it too ([details](docs/findings/05-patient-leakage.md)).

**A fourth constraint bounds every exact-match number above.** Only **75 of 129 test cases (58.1%)**
have their correct DRG present anywhere in their own fold's training pool — 105 of the 145 unique
diagnoses occur exactly once in the dataset. A *perfect* retriever therefore caps at 58.1% under
exact matching, which is the context in which the threshold-1.0 scores of 0.24–0.29 should be read.

Start at [docs/](docs/README.md); the synthesis is
[07-comparison-validity.md](docs/findings/07-comparison-validity.md).

---

## Baseline Reproduction

The original paper uses **BioSentVec** (700-dimensional sent2vec embeddings trained on PubMed + MIMIC-III) to compute symptom-level pairwise cosine similarities between patients. Diagnosis similarity is determined by taking the MAX similarity across the Cartesian product of ground-truth and predicted diagnosis descriptions, then applying a threshold to classify true/false positives.

**Published results at threshold = 0.6** (Comito et al., not this repo's output):

| Method | F1 Score |
|--------|:--------:|
| TOP-10 | 0.489 |
| TOP-20 | 0.512 |
| TOP-30 | 0.521 |

**Our reproduction — 2026-08-05.** The baseline arm had never executed in this checkout: it crashed
on a wrong data path and an unbound name, and `sent2vec` cannot be built under MSVC, making the arm
Linux-only. Both were fixed and the arm ran end to end on a rented Linux box (10 folds, ~13 min):

| Method | Precision | Recall | **F1** | Published F1 |
|--------|:---------:|:------:|:------:|:------------:|
| TOP-10 @ 0.6 | 0.562 | 0.426 | **0.482** | 0.489 |

**Within 0.007 of the published figure** — the first successful reproduction of the paper's headline
number by this codebase. Note precision ≠ recall here, unlike every BERT row: the baseline declines
to predict on 23.2% of cases. Details: [09-baseline-first-run.md](docs/findings/09-baseline-first-run.md).

Threshold 0.6 is the paper's own operating point, so this row is the correct comparison against the
paper — but it is still a saturated threshold for the BERT arms, and it is not where the four
encoders can be compared to each other. For that, see the threshold-1.0 table above.

```bash
python scripts/run_baseline.py    # Linux only; needs the 20.93 GiB BioSentVec model
```

## BERT Extension (Original Contribution)

We replace BioSentVec with three biomedical BERT models that produce 768-dimensional embeddings.
Everything else in the pipeline — fold splits, pruning, aggregation, scoring, thresholds — is
unchanged, so the encoder is the only moving part:

| Model | HuggingFace Path | Training Data | Size |
|-------|-------------------|---------------|:----:|
| Bio_ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | MIMIC-III clinical notes | 416 MB |
| BiomedBERT | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | PubMed abstracts | 420 MB |
| BlueBERT | `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` | PubMed + MIMIC-III | 420 MB |

**Results at threshold = 0.6 — SATURATED across all three BERT arms. This table measures the metric,
not the models; it is included only because 0.6 is the paper's threshold.**

| Method | BioSentVec (published) | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------------:|:-----------------:|:----------:|:--------:|
| TOP-10 @ 0.6 | 0.489 | 1.000 *(saturated)* | 1.000 *(saturated)* | 1.000 *(saturated)* |
| TOP-20 @ 0.6 | 0.512 | 1.000 *(saturated)* | 1.000 *(saturated)* | 1.000 *(saturated)* |
| TOP-30 @ 0.6 | 0.521 | 1.000 *(saturated)* | 1.000 *(saturated)* | 1.000 *(saturated)* |

All three BERT models reach F1 = 1.000 at threshold 0.6, and BiomedBERT holds it through threshold
0.9. **This is not diagnostic accuracy** — see the score distribution analysis below for the
mechanism. Note also that TOP-K scores rise monotonically with K because a single hit inside K
suffices and there is no penalty for the other K−1 predictions; that curve is an artifact of the
metric, not evidence that larger K retrieves better.

The informative comparison is the [threshold-1.0 headline table](#threshold-10-top-10--the-only-setting-where-all-four-separate)
at the top of this README.

### Runtime — the encoder is not the bottleneck

On the four-arm pod run: **14.27 / 14.18 / 14.05 minutes** for Bio_ClinicalBERT / BiomedBERT /
BlueBERT, and **~12.7 minutes** for the BioSentVec baseline. Measured across the committed runs,
**embedding is 0.17–0.45% of wall-clock**, while the single-threaded pure-Python cosine loop is over
93%. Total fold time varies only **1.7%** across the three transformers. Making embedding
instantaneous would cut a 21.98-minute run to 21.88 minutes, so **a GPU buys essentially nothing for
these arms** ([details](docs/findings/08-runtime-and-cost.md)).

Results are also **platform-independent**: re-running the BERT arms on x86 Linux reproduced the
Apple-silicon numbers **bit-for-bit, to all 17 significant figures.**

```bash
python scripts/run_all_bert_models.py
```

See [docs/bert_model_comparison.md](docs/bert_model_comparison.md) for full results at all thresholds.

## Visual Summary (README Charts)

These plots are generated directly from the three experiment outputs in:
- `Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/PerformanceIndex.txt`
- `Prediction_Output_BiomedBERT_15022026_12-03-36/PerformanceIndex.txt`
- `Prediction_Output_BlueBERT_15022026_12-24-38/PerformanceIndex.txt`

![F1 vs threshold (TOP-10)](docs/readme_plots/f1_vs_threshold_top10.svg)
*At TOP-10, BiomedBERT stays saturated through 0.9 while BlueBERT drops earlier. The 1.000 region on
the left of this chart is metric saturation, not accuracy.*

![F1 vs threshold (TOP-50)](docs/readme_plots/f1_vs_threshold_top50.svg)
*At TOP-50, all models improve at strict thresholds, but separation remains at 0.9/1.0.*

![F1 vs TOP-K at threshold 0.9](docs/readme_plots/f1_vs_topk_t09.svg)
*Top-K expansion strongly helps Bio_ClinicalBERT and BlueBERT at threshold 0.9 — but one hit inside K
suffices and the other K−1 predictions are unpenalised, so the upward slope is a metric artifact.*

![F1 vs TOP-K at threshold 1.0](docs/readme_plots/f1_vs_topk_t10.svg)
*At exact-match threshold 1.0, model differences are modest and increase gradually with K.*

![Runtime breakdown](docs/readme_plots/runtime_breakdown.svg)
*10-fold evaluation dominates runtime; startup overhead differs mostly by model loading time.*

![Saturation by threshold](docs/readme_plots/saturation_by_threshold.svg)
*Per-patient MAX similarity saturation explains the perfect F1 at threshold 0.6.*

Regenerate these charts:

```bash
python3 scripts/build_readme_plots.py
```

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

### Visualizations

![Diagnosis score distributions](docs/score_distribution_analysis/score_distributions.png)
*Diagnosis score distributions across baseline and BERT models.*

![Per-patient maximum similarity distributions](docs/score_distribution_analysis/per_patient_max_distributions.png)
*Per-patient MAX similarity distributions showing saturation behavior.*

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
  compare_models.py      # Four-arm comparison PDF -> results/model_comparison.pdf
  analyze_score_distributions.py  # Score distribution analysis
data/                    # Data files
  folds/                 # 10-fold cross-validation splits
  raw/                   # Raw data files
  models/                # Pre-trained model files
docs/                    # Documentation and analysis reports
results/                 # Comparison outputs (model_comparison.pdf, per-arm runs)
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

**For baseline only:** Also requires sent2vec and the BioSentVec pre-trained model (20.93 GiB). See [docs/guides/setup.md](docs/guides/setup.md) for details. The baseline arm is **Linux-only** — `sent2vec` cannot be built under MSVC — and first ran successfully on 2026-08-05; see [docs/findings/09-baseline-first-run.md](docs/findings/09-baseline-first-run.md) for that run and [docs/findings/01-baseline-reproduction.md](docs/findings/01-baseline-reproduction.md) for the history of the crash bugs.

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

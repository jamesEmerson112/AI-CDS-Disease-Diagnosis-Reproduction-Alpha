# Encoder Comparison: Bio_ClinicalBERT vs. BiomedBERT vs. BlueBERT

## Bottom line

This is the repo owner's original contribution: swapping the paper's BioSentVec baseline for
three biomedical BERT sentence encoders and re-running the same 10-fold pipeline against them.
The headline numbers — "all three hit **F1 = 1.000** at the paper's threshold of 0.6" — are real
numbers pulled straight from the committed output files, but they are not evidence that any of
these encoders diagnoses well. Two separate problems sit underneath every table in this document:

1. **The metric is not an F1 score.** Precision, recall, and F-score are numerically identical in
   every one of the 12,600 metric rows across the three committed result files (verified below,
   zero violations). The pipeline's own confusion-matrix code guarantees `TP + FP == n` for every
   fold, so `P = R = F1 = TP / n`, i.e. plain accuracy under three different column headers. Full
   derivation: [`04-metric-degeneracy.md`](04-metric-degeneracy.md).
2. **The metric saturates at 0.6.** Biomedical BERT embeddings pack diagnosis text into a narrow
   region of the embedding space — even *unrelated* diagnoses score 0.65–0.93 cosine similarity —
   and the MAX-over-Cartesian-product aggregator amplifies that further, so essentially every
   patient pair clears 0.6 regardless of whether the predicted diagnosis is right. Full
   derivation: [`03-metric-saturation.md`](03-metric-saturation.md).

Given both, the 0.6 and 0.7 rows below carry **no information about which encoder is better** —
they are within rounding distance of 1.000 for every model and every prediction strategy. The
only rows worth reading are 0.9 and 1.0, and even those measure something narrower than
"diagnostic accuracy": whether the nearest-neighbor patient's diagnosis text embeds almost
identically to the ground truth, not whether it is clinically correct. See
["What this comparison can and cannot tell you"](#what-this-comparison-can-and-cannot-tell-you)
below before citing any number in this document as a model-quality result.

All figures in this document are read directly from
[`docs/Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/`](../Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/),
[`docs/Prediction_Output_BiomedBERT_15022026_12-03-36/`](../Prediction_Output_BiomedBERT_15022026_12-03-36/),
and [`docs/Prediction_Output_BlueBERT_15022026_12-24-38/`](../Prediction_Output_BlueBERT_15022026_12-24-38/)
(`PerformanceIndex.txt`, `timing_report.txt`), cross-checked against
[`docs/score_distribution_analysis/score_distribution_summary.txt`](../score_distribution_analysis/score_distribution_summary.txt).
They supersede the equivalent tables in [`docs/bert_model_comparison.md`](../bert_model_comparison.md),
which this document draws from but corrects on dataset size and metric framing.

## The three encoders

| # | Model | HuggingFace path | Training corpus | Dim |
|---|-------|-------------------|------------------|:---:|
| 1 | Bio_ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | MIMIC-III clinical notes | 768 |
| 2 | BiomedBERT | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | PubMed abstracts | 768 |
| 3 | BlueBERT | `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` | PubMed + MIMIC-III | 768 |

(Source: the `MODELS` dict in [`src/models/bert_models.py:30-49`](../../src/models/bert_models.py),
which matches [`docs/bert_model_comparison.md`](../bert_model_comparison.md).)

All three replace BioSentVec's 700-dimensional sent2vec vectors with 768-dimensional
sentence-transformers embeddings, loaded via `SentenceTransformer(...)` with
`normalize_embeddings=False` at encode time (`bert_models.py:294`, `:337-343`). That flag does
**not** affect any result in this document: `cosine_similarity()`
([`cython_utils.py:17-32`](../../src/utils/cython_utils.py)) explicitly divides by both vector
norms, so pre-normalizing at encode time is a no-op for a cosine score either way. The original
comparison draft raised this as an open question about BiomedBERT's saturation; it is not the
cause — see [Where the encoders actually differ](#where-the-encoders-actually-differ) for the real
one.

## Method (shared with the baseline arm)

The BERT arm reuses the same retrieval-and-scoring pipeline as the baseline
([`01-baseline-reproduction.md`](01-baseline-reproduction.md) documents it in full); only the
encoder changes. In brief:

- **Symptom-level patient similarity**: for a test admission against a candidate training
  admission, take the mean of per-symptom maximum cosine similarities
  (`compute_patient_similarity_pairwise()`, `bert_models.py:94-137`), gated by
  `PRUNING_SIMILARITY = 0.5` (`Constants.py:14`).
- **Retrieval strategy**: **MAX** takes the single best-matching training patient; **TOP-K** (K ∈
  {10, 20, 30, 40, 50}) pools the K best-matching training patients and counts a hit if *any* of
  their diagnoses clears the threshold (`predict_topk_diagnoses_pure()`,
  `containGreaterOrEqualsValue()`, `bert_models.py:144-197`).
- **Diagnosis-level scoring**: MAX cosine similarity over the full Cartesian product of
  ground-truth × predicted diagnosis strings
  (`util_cy.get_diagnosis_similarity_by_description_max()`), compared against each threshold in
  `{0.6, 0.7, 0.8, 0.9, 1.0}` (`bert_models.py:466`).
- **Cross-validation**: 10 pre-computed static folds (`K_FOLD = 10`, `Constants.py:19`), 129
  admissions total, averaging ~12.9 test cases per fold. Fold 0 is 116 train / 13 test.

Dataset size is **129 admissions, not 128**. `wc -l data/raw/Symptoms-Diagnosis.txt` reports 128
only because the file has no trailing newline; `grep -c ';'` gives 129, Fold 0's 116+13 sums to
129, and the golden (10-fold-averaged) TP row for MAX at threshold 0.6 reads `12.9`, i.e.
`129 / 10` — all three of the BERT `PerformanceIndex.txt` files agree on this independently. The
score-distribution analysis confirms it a fourth way: "Total patients = 129" in
[`score_distribution_summary.txt`](../score_distribution_analysis/score_distribution_summary.txt).

## What "F1" means in these tables

Every number below labeled with a threshold is what the pipeline writes to the `FS` column of its
`PerformanceIndex.txt` output. It is not a genuine F1 score. Directly checking all three committed
result files:

```
grep pattern: 7-field aggregate rows (threshold, TP, FP, P, R, FS, PR)
Bio_ClinicalBERT: 4,200 rows, 0 have P != R or FS != P
BiomedBERT:       4,200 rows, 0 have P != R or FS != P
BlueBERT:         4,200 rows, 0 have P != R or FS != P
Total:            12,600 rows, 0 violations
```

The reason is structural, not coincidental. In
[`compute_aggregated_performance_index()`](../../src/utils/cython_utils.py) (`cython_utils.py:401-436`),
`precision = tp / (tp + fp)` and `recall = tp / nrow`. Every test case in every fold is scored as
exactly one of TP or FP — there is no abstention path in the current code — so `tp + fp == nrow`
always, which forces `precision == recall` and, since the F-score is their harmonic mean,
`F1 == precision == recall == tp / nrow`, i.e. **accuracy**. The `PR` column the code writes for
these aggregate rows (`prediction_rate = (tp + fp) / nrow`) is 1.0 in all 330 aggregate rows per
file for the same reason. (The per-test-case rows earlier in each file reuse the column header
`PR` for something different — a `1.0 / nrow` per-case weight, not a prediction rate — so it is
not 1.0 there; that is a labeling quirk of the raw output, not a second data point about the
metric.) Full derivation, including why the same problem affects any threshold you pick — not just
0.6: [`04-metric-degeneracy.md`](04-metric-degeneracy.md).

Because of this, the rest of this document reports the shared P = R = F1 value as **accuracy**
rather than repeating three identical numbers under three different headers, and rather than
calling it "F1."

## Results across thresholds 0.6–1.0

Values are 10-fold-averaged accuracy (`TP / n`, see above). BioSentVec's published numbers are
included at 0.6 only, where the original paper reports them — they were **not** reproduced against
this repo's code (see [`01-baseline-reproduction.md`](01-baseline-reproduction.md); the baseline
arm currently crashes before it can run) and are shown only as the historical reference point the
BERT arm is informally compared against.

### Threshold = 0.6 (the paper's operating point — fully saturated)

| Method | BioSentVec (published, unverified) | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:-----------------------------------:|:-----------------:|:----------:|:--------:|
| MAX | -- | 1.000 | 1.000 | 1.000 |
| TOP-10 | 0.489 | 1.000 | 1.000 | 1.000 |
| TOP-20 | 0.512 | 1.000 | 1.000 | 1.000 |
| TOP-30 | 0.521 | 1.000 | 1.000 | 1.000 |
| TOP-40 | -- | 1.000 | 1.000 | 1.000 |
| TOP-50 | -- | 1.000 | 1.000 | 1.000 |

Every BERT cell here is 1.000. This table says the three encoders are indistinguishable at 0.6 —
which is true, but only because 0.6 is below the minimum pairwise diagnosis similarity for two of
the three encoders (see [Where the encoders actually differ](#where-the-encoders-actually-differ)).
It is not evidence any BERT encoder out-diagnoses BioSentVec.

### Threshold = 0.7 (still nearly saturated)

| Method | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------:|:----------:|:--------:|
| MAX | 1.000 | 1.000 | 0.907 |
| TOP-10 | 1.000 | 1.000 | 1.000 |
| TOP-20 | 1.000 | 1.000 | 1.000 |
| TOP-30 | 1.000 | 1.000 | 1.000 |
| TOP-40 | 1.000 | 1.000 | 1.000 |
| TOP-50 | 1.000 | 1.000 | 1.000 |

The only cell that moves is BlueBERT at MAX (K = 1); pooling to TOP-10 or wider erases even that
gap.

### Threshold = 0.8 (BlueBERT starts to separate)

| Method | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------:|:----------:|:--------:|
| MAX | 0.969 | 1.000 | 0.395 |
| TOP-10 | 1.000 | 1.000 | 0.813 |
| TOP-20 | 1.000 | 1.000 | 0.907 |
| TOP-30 | 1.000 | 1.000 | 0.930 |
| TOP-40 | 1.000 | 1.000 | 0.946 |
| TOP-50 | 1.000 | 1.000 | 0.962 |

BiomedBERT is still perfectly saturated. Bio_ClinicalBERT drops fractionally, only at K = 1.

### Threshold = 0.9 (the first threshold with real signal)

| Method | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------:|:----------:|:--------:|
| MAX | 0.285 | **1.000** | 0.194 |
| TOP-10 | 0.797 | **1.000** | 0.340 |
| TOP-20 | 0.906 | **1.000** | 0.487 |
| TOP-30 | 0.930 | **1.000** | 0.517 |
| TOP-40 | 0.946 | **1.000** | 0.541 |
| TOP-50 | 0.953 | **1.000** | 0.572 |

BiomedBERT is *still* perfectly saturated at 0.9, across every K. This is the clearest single
number in the whole comparison, and it is explained, not mysterious: see below.

### Threshold = 1.0 (exact cosine match)

| Method | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------:|:----------:|:--------:|
| MAX | 0.146 | 0.169 | 0.178 |
| TOP-10 | 0.285 | 0.254 | 0.239 |
| TOP-20 | 0.331 | 0.324 | 0.339 |
| TOP-30 | 0.363 | 0.340 | 0.347 |
| TOP-40 | 0.394 | 0.371 | 0.378 |
| TOP-50 | 0.401 | 0.378 | 0.417 |

At exact match, all three encoders land in the same 0.15–0.42 range, with no consistent winner —
BlueBERT edges ahead at TOP-50 (0.417), Bio_ClinicalBERT at TOP-10/TOP-20/TOP-30. A cosine
similarity of exactly 1.0 between two float32 embeddings essentially requires the underlying
diagnosis text to be identical (or the same string via `MAX` picking out a shared diagnosis
between two patients), so this threshold is closer to a lexical exact-match rate than a semantic
one — it is not a finer-grained version of the 0.9 result.

## Where the encoders actually differ

The three encoders are not interchangeable — they differ sharply in how *compact* their embedding
space is, and that compactness is what determines where each one saturates. From
[`score_distribution_summary.txt`](../score_distribution_analysis/score_distribution_summary.txt)
(all 10,440 unique-diagnosis-pair cosine similarities, computed independently of the fold
evaluation, and cross-checked to 2e-7–9e-7 against the pipeline's own `cosine_similarity()`):

| Model | Mean pairwise similarity | Std | Min | % of diagnosis pairs >= 0.9 |
|-------|:-------------------------:|:---:|:---:|:----------------------------:|
| BiomedBERT | 0.9282 | 0.0303 | 0.7246 | 87.62% |
| Bio_ClinicalBERT | 0.8348 | 0.0450 | 0.6454 | 5.58% |
| BlueBERT | 0.7170 | 0.0652 | 0.4810 | 0.59% |

BiomedBERT's diagnosis embeddings are the most tightly clustered of the three — even the least
similar pair of diagnoses in the entire 145-diagnosis dataset scores 0.7246, and 87.62% of all
pairs already exceed 0.9 before any patient-level aggregation happens. That is a direct,
independently-verified explanation for the threshold-0.9 table above: BiomedBERT can't help but
stay saturated, because almost no diagnosis pair in the dataset embeds far enough apart to fail a
0.9 cutoff. BlueBERT sits at the opposite end — the widest spread (std 0.0652 vs. 0.0303–0.0450)
and the lowest floor (min 0.4810) — which is why it is the only model with headroom below 0.8 at
all, and the only one for which the 0.7–0.9 range carries any discriminative signal.

Read literally, "BiomedBERT wins" at every threshold up to 0.9 in the tables above. Read correctly,
BiomedBERT's embedding space is the *least* able to tell diagnoses apart, and its perfect scores
are a ceiling effect, not evidence of better retrieval. BlueBERT's lower numbers reflect a wider,
more discriminative embedding space, not worse performance. Full mechanism (including the
MAX-operator amplification that turns these pairwise numbers into the per-patient scores actually
used for classification): [`03-metric-saturation.md`](03-metric-saturation.md).

## Runtime

| Phase | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|-------|:----------------:|:----------:|:--------:|
| Model loading | 81.32s | 15.31s | 11.12s |
| Symptom embeddings | 5.04s | 2.84s | 1.49s |
| Diagnosis embeddings (145 unique) | 0.94s | 0.62s | 0.63s |
| 10-fold evaluation | 1231.52s (20.53 min) | 1248.55s (20.81 min) | 1228.11s (20.47 min) |
| **Total (this model)** | **1318.82s (21.98 min)** | **1267.33s (21.12 min)** | **1241.36s (20.69 min)** |

(Source: each model's own `timing_report.txt`.)

Sum of the three models' total execution times is **3,827.51s ≈ 63.8 minutes**. Fold-evaluation
time dominates every run and is roughly constant across encoders (~20.5–20.8 min) — it is
CPU-bound pure-Python similarity computation, not GPU/MPS-bound; only model loading and embedding
generation vary by encoder, and Bio_ClinicalBERT's 81s load time (vs. 11–15s for the other two) is
a first-download-and-no-native-sentence-transformers-config cost, not a per-run cost on subsequent
loads.

`bert_model_comparison.md` additionally cited an "Overall Pipeline: 76.4 min" figure. That number
does not appear in any `timing_report.txt` and this document cannot reconcile it: the three output
directories' own timestamps (`Prediction_Output_..._15022026_11-33-48`, `_12-03-36`, `_12-24-38`)
put the first run's start and the third run's completion (12:24:38 start + 20.69 min) about 71.5
minutes apart, and the sum of the three self-reported totals is 63.8 minutes — neither matches
76.4 minutes. Treat 76.4 as unverified; the 63.8-minute figure above is what the committed timing
reports actually support.

## What this comparison can and cannot tell you

**It can tell you:**
- The *relative compactness* of each encoder's diagnosis-text embedding space (BiomedBERT most
  compact, BlueBERT least), independently confirmed two ways — the raw pairwise-similarity
  statistics above, and the threshold at which each encoder's accuracy table stops reading 1.000.
- That any conclusion drawn from the 0.6 or 0.7 rows specifically is not a real result — those
  rows are saturated for all three encoders (BlueBERT-MAX at 0.7 excepted) and would be saturated
  for almost any encoder producing similarly compact biomedical embeddings.
- A rough sense of how the encoders order at threshold 1.0 (near-lexical exact match), where no
  single encoder dominates and the spread across TOP-K is modest (0.15–0.42).

**It cannot tell you:**
- Which encoder makes *better clinical predictions*. The metric collapses to `TP / n` (accuracy)
  with no precision/recall trade-off, no false-negative concept, and no case where a patient's
  diagnosis is scored as anything other than a binary hit-or-miss against a single similarity
  number — see [`04-metric-degeneracy.md`](04-metric-degeneracy.md).
- Whether any encoder beats the BioSentVec baseline. The baseline's 0.489/0.512/0.521 numbers were
  never regenerated against this repo's code and there is no committed baseline
  `PerformanceIndex.txt` to compare against on equal footing — see
  [`01-baseline-reproduction.md`](01-baseline-reproduction.md).
- Whether a "correct" diagnosis match at 0.9+ cosine similarity means anything clinically. Cosine
  similarity between two diagnosis-text embeddings is not validated anywhere in this repo against
  a clinician's judgment of diagnostic relatedness; it is only validated as a self-consistent
  numpy-vs-`cython_utils` computation.
- Anything about generalization beyond this specific 129-admission dataset and 10 static folds.

If the goal becomes making this comparison meaningful, the fix has to address saturation
(`03-metric-saturation.md`'s remedies: MEAN aggregation, DRG-code exact match, Hungarian matching,
or set-level Jaccard/F1 — see
[`score_distribution_analysis/next_steps.md`](../score_distribution_analysis/next_steps.md)) *and*
degeneracy (`04-metric-degeneracy.md`) together. Fixing only one leaves the other in place: a
wider-spread metric that still collapses to accuracy, or a non-degenerate metric that is still
saturated at 0.6.

## Related documents

- [`01-baseline-reproduction.md`](01-baseline-reproduction.md) — why the BioSentVec baseline
  cannot currently be run or checked against in this repo.
- [`03-metric-saturation.md`](03-metric-saturation.md) — full derivation of why threshold 0.6
  saturates, including the embedding-compactness and MAX-operator mechanisms summarized above.
- [`04-metric-degeneracy.md`](04-metric-degeneracy.md) — full derivation of why P = R = F1 at
  every threshold, not only 0.6.
- [`../bert_model_comparison.md`](../bert_model_comparison.md) — the original results write-up
  this document supersedes; its result tables are preserved above, corrected for dataset size and
  reframed away from "F1" language.
- [`../score_distribution_analysis/next_steps.md`](../score_distribution_analysis/next_steps.md) —
  proposed alternative evaluation strategies.
- [`../../README.md`](../../README.md) — project overview and reproduction instructions.

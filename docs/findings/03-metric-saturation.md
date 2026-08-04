# Metric saturation: why every BERT model scores F1 = 1.000 at threshold 0.6

## Conclusion

All three BERT arms — Bio_ClinicalBERT, BiomedBERT, BlueBERT — report **F1 = 1.000 at the
paper's threshold of 0.6**, across every TOP-K prediction cut the pipeline evaluates
(TOP-10 through TOP-50). This is visible directly in the committed results, e.g. the
10-fold TOP-10 block for Bio_ClinicalBERT:

```
0.6	12.9	0.0	1.0	1.0	1.0	1.0
```
(`TP FP P R FS PR`, from
[`docs/Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/PerformanceIndex.txt`](../Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/PerformanceIndex.txt))

A perfect score is not evidence the BERT extension out-diagnoses BioSentVec (published
0.489–0.521 at the same threshold). It is an artifact of two compounding properties of the
evaluation pipeline:

1. **Embedding-space compactness** — biomedical BERT models map diagnosis text into a narrow
   region of the embedding space, so even *unrelated* diagnoses score a high cosine similarity.
2. **MAX-operator amplification** — the scoring function takes the single best-matching pair out
   of the full Cartesian product of ground-truth × predicted diagnosis descriptions, which pushes
   the effective score higher still.

Together, these put essentially 100% of patient pairs above 0.6 regardless of whether the
predicted diagnoses have anything to do with the true ones. The number the pipeline reports is
real; what it measures is not diagnostic accuracy.

All figures below come from `scripts/analyze_score_distributions.py`, whose full output is
[`docs/score_distribution_analysis/score_distribution_summary.txt`](../score_distribution_analysis/score_distribution_summary.txt).
The script re-implements the scoring path in numpy and spot-checks it against the pipeline's own
`cython_utils.cosine_similarity()`; the two agree to within 2e-7–9e-7 (float32 rounding), so the
numbers below describe the actual production scoring function, not an approximation of it.

**This is a different problem from the one in [04-metric-degeneracy.md](04-metric-degeneracy.md).**
Saturation means the 0.6 threshold is too lenient for these embeddings — almost every score clears
it. Degeneracy means precision, recall, and F1 collapse to the same value (`TP/n`) **at every
threshold**, because each test case can only ever be scored as a single TP-or-FP, leaving one
degree of freedom no matter where the threshold sits. Every fix described below removes
saturation without touching degeneracy: raise the threshold, change the aggregator, do whatever
you like, and P, R, and F1 will still be forced equal, and the `PR` column will still read 1.0.
Fixing saturation makes the number at a given threshold discriminate between models; fixing
degeneracy is what is required before that number can honestly be called an F1 score at all. They
have to be fixed together, and fixing one is not a substitute for the other.

## Cause 1: embedding-space compactness

Each model embeds all 145 unique diagnosis descriptions in the dataset and every pair (excluding
self-pairs, N = 10,440) is scored by cosine similarity. All three models push the bulk of that
mass above 0.6 before any patient-level aggregation happens at all:

| Model | Mean | Median | Std | Min | % ≥ 0.6 | % ≥ 0.7 | % ≥ 0.8 | % ≥ 0.9 |
|---|---|---|---|---|---|---|---|---|
| Bio_ClinicalBERT | 0.8348 | 0.8371 | 0.0450 | 0.6454 | 100.00% | 99.55% | 79.26% | 5.58% |
| BiomedBERT | 0.9282 | 0.9341 | 0.0303 | 0.7246 | 100.00% | 100.00% | 99.32% | 87.62% |
| BlueBERT | 0.7170 | 0.7176 | 0.0652 | 0.4810 | 96.35% | 60.98% | 9.02% | 0.59% |

(Source: `score_distribution_summary.txt`, Section 1.)

Two things stand out. First, the *minimum* pairwise similarity across all 10,440 diagnosis pairs
is 0.6454 for Bio_ClinicalBERT and 0.7246 for BiomedBERT — meaning **no two diagnoses in the
entire dataset embed further apart than the paper's 0.6 threshold** for those models, whether or
not they are clinically related. BlueBERT is the outlier, with a noticeably wider spread (std
0.0652 vs. 0.0303–00450) and a minimum of 0.4810, which is why it is the only model where
saturation is incomplete even before the MAX operator is applied. Second, BiomedBERT is the most
compact by a wide margin: mean 0.9282, with 87.62% of all diagnosis pairs already above 0.9 before
any patient-level aggregation.

## Cause 2: MAX-operator amplification

The scoring function actually used at inference time,
`get_diagnosis_similarity_by_description_max()` in `src/utils/cython_utils.py` (line 291), takes
the ground-truth diagnosis list for one patient and the predicted diagnosis list for another, and
returns the single **maximum** cosine similarity over the full Cartesian product:

```python
def get_diagnosis_similarity_by_description_max(embendings_diagnosis, gt_diagnosis, predicted_diagnosis, method):
    max_similarity = MIN_SIMILARITY
    for x in gt_diagnosis:
        x_description = x[x.index(':') + 1:len(x)]
        for y in predicted_diagnosis:
            y_description = y[y.index(':') + 1:len(y)]
            emb_diagnosis_to_predict = embendings_diagnosis.get(x_description)
            emb_diagnosis_predicted = embendings_diagnosis.get(y_description)
            diagnosis_similarity = cosine_similarity(emb_diagnosis_to_predict[0], emb_diagnosis_predicted[0])
            if diagnosis_similarity > max_similarity:
                max_similarity = diagnosis_similarity
    return max_similarity
```

Patients in this dataset average 1.74 diagnoses each (min 1, max 3, 129 patients, 145 unique
descriptions — `score_distribution_summary.txt`, diagnosis-count block), so a typical patient pair
contributes roughly a 1.74 × 1.74 ≈ 3-cell Cartesian product. Taking the max over even a handful
of already-compact similarities is a one-sided operation: it can only push the score up from the
pairwise mean, never down. Simulating this over all 129 × 128 = 16,512 ordered patient pairs shows
exactly that shift:

| Model | Pairwise mean → Per-patient MAX mean | % ≥ 0.6 | % ≥ 0.7 | % ≥ 0.8 | % ≥ 0.9 | % ≥ 1.0 |
|---|---|---|---|---|---|---|
| Bio_ClinicalBERT | 0.8348 → 0.8586 | 100.00% | 100.00% | 93.81% | 11.63% | 1.49% |
| BiomedBERT | 0.9282 → 0.9447 | 100.00% | 100.00% | 100.00% | 99.71% | 1.62% |
| BlueBERT | 0.7170 → 0.7565 | 99.96% | 84.16% | 18.87% | 2.47% | 1.31% |

(Source: `score_distribution_summary.txt`, Section 2.)

At threshold 0.6 — the paper's own operating point — all three models are effectively saturated:
100.00%, 100.00%, and 99.96% of all 16,512 possible patient pairs clear it. Even at 0.7, two of
three models are still at 100.00%. Note also that 1.3–1.6% of patient pairs hit an *exact* 1.0000
similarity even though the underlying diagnosis text differs — a direct consequence of MAX
picking out shared or near-duplicate diagnosis strings (e.g. two patients who share one identical
diagnosis code but differ on the other) rather than any property of the model's semantic
resolution.

### The pipeline's own K compounds this further

`analyze_score_distributions.py` measures the single-diagnosis-list-vs-single-diagnosis-list case.
The production pipeline's TOP-K prediction modes make the Cartesian product larger still, because
the predicted set is drawn from the K nearest training patients' diagnoses pooled together, not
from a single patient. The committed `PerformanceIndex.txt` files show the effect directly. The
strictest mode the pipeline evaluates — K = 1, labeled `MAX SIMILARITY by MAX` in the output — is
already fully saturated at 0.6–0.7 for every model, and gives the clearest look at where each
model's saturation ceiling actually is:

| Model | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|
| Bio_ClinicalBERT | 1.000 | 1.000 | 0.969 | 0.285 | 0.146 |
| BiomedBERT | 1.000 | 1.000 | 1.000 | 1.000 | 0.169 |
| BlueBERT | 1.000 | 0.907 | 0.395 | 0.194 | 0.178 |

(F1 values, 10-fold averages, K = 1 nearest neighbor; sources:
[`Bio_ClinicalBERT`](../Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/PerformanceIndex.txt),
[`BiomedBERT`](../Prediction_Output_BiomedBERT_15022026_12-03-36/PerformanceIndex.txt),
[`BlueBERT`](../Prediction_Output_BlueBERT_15022026_12-24-38/PerformanceIndex.txt), each at line
~5921, `10-FOLD PERFORMANCE INDEX of MAX SIMILARITY by MAX`.)

This ordering — BiomedBERT most saturated, BlueBERT least — exactly matches the compactness
ranking measured offline. But once K grows to 10, saturation gets worse, not better: the TOP-10
block in the same files shows Bio_ClinicalBERT and BiomedBERT both still at F1 = 1.000 at 0.8, and
even BlueBERT — the one model with real headroom at K = 1 — jumps from 0.395 to 0.813 at 0.8
simply because pooling ten candidate patients' diagnoses gives the MAX operator ten times as many
chances to find a high-scoring pair. The paper's actual evaluation runs TOP-10 through TOP-50, so
production results are evaluated at a saturation level at least as bad as the K = 1 numbers above,
and generally worse.

See also:
[`score_distributions.png`](../score_distribution_analysis/score_distributions.png) — histograms
and CDFs of the all-pairwise distributions in the first table above — and
[`per_patient_max_distributions.png`](../score_distribution_analysis/per_patient_max_distributions.png)
— histograms and a threshold-sensitivity curve for the per-patient MAX distributions in the second
table.

## What "F1 = 1.000" actually reflects

Put together: the embeddings are compact enough that almost no diagnosis pair in the dataset
scores below 0.6 to begin with (Cause 1), and the MAX-over-Cartesian-product aggregator — applied
once per predicted diagnosis list, and again implicitly by pooling K candidate patients — turns
"almost no pair" into "effectively zero pairs" (Cause 2). The reported F1 = 1.000 therefore
reflects the geometry of the embedding space and the leniency of the aggregator, not the model's
ability to retrieve clinically similar patients or predict correct diagnoses. It is a ceiling
effect, not a result.

## Proposed remedies

`docs/score_distribution_analysis/next_steps.md` lays out four alternative evaluation strategies,
ranked by impact-to-complexity ratio. Summarized:

| # | Strategy | What changes | Complexity |
|---|---|---|---|
| A | **MEAN aggregation** instead of MAX | Replace the running max in `get_diagnosis_similarity_by_description_max()` with a running sum divided by pair count; add a parallel call site in `bert_models.py` | Low — one function, no new dependencies |
| B | **DRG-code exact match** | Parse the `HCFA:`/`APR:`/`MS:` prefixes already present in `data/raw/Symptoms-Diagnosis.txt` and score exact code equality instead of cosine similarity | Low–Medium — straightforward parsing, but needs care around multi-code patients (`APR:...--HCFA:...`) |
| C | **Hungarian optimal matching** | Build a per-patient-pair cost matrix and use `scipy.optimize.linear_sum_assignment` for a 1-to-1 assignment instead of a single best pair | Medium — new function, unequal-set-size handling |
| D | **Set-level Jaccard / F1** | Treat ground-truth and predicted diagnoses as multi-label sets and score set overlap directly | Medium — the metric itself is simple, but it needs a match criterion borrowed from A or B, and integration with the existing fold loop |

Strategy A is the only one of the four that is a genuine one-line swap inside the existing scoring
function — it directly targets Cause 2 above and needs no new data handling or dependency. B
through D are real surgery: B requires new parsing logic against a data format that already has
edge cases (multi-DRG patients), and C and D both require restructuring the per-patient scoring
call site in `src/models/bert_models.py:542` rather than editing a single function in place. None
of A–D touches the degeneracy problem in
[04-metric-degeneracy.md](04-metric-degeneracy.md) — that requires changing what a "test case" is
counted as, not how similarity is aggregated.

## Reproducing this analysis

```bash
conda activate disease-diagnosis
python scripts/analyze_score_distributions.py
```

This regenerates `docs/score_distribution_analysis/score_distributions.png`,
`per_patient_max_distributions.png`, and `score_distribution_summary.txt` in place. It does not
touch any prediction pipeline or write to `Prediction_Output_*`; it only re-embeds the 145 unique
diagnosis descriptions and replays the scoring math offline.

# Metric saturation: why every BERT model scores F1 = 1.000 at threshold 0.6

> **In plain words.** The paper's scoring rule says a predicted diagnosis counts as correct if
> its similarity to the true diagnosis is at least 0.6, on a 0-to-1 scale. The problem: to a
> biomedical BERT model, *all* medical phrases look alike. In this dataset, even completely
> unrelated diagnoses score above 0.68 for one of the three models and above 0.59 for a second —
> meaning the 0.6 bar sits at or **below the lowest score any pair can produce**. (Re-measured
> 2026-08-12 under `corrected`: this sentence used to say "above 0.64 for two of the three", which
> was the `legacy` measurement. The 0.6 bar is now clear of *every* pair for BiomedBERT alone; for
> Bio_ClinicalBERT two pairs of 10,440 fall below it.) Everything passes. Every model gets a perfect
> 1.000, and the perfect score means nothing: it is a limbo contest with the bar lying on the
> ground. Two things stack to cause it: the models crowd every diagnosis into a small corner of
> their embedding space, and the scorer then takes the single *best*-matching pair of (true,
> predicted) diagnoses, which pushes scores higher still. The threshold knob this finding
> complains about was later removed entirely — see [12](12-drg-grader.md).

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
   of the full Cartesian product of ground-truth × predicted diagnosis descriptions (every true
   diagnosis crossed with every predicted one), which pushes the effective score higher still.

Together, these put essentially 100% of patient pairs above 0.6 regardless of whether the
predicted diagnoses have anything to do with the true ones. The number the pipeline reports is
real; what it measures is not diagnostic accuracy.

All figures below come from `scripts/analyze_score_distributions.py`, whose full output is
[`docs/score_distribution_analysis/score_distribution_summary.txt`](../score_distribution_analysis/score_distribution_summary.txt).

> **RE-MEASURED 2026-08-12 (TODO P37). Every table below now carries `corrected` numbers; the
> `legacy` ones it used to carry are kept beside them.** Two things changed at once and they must
> not be confused.
>
> 1. **The script stopped simulating the grader.** It used to re-derive the scoring path in its own
>    numpy cosine — a fourth cosine implementation in this repository — and imported no
>    `PipelineConfig` at all. Section 2 now calls `cython_utils.get_diagnosis_relevance(..., config)`,
>    the same dispatch point both arms grade with, and Section 1 calls
>    `cython_utils.cosine_similarity`. **The old spot-check said the two agreed to 2e-7–9e-7 and
>    that was true of the kernel — but not of the answer at threshold 1.0.** For a vector against
>    itself the shipped kernel returns exactly 1.0 in 2000 of 2000 trials while numpy
>    normalise-then-dot does so in only 1356, so the simulated path *undercounted exact matches*.
>    Corrected below.
> 2. **The measurement moved off `legacy` text handling.** The canonical artifact is now measured
>    under `--pipeline corrected`. Note that `fold_dir` is **not read** by this analysis — it scores
>    every ordered patient pair, not fold test cases — so a corrected-vs-legacy delta here is a
>    *preprocessing* delta and contains no part of the patient-leakage effect.
>
> **The claim this paragraph used to make — "saturation persists identically under `corrected`" —
> was an assertion, and is now a measurement. It persists; "identically" was too strong.** At the
> paper's 0.6 threshold the three arms move 100.00 → 100.00, 100.00 → 100.00 and 99.96 → 99.65;
> the largest movement in either table is Bio_ClinicalBERT's all-pairwise ≥ 0.8 row, 79.26% →
> 72.95% (6.31 points), with its per-patient ≥ 0.8 row second at 93.81% → 89.47%.
>
> The mechanism is untouched and the document's conclusion — that F1 = 1.000 at 0.6 is a ceiling
> effect rather than a result — is untouched. **Two supporting claims did change**, and both are
> marked where they appear: the "no two diagnoses embed further apart than 0.6" observation now
> holds for BiomedBERT alone, and the exact-1.0 share was not stale but *wrong*.

**This is a different problem from the one in [04-metric-degeneracy.md](04-metric-degeneracy.md),
and fixing one does not fix the other.** Saturation means the 0.6 threshold is too lenient for
these embeddings — almost every score clears it. Degeneracy means precision, recall, and F1
collapse to the same value (`TP/n`) **at every threshold**, because each test case can only ever
be scored as a single TP-or-FP. Raise the threshold, change the aggregator, do whatever you like:
P, R, and F1 will still be forced equal, and the `PR` column will still read 1.0. Fixing
saturation makes the number at a given threshold discriminate between models; fixing degeneracy
is what would be required before that number could honestly be called an F1 score at all.

## Cause 1: embedding-space compactness

Each model embeds all 145 unique diagnosis descriptions in the dataset, and every pair (excluding
self-pairs, N = 10,440) is scored by cosine similarity. All three models push the bulk of that
mass above 0.6 before any patient-level aggregation happens at all:

**`corrected` — the canonical measurement, 2026-08-12:**

| Model | Mean | Median | Std | Min | % ≥ 0.6 | % ≥ 0.7 | % ≥ 0.8 | % ≥ 0.9 |
|---|---|---|---|---|---|---|---|---|
| Bio_ClinicalBERT | 0.8253 | 0.8283 | 0.0489 | 0.5856 | 99.98% | 98.43% | 72.95% | 4.59% |
| BiomedBERT | 0.9267 | 0.9348 | 0.0338 | 0.6834 | 100.00% | 99.98% | 98.95% | 84.89% |
| BlueBERT | 0.7160 | 0.7164 | 0.0648 | 0.4736 | 96.15% | 60.73% | 8.40% | 0.71% |

**`legacy` — what this table used to hold, retained for comparison:**

| Model | Mean | Median | Std | Min | % ≥ 0.6 | % ≥ 0.7 | % ≥ 0.8 | % ≥ 0.9 |
|---|---|---|---|---|---|---|---|---|
| Bio_ClinicalBERT | 0.8348 | 0.8371 | 0.0450 | 0.6454 | 100.00% | 99.55% | 79.26% | 5.58% |
| BiomedBERT | 0.9282 | 0.9341 | 0.0303 | 0.7246 | 100.00% | 100.00% | 99.32% | 87.62% |
| BlueBERT | 0.7170 | 0.7176 | 0.0652 | 0.4810 | 96.35% | 60.98% | 9.02%¹ | 0.59% |

(Source: `score_distribution_summary.txt`, Section 1 — the committed file now holds the
`corrected` numbers and names that pipeline in its header. ¹ Re-running `legacy` through the fixed
code path reproduced its own table to the digit except for **three** boundary cells — all of them
BlueBERT's, all moved by float32 rounding rather than by anything structural: this one,
9.02% → 9.03%; its per-patient P25 in the next table, 0.7163 → 0.7162; and its per-patient
% ≥ 0.8, 18.87% → 18.88%, marked ² below. The three ≥ 1.0 cells moved too, but those are the
grader correction described further down, not rounding.
*Corrected 2026-08-12: this note read "two boundary cells" and omitted the third, which had been
substituted into the Section 2 table unmarked instead of recorded here. Verified by re-running
`--pipeline legacy` and diffing against the committed summary at `34d6d43`.*)

Two things stand out, and one of them **changed under `corrected`**. First, the *minimum* pairwise
similarity across all 10,440 diagnosis pairs. Under `legacy` it was 0.6454 for Bio_ClinicalBERT and
0.7246 for BiomedBERT, which supported a striking claim: **no two diagnoses in the entire dataset
embed further apart than the paper's 0.6 threshold** for those models. **That claim is now true of
BiomedBERT only** — under `corrected` Bio_ClinicalBERT's minimum falls to 0.5856 and 0.02% of its
pairs (2 of 10,440) sit below 0.6. The correct statement is that BiomedBERT still has no pair below
0.6 (min 0.6834) and Bio_ClinicalBERT has essentially none. BlueBERT remains the outlier: the
widest spread (std 0.0648 vs 0.0338–0.0489) and a minimum of 0.4736, the only model where
saturation is incomplete before the MAX operator is applied. Second — unchanged — BiomedBERT is the
most compact by a wide margin: mean 0.9267, with 84.89% of all diagnosis pairs already above 0.9
before any patient-level aggregation.

**One new row appears under `corrected`: `Max = 1.0000`, `% ≥ 1.0 = 0.01%`.** That is not an
encoder property. Under `corrected` the *encoded* text is preprocessed (`bert_models`' FIX 4), and
exactly one pair of distinct raw descriptions preprocesses to the same string — `"respiratory
system diagnosis w/ ventilator support 96+ hours"` and `"...w ventilator support 96+ hours"` —
so their vectors are identical and their cosine is exactly 1.0. It is 1 pair of 10,440 for all
three models, which is why the figure is the same 0.01% in all three rows. Under `legacy` the
encoder sees raw text, all 145 descriptions are distinct strings, and the row reads 0.00%.
Measured the same day: `legacy` preprocessing produces **two** such collisions rather than one,
the extra being `w/o extensive procedure` collapsing onto `w extensive procedure` — finding
[06](06-preprocessing-defects.md)'s headline defect, visible here as a geometry artifact. It does
not reach the `legacy` numbers only because `legacy` never hands preprocessed text to the encoder.

## Cause 2: MAX-operator amplification

The scoring function actually used at inference time,
`get_diagnosis_similarity_by_description_max()` (now in `src/aicds/utils/cython_utils.py`; this
document predates the package move), takes the ground-truth diagnosis list for one patient and
the predicted diagnosis list for another, and returns the single **maximum** cosine similarity
over the full Cartesian product:

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
descriptions — `score_distribution_summary.txt`, diagnosis-count block), so a typical patient
pair contributes roughly a 1.74 × 1.74 ≈ 3-cell Cartesian product. Taking the max over even a
handful of already-compact similarities is a one-sided operation: it can only push the score up
from the pairwise mean, never down. Calling that function — the real one, once per pair, since
2026-08-12 — over all 129 × 128 = 16,512 ordered patient pairs shows exactly that shift:

**`corrected` — the canonical measurement, 2026-08-12:**

| Model | Pairwise mean → Per-patient MAX mean | % ≥ 0.6 | % ≥ 0.7 | % ≥ 0.8 | % ≥ 0.9 | % ≥ 1.0 |
|---|---|---|---|---|---|---|
| Bio_ClinicalBERT | 0.8253 → 0.8502 | 100.00% | 100.00% | 89.47% | 9.52% | **1.89%** |
| BiomedBERT | 0.9267 → 0.9463 | 100.00% | 100.00% | 100.00% | 99.06% | **1.89%** |
| BlueBERT | 0.7160 → 0.7564 | 99.65% | 82.92% | 19.45% | 2.69% | **1.89%** |

**`legacy` — what this table used to hold, retained for comparison:**

| Model | Pairwise mean → Per-patient MAX mean | % ≥ 0.6 | % ≥ 0.7 | % ≥ 0.8 | % ≥ 0.9 | % ≥ 1.0 |
|---|---|---|---|---|---|---|
| Bio_ClinicalBERT | 0.8348 → 0.8586 | 100.00% | 100.00% | 93.81% | 11.63% | ~~1.49%~~ 1.89% |
| BiomedBERT | 0.9282 → 0.9447 | 100.00% | 100.00% | 100.00% | 99.71% | ~~1.62%~~ 1.89% |
| BlueBERT | 0.7170 → 0.7565 | 99.96% | 84.16% | 18.87%² | 2.47% | ~~1.31%~~ 1.89% |

(Source: `score_distribution_summary.txt`, Section 2. The struck values are what the *simulated*
grader reported; the corrections are the shipped grader's answer on the same `legacy` inputs — see
the ≥ 1.0 paragraph below. ² The third boundary cell of footnote ¹: the `legacy` re-run reports
**18.88%** here. The committed `legacy` value is kept in the table, as 9.02% is in Section 1, because
this table is what the document *used to hold*. *Corrected 2026-08-12: the cell read an unmarked
18.88%, the only cell in either `legacy` table not matching the committed summary at `34d6d43`.*)

At threshold 0.6 — the paper's own operating point — all three models remain effectively saturated:
100.00%, 100.00%, and 99.65% of all 16,512 possible patient pairs clear it. Even at 0.7, two of
three models are still at 100.00%.

**The ≥ 1.0 column was wrong, in a way worth understanding, and it is the clearest thing P37 found.**
It used to read 1.49% / 1.62% / 1.31%, three different values that invited an encoder-level reading:
*this model resolves near-duplicates more finely than that one*. The true value is **1.89% for all
three**, and it cannot be anything else: two patients score exactly 1.0 iff they share a diagnosis
description, in which case the two lookups return **the same vector**, and
`cython_utils.cosine_similarity` returns exactly 1.0 for a vector against itself. Counted directly
from the data with no model involved, **312 of the 16,512 ordered patient pairs share a description
— 1.89%.** The old spread was an artifact of the simulated scorer: normalising in float32 and then
taking a dot product returns exactly 1.0 for a vector against itself in only **1,356 of 2,000**
random trials (the shipped kernel: 2,000 of 2,000), so the simulation dropped a different arbitrary
subset of the 312 for each model. The three-way tie is the measurement; the spread was rounding.

**A consequence worth stating, because it corroborates [finding 12](12-drg-grader.md) from a
different direction.** That same 312/16,512 = 1.89% is *also* what `--pipeline drg` returns for the
fraction of ordered patient pairs it scores 1.0 (measured: mean relevance 0.0189 = 312/16,512, on
every threshold row and identical across all three models, because the DRG grader consults no
embedding at all). So at
threshold 1.0 the cosine grader and the DRG grader are counting the same pairs — which is exactly
why finding 12's four arms reproduced their threshold-1.0 cosine numbers *bit-exactly* under the
DRG grader. That result was reported as an empirical surprise; this is the mechanism behind it.

### The pipeline's own K compounds this further

`analyze_score_distributions.py` measures the one-diagnosis-list-vs-one-diagnosis-list case. The
production pipeline's TOP-K prediction modes make the Cartesian product larger still, because the
predicted set is pooled from the K nearest training patients' diagnoses, not drawn from a single
patient. The committed `PerformanceIndex.txt` files show the effect directly. The strictest mode
the pipeline evaluates — K = 1, labeled `MAX SIMILARITY by MAX` in the output — is already fully
saturated at 0.6–0.7 for every model, and gives the clearest look at where each model's
saturation ceiling actually sits:

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
block in the same files shows Bio_ClinicalBERT and BiomedBERT both still at F1 = 1.000 at 0.8,
and even BlueBERT — the one model with real headroom at K = 1 — jumps from 0.395 to 0.813 at 0.8,
simply because pooling ten candidate patients' diagnoses gives the MAX operator ten times as many
chances to find a high-scoring pair. The paper's actual evaluation runs TOP-10 through TOP-50, so
production results are evaluated at a saturation level at least as bad as the K = 1 numbers
above, and generally worse.

See also:
[`score_distributions.png`](../score_distribution_analysis/score_distributions.png) — histograms
and CDFs of the all-pairwise distributions in the first table above — and
[`per_patient_max_distributions.png`](../score_distribution_analysis/per_patient_max_distributions.png)
— histograms and a threshold-sensitivity curve for the per-patient MAX distributions in the
second table.

## What "F1 = 1.000" actually reflects

Put together: the embeddings are compact enough that almost no diagnosis pair in the dataset
scores below 0.6 to begin with (Cause 1), and the MAX-over-Cartesian-product aggregator —
applied once per predicted diagnosis list, and again implicitly by pooling K candidate patients
— turns "almost no pair" into "effectively zero pairs" (Cause 2). The reported F1 = 1.000
therefore reflects the geometry of the embedding space and the leniency of the aggregator, not
the model's ability to retrieve clinically similar patients or predict correct diagnoses. It is
a ceiling effect, not a result.

## Proposed remedies — status as of 2026-08-08

This section was written as a menu of future options. One of them has since shipped, which
changes how the list should be read:

| # | Strategy | What changes | Status |
|---|---|---|---|
| A | **MEAN aggregation** instead of MAX | Replace the running max in `get_diagnosis_similarity_by_description_max()` with a running sum divided by pair count | Not implemented. Note a mean-based aggregator would reintroduce a hash-order nondeterminism — see the ULP warning in [12](12-drg-grader.md) before attempting it |
| B | **DRG-code exact match** | Score exact label equality instead of cosine similarity | **SHIPPED as `--pipeline drg`** (`grader="drg-exact"`, [finding 12](12-drg-grader.md)). It removes the threshold knob entirely — the five threshold rows collapse to one — so saturation does not arise under it at all |
| C | **Hungarian optimal matching** | Build a per-patient-pair cost matrix and use `scipy.optimize.linear_sum_assignment` for a 1-to-1 assignment instead of a single best pair | Not implemented |
| D | **Set-level Jaccard / F1** | Treat ground-truth and predicted diagnoses as multi-label sets and score set overlap directly | Not implemented — tracked as **P6** in `docs/plans/correctness-fixes.md`, still the more complete fix |

Rank-aware metrics (MRR, Precision@K — [finding 13](13-rank-aware-metrics.md)) also landed since
this was written; they remove the *K* knob the last section complains about. None of A–D touches
the degeneracy problem in [04-metric-degeneracy.md](04-metric-degeneracy.md) — that would require
changing what a "test case" is counted as, not how similarity is aggregated.

## Reproducing this analysis

```bash
conda activate disease-diagnosis
python scripts/analyze_score_distributions.py --pipeline corrected     # the committed artifact
python scripts/analyze_score_distributions.py --pipeline legacy --out /tmp/sd_legacy --no-plots
python scripts/analyze_score_distributions.py --pipeline drg    --out /tmp/sd_drg    --no-plots
```

The first command regenerates `docs/score_distribution_analysis/score_distributions.png`,
`per_patient_max_distributions.png`, and `score_distribution_summary.txt` in place. It does not
touch any prediction pipeline or write to `Prediction_Output_*`; it only re-embeds the 145 unique
diagnosis descriptions and replays the scoring math offline — against the *shipped* grader, since
2026-08-12.

**Always pass `--out` for a pipeline whose numbers are not the committed ones.** The three
artifacts have fixed names, so a second pipeline written into the default directory silently
replaces the canonical set with numbers nobody quotes. The header inside the summary names the
pipeline, and both consumers (`build_readme_plots.py`, `build_dashboard_data.py`) now carry that
name through — into the saturation chart's title and into `meta.saturationPipeline` — so a swap is
visible rather than silent, but it is still a swap.

The caveat that used to close this section — that the script re-implements the grader against its
own cosine and imports no config, so it keeps generating saturation evidence for a retired ruler —
**was TODO P37 and is closed as of 2026-08-12.** What replaces it is narrower and still worth
knowing: under `--pipeline drg` there is no distribution to plot. Relevance is 0/1, every threshold
row reads the same 1.89%, and all three models are identical by construction. Saturation is a
property of the cosine grader, not of the encoders.

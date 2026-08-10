# Encoder comparison: BioSentVec vs. Bio_ClinicalBERT vs. BiomedBERT vs. BlueBERT

> **In plain words.** This document compares the paper's original text model (BioSentVec, a 2019
> model that stores one fixed vector per word/word-pair in a 21 GB table) against three modern
> BERT models (which compute a vector from context, from a ~420 MB file). It was written when
> only the three BERT models had ever produced results here, and its job was to explain why
> their perfect-looking scores could not be trusted and what a fair four-way comparison would
> require. Almost everything it demanded has since been done — the baseline ran, the data split
> was fixed, the text handling was unified, the grader was replaced — and the corrected results
> live in [11](11-corrected-pipeline-first-results.md), [12](12-drg-grader.md) and
> [13](13-rank-aware-metrics.md). Read this document for the *legacy* analysis: what the
> February 2026 BERT numbers actually measured, and the four defects found underneath them.

> **Status, 2026-08-08.** Superseded in part. The tables below are the committed legacy
> (February 2026) results and remain the correct record of that pipeline. But the blocking
> claims are resolved: the baseline **ran** on 2026-08-05 ([09](09-baseline-first-run.md),
> reproducing the published TOP-10 figure to within 0.007), the folds are regrouped by patient
> ([05](05-patient-leakage.md)), preprocessing is unified under `corrected`
> ([06](06-preprocessing-defects.md)), and the metric knobs are gone
> ([12](12-drg-grader.md), [13](13-rank-aware-metrics.md)). **For any cross-arm claim, quote the
> corrected numbers, not the tables here.** Path note: this document predates the
> `src/aicds` package move; file paths and line numbers below have drifted — go by function
> name.

## Bottom line

Four sentence encoders are named in this project. The paper's own encoder, **BioSentVec**, is a
sent2vec model — a shallow, non-contextual, log-linear sentence embedder. The repo owner's
contribution swaps it for three biomedical **BERT** encoders behind the same 10-fold pipeline.
The headline numbers — "all three BERT models hit **F1 = 1.000** at the paper's threshold of
0.6" — are real numbers pulled straight from the committed output files, but they are not
evidence that any of these encoders diagnoses well, and they are not evidence that any of them
beats BioSentVec.

Four separate problems sit underneath everything in this document. The first two are properties
of the metric; the second two are properties of the comparison itself.

1. **The metric is not an F1 score.** Precision, recall, and F-score are numerically identical
   in every one of the 12,600 metric rows across the three committed BERT result files (verified
   below, zero violations). The confusion-matrix code guarantees `TP + FP == n` for every BERT
   fold, so `P = R = F1 = TP / n`, i.e. plain accuracy under three different column headers.
   Full derivation: [`04-metric-degeneracy.md`](04-metric-degeneracy.md).
2. **The metric saturates at 0.6.** Biomedical BERT embeddings pack diagnosis text into a narrow
   region of the embedding space — even *unrelated* diagnoses score 0.65–0.93 cosine similarity
   — and the MAX-over-Cartesian-product aggregator amplifies that further, so essentially every
   patient pair clears 0.6 regardless of whether the predicted diagnosis is right. Full
   derivation: [`03-metric-saturation.md`](03-metric-saturation.md).
3. **Only three of the four encoders had been measured when this was written.** There was no
   committed BioSentVec `PerformanceIndex.txt` anywhere in the repository, and the
   0.489 / 0.512 / 0.521 figures were an inherited citation with no artifact behind them.
   **Since resolved:** the baseline ran end-to-end on 2026-08-05 and reproduced TOP-10 to within
   0.007 ([09-baseline-first-run.md](09-baseline-first-run.md)). See
   [The BioSentVec numbers, and what they are not](#the-biosentvec-numbers-and-what-they-are-not)
   for what the citation was before that run existed.
4. **Even the three-way BERT comparison is inside its own noise floor.** Fold-to-fold standard
   deviation at the only non-saturated threshold is 0.071–0.124; the largest gap between any two
   encoders at that threshold is 0.046. The ranking flips depending on K. And a third of the
   test set retrieves the same patient's own prior admission, worth +0.11 to +0.26 — an effect
   several times larger than anything separating the encoders. See
   [The differences are inside the noise floor](#the-differences-are-inside-the-noise-floor) and
   [Patient leakage inflates every arm](#patient-leakage-inflates-every-arm).

Given all four, the 0.6 and 0.7 rows below carry **no information about which encoder is
better**, the 0.8–1.0 rows carry less than they appear to, and the BioSentVec column that a
reader expects to find is deliberately not in the tables. See
["What this comparison can and cannot tell you"](#what-this-comparison-can-and-cannot-tell-you)
before citing any number in this document as a model-quality result.

All BERT figures in this document are read directly from
[`docs/Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/`](../Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/),
[`docs/Prediction_Output_BiomedBERT_15022026_12-03-36/`](../Prediction_Output_BiomedBERT_15022026_12-03-36/),
and [`docs/Prediction_Output_BlueBERT_15022026_12-24-38/`](../Prediction_Output_BlueBERT_15022026_12-24-38/)
(`PerformanceIndex.txt`, `timing_report.txt`), cross-checked against
[`docs/score_distribution_analysis/score_distribution_summary.txt`](../score_distribution_analysis/score_distribution_summary.txt).

## Provenance: what was actually measured for each encoder

This table is the most important one in the document, and it is deliberately placed before any
results. It reflects the state **at the time of writing**; the BioSentVec row changed on
2026-08-05.

| Encoder | Committed artifact in this repo | Runnable in this checkout? | Status of its published numbers |
|---|---|---|---|
| **BioSentVec** | **none** (run output is DUA-covered and gitignored; the aggregates are recorded in [09](09-baseline-first-run.md)) | **Yes, since `c2fee6e` — Linux only** (sent2vec will not build under MSVC; needs the 21 GB model file) | **Reproduced 2026-08-05**: legacy TOP-10 = 0.4824 vs published 0.489 |
| Bio_ClinicalBERT | `PerformanceIndex.txt` + `timing_report.txt`, 15 Feb 2026 | Yes — `python scripts/run_bert_analysis.py --model 1` | Measured, reproducible |
| BiomedBERT | `PerformanceIndex.txt` + `timing_report.txt`, 15 Feb 2026 | Yes — `--model 2` | Measured, reproducible |
| BlueBERT | `PerformanceIndex.txt` + `timing_report.txt`, 15 Feb 2026 | Yes — `--model 3` | Measured, reproducible |

The absence of a BioSentVec artifact was checked three ways at the time: `find . -iname
"*PerformanceIndex*"` outside `.git` returned exactly the three BERT files; `git log --all
--diff-filter=A --name-only` showed those same three as the only `PerformanceIndex.txt` files
ever added to history; and no file named `CS2V.py` — the script the baseline numbers are
attributed to — appears in any commit. The detailed reconstruction of where the published
numbers came from is in [`01-baseline-reproduction.md`](01-baseline-reproduction.md).

## The four encoders

### One shallow log-linear model and three transformers

The BERT-vs-BERT part of this comparison is a **domain-adaptation** study: three checkpoints of
the same architecture, differing in what they were pretrained on. The BioSentVec-vs-BERT part is
an **architecture** study, and a much larger jump than the three-way table suggests.

| Encoder | Family | Architecture | Contextual? | Training objective | Dim | Corpus |
|---|---|---|:---:|---|:---:|---|
| BioSentVec | sent2vec (word2vec/fastText lineage) | shallow log-linear; a sentence vector is the average of its word and bigram embeddings | **No** — a word's vector is fixed regardless of context | sentence-level (predicts a held-out word from the averaged sentence context) | 700 | PubMed + MIMIC-III |
| Bio_ClinicalBERT | BERT-base encoder | 12 layers, 12 heads, 768 hidden, self-attention | Yes | masked LM + NSP; no sentence-level objective | 768 | MIMIC-III clinical notes |
| BiomedBERT | BERT-base encoder | 12 layers, 12 heads, 768 hidden, self-attention | Yes | masked LM; no sentence-level objective | 768 | PubMed abstracts |
| BlueBERT | BERT-base encoder | 12 layers, 12 heads, 768 hidden, self-attention | Yes | masked LM + NSP; no sentence-level objective | 768 | PubMed + MIMIC-III |

Sources: the `MODELS` dict in `bert_models.py` for the three HuggingFace paths
(`emilyalsentzer/Bio_ClinicalBERT`, `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`,
`bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12`); the layer/head/hidden figures are read
from the cached `config.json` (`num_hidden_layers: 12`, `num_attention_heads: 12`,
`hidden_size: 768`); and BioSentVec's 700 dimensions and bigram n-grams are legible in the model
filename the baseline loads, `BioSentVec_PubMed_MIMICIII-bigram_d700.bin`. The architecture and
objective descriptions are from the published model papers (Pagliardini et al. 2018 for
sent2vec; Chen, Peng & Lu 2019 for BioSentVec; Devlin et al. 2019 for BERT), not measured here.

Three consequences worth stating plainly:

- **BERT is encoder-only.** BERT discards the decoder half of the original Transformer, so all
  three BERT arms are *encoders* in exactly the sense this pipeline needs — text in, one vector
  out. That is the only structural thing they share with sent2vec.
- **700D vs. 768D is incidental and affects nothing.** Cosine similarity is only ever computed
  between two vectors from the *same* encoder, inside a single program run
  (`cosine_similarity()` asserts `len(u) == len(v)`). The two arms are separate runs writing
  separate output directories; no 700D vector is ever compared to a 768D one. Dimensionality is
  not a confound here.
- **The pooling asymmetry runs *against* the BERT arm.** None of the three HuggingFace
  checkpoints ships a sentence-transformers configuration — the local HF cache records a
  `.no_exist/…/modules.json` entry for `emilyalsentzer/Bio_ClinicalBERT`, i.e. the library
  looked for the config and the repo does not have one — so `SentenceTransformer(...)`
  auto-wraps each model with **mean pooling** over token embeddings. Reimers & Gurevych's SBERT
  paper (2019) measured mean-pooled vanilla BERT as a *weak* sentence encoder, below averaged
  GloVe on STS, precisely because masked-LM pretraining never optimizes the pooled vector for
  cosine comparison. sent2vec, by contrast, was trained with a sentence-level objective in the
  first place. **So the baseline has the shallower architecture but the better-matched training
  objective.** Whichever way a genuine comparison came out, "BERT is newer, therefore better
  here" is not a safe prior. (The corrected results bore this out: no encoder separates —
  [13](13-rank-aware-metrics.md).)

`normalize_embeddings=False` is passed at encode time. That flag does **not** affect any result
in this document: `cosine_similarity()` explicitly divides by both vector norms, so
pre-normalizing is a no-op for a cosine score either way. The original comparison draft raised
this as an open question about BiomedBERT's saturation; it is not the cause — see
[Where the encoders actually differ](#where-the-encoders-actually-differ-embedding-space-geometry).

## Method (shared with the baseline arm — with documented exceptions)

The BERT arm reuses the same retrieval-and-scoring pipeline as the baseline
([`01-baseline-reproduction.md`](01-baseline-reproduction.md) documents it in full); the intent
is that only the encoder changes. In brief:

- **Symptom-level patient similarity**: for a test admission against a candidate training
  admission, take the mean of per-symptom maximum cosine similarities
  (`compute_patient_similarity_pairwise()`), gated by `PRUNING_SIMILARITY = 0.5`
  (`Constants.py`).
- **Retrieval strategy**: **MAX** takes the single best-matching training patient; **TOP-K**
  (K ∈ {10, 20, 30, 40, 50}) pools the K best-matching training patients and counts a hit if
  *any* of their diagnoses clears the threshold (`predict_topk_diagnoses_pure()`,
  `containGreaterOrEqualsValue()`).
- **Diagnosis-level scoring**: MAX cosine similarity over the full Cartesian product of
  ground-truth × predicted diagnosis strings
  (`util_cy.get_diagnosis_similarity_by_description_max()`), compared against each threshold in
  `{0.6, 0.7, 0.8, 0.9, 1.0}`.
- **Cross-validation**: 10 pre-computed static folds (`K_FOLD = 10`), 129 admissions total,
  averaging ~12.9 test cases per fold. Fold 0 is 116 train / 13 test.

"Only the encoder changes" is the repo's stated design constraint
([`reference/architecture.md`](../reference/architecture.md), *The central design constraint*).
**At the time of writing it was not true**; the three failures are documented in
[Where the shared-pipeline constraint actually fails](#where-the-shared-pipeline-constraint-actually-fails),
and the destructive one has since been fixed under the `corrected` pipeline.

Dataset size is **129 admissions, not 128**. `wc -l data/raw/Symptoms-Diagnosis.txt` reports 128
only because the file has no trailing newline; `grep -c ';'` gives 129, Fold 0's 116+13 sums to
129, and the 10-fold-averaged TP row for MAX at threshold 0.6 reads `12.9`, i.e. `129 / 10` —
all three of the BERT `PerformanceIndex.txt` files agree on this independently. The
score-distribution analysis confirms it a fourth way: "Total patients = 129" in
[`score_distribution_summary.txt`](../score_distribution_analysis/score_distribution_summary.txt).

## What "F1" means in these tables

Every number below labeled with a threshold is what the pipeline writes to the `FS` column of
its `PerformanceIndex.txt` output. For the BERT arm it is not a genuine F1 score. Directly
checking all three committed result files:

```
grep pattern: 7-field aggregate rows (threshold, TP, FP, P, R, FS, PR)
Bio_ClinicalBERT: 4,200 rows, 0 have P != R or FS != P
BiomedBERT:       4,200 rows, 0 have P != R or FS != P
BlueBERT:         4,200 rows, 0 have P != R or FS != P
Total:            12,600 rows, 0 violations
```

The reason is structural, not coincidental. In `compute_aggregated_performance_index()`
(`cython_utils.py`), `precision = tp / (tp + fp)` and `recall = tp / nrow`. Every BERT test case
in every fold is scored as exactly one of TP or FP — no test case fails to find a candidate
above `PRUNING_SIMILARITY`, and there is no abstention path in the BERT code — so
`tp + fp == nrow` always, which forces `precision == recall` and, since the F-score is their
harmonic mean, `F1 == precision == recall == tp / nrow`, i.e. **accuracy**. The `PR` column the
code writes for these aggregate rows (`prediction_rate = (tp + fp) / nrow`) is 1.0 in all 330
aggregate rows per file for the same reason. (The per-test-case rows earlier in each file reuse
the column header `PR` for something different — a `1.0 / nrow` per-case weight, not a
prediction rate — so it is not 1.0 there; that is a labeling quirk of the raw output, not a
second data point about the metric.) Full derivation:
[`04-metric-degeneracy.md`](04-metric-degeneracy.md).

Because of this, the rest of this document reports the shared P = R = F1 value as **accuracy**
rather than repeating three identical numbers under three different headers, and rather than
calling it "F1."

**Whether the same was true of the BioSentVec number was genuinely unsettled when this was
written — it has since been settled by running it: the baseline is NOT degenerate.** The
2026-08-05 run ([09](09-baseline-first-run.md)) shows P ≠ R in every row, with a constant
prediction rate of 0.7679 — 30 of 129 test cases abstain because nothing clears the pruning
floor. The original reasoning below survives as the correct prediction of exactly that outcome:
the arithmetic is shared (the original Cython source carries the identical formulas, visible in
the archived generated C at `archive/cython_source/util_cy.c:10776`, `:10814`, `:10886`), so a
BioSentVec run *would* be degenerate **if** `tp + fp == nrow` held for it — and the only
surviving pre-run record,
[`archive/stale-docs/Reproduce_w_transformers.md:134-143`](../../archive/stale-docs/Reproduce_w_transformers.md),
reported P = 0.621 and R = 0.412 at TOP-10 — *different numbers*, with `R / P = 0.663`, the
signature of a run in which only ~66% of test cases cleared the pruning floor. That is what
sent2vec's less compressed similarity distribution predicts, and what the real run confirmed.

So the two arms' numbers are **not the same kind of quantity**: the BERT numbers are measured
accuracy over all cases; the baseline number is a genuine F-score computed over a
partially-abstaining run. [13](13-rank-aware-metrics.md) later gave the columns their true
names — `P` is the answered-cases hit rate, `R` the all-cases hit rate, `PR` the coverage — and
proved it bit-exactly. (The archived record remains internally suspect: the implied prediction
rate `R / P` should not depend on K, since MAX and every TOP-K share one pruning guard, yet the
table implies 0.663, 0.749, and 0.795 for TOP-10/20/30. The 2026-08-05 run shows a constant rate
at every K, as the code requires.)

## Results across thresholds 0.6–1.0 (BERT arm only)

Values are 10-fold-averaged accuracy (`TP / n`, see above), read from the `10-FOLD PERFORMANCE
INDEX` blocks at line 5921 onward in each file. **BioSentVec is not a column in these tables**,
by design — see [the next section](#the-biosentvec-numbers-and-what-they-are-not) for why.
**These are legacy-pipeline numbers** (leaky folds, divergent preprocessing); the corrected
equivalents are in [11](11-corrected-pipeline-first-results.md).

### Threshold = 0.6 (the paper's operating point — fully saturated)

| Method | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------:|:----------:|:--------:|
| MAX | 1.000 | 1.000 | 1.000 |
| TOP-10 | 1.000 | 1.000 | 1.000 |
| TOP-20 | 1.000 | 1.000 | 1.000 |
| TOP-30 | 1.000 | 1.000 | 1.000 |
| TOP-40 | 1.000 | 1.000 | 1.000 |
| TOP-50 | 1.000 | 1.000 | 1.000 |

Every cell here is 1.000. This table says the three BERT encoders are indistinguishable at 0.6 —
which is true, but only because 0.6 is below the minimum pairwise diagnosis similarity for two
of the three (see [Where the encoders actually
differ](#where-the-encoders-actually-differ-embedding-space-geometry)).

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
number in the whole comparison, and it is explained, not mysterious: 99.71% of BiomedBERT's
per-patient MAX similarities already exceed 0.9 **regardless of whether the retrieved diagnosis
is correct** (`score_distribution_summary.txt`, Section 2). The metric is rewarding
embedding-space compression, not skill.

### Threshold = 1.0 (exact cosine match)

| Method | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------:|:----------:|:--------:|
| MAX | 0.146 | 0.169 | 0.178 |
| TOP-10 | 0.285 | 0.254 | 0.239 |
| TOP-20 | 0.331 | 0.324 | 0.339 |
| TOP-30 | 0.363 | 0.340 | 0.347 |
| TOP-40 | 0.394 | 0.371 | 0.378 |
| TOP-50 | 0.401 | 0.378 | 0.417 |

At exact match, all three encoders land in the same 0.15–0.42 range with no consistent winner —
each of the three is "best" at at least one K. A cosine similarity of exactly 1.0 between two
float32 embeddings essentially requires the underlying diagnosis text to be identical (or the
same string via `MAX` picking out a shared diagnosis between two patients), so this threshold is
closer to a lexical exact-match rate than a semantic one — it is not a finer-grained version of
the 0.9 result. It is, however, the only threshold at which all three encoders are
simultaneously unsaturated, which makes it the one place a three-way comparison could in
principle be attempted. ([12](12-drg-grader.md) later proved this threshold *is* an
encoder-independent exact-match metric: the DRG grader reproduces it bit-exactly.)
[The next-but-one section](#the-differences-are-inside-the-noise-floor) shows why the ranking
still fails there.

## The BioSentVec numbers, and what they are not

**Status: superseded on 2026-08-05.** The block below is preserved as the record of what the
published figures were before this repository could check them — an inherited citation with no
artifact. That changed when the baseline ran: legacy TOP-10 = **0.4824** against the published
0.489, i.e. reproduced to within 0.007 ([09-baseline-first-run.md](09-baseline-first-run.md)).

> ### `[UNVERIFIED — INHERITED CITATION, NO ARTIFACT]` *(historical, pre-2026-08-05)*
>
> | Method | Threshold | Reported value |
> |---|:---:|:---:|
> | TOP-10 | 0.6 | 0.489 |
> | TOP-20 | 0.6 | 0.512 |
> | TOP-30 | 0.6 | 0.521 |
>
> **Source:** the README, which in turn traces to
> [`archive/stale-docs/Reproduce_w_transformers.md:134-143`](../../archive/stale-docs/Reproduce_w_transformers.md),
> which attributes them to a `Prediction_Output_22112025_04-41-14_ORIGINAL_OUTPUTS/` directory
> produced by a script `CS2V.py`. **Neither the directory nor the script exists in this
> repository or in any commit in its history.**
>
> **What was not known about these three numbers at the time:** which encoder revision produced
> them, on which folds, with which preprocessing, under which `PRUNING_SIMILARITY`, whether
> `tp + fp == nrow` held, and therefore whether they were accuracy or a genuine F-score. There
> was no `PerformanceIndex.txt` to re-read, no per-fold block to check variance against, no
> timing report, and no way to recompute them.

### What it would take to make the fourth column real

**All four requirements have since been met** — the baseline ran end-to-end on rented Linux on
2026-08-05. Kept for the record:

| # | Requirement | State then | State now |
|---|---|---|---|
| 1 | Fix the dataset path (`<repo-root>/Symptoms-Diagnosis.txt` vs `data/raw/`) | Unfixed | **Fixed in `c2fee6e`** |
| 2 | Fix the unbound `entity` name (`NameError`) | Unfixed | **Fixed in `c2fee6e`** |
| 3 | Obtain `BioSentVec_PubMed_MIMICIII-bigram_d700.bin` (~21 GB; 20.93 GiB measured) | Not present | **Obtained on the RunPod box** (deliberately not in the repo) |
| 4 | A working `sent2vec` Python binding | Unproven — [`archive/baseline_debug.txt`](../../archive/baseline_debug.txt) captures `AttributeError: module 'sent2vec' has no attribute 'Sent2vecModel'` from installing the WRONG PyPI package of the same name | **Working: epfml/sent2vec built from source** (see `pyproject.toml`'s `baseline` extra) |

The preprocessing divergence that would have confounded the comparison even after those four
(see [the constraint-failure section](#where-the-shared-pipeline-constraint-actually-fails)) was
fixed in `c2115ba` behind the `corrected` config.

## Where the encoders actually differ: embedding-space geometry

The three BERT encoders are not interchangeable — they differ sharply in how *compact* their
embedding space is, and that compactness is what determines where each one saturates. From
[`score_distribution_summary.txt`](../score_distribution_analysis/score_distribution_summary.txt)
(all 10,440 unique-diagnosis-pair cosine similarities over the 145 unique diagnosis
descriptions, computed independently of the fold evaluation, and cross-checked to 2e-7–9e-7
against the pipeline's own `cosine_similarity()`):

| Model | Min | Mean | Median | Std | % of pairs ≥ 0.6 | % ≥ 0.9 |
|-------|:---:|:----:|:------:|:---:|:----------------:|:-------:|
| BiomedBERT | 0.7246 | 0.9282 | 0.9341 | 0.0303 | 100.00% | 87.62% |
| Bio_ClinicalBERT | 0.6454 | 0.8348 | 0.8371 | 0.0450 | 100.00% | 5.58% |
| BlueBERT | 0.4810 | 0.7170 | 0.7176 | 0.0652 | 96.35% | 0.59% |
| **BioSentVec** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** | **not measured** |

The BioSentVec row is empty because `scripts/analyze_score_distributions.py` only loads the
three HuggingFace models. **This is still, as of 2026-08-08, the single most useful missing
measurement in the project** — the baseline now runs, but its diagnosis-embedding geometry has
never been computed. It is much cheaper than a full 10-fold run — 145 sentence embeddings and a
pairwise cosine matrix, no folds, no retrieval — and it would settle directly whether the
paper's own 0.6 threshold was saturated for the paper's own encoder. If it was, the published
0.489 / 0.512 / 0.521 are surprisingly *low* for a saturated metric and something else is going
on; if it was not, then the paper's threshold was calibrated for an embedding space the BERT arm
does not share, and reusing 0.6 unchanged was the original mistake. (The baseline's 23–24%
abstention rate is indirect evidence for the latter.)

For the three models that were measured: BiomedBERT's diagnosis embeddings are the most tightly
clustered — even the least similar pair of diagnoses in the entire dataset scores 0.7246, and
87.62% of all pairs already exceed 0.9 before any patient-level aggregation. That is a direct,
independently-verified explanation for the threshold-0.9 table above. BlueBERT sits at the
opposite end — the widest spread (std 0.0652 vs. 0.0303–0.0450) and the lowest floor (min
0.4810) — which is why it is the only model with headroom below 0.8, and the only one for which
the 0.7–0.9 range carries discriminative signal.

Read literally, "BiomedBERT wins" at every threshold up to 0.9 in the tables above. Read
correctly, BiomedBERT's embedding space is the *least* able to tell diagnoses apart, and its
perfect scores are a ceiling effect. BlueBERT's lower numbers reflect a wider, more
discriminative embedding space, not worse performance. Full mechanism (including the
MAX-operator amplification that turns these pairwise numbers into the per-patient scores
actually used for classification): [`03-metric-saturation.md`](03-metric-saturation.md).

## The differences are inside the noise floor

Threshold 1.0 is the only threshold at which all three BERT encoders are unsaturated, so it is
the only place a ranking could be read. It does not survive contact with the fold-to-fold
variance. Each cell below is the 10-fold mean; `s.d.` is the sample standard deviation of that
model's ten per-fold values, parsed from the per-fold `PERFORMANCE INDEX` blocks in the same
files.

| Method | Bio_ClinicalBERT (s.d.) | BiomedBERT (s.d.) | BlueBERT (s.d.) | Best–worst gap |
|---|:---:|:---:|:---:|:---:|
| MAX | 0.146 (0.085) | 0.169 (0.079) | **0.178** (0.081) | 0.031 |
| TOP-10 | **0.285** (0.102) | 0.254 (0.071) | 0.239 (0.083) | 0.046 |
| TOP-20 | 0.331 (0.124) | 0.324 (0.112) | **0.339** (0.120) | 0.015 |
| TOP-30 | **0.363** (0.106) | 0.340 (0.095) | 0.347 (0.114) | 0.023 |
| TOP-40 | **0.394** (0.114) | 0.371 (0.098) | 0.378 (0.096) | 0.023 |
| TOP-50 | 0.401 (0.116) | 0.378 (0.102) | **0.417** (0.118) | 0.039 |

**Every gap in the right-hand column is smaller than every standard deviation in the table** —
row by row, by factors of 1.5× (TOP-10) to 8× (TOP-20). The per-fold spread is 0.071–0.124; the
encoder-to-encoder spread is 0.015–0.046. And the bolded winner moves: BlueBERT at MAX,
Bio_ClinicalBERT at TOP-10, BlueBERT at TOP-20, Bio_ClinicalBERT at TOP-30 and TOP-40, BlueBERT
at TOP-50 — with BiomedBERT last at four of the six Ks and second at the other two. There is no
consistent ordering to report.

The same holds at threshold 0.9 for the two unsaturated models: per-fold s.d. runs 0.038–0.136
there (Bio_ClinicalBERT TOP-40 lowest at 0.038, BlueBERT TOP-30 highest at 0.136), against a
BiomedBERT column that is a flat 1.000 with zero variance because it is saturated.

With n = 129 admissions and 10 folds, **no claim that one encoder beats another is supported by
this data.** That conclusion survived every subsequent correction — the grouped folds, the
unified preprocessing, the DRG grader, and the rank-aware metrics all reconfirmed it
([11](11-corrected-pipeline-first-results.md), [13](13-rank-aware-metrics.md)).

## Patient leakage inflates every arm

The 10 folds are static committed files (`data/folds/Fold{0..9}/`) shared by both arms — the
baseline and all three BERT runs see the identical splits — and they are split by **admission**,
not by patient. The 129 admissions come from only **100 distinct `SUBJECT_ID`s**: 14 patients
contribute more than one admission, and one patient alone contributes 15. As a result **41 of
the 129 test cases (31.8%) have another admission from the same patient sitting in their own
retrieval pool.** Measured directly by joining `data/raw/Symptoms-Diagnosis.txt`
(`HADM_ID;SUBJECT_ID;…`) against each fold's `TrainingSet.txt`/`TestSet.txt`.

Splitting the per-test-case rows of the committed `PerformanceIndex.txt` files into the 41
leaked and 88 clean cases, at threshold 1.0:

| Method | Group | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|---|---|:---:|:---:|:---:|
| MAX | leaked (n=41) | 0.293 | 0.293 | 0.293 |
| MAX | clean (n=88) | 0.080 | 0.114 | 0.125 |
| TOP-10 | leaked (n=41) | 0.415 | 0.415 | 0.415 |
| TOP-10 | clean (n=88) | 0.227 | 0.182 | 0.159 |

Across all 18 model × strategy combinations at threshold 1.0, leaked cases score **+0.11 to
+0.26** above clean cases. Compare that to the 0.015–0.046 spread between encoders in the
previous section: the leakage effect is **5–10× larger than the effect being studied.**

The most telling detail is in the leaked rows: **all three encoders score identically on the
leaked cases** — 0.293 at MAX, 0.415 at TOP-10, for every model — while differing on the clean
cases (0.080 / 0.114 / 0.125 and 0.227 / 0.182 / 0.159). When the retrieval pool contains the
patient's own prior chart, retrieval finds it no matter which encoder is doing the retrieving,
and the diagnosis text matches trivially. That is not a measurement of the encoder; it is a
measurement of the fold construction.

Because the folds are shared, **this applies to the baseline's 0.489 / 0.512 / 0.521 exactly as
much as it applies to the BERT numbers.** It was the one thing in this document genuinely common
to all four arms. **Fixed 2026-08-05 (`c2115ba`)**: `scripts/make_folds.py` regroups the folds by
`SUBJECT_ID` with `GroupKFold`, taking leaked cases from 41 to 0; the `legacy` config keeps the
old folds on purpose so the golden regression never moves. The full write-up is
[`05-patient-leakage.md`](05-patient-leakage.md), and the measured cost of removing the leak —
every arm loses ground, the baseline most — is in
[`11-corrected-pipeline-first-results.md`](11-corrected-pipeline-first-results.md).

## Where the shared-pipeline constraint actually fails

The repo's design constraint is that both arms share preprocessing, fold loading, and evaluation
so that the embedding model is the only variable
([`reference/architecture.md`](../reference/architecture.md)). Symptom text does honor this:
both arms call `util_cy.preprocess_sentence()` on every symptom. Diagnosis text did not.
**Failure 1 below — the destructive one — was fixed in `c2115ba` behind the `corrected` config**
(`legacy` keeps the old behaviour so the golden stays byte-exact; under `corrected`, 145/145
descriptions reach both encoders identically). Failures 2 and 3 still stand.

### 1. The two arms embed different diagnosis text (the serious one — FIXED under `corrected`)

The baseline preprocesses the diagnosis description before embedding it:

```python
embs = model.embed_sentence(preprocess_sentence(diagnosis_description))
```
(`cython_utils.py`, inside `embending_diagnosis()`)

The BERT arm did not: `compute_bert_diagnosis_embeddings()` sliced the description off after the
`:` and handed the **raw** string straight to `model.encode()`. (It now preprocesses too, gated
on `use_corrected_preprocessing(config)` — under `legacy` it still encodes raw text, which is
why a legacy baseline-vs-BERT delta is confounded by preprocessing and a corrected one is not.)
Measured over the 145 unique diagnosis descriptions, running them through
`preprocess_sentence()`:

| Quantity | Value |
|---|---:|
| Unique diagnosis descriptions | 145 |
| Descriptions whose text **changes** under `preprocess_sentence()` | **119 (82.1%)** |
| Descriptions containing `w/o` (i.e. "without") | 14 |
| Distinct descriptions that **collide** into one string after preprocessing | 2 pairs |

The changes are not cosmetic. `preprocess_sentence()` pads `/` into its own token, then drops
punctuation *and NLTK English stopwords* — and `"o"` is in NLTK's English stopword list. So in
the baseline arm, and only in the baseline arm:

```
'cardiac valve procedures w/o cardiac catheterization'
    -> 'cardiac valve procedures w cardiac catheterization'
'coronary bypass w/o cardiac cath w/o mcc'
    -> 'coronary bypass w cardiac cath w mcc'
'multiple significant trauma w/o o.r. procedure'
    -> 'multiple significant trauma w r procedure'
```

**"without" becomes "with" in all 14 cases.** The two collisions are worse still: after
preprocessing, `'tracheostomy w long term mechanical ventilation w/o extensive procedure'` and
`'tracheostomy w long term mechanical ventilation w extensive procedure'` are the *same string*,
so the baseline embeds two clinically opposite DRG descriptions to the identical vector (the
embeddings dict is keyed by the raw description, so they stay two entries — with one shared
value). The other collision is the benign `w/` vs. `w` pair.

The consequence for this document: **any legacy BioSentVec-vs-BERT number is confounded by a
preprocessing difference, not only by an encoder difference** — it touches 82% of the diagnosis
vocabulary and inverts the polarity of 14 descriptions. The resolution landed in
`cython_utils.py` so both arms move together, exactly as this section demanded; under
`corrected`, `w/o` survives as `without`. Details: [`06-preprocessing-defects.md`](06-preprocessing-defects.md).

### 2. The per-test-case rows mean different things in the two arms

Both arms write a `PERFORMANCE INDEX` block per test case, but they compute different
quantities:

| | Baseline | BERT |
|---|---|---|
| Where computed | `util_cy.compute_performance_index()` | inline in `bert_models.py` |
| Confusion matrix used | the **fold-level running total** (initialized once per fold; `predictS2V` prints after each case) | `(tp, fp)` **local to that one test case**, always `(1,0)` or `(0,1)` |
| `precision` | `tp / (tp + fp)` | `tp / (tp + fp)` |
| `recall` | `tp / nrow` | `tp / (tp + fp)` — the same expression as precision |
| `PR` | `(tp + fp) / nrow` (a real prediction rate) | `1.0 / nrow` (a per-case weight) |

So a baseline per-case row is a cumulative fold-to-date index in which P and R genuinely differ
until the last case of the fold, while a BERT per-case row is a single binary verdict printed
three times. The two files are not row-comparable even though they share a column layout. Only
the fold-aggregate and 10-FOLD blocks — which both arms route through the shared
`compute_aggregated_performance_index()` — are computed the same way.

### 3. The output headers differ

The baseline writes `Constants.PERFORMANCE_INDEX_HEADER`, tab-separated
(`"\t TP \t FP \t  P \t R \t FS \t PR\n"`). The BERT arm writes its own space-padded literal for
per-case blocks and only uses the shared constant for the aggregate blocks. Cosmetic on its own,
but it means any parser written against one arm's output needs adjusting for the other's —
including the verification script in [`04-metric-degeneracy.md`](04-metric-degeneracy.md). (As
of 2026-08-08 the repo has four such parsers, none shared — see the Phase 3 notes in
`docs/plans/revival-roadmap.md`.)

## Runtime

| Phase | Bio_ClinicalBERT | BiomedBERT | BlueBERT | BioSentVec |
|-------|:----------------:|:----------:|:--------:|:----------:|
| Model loading | 81.32s | 15.31s | 11.12s | not measured |
| Symptom embeddings | 5.04s | 2.84s | 1.49s | not measured |
| Diagnosis embeddings (145 unique) | 0.94s | 0.62s | 0.63s | not measured |
| 10-fold evaluation | 1231.52s (20.53 min) | 1248.55s (20.81 min) | 1228.11s (20.47 min) | not measured |
| **Total (this model)** | **1318.82s (21.98 min)** | **1267.33s (21.12 min)** | **1241.36s (20.69 min)** | **not measured** |

(Source: each model's own `timing_report.txt`. The "not measured" BioSentVec column was filled
in later on different hardware — ~13 min for the full run on the RunPod box — so it cannot be
inserted into this table without mixing machines; see
[`08-runtime-and-cost.md`](08-runtime-and-cost.md) for the cross-arm runtime story.)

Sum of the three measured models' total execution times is **3,827.51s ≈ 63.8 minutes**.
Fold-evaluation time dominates every run and is roughly constant across encoders
(~20.5–20.8 min) — it is CPU-bound pure-Python similarity computation, not GPU/MPS-bound; only
model loading and embedding generation vary by encoder, and Bio_ClinicalBERT's 81s load time
(vs. 11–15s for the other two) is a first-download cost, not a per-run cost on subsequent loads.

`bert_model_comparison.md` additionally cited an "Overall Pipeline: 76.4 min" figure. That
number does not appear in any `timing_report.txt` and this document cannot reconcile it: the
three output directories' own timestamps put the first run's start and the third run's
completion about 71.5 minutes apart, and the sum of the three self-reported totals is 63.8
minutes — neither matches 76.4 minutes. Treat 76.4 as unverified; the 63.8-minute figure above
is what the committed timing reports actually support.

## What this comparison can and cannot tell you

**It can tell you:**
- **How the four encoders differ architecturally** — one shallow, non-contextual, sentence-level
  log-linear model versus three 12-layer contextual transformer encoders that were never trained
  to produce a cosine-comparable pooled vector. That asymmetry is documented above from the
  model papers and the loading code, and it does not depend on any measurement in this repo.
- The *relative compactness* of the three BERT encoders' diagnosis-text embedding spaces
  (BiomedBERT most compact, BlueBERT least), independently confirmed two ways — the raw
  pairwise-similarity statistics, and the threshold at which each encoder's accuracy table stops
  reading 1.000.
- That any conclusion drawn from the 0.6 or 0.7 rows specifically is not a real result — those
  rows are saturated for all three BERT encoders (BlueBERT-MAX at 0.7 excepted) and would be
  saturated for almost any encoder producing similarly compact biomedical embeddings.

**It cannot tell you** (and the corrected pipeline later confirmed each of these the hard way):
- **Whether any encoder beats the BioSentVec baseline.** At the time: no artifact, no
  reproducible run, a preprocessing confound. Now: all fixed, and the answer is that **no
  encoder separates from any other** — largest paired |t| = 1.718 vs the 2.262 needed
  ([13](13-rank-aware-metrics.md)).
- **Which of the three BERT encoders is better.** The between-encoder gaps (0.015–0.046 at the
  one unsaturated threshold) are smaller than the fold-to-fold standard deviation (0.071–0.124),
  and the ranking flips with K.
- **How much of any score is retrieval and how much is leakage.** 31.8% of test cases retrieve
  the same patient's own prior admission, worth +0.11 to +0.26 — and on exactly those cases all
  three encoders score identically. (Answered in [11](11-corrected-pipeline-first-results.md):
  removing the leak costs every arm 0.036–0.064 at TOP-10 threshold 1.0.)
- Which encoder makes *better clinical predictions*. The metric collapses to `TP / n` (accuracy)
  with no precision/recall trade-off, no false-negative concept, and no case where a patient's
  diagnosis is scored as anything other than a binary hit-or-miss against a single similarity
  number — see [`04-metric-degeneracy.md`](04-metric-degeneracy.md).
- Whether a "correct" diagnosis match at 0.9+ cosine similarity means anything clinically.
  Cosine similarity between two diagnosis-text embeddings is not validated anywhere in this repo
  against a clinician's judgment of diagnostic relatedness; it is only validated as a
  self-consistent numpy-vs-`cython_utils` computation.
- Anything about generalization beyond this specific 129-admission dataset and 10 static folds.

## What a genuine four-way comparison would require

This was the document's to-do list, in dependency order. **Five of the six steps have since been
done**; their outcomes are linked.

1. ~~**Regroup the folds by `SUBJECT_ID`**~~ — **DONE** (`c2115ba`, `scripts/make_folds.py`,
   41 → 0 leaked cases; [05](05-patient-leakage.md)).
2. ~~**Unify diagnosis preprocessing across both arms**, and decide deliberately whether
   `w/o → w` is acceptable at all — it is not~~ — **DONE** (`c2115ba`; `w/o` survives as
   `without` under `corrected`; [06](06-preprocessing-defects.md)).
3. ~~**Restore the baseline arm**~~ — **DONE** (`c2fee6e` + the 2026-08-05 run;
   [09](09-baseline-first-run.md)).
4. **Measure BioSentVec's diagnosis-embedding geometry** — **STILL OPEN**, and still the
   cheapest high-value measurement in the project: 145 embeddings, one pairwise cosine matrix,
   no folds. It answers whether the paper's 0.6 threshold was ever unsaturated for the paper's
   own encoder.
5. ~~**Replace the metric**, fixing saturation and degeneracy together, with rank-aware metrics
   against an encoder-independent relevance label~~ — **DONE** in exactly that shape: the
   `drg-exact` grader ([12](12-drg-grader.md)) plus MRR/Precision@K
   ([13](13-rank-aware-metrics.md)). The prediction in the original text — that this removes
   the "grader is the model being graded" problem — held, with a twist: the threshold-1.0
   column turned out never to have been self-grading-inflated.
6. ~~**Report effect sizes with per-fold variance**, not point estimates~~ — **DONE**
   throughout [11](11-corrected-pipeline-first-results.md), [13](13-rank-aware-metrics.md), and
   the README's paired-t tables.

## Related documents

- [`01-baseline-reproduction.md`](01-baseline-reproduction.md) — the original paper's method
  walkthrough, the provenance trail for the 0.489 / 0.512 / 0.521 figures, and their successful
  reproduction (run record: [`09-baseline-first-run.md`](09-baseline-first-run.md)).
- [`03-metric-saturation.md`](03-metric-saturation.md) — full derivation of why threshold 0.6
  saturates.
- [`04-metric-degeneracy.md`](04-metric-degeneracy.md) — full derivation of why P = R = F1 at
  every threshold in the committed BERT results; its baseline question was settled by the
  2026-08-05 run.
- [`05-patient-leakage.md`](05-patient-leakage.md) — the leakage finding this document
  previewed, and its fix.
- [`11-corrected-pipeline-first-results.md`](11-corrected-pipeline-first-results.md) — the
  four-arm leakage-free results this document said were required.
- [`12-drg-grader.md`](12-drg-grader.md) · [`13-rank-aware-metrics.md`](13-rank-aware-metrics.md)
  — the two knob removals.
- [`../plans/metric-redesign.md`](../plans/metric-redesign.md) — the metric options as they were
  weighed.
- [`../reference/architecture.md`](../reference/architecture.md) — the one-evaluation-core,
  two-embedding-arms design constraint.
- [`../score_distribution_analysis/next_steps.md`](../score_distribution_analysis/next_steps.md)
  — proposed alternative evaluation strategies (Strategy B has since shipped as `drg-exact`).
- [`../../README.md`](../../README.md) — project overview; its headline tables are now the
  corrected ones.

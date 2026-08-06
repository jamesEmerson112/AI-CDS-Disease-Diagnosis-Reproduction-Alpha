# AI-CDS Disease Diagnosis System

Author: An Thien Vo (James)

Clinical Decision Support System for disease diagnosis prediction using patient symptom similarity.

This project reproduces the clinical decision support system of *"AI-Driven Clinical Decision
Support: Enhancing Disease Diagnosis Exploiting Patients Similarity"* (Comito et al., 2022) and then
swaps its 2019 BioSentVec encoder for three modern biomedical BERT models behind the **identical**
retrieval and scoring pipeline. The result is a null result, and the null result is the
contribution: under this evaluation the old encoder and the new ones cannot be told apart.

**The headline numbers below are leakage-free.** The folds originally split on `HADM_ID`, letting 41
of 129 test cases retrieve the same patient's own other admission; they now split on `SUBJECT_ID`
with `GroupKFold`, and the two arms now preprocess diagnosis text identically. Both corrections are
selectable rather than destructive — `legacy` still reproduces the original pipeline bit-for-bit.

---

## Headline comparison — four encoders, one pipeline, leakage-free

All four arms were run on **one machine** (RunPod Linux, 32 vCPU) under `AICDS_PIPELINE=corrected`
on **2026-08-06**, from commits `c2115ba` + `31bea66`. Runtime was ~48 minutes for all four
(baseline ~13 min, each BERT arm ~11.5 min). The three transformer runs previously reproduced
Apple-silicon runs **bit-for-bit to all 17 significant figures**, so hardware is not a confound.

`corrected` changes **two things at once** — the fold split *and* the preprocessing — so the deltas
below cannot be attributed to one or the other. One-change-at-a-time configs (`folds-only`,
`preprocess-only`) exist for that and have not been run.

### Threshold 1.0, TOP-10 — the informative setting

The only threshold at which no model sits on the ceiling, and therefore the only one where the four
arms can be compared at all.

| Encoder | Dim | Model size | Precision | Recall | F | TP | FP | `TP+FP` | Pred. rate | Legacy F | Δ |
|---------|:---:|:----------:|:---------:|:------:|:------:|:---:|:---:|:-------:|:----------:|:--------:|:------:|
| Bio_ClinicalBERT | 768 | 416 MB | 0.2491 | 0.2491 | **0.2491** | 3.3 | 9.6 | 12.9 | 1.0000 | 0.2853 | −0.0362 |
| **BioSentVec — the 2019 baseline** | 700 | **20.93 GiB** | **0.2512** | 0.1923 | **0.2163** | 2.5 | 7.3 | **9.8** | **0.7558** | 0.2801 | −0.0638 |
| BiomedBERT | 768 | 420 MB | 0.1981 | 0.1981 | **0.1981** | 2.6 | 10.3 | 12.9 | 1.0000 | 0.2545 | −0.0564 |
| BlueBERT | 768 | 420 MB | 0.1821 | 0.1821 | **0.1821** | 2.4 | 10.5 | 12.9 | 1.0000 | 0.2391 | −0.0570 |

*TP and FP are means across the 10 folds (~12.9 test cases per fold). Pred. rate is the fraction of
test cases on which the system predicts at all. "Legacy F" is the same measurement under the
original leaky folds and divergent preprocessing. Model size is the on-disk weight file:
BioSentVec is **17× larger than all three transformers combined**, because sent2vec stores an
explicit unigram + bigram embedding table while BERT computes representations from ~110M parameters.*

**Every arm loses ground once leakage is removed — the baseline most of all (−0.0638).** That is the
expected direction: the leaked cases were free wins, and the baseline had the most to gain from
them.

**Verdict — no encoder ranking is supported by this experiment.** The 700-dimensional,
non-contextual, 2019 baseline lands **second of four** and holds the **highest precision of all four
arms** (0.2512). The next three sections are why the word "second" should not be trusted at all.

### None of these gaps clears the noise

The single most important table in this README. Every score above is a mean over 10 folds, and the
folds disagree wildly: per-fold F ranges from **0.00 to 0.67 on identical data**, with per-fold
standard deviations of **0.054–0.139**. Those sds are *larger than every gap between encoders*, so
before reading any ranking, ask whether the gap is bigger than the fold-to-fold wobble.

All four arms run on the **same** folds, so the right instrument is a **paired** *t*-test on the
per-fold differences (`t = mean(diff) / (sd(diff)/√10)`, 9 degrees of freedom, so **|t| > 2.262** is
needed for p < 0.05):

| Aggregator @ 1.0 | 1st | 2nd | Gap | Paired *t* | p < 0.05? |
|---|---|---|:---:|:---:|:---:|
| MAX | BioSentVec | Bio_ClinicalBERT | 0.0015 | 0.07 | **no** |
| TOP-10 | Bio_ClinicalBERT | BioSentVec | 0.0328 | 0.87 | **no** |
| TOP-20 | Bio_ClinicalBERT | BlueBERT | 0.0010 | 0.10 | **no** |
| TOP-30 | BlueBERT | Bio_ClinicalBERT | 0.0167 | 1.04 | **no** |

**Not one first-place margin is statistically significant.** The largest *t* anywhere is 1.04, less
than half the threshold. Pairing is the *generous* choice here too — fold difficulty is genuinely
shared, with per-fold F correlating at r = 0.72–0.96 among the three BERT arms — and the gaps still
do not survive it.

A leave-one-fold-out check makes the same point without any statistics: at MAX, dropping a single
fold hands first place to **BioSentVec in 5 of 10 cases and Bio_ClinicalBERT in the other 5.** The
baseline's "win" is a coin flip. Bio_ClinicalBERT's TOP-10 lead is the one that survives all ten
leave-one-out runs, but dropping fold 0 alone shrinks it from 0.0328 to 0.0068 — so it rests largely
on one fold in which it scored 0.60 while every other encoder scored 0.33.

**With n = 129 and 10 folds, this experiment does not have the statistical power to separate four
encoders whose true differences are this small.** That is a design finding about the study, not a
defect of the encoders, and it applies equally to the original paper.

**The leakage fix did not change any encoder's rank.** BioSentVec was 1st at MAX, 2nd at TOP-10, and
4th at TOP-20 through TOP-50 *both before and after*. What the fix changed is the magnitude — and it
moved **against** the baseline everywhere: its deficit to 1st widened at every `K` (TOP-10
0.0051 → 0.0328; TOP-30 0.0744 → 0.1383), and at MAX its margin over 2nd *shrank* from 0.0067 to
0.0015. So the correct reading is not "leakage removal let the old encoder catch up." It is that the
old encoder was **never distinguishable from the new ones in this data**, and removing leakage
removed the confound that could have explained that away.

**Read the `TP+FP` and prediction-rate columns together — they are the whole degeneracy story.**
Every BERT arm sums to exactly **12.9**, the mean fold test size, so `tp + fp == nrow`; precision
therefore reduces to `tp/nrow`, which *is* recall, and their harmonic mean is that same number.
**Every BERT "F" in this table is accuracy.** The baseline sums to **9.8** because it abstains on
24.4% of cases when nothing clears the pruning gate, which is why it alone has P ≠ R.

### The ranking inverts when you change the aggregator

This is the sharpest evidence that the ranking is not a property of the encoders. Holding the
threshold at 1.0 and changing only `K` — an arbitrary knob — **reverses the order**:

| Encoder | MAX | TOP-10 | TOP-20 | TOP-30 |
|---------|:---:|:------:|:------:|:------:|
| **BioSentVec (baseline)** | **0.0877 — 1st** | 0.2163 — 2nd | 0.2229 — 4th | 0.2296 — **4th** |
| Bio_ClinicalBERT | 0.0862 — 2nd | **0.2491 — 1st** | **0.3049 — 1st** | 0.3513 — 2nd |
| BiomedBERT | 0.0855 — 3rd | 0.1981 — 3rd | 0.2888 — 3rd | 0.3353 — 3rd |
| BlueBERT | 0.0785 — **4th** | 0.1821 — 4th | 0.3038 — 2nd | **0.3679 — 1st** |

**BioSentVec goes 1st → 4th and BlueBERT goes 4th → 1st, on the same data, at the same threshold.**

**And there is a mechanism, not just noise.** The baseline abstains on 24.4% of cases; every BERT arm
predicts on 100%. Widening `K` cannot help the baseline on a case where it declined to predict, but
it hands each BERT arm another free guess — and since one hit inside `K` suffices with no penalty for
the other `K−1`, **TOP-K structurally rewards not abstaining.** The metric is scoring willingness to
guess, and calling it retrieval quality.

This also means the between-encoder spread does **not** move in one direction when leakage is
removed. It shrinks under MAX and grows under TOP-K:

| Aggregator @ threshold 1.0 | Legacy spread | Corrected spread | Direction |
|---|:---:|:---:|---|
| MAX | 0.0381 | 0.0092 | ↓ 4.1× |
| TOP-10 | 0.0462 | 0.0671 | ↑ 1.45× |
| TOP-20 | 0.0506 | 0.0819 | ↑ 1.62× |
| TOP-30 | 0.0744 | 0.1383 | ↑ 1.86× |

Normalising by the leading value does not rescue it (MAX 20.7% → 10.5%; TOP-10 16.2% → 26.9%). Any
statement of the form "the encoders converged" or "the encoders separated" is really a statement
about which aggregator was picked.

### It inverts when you move the threshold, too

Same phenomenon on the other knob. TOP-10, corrected:

| Encoder | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---------|:---:|:---:|:---:|:---:|:---:|
| Bio_ClinicalBERT | 1.0000 | 1.0000 | 1.0000 | 0.7285 | **0.2491** |
| BiomedBERT | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.1981 |
| BlueBERT | 1.0000 | 1.0000 | 0.8194 | 0.2923 | 0.1821 |
| BioSentVec | 0.3922 | 0.3077 | 0.2620 | 0.2163 | 0.2163 |

At 0.9 the order is BiomedBERT → Bio_ClinicalBERT → BlueBERT → BioSentVec. At 1.0 it is
Bio_ClinicalBERT → BioSentVec → BiomedBERT → BlueBERT. **BiomedBERT drops 1st → 3rd and the baseline
climbs 4th → 2nd.** BiomedBERT tops the table at 0.9 only because it is still pinned at the ceiling
there — it is the most compact of the three embedding spaces, so it is the last to fall off 1.000,
and grading it with its own cosine rewards exactly that compactness.

### Threshold 0.6 — still saturated. This is the metric, not the model.

| Encoder | Precision | Recall | F | Pred. rate | Reading |
|---------|:---------:|:------:|:------:|:----------:|---------|
| BioSentVec (baseline) | 0.4692 | 0.3410 | **0.3922** | 0.7558 | published figure is 0.489 |
| Bio_ClinicalBERT | 1.0000 | 1.0000 | **1.000** | 1.0000 | **saturated — not a result** |
| BiomedBERT | 1.0000 | 1.0000 | **1.000** | 1.0000 | **saturated — not a result** |
| BlueBERT | 1.0000 | 1.0000 | **1.000** | 1.0000 | **saturated — not a result** |

**The three 1.000s are an artifact and must not be read as an achievement.** At threshold 0.6 the
compactness of biomedical embedding space combined with the MAX-over-Cartesian-product aggregator
puts ~100% of patient pairs above the bar, so essentially everything counts as a match. 0.6 is the
paper's threshold, which is why it appears here at all; it cannot discriminate between models.

### The one asymmetry that *is* real: cost

The accuracy difference does not survive a significance test. The cost difference is not close.

| | BioSentVec | one BERT-base arm |
|---|---|---|
| On disk | 20.93 GiB | 416–420 MB |
| Parameters | **~5.6 billion** (≈8M n-gram vectors × 700 dims) | **~110 million** |
| Where the information lives | memorised: one fixed vector per unigram/bigram | computed: contextual, from a small parameter set |
| RAM to run | full 20.93 GiB resident | a few hundred MB |
| Builds on Windows | **no** — MSVC rejects sent2vec's GCC-only flags | yes |

*Parameter count derived from file size and dimensionality assuming float32, not read from the model.*

**Three ~110M-parameter transformers match a ~5.6-billion-parameter n-gram lookup table to within
statistical noise, using roughly 1/51 the parameters and 1/17 the disk.** Note the contrast is *not*
neural versus non-neural — sent2vec is itself a shallow neural embedding model in the word2vec /
fastText lineage. It is **shallow and non-contextual** (one stored vector per n-gram, hence the 21 GB
table) versus **deep and contextual** (representations computed on demand). That efficiency result is
the practical claim this experiment genuinely supports.

**What the baseline uniquely offers, and it is not accuracy: it abstains.** It is the only arm that
declines to predict when nothing clears the pruning gate (24.4% of cases), which gives it the highest
precision of the four and makes it the **only arm producing an interpretable number at the paper's own
threshold of 0.6** — all three BERT arms are pinned at a meaningless 1.000 there. For a clinical
decision support system, "I don't know" is arguably the more useful behaviour, and the compact BERT
embedding spaces have lost the ability to say it.

---

## What the correction changed, and what it did not

**Fixed.**

| Defect | Before | After |
|---|---|---|
| Patient leakage in the folds | 41 of 129 test cases could retrieve the same patient's other admission | **0** — `GroupKFold` on `SUBJECT_ID` |
| Divergent cross-arm preprocessing | 26 of 145 diagnosis descriptions identical between arms | **145 of 145** |
| `w/o` → `w`, destroying negation | `"Tracheostomy w/o Extensive Procedure"` collapsed onto the `w` variant | `w/o` survives as `without` |
| Comma-shredded symptom fragments | 1805 tokens, 89 orphan fragments | 1725 tokens, **9** fragments remain (see TODO P27) |

A side effect worth knowing: the folds are now **uneven** (114/15 through 117/12) because one subject
holds 15 admissions and whole patients must stay together. Per-fold *n* varies, so treat per-fold σ
accordingly.

**Survived, exactly as expected — these are metric design, not data splitting.**

- **Saturation.** BiomedBERT is still 1.000 at every threshold from 0.6 to 0.9 on TOP-10;
  Bio_ClinicalBERT at 0.6–0.8; BlueBERT at 0.6–0.7.
- **Degeneracy.** All three BERT arms still report prediction rate exactly 1.0000, so P == R == F in
  every row and every BERT "F1" is still accuracy.
- **The baseline still abstains**, at `PR` = 0.7558, so it alone has P ≠ R. This confirms degeneracy
  is a consequence of BERT's compact embedding space, not a structural property of the code.

---

## Caveats — what is fixed, and what still is not

> **Fixed: the folds no longer leak patients.** This was the largest single correction available and
> it applied to both arms, which means the published 0.489/0.512/0.521 carry the contamination too.
> Measured cost of removing it: −0.036 to −0.064 at TOP-10 threshold 1.0, and −0.090 on the
> baseline's headline TOP-10 @ 0.6 ([details](docs/findings/05-patient-leakage.md),
> [results](docs/findings/11-corrected-pipeline-first-results.md)).
>
> **Still broken 1 — the metric saturates.** At threshold 0.6 nearly every diagnosis pair counts as a
> match, so F = 1.000 measures the metric, not the model
> ([details](docs/findings/03-metric-saturation.md)).
>
> **Still broken 2 — the metric is degenerate; every BERT "F1" here is accuracy.** Precision, recall,
> and F-score are the *same number* in every BERT row, because every test case increments exactly one
> of TP or FP. This is a property of the embedding space, not the code: the baseline's looser 700D
> space abstains and its precision and recall *do* diverge. Saturation and degeneracy are **one root
> cause at two different gates** ([details](docs/findings/04-metric-degeneracy.md)).
>
> **Still broken 3 — the system grades itself.** The same embedding space both retrieves candidates
> and judges whether a prediction is correct, so a more compressed space marks its own work more
> leniently. Every number in this README inherits that. This is the top open item (TODO P4) and the
> largest remaining threat to any cross-encoder claim.
>
> **Still broken 4 — rank is discarded.** A hit at rank 1 and a hit at rank 50 count identically, so
> TOP-K rises with K by construction. That curve is an artifact of the metric, not evidence that
> larger K retrieves better (TODO P5).

**A fifth constraint bounds every exact-match number above.** Only **75 of 129 test cases (58.1%)**
have their correct DRG present anywhere in their own fold's training pool — 105 of the 145 unique
diagnoses occur exactly once in the dataset. A *perfect* retriever therefore caps at 58.1% under
exact matching, which is the context in which the threshold-1.0 scores of 0.18–0.25 should be read.
**This figure was measured on the old folds and has not been re-measured on the grouped ones.**

**Every remaining defect biases in the same direction.** This is the most important sentence in the
README, and it is why "the transformers look slightly better" cannot be reported as a result:

| Defect | Which arm it favours | Why |
|---|---|---|
| Self-grading | **BERT** | each arm judges its own predictions with its own cosine, and a compact space marks its own work leniently (Bio_ClinicalBERT's mean pairwise cosine between *unrelated* diagnoses is 0.83) |
| TOP-K rewards guessing | **BERT** | the baseline abstains on 24.4% of cases, so extra `K` cannot help it; every BERT arm always predicts and gains a free guess per unit of `K` |
| Saturation | **BERT** | all three BERT arms sit at 1.000 at the paper's own threshold, so their behaviour there is unobservable |

Bio_ClinicalBERT is nominally ahead of the baseline at **5 of the 6 aggregators**, and its margin grows
monotonically with `K` (+0.0328 at TOP-10 rising to +0.1681 at TOP-50 — while the baseline's score
plateaus at 0.2383 from TOP-40 on, because it is abstaining). **That is precisely the pattern the three
biases above predict with zero capability difference**, which is also why no paired test reaches
significance. The observed result is currently unfalsifiable, and making it falsifiable is exactly what
the encoder-independent grader (TODO P4) is for.

**The bottom line for anyone quoting this repo.** Two claims are defensible and one is not:

- ✅ **Efficiency.** The transformers match the baseline using ~1/51 the parameters and ~1/17 the disk.
  This is a hardware fact, independent of the metric.
- ✅ **Non-inferiority.** No significant difference between any pair of encoders, at any aggregator or
  threshold. A null result, and the honest headline.
- ❌ **Superiority.** Not supported for *any* encoder, and specifically not for the transformers, since
  every known bias in the metric points that way already.

These numbers are leakage-free and preprocessing-unified, which makes them a defensible
*reproduction*. They remain self-graded and rank-blind, which means they are **not** an encoder
ranking. Say so explicitly rather than letting a reader infer otherwise.

Start at [docs/](docs/README.md); the synthesis is
[07-comparison-validity.md](docs/findings/07-comparison-validity.md) and the corrected results are
[11-corrected-pipeline-first-results.md](docs/findings/11-corrected-pipeline-first-results.md).

---

## Reproducing

```bash
python scripts/make_folds.py --verify                                # regenerate data/folds_grouped/

# corrected pipeline — the numbers in this README
AICDS_PIPELINE=corrected python scripts/run_baseline.py              # Linux only; 20.93 GiB model
python scripts/run_bert_analysis.py --model all --pipeline corrected

# legacy pipeline — bit-identical to the original, for comparison
python scripts/run_baseline.py
python scripts/run_bert_analysis.py --model all
```

`--pipeline` also accepts `folds-only` and `preprocess-only`, which isolate the two halves of the
correction. Neither has been run yet, which is why this README states the combined delta only.

**Comparison PDF.** `results/model_comparison.pdf` covers the **legacy** four-arm run — provenance,
summary table, threshold curves, TOP-K curves, the prediction-rate (degeneracy) page, and the full
threshold × TOP-K grid. **There is no corrected equivalent yet:** `scripts/compare_models.py` carries
16 sanity assertions pinned to legacy values, and they correctly fire and abort when pointed at
`results_corrected/` (TODO P34). The assertions are the only thing verifying the parser reads the
columns correctly, so they get keyed by pipeline rather than deleted.

---

## Baseline Reproduction

The original paper uses **BioSentVec** (700-dimensional sent2vec embeddings trained on PubMed +
MIMIC-III) to compute symptom-level pairwise cosine similarities between patients. Diagnosis
similarity is the MAX similarity across the Cartesian product of ground-truth and predicted
diagnosis descriptions, thresholded to classify true/false positives.

The baseline arm had never executed in this checkout: it crashed on a wrong data path and an unbound
name, and `sent2vec` cannot be built under MSVC, making the arm **Linux-only**. Both were fixed and
it ran end to end on a rented Linux box on 2026-08-05 (10 folds, ~13 min).

**Threshold 0.6, against the published figures:**

| Method | Published | Legacy repro | Corrected |
|--------|:---------:|:------------:|:---------:|
| TOP-10 | 0.489 | **0.4824** | 0.3922 |
| TOP-20 | 0.512 | 0.4824 | 0.4163 |
| TOP-30 | 0.521 | 0.4920 | 0.4316 |

**Under `legacy`, TOP-10 lands within 0.007 of the published figure** — the first successful
reproduction of the paper's headline number by this codebase, and the artifact that settled the
degeneracy question ([09-baseline-first-run.md](docs/findings/09-baseline-first-run.md)). Note P ≠ R
here, unlike every BERT row: the baseline declines to predict on 23.2% of cases under `legacy` and
24.4% under `corrected`.

**Under `corrected`, TOP-10 falls to 0.3922 — about a fifth of the published number was
contamination.** That drop is the finding, not a regression: the legacy path still reproduces 0.4824
bit-for-bit on demand.

Threshold 0.6 is the paper's own operating point, so this table is the correct comparison against
the paper. It is not where the four encoders can be compared to each other — for that, see the
[threshold-1.0 table](#threshold-10-top-10--the-informative-setting) above.

## BERT Extension (Original Contribution)

We replace BioSentVec with three biomedical BERT models that produce 768-dimensional embeddings.
Everything else — fold splits, preprocessing, pruning, aggregation, scoring, thresholds — is shared,
so the encoder is the only intended moving part:

| Model | HuggingFace Path | Training Data | Size |
|-------|-------------------|---------------|:----:|
| Bio_ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | MIMIC-III clinical notes | 416 MB |
| BiomedBERT | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | PubMed abstracts | 420 MB |
| BlueBERT | `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` | PubMed + MIMIC-III | 420 MB |

**Corrected pipeline at threshold 0.6 — still SATURATED across all three BERT arms. This table
measures the metric, not the models; it is included only because 0.6 is the paper's threshold.**

| Method | BioSentVec (corrected) | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------------:|:-----------------:|:----------:|:--------:|
| TOP-10 @ 0.6 | 0.3922 | 1.000 *(saturated)* | 1.000 *(saturated)* | 1.000 *(saturated)* |
| TOP-20 @ 0.6 | 0.4163 | 1.000 *(saturated)* | 1.000 *(saturated)* | 1.000 *(saturated)* |
| TOP-30 @ 0.6 | 0.4316 | 1.000 *(saturated)* | 1.000 *(saturated)* | 1.000 *(saturated)* |

Removing patient leakage moved none of the three BERT arms here, because they were already pinned at
the ceiling. **That is the cleanest demonstration available that saturation and leakage are
independent defects:** fixing the folds cannot rescue a metric that has no headroom.

The informative comparison is the
[threshold-1.0 table](#threshold-10-top-10--the-informative-setting) at the top of this README.

### Runtime — the encoder is not the bottleneck

On the corrected four-arm pod run: **~11.5 minutes** for each of Bio_ClinicalBERT / BiomedBERT /
BlueBERT, and **~13 minutes** for the BioSentVec baseline. Measured across the committed runs,
**embedding is 0.17–0.45% of wall-clock**, while the single-threaded pure-Python cosine loop is over
93%. Total fold time varies only **1.7%** across the three transformers. Making embedding
instantaneous would cut a 21.98-minute run to 21.88 minutes, so **a GPU buys essentially nothing for
these arms** ([details](docs/findings/08-runtime-and-cost.md)).

Results are also **platform-independent**: re-running the BERT arms on x86 Linux reproduced the
Apple-silicon numbers **bit-for-bit, to all 17 significant figures.**

## Visual Summary (README Charts)

> **These charts are from the LEGACY pipeline** — three BERT arms only, no baseline, generated from
> the committed February 2026 runs under `docs/`. They have not been regenerated under `corrected`,
> and `scripts/build_readme_plots.py` currently globs the wrong directory
> ([10-output-path-fragmentation.md](docs/findings/10-output-path-fragmentation.md)). Read them for
> the *shape* of the saturation and TOP-K artifacts, which `corrected` did not change; do not read
> the values as current.

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

## Score Distribution Analysis (Key Finding)

The perfect F1 scores are an artifact of **embedding space compactness** combined with the
**MAX-over-Cartesian-product** evaluation strategy, not genuine diagnostic accuracy.

> **Measured under legacy preprocessing.** The statistics below have not been recomputed since
> diagnosis-text handling was unified, so the exact figures apply to the legacy text. The mechanism
> is unaffected — saturation persists identically under `corrected`, as the tables above show.

**Why the metric saturates:**

1. **Compact embedding spaces** — Biomedical BERT models map diagnosis text into a narrow region.
   Even *unrelated* diagnoses have high cosine similarity:

   | Model | Mean Pairwise Sim | Min Pairwise Sim | Std |
   |-------|:-----------------:|:----------------:|:---:|
   | BiomedBERT | 0.93 | 0.72 | 0.03 |
   | Bio_ClinicalBERT | 0.83 | 0.65 | 0.05 |
   | BlueBERT | 0.72 | 0.48 | 0.07 |

2. **MAX operator amplification** — Taking the maximum similarity across all diagnosis pairs inflates
   scores further. Per-patient MAX similarity exceeds 0.6 for virtually all patient pairs:

   | Model | % of patient pairs with MAX >= 0.6 |
   |-------|:----------------------------------:|
   | Bio_ClinicalBERT | 100.00% |
   | BiomedBERT | 100.00% |
   | BlueBERT | 99.96% |

3. **Conclusion** — The evaluation metric is saturated at threshold 0.6 for BERT models. The F1
   scores cannot discriminate between models or meaningfully compare against the baseline.
   Alternative evaluation strategies (MEAN instead of MAX, DRG code matching, higher thresholds) are
   needed. Note that the ordering inversions documented above show a stricter threshold alone is
   *not* sufficient — it removes the ceiling without making the ranking stable.

Visualizations and full statistics are in
[`docs/score_distribution_analysis/`](docs/score_distribution_analysis/).

![Diagnosis score distributions](docs/score_distribution_analysis/score_distributions.png)
*Diagnosis score distributions across baseline and BERT models.*

![Per-patient maximum similarity distributions](docs/score_distribution_analysis/per_patient_max_distributions.png)
*Per-patient MAX similarity distributions showing saturation behavior.*

```bash
python scripts/analyze_score_distributions.py
```

## Project Structure

```
src/aicds/               # Installable package (src layout; pip install -e .)
  config.py              # PipelineConfig: LEGACY / CORRECTED / FOLDS_ONLY / PREPROCESS_ONLY
  models/                # Baseline (sent2vec) and BERT implementations
  utils/                 # Constants, runtime helpers, cython_utils (pure Python, shared math)
  entity/                # Data classes
  evaluation/            # Evaluation modules
scripts/                 # Entry points
  make_folds.py          # GroupKFold on SUBJECT_ID -> data/folds_grouped/
  run_baseline.py        # BioSentVec baseline (Linux only)
  run_bert_analysis.py   # --model 1|2|3|all  --pipeline legacy|corrected|...
  compare_models.py      # Four-arm comparison PDF
  analyze_score_distributions.py
data/
  folds/                 # Committed 10-fold splits (split on HADM_ID; leaky, pinned for legacy)
  folds_grouped/         # GroupKFold on SUBJECT_ID (generated, gitignored)
  raw/                   # Raw data files
docs/                    # findings/ guides/ reference/ plans/
results/                 # Legacy runs (gitignored)
results_corrected/       # Corrected runs (gitignored)
tests/                   # Includes the byte-exact golden regression net
```

The central design constraint: **both arms share everything except the embedding model.**
Preprocessing, fold loading, diagnosis scoring, and all confusion-matrix math live in
`src/aicds/utils/cython_utils.py` so the two arms stay comparable — that comparability is the point.
See [docs/reference/architecture.md](docs/reference/architecture.md).

## Setup

```bash
conda env create -f config/environment.yml   # env "disease-diagnosis", Python 3.9
conda activate disease-diagnosis
pip install -e .
git config core.hooksPath .githooks          # data-use guard; hooks are not cloned
```

**Key dependencies:** sentence-transformers, torch, matplotlib, numpy, scikit-learn

**For the baseline only:** also requires `sent2vec` and the BioSentVec pre-trained model
(20.93 GiB). The baseline arm is **Linux-only** — `sent2vec` cannot be built under MSVC, which
rejects its GCC-only compiler flags. See [docs/guides/setup.md](docs/guides/setup.md) for the full
procedure, including the macOS/ARM OpenMP conflict and the Linux torch/nltk `LD_LIBRARY_PATH` trap.

**Testing:** `pytest` runs the fast suite in seconds. `pytest -m golden` re-runs the full 10-fold
pipeline and compares it **byte-for-byte** against a committed reference (~44 min). Nothing here is
trained, so every emitted number is a pure function of the input data and the arithmetic in
`cython_utils.py` — meaning any behaviour change is a numerical change, and the realistic failure
mode of refactoring is the numbers moving while every other test stays green. The golden is the only
thing that catches that. If it fails, read the diff; a changed number is the finding.

## Data handling

This repository contains committed MIMIC-III records under a PhysioNet DUA that prohibits
redistribution. **Do not add clinical data to this repository.** `.githooks/pre-commit` blocks new
files under `data/raw`/`data/folds` and any new file containing 20+ distinct `HADM_ID`s. Run
directories under `results/` and `results_corrected/` are gitignored for the same reason — only
aggregates appear in documentation. See [docs/guides/data-use.md](docs/guides/data-use.md).

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

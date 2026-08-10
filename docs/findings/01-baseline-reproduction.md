# Baseline Reproduction Status: BioSentVec (Comito et al. 2022)

> **In plain words.** This document describes the original authors' system — how it predicts a
> diagnosis by finding past patients with similar symptoms — and tracks the effort to reproduce
> their published score. **That effort succeeded**: on 2026-08-05 the baseline ran end-to-end
> and landed within 0.007 of the paper's headline figure
> ([09-baseline-first-run.md](09-baseline-first-run.md)). The method walkthrough below is the
> best plain description of the pipeline in the repo; the history section at the bottom records
> the crash bugs and provenance gaps that had to be closed first, kept short because they are
> closed.

**Status: REPRODUCED.** Legacy TOP-10 F1 = **0.4824** against the published **0.489** — within
0.007 (1.4% relative) — from the full 10-fold run of 2026-08-05 on rented Linux
([09](09-baseline-first-run.md)). The corrected (leakage-free) figure is **0.3922**
([11](11-corrected-pipeline-first-results.md)); the 18.4% drop between the two is a finding
about the published number, not a reproduction failure. The arm is **Linux-only** (`sent2vec`
will not build under MSVC) and needs the 20.93 GiB model file. Path note: written before the
`src/aicds` package move — go by function name, not line number.

This arm is scaffolded from the original authors' released code — it is reference material for
comparison, not the repo owner's contribution. The owner's contribution is the BERT extension
([02-encoder-comparison.md](02-encoder-comparison.md)).

## What this arm is

The paper (Comito, Falcone, Forestiero, *IEEE Access* 2022) predicts a patient's discharge
diagnosis by finding the most similar prior patients based on free-text symptom descriptions,
using pre-trained biomedical sentence embeddings (**BioSentVec**, 700-dimensional sent2vec
vectors trained on PubMed + MIMIC-III) and cosine similarity. `baseline_sent2vec.py` is this
repo's port of that pipeline, driven by `python scripts/run_baseline.py`. The
similarity/aggregation/scoring routines live in `cython_utils.py`, imported under the alias
`util_cy` — a pure-Python re-implementation of what was originally a compiled Cython module
(`util_cy.pyx`; the original filename surfaces in a captured traceback in
[`archive/baseline_debug.txt`](../../archive/baseline_debug.txt)).

## Method, as implemented in this repo's code

### Data

Each admission is one line of `HADM_ID;SUBJECT_ID;ADMITTIME;DISCHTIME;SYMPTOMS;DIAGNOSIS` in
[`data/raw/Symptoms-Diagnosis.txt`](../../data/raw/Symptoms-Diagnosis.txt) (field offsets
defined in `SymptomsDiagnosis.py`). The file contains **129 admissions**
(`grep -c ';' data/raw/Symptoms-Diagnosis.txt` → 129; `wc -l` reports 128 only because the file
has no trailing newline). This is confirmed independently by a captured run log,
`archive/baseline_debug.txt`, which logged `[INFO] Dataset loaded: 129 admissions`.

Symptoms and diagnoses are pre-split into 10 static folds under `data/folds/Fold{0..9}/` (e.g.
`TrainingSet.txt` + `TestSet.txt`), generated once rather than re-split per run. Fold 0 has 116
training / 13 test admissions, which sums to 129 — consistent across folds. Folds are read by
`util_cy.load_dataset()`. (These are the *legacy* folds; the leakage-free `data/folds_grouped/`
came later — [05-patient-leakage.md](05-patient-leakage.md).)

### Preprocessing

- `preprocess_sentence()` lowercases the symptom text, pads a few punctuation marks (`/`, `.`,
  `.-`, `'`) with spaces so they tokenize as separate tokens, then tokenizes with NLTK
  `word_tokenize` and drops punctuation tokens and English stopwords. (This is the function
  whose stopword step destroys `w/o` — [06-preprocessing-defects.md](06-preprocessing-defects.md).)
- `preprocess_diagnosis()` lowercases the diagnosis field, splits it on `--` into individual
  DRG-coded diagnosis strings, strips the `apr:`/`hcfa:`/`ms:` DRG-type prefixes to
  de-duplicate identical diagnosis descriptions that appear under multiple DRG systems, then
  re-attaches the surviving DRG-type prefix(es) to each unique description. A single admission
  can therefore carry more than one diagnosis string — this is why diagnosis scoring below is a
  Cartesian product, not a 1:1 comparison.

### Patient similarity: mean-of-max over symptom pairs

For a test admission and a candidate training admission, `predictS2V()` computes, for every
*test* symptom, the maximum cosine similarity against every *train* symptom of that candidate.
Those per-symptom maxima are summed and divided by
`max(len(test_symptoms), len(train_symptoms))` — not the test-symptom count alone — so patients
with very different symptom-list lengths are penalized. This produces one scalar similarity per
(test admission, candidate training admission) pair; the full `nrow × ncol` matrix is
`similarity_matrix` in `baseline_sent2vec.py`.

### Retrieval: MAX and TOP-K, gated by `PRUNING_SIMILARITY`

`PRUNING_SIMILARITY = 0.5` (`Constants.py`) is a hard floor: a candidate training admission is
eligible to contribute a predicted diagnosis only if its patient-similarity score is ≥ 0.5.
(This floor is also the abstention mechanism: a test case with *no* candidate above it produces
no prediction at all — the behaviour that turned out to separate the baseline from every BERT
arm, [09](09-baseline-first-run.md).)

- **MAX**: the single training admission with the highest similarity above the pruning floor is
  taken as the match; its diagnosis becomes the sole prediction.
- **TOP-K**: the K highest-similarity training admissions above the pruning floor (K ∈
  {10, 20, 30, 40, 50}, from `TOP_K_LOWER_BOUND=10`, `TOP_K_UPPER_BOUND=60`, `TOP_K_INCR=10` in
  `Constants.py`) each contribute a diagnosis-similarity score; the case counts as a hit if
  **any** of the top-K contributes a score ≥ the threshold being evaluated
  (`containGreaterOrEqualsValue`).

### Diagnosis scoring: MAX over the ground-truth × predicted Cartesian product

`get_diagnosis_similarity_by_description_max()` takes the maximum cosine similarity across
**every pair** of (ground-truth diagnosis string, predicted diagnosis string) — the full
Cartesian product, since either side can carry multiple DRG-derived diagnosis strings for one
admission. This single MAX value is then compared against each threshold to decide TP vs. FP.

### Thresholds and metrics

Thresholds `{0.6, 0.7, 0.8, 0.9, 1.0}` are each evaluated independently. For each threshold,
`compute_performance_index()` / `compute_aggregated_performance_index()` computes
`precision = tp/(tp+fp)`, `recall = tp/nrow`, and the harmonic-mean F-score from those two —
the same confusion-matrix/scoring code both arms share. (How the two arms behave differently
under the *same* formulas became finding [04](04-metric-degeneracy.md): the baseline abstains,
so its P ≠ R; the BERT arms never do, so theirs collapse.)

### Cross-validation

10-fold CV (`K_FOLD = 10`, `Constants.py`) over the **pre-computed static folds** described
above — `baseline_sent2vec.py` loops `nFold in range(0, K_FOLD)`, loads that fold's train/test
split, runs `predictS2V` per test admission, then aggregates into per-fold and 10-fold-mean
`PerformanceIndex.txt` output.

## The published numbers, and how the reproduction landed

At threshold 0.6:

| Method | Published | Reproduced (legacy, 2026-08-05) | Corrected |
|--------|:---------:|:-------------------------------:|:---------:|
| TOP-10 | 0.489 | **0.4824** | 0.3922 |
| TOP-20 | 0.512 | 0.4824 | 0.4163 |
| TOP-30 | 0.521 | 0.4920 | 0.4316 |

The TOP-10 row is the headline reproduction claim (within 0.007); TOP-20/30 land within ~0.03.
Precision and recall individually differ more than F1 does (0.562 vs 0.621, 0.426 vs 0.412) —
the signature of a slightly different abstention rate, with the errors partly cancelling in the
harmonic mean. Full run record: [09-baseline-first-run.md](09-baseline-first-run.md).

## History: why reproduction took until 2026-08-05

Kept brief, because it is closed — but the provenance half remains the canonical record of
where the published numbers came from.

**The code had two crash bugs** (a dataset path pointing at the repo root instead of
`data/raw/`, then an unbound name `entity` where the import bound `entity_module`), **fixed in
`c2fee6e`**. Beyond the bugs, running it required the 20.93 GiB
`BioSentVec_PubMed_MIMICIII-bigram_d700.bin` — which loads fully RAM-resident, from the
*current working directory* rather than `data/models/` as that directory's README claims (a
still-open defect) — and a working `sent2vec` binding. The archived
`AttributeError: module 'sent2vec' has no attribute 'Sent2vecModel'`
([`archive/baseline_debug.txt`](../../archive/baseline_debug.txt)) turned out to be caused by
PyPI's unrelated `sent2vec` package; the baseline needs `epfml/sent2vec` built from source,
which is what `pyproject.toml`'s `baseline` extra points at. On Windows even that fails — MSVC
rejects the package's GCC-only compiler flags — so the arm is **Linux-only**
([07-comparison-validity.md](07-comparison-validity.md) has the build log).

**Provenance of the published figures.** The paper trail
([`archive/stale-docs/Reproduce_w_transformers.md`](../../archive/stale-docs/Reproduce_w_transformers.md))
attributes 0.489/0.512/0.521 to a `Prediction_Output_22112025_04-41-14_ORIGINAL_OUTPUTS/`
directory produced by a script `CS2V.py` — neither of which exists in this repository or in any
commit (`git log --all` finds no `CS2V.py`). Until 2026-08-05 the figures were therefore an
inherited citation with no artifact behind them; the run is what graduated them to an
independently reproduced result.

## Summary

| Question | Answer (2026-08-09) |
|---|---|
| Does `python scripts/run_baseline.py` run? | **Yes, on Linux** — fixed in `c2fee6e`, verified by the full 10-fold run of 2026-08-05. Windows is impossible (MSVC rejects sent2vec's GCC-only flags). |
| Were the published numbers reproduced? | **Yes** — TOP-10 within 0.007 (0.4824 vs 0.489); TOP-20/30 within ~0.03. |
| Is there a committed baseline output file? | No — run output is DUA-covered and gitignored; the aggregates live in [09](09-baseline-first-run.md). |
| What does it need to run? | Python 3.9, `pip install -e ".[baseline]"` (epfml/sent2vec, **not** PyPI's), the 20.93 GiB model at the working directory, Linux. |
| Is this arm the repo owner's original contribution? | No — scaffolded from the original authors' code; the reference point for the BERT extension. |
| Is the dataset size 129 or 128 admissions? | 129 (`grep -c ';'`, fold sizes, and a captured run log all agree; `wc -l` = 128 is a trailing-newline artifact). |

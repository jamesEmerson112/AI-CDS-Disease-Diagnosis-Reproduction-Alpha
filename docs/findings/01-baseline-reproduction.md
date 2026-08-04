# Baseline Reproduction Status: BioSentVec (Comito et al. 2022)

**Bottom line: the baseline arm does not run in the current checkout.** It crashes during
dataset loading with a `FileNotFoundError`, and would hit a `NameError` immediately after even
if that were fixed. The 0.489 / 0.512 / 0.521 F1 numbers quoted in [`README.md`](../../README.md)
were **not** produced by the code currently in `src/models/baseline_sent2vec.py` — they are
inherited from an earlier, pre-reorganization version of the script (`CS2V.py`) that no longer
exists in this repository. No baseline `PerformanceIndex.txt` is committed anywhere in this repo
to check them against. Treat the published numbers as an unverified citation from the original
project history, not as a reproduced result.

This arm is scaffolded from the original authors' released code — it is reference material for
comparison, not the repo owner's contribution. The owner's contribution is the [BERT
extension](../bert_model_comparison.md).

## What this arm is

The paper (Comito, Falcone, Forestiero, *IEEE Access* 2022) predicts a patient's discharge
diagnosis by finding the most similar prior patients based on free-text symptom descriptions,
using pre-trained biomedical sentence embeddings (**BioSentVec**, 700-dimensional sent2vec
vectors trained on PubMed + MIMIC-III) and cosine similarity. `src/models/baseline_sent2vec.py`
is this repo's port of that pipeline, driven by `python scripts/run_baseline.py`
([`scripts/run_baseline.py`](../../scripts/run_baseline.py)). The similarity/aggregation/scoring
routines live in [`src/utils/cython_utils.py`](../../src/utils/cython_utils.py), imported under
the alias `util_cy` — a pure-Python re-implementation of what was originally a compiled Cython
module (`util_cy.pyx`; the original filename surfaces in a captured traceback in
[`archive/baseline_debug.txt`](../../archive/baseline_debug.txt)).

## Method, as implemented in this repo's code

### Data

Each admission is one line of `HADM_ID;SUBJECT_ID;ADMITTIME;DISCHTIME;SYMPTOMS;DIAGNOSIS` in
[`data/raw/Symptoms-Diagnosis.txt`](../../data/raw/Symptoms-Diagnosis.txt) (field offsets defined
in [`src/entity/SymptomsDiagnosis.py`](../../src/entity/SymptomsDiagnosis.py)). The file contains
**129 admissions** (`grep -c ';' data/raw/Symptoms-Diagnosis.txt` → 129; `wc -l` reports 128
only because the file has no trailing newline). This is confirmed independently by a captured
run log, `archive/baseline_debug.txt`, which logged `[INFO] Dataset loaded: 129 admissions`
before failing at a later step in the old (pre-reorg) code path.

Symptoms and diagnoses are pre-split into 10 static folds under `data/folds/Fold{0..9}/` (e.g.
`TrainingSet.txt` + `TestSet.txt`), generated once rather than re-split per run. Fold 0 has 116
training / 13 test admissions, which sums to 129 —consistent across folds. Folds are read by
`util_cy.load_dataset()` (`src/utils/cython_utils.py:176-192`).

### Preprocessing

- `preprocess_sentence()` (`cython_utils.py:251-260`) lowercases the symptom text, pads a few
  punctuation marks (`/`, `.`, `.-`, `'`) with spaces so they tokenize as separate tokens, then
  tokenizes with NLTK `word_tokenize` and drops punctuation tokens and English stopwords.
- `preprocess_diagnosis()` (`cython_utils.py:263-288`) lowercases the diagnosis field, splits it
  on `--` into individual DRG-coded diagnosis strings, strips the `apr:`/`hcfa:`/`ms:` DRG-type
  prefixes to de-duplicate identical diagnosis descriptions that appear under multiple DRG
  systems, then re-attaches the surviving DRG-type prefix(es) to each unique description. A
  single admission can therefore carry more than one diagnosis string — this is why diagnosis
  scoring below is a Cartesian product, not a 1:1 comparison.

### Patient similarity: mean-of-max over symptom pairs

For a test admission and a candidate training admission, `predictS2V()`
(`cython_utils.py:40-165`) computes, for every *test* symptom, the maximum cosine similarity
against every *train* symptom of that candidate (`cython_utils.py:59-75`). Those per-symptom
maxima are summed and divided by `max(len(test_symptoms), len(train_symptoms))`
(`cython_utils.py:78-88`) — not the test-symptom count alone — so patients with very different
symptom-list lengths are penalized. This produces one scalar similarity per
(test admission, candidate training admission) pair; the full `nrow × ncol` matrix is
`similarity_matrix` in `src/models/baseline_sent2vec.py:342`.

### Retrieval: MAX and TOP-K, gated by `PRUNING_SIMILARITY`

`PRUNING_SIMILARITY = 0.5` (`src/utils/Constants.py:14`) is a hard floor: a candidate training
admission is eligible to contribute a predicted diagnosis only if its patient-similarity score is
≥ 0.5.

- **MAX**: the single training admission with the highest similarity above the pruning floor is
  taken as the match (`cython_utils.py:93-99`); its diagnosis becomes the sole prediction.
- **TOP-K**: the K highest-similarity training admissions above the pruning floor (K ∈
  {10, 20, 30, 40, 50}, from `TOP_K_LOWER_BOUND=10`, `TOP_K_UPPER_BOUND=60`, `TOP_K_INCR=10` in
  `Constants.py:16-18`) each contribute a diagnosis-similarity score
  (`cython_utils.py:121-146`); the case counts as a hit if **any** of the top-K contributes a
  score ≥ the threshold being evaluated (`containGreaterOrEqualsValue`, `cython_utils.py:366-370`).

### Diagnosis scoring: MAX over the ground-truth × predicted Cartesian product

`get_diagnosis_similarity_by_description_max()` (`cython_utils.py:291-307`) takes the maximum
cosine similarity across **every pair** of (ground-truth diagnosis string, predicted diagnosis
string) — the full Cartesian product, since either side can carry multiple DRG-derived diagnosis
strings for one admission. This single MAX value is then compared against each threshold to
decide TP vs. FP.

### Thresholds and metrics

Thresholds `{0.6, 0.7, 0.8, 0.9, 1.0}` are each evaluated independently
(`init_confusion_matrix()` / `init_performance_matrix()`, `cython_utils.py:350-363`). For each
threshold, `compute_performance_index()` / `compute_aggregated_performance_index()`
(`cython_utils.py:373-436`) computes `precision = tp/(tp+fp)`, `recall = tp/nrow`, and the
harmonic-mean F-score from those two — the same confusion-matrix/scoring code both arms share, so
whatever precision/recall degeneracy exists there (see the metric-saturation analysis in
[`docs/score_distribution_analysis/`](../score_distribution_analysis/)) applies to the baseline
arm too, not only to the BERT arm.

### Cross-validation

10-fold CV (`K_FOLD = 10`, `Constants.py:19`) over the **pre-computed static folds** described
above — `src/models/baseline_sent2vec.py:322-389` loops `nFold in range(0, K_FOLD)`, loads that
fold's train/test split, runs `predictS2V` per test admission, then aggregates into per-fold and
10-fold-mean `PerformanceIndex.txt` output.

## Published baseline numbers

At threshold 0.6, per [`README.md`](../../README.md) (lines 13-19):

| Method | F1 Score |
|--------|:--------:|
| TOP-10 | 0.489 |
| TOP-20 | 0.512 |
| TOP-30 | 0.521 |

These are the numbers the BERT arm is informally compared against. **They cannot currently be
regenerated or checked from anything committed in this repository** — see below.

## Reproduction status

### The current crash

Running `python scripts/run_baseline.py` fails during dataset loading, before the script ever
attempts to load the 700D BioSentVec model:

1. **`FileNotFoundError`** — `src/models/baseline_sent2vec.py:236` builds the dataset path as
   `os.path.join(CH_DIR, "Symptoms-Diagnosis.txt")`, i.e. `<repo-root>/Symptoms-Diagnosis.txt`.
   The file actually lives at `data/raw/Symptoms-Diagnosis.txt`. The `open()` call on the next
   line (`:237`) raises immediately.
2. **`NameError`** — even with the path fixed, `src/models/baseline_sent2vec.py:244-247`
   constructs each admission via `entity.SymptomsDiagnosis.SymptomsDiagnosis(...)`, but the bare
   name `entity` is never bound anywhere in the file. The imports at the top are
   `import src.entity.SymptomsDiagnosis as entity_module` and
   `from src.entity.SymptomsDiagnosis import SymptomsDiagnosis` (`:16-17`) — neither creates an
   `entity` symbol. This is the next thing that would fail.

Both are one-line-scale fixes (correct the path; use `entity_module` or the already-imported
`SymptomsDiagnosis` class directly), but neither has been applied, so the arm has not been run
end-to-end since the repository was reorganized into its current `src/`/`data/` layout.

### What verifying a fix would additionally require

Even with both bugs patched, the script would go on to call `util_cy.load_model()`
(`cython_utils.py:237-248`), which loads
`BioSentVec_PubMed_MIMICIII-bigram_d700.bin` — a **~21 GB** pre-trained model file
(20.9 GB per the archived [`SETUP_GUIDE.md`](../../archive/stale-docs/SETUP_GUIDE.md), line 128) that is not, and should not
be, committed to this repository. It must be downloaded separately before the baseline arm can
execute at all. No such file is present anywhere in this checkout
(`find . -iname "*.bin"` returns nothing).

Separately, a captured debug log from a prior run of the pre-reorg script,
[`archive/baseline_debug.txt`](../../archive/baseline_debug.txt), shows a *different* failure at
that same model-loading step — `AttributeError: module 'sent2vec' has no attribute
'Sent2vecModel'` — which points at a fragile/version-sensitive `sent2vec` Python binding as a
further obstacle independent of the two bugs above. This is evidence from a different
environment (a GT cluster path, per the log) at a different point in time, so it doesn't
necessarily apply to whatever environment reproduces the published numbers, but it does mean a
successful end-to-end baseline run has not been demonstrated in *any* artifact currently in this
repo.

### Provenance of the published numbers

`docs/Reproduce_w_transformers.md` — itself stale on the current file layout, but the closest
thing to a paper trail — states the baseline "has been successfully implemented and evaluated,
with results stored in `Prediction_Output_22112025_04-41-14_ORIGINAL_OUTPUTS/`" and gives a
performance table matching the numbers in `README.md`. Neither that output directory nor the
`CS2V.py` script it refers to exists in this repository; `git log --all` finds no history for a
file named `CS2V.py`. The three `PerformanceIndex.txt` files actually committed in this repo
(`docs/Prediction_Output_{Bio_ClinicalBERT,BiomedBERT,BlueBERT}_*/PerformanceIndex.txt`) are all
from BERT-model runs — **there is no committed baseline `PerformanceIndex.txt` to check the
0.489/0.512/0.521 figures against.** They should be treated as inherited documentation from the
original project's pre-reorganization phase, not as a result this repo can currently reproduce or
verify.

## Summary

| Question | Answer |
|---|---|
| Does `python scripts/run_baseline.py` run today? | No — crashes at dataset load (`FileNotFoundError`, then a masked `NameError`). |
| Are the two crash bugs simple to fix? | Yes, both are one-line path/import fixes. |
| Would fixing them be enough to reproduce the published numbers? | No — also needs the ~21 GB BioSentVec model file (not in repo) and a working `sent2vec` install. |
| Is there a committed baseline output file to check against? | No. Only BERT-model `PerformanceIndex.txt` files are committed. |
| Is this arm the repo owner's original contribution? | No — it is scaffolded from the original authors' code and serves as the reference point for the [BERT extension](../bert_model_comparison.md). |
| Is the dataset size 129 or 128 admissions? | 129 (`grep -c ';'`, fold sizes, and a captured run log all agree; `wc -l` = 128 is a trailing-newline artifact). |

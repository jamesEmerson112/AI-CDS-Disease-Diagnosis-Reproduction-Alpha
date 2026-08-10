# Patient-level leakage in the cross-validation folds

> **In plain words.** The evaluation is supposed to test whether the system can find *other*
> patients who resemble a new one. But the data split was cut by hospital *visit*, not by
> *person* — and 29 of the 100 people in the dataset have more than one visit (one person has
> fifteen). So for a third of the test cases, the "similar patient" the system found was
> literally the same person on a different visit, with nearly the same symptoms and the same
> recurring diagnosis. That's not prediction; it's finding someone's own chart under a
> different number. The smoking gun: on exactly those cases, all three AI models score
> *identically* — because when the answer key is in the pool, it doesn't matter who's
> searching. **Fixed on 2026-08-05** by re-cutting the folds so a person's visits always stay
> on the same side of the split.

> **Status: FIXED 2026-08-05 (`c2115ba`).** `scripts/make_folds.py` regenerates the folds with
> `GroupKFold` on `SUBJECT_ID` → `data/folds_grouped/`; leaked test cases went from **41 to 0**,
> recounted independently rather than trusting the tool. The `legacy` config keeps the old
> leaky folds on purpose so the golden regression never moves. The measured cost of the fix —
> every arm's score drops, the baseline's most — is in
> [11-corrected-pipeline-first-results.md](11-corrected-pipeline-first-results.md).

**Bottom line: 41 of 129 test cases (31.8%) had another admission from the same patient sitting
in their own retrieval pool.** The folds split on `HADM_ID` (admission) while the data contains
repeat patients, so a test admission could retrieve the same person's earlier or later chart.
The measured inflation at threshold 1.0 is **+0.11 to +0.26** — roughly an order of magnitude
larger than the differences between the encoders this project set out to compare.

This affected **both arms equally**. The folds are static committed files shared by the
BioSentVec baseline and the BERT extension, so the published 0.489 / 0.512 / 0.521 carry the
same bias as the BERT results.

## The mechanism

`data/raw/Symptoms-Diagnosis.txt` has two identity columns (`SymptomsDiagnosis.py`):

- `HADM_ID` — one hospital **admission** (one stay)
- `SUBJECT_ID` — one **patient** (one human being)

**129 admissions come from 100 distinct patients.** 86 patients contributed one admission, 11
contributed two, 2 contributed three, and one patient — `SUBJECT_ID 41976` — contributed
**15 admissions, 11.6% of the entire dataset.**

The folds partition on admission. So admission #3 of patient 41976 can land in a test set while
admissions #1, #2, #4 … #15 remain in the training pool. Retrieval then finds the same person,
and "find a clinically similar patient" degenerates into "find this patient's own chart under a
different admission number."

That patient's own admissions overlap substantially: mean Jaccard similarity between their
symptom lists is **0.27**, maximum **0.69**. Their diagnoses are largely recurrent sepsis.

Nothing was improperly accessed. The answer key was simply duplicated inside the searchable
pool under a different ID.

## Measurement

Contamination per fold — a test case counts as contaminated when its `SUBJECT_ID` also appears
in that fold's `TrainingSet.txt`:

| Fold | test n | contaminated | rate |
|---|---:|---:|---:|
| 0 | 13 | 5 | 38.5% |
| 1 | 13 | 3 | 23.1% |
| 2 | 13 | 2 | 15.4% |
| 3 | 13 | 2 | 15.4% |
| 4 | 13 | 7 | 53.8% |
| 5 | 13 | 5 | 38.5% |
| 6 | 13 | 5 | 38.5% |
| 7 | 13 | 5 | 38.5% |
| 8 | 13 | 4 | 30.8% |
| 9 | 12 | 3 | 25.0% |
| **total** | **129** | **41** | **31.8%** |

13 of the 14 multi-admission patients leak in at least one fold. `SUBJECT_ID 41976` leaks in
all ten.

## What it is worth

The per-patient blocks in the committed `PerformanceIndex.txt` files carry `HADM_ID`, so the
contaminated and clean subsets can be scored separately. At threshold 1.0 — the only threshold
nothing saturates:

| Model | Method | leaked (n=41) | clean (n=88) | gap |
|---|---|---:|---:|---:|
| Bio_ClinicalBERT | MAX | 0.293 | 0.080 | +0.213 |
| BiomedBERT | MAX | 0.293 | 0.114 | +0.179 |
| BlueBERT | MAX | 0.293 | 0.125 | +0.168 |
| Bio_ClinicalBERT | TOP-10 | 0.415 | 0.227 | +0.187 |
| BiomedBERT | TOP-10 | 0.415 | 0.182 | +0.233 |
| BlueBERT | TOP-10 | 0.415 | 0.159 | +0.256 |
| Bio_ClinicalBERT | TOP-50 | 0.512 | 0.352 | +0.160 |
| BiomedBERT | TOP-50 | 0.488 | 0.330 | +0.158 |
| BlueBERT | TOP-50 | 0.512 | 0.375 | +0.137 |

**The strongest evidence that this is the mechanism, not a coincidence:** on contaminated cases
all three encoders score *identically* — 0.293 at MAX (12/41), 0.415 at TOP-10 (17/41). On
clean cases they diverge (0.080 / 0.114 / 0.125). When a patient's own prior chart is in the
pool, retrieval finds it regardless of which encoder computes the distance. The encoder stops
mattering.

## Why this dominates the encoder comparison

| Quantity | Magnitude |
|---|---|
| Leakage inflation (threshold 1.0) | **+0.11 to +0.26** |
| Per-fold standard deviation (threshold 1.0) | 0.071 – 0.124 |
| Difference *between encoders* (threshold 1.0) | 0.015 – 0.046 |

The contamination effect is 5–10× the effect under study, and the encoder differences do not
clear the fold-to-fold noise floor at all. No encoder ranking drawn from the legacy results is
supportable.

## What is *not* wrong

The fold partition is valid at the admission level: every one of the 129 `HADM_ID`s appears in
exactly one `TestSet.txt`, with no duplicates and no omissions, and every train/test pair sums
to 129 (folds 0–8 are 116/13; fold 9 is 117/12). There are also no exact duplicate records — no
two admissions share an identical SYMPTOMS string or an identical (SYMPTOMS, DIAGNOSIS) pair.

**The data is not dirty.** It correctly records that one person had 15 hospital stays. The
defect is in how the folds were cut from it, which is why the fix is a fold regeneration rather
than a data cleaning pass.

## Fix — landed 2026-08-05 (`c2115ba`)

Regrouped on `SUBJECT_ID` using `sklearn.model_selection.GroupKFold`:
`python scripts/make_folds.py --verify` writes `data/folds_grouped/` (gitignored, regenerated
deterministically — never overwrite `data/folds/`, which the golden depends on). The
regenerated files respect the trailing-newline hazard in `load_dataset` (it silently drops each
line's last character; a fold file without a final newline loses its last symptom).

Two consequences, both anticipated in writing:

- **The headline numbers fell.** Baseline TOP-10 @ 0.6 went 0.4824 → 0.3922; every arm dropped
  at threshold 1.0. That is the correction working — see
  [11](11-corrected-pipeline-first-results.md).
- **The grouped folds are uneven** (114/15 through 117/12), because whole patients must stay
  together and one patient holds 15 admissions. Per-fold *n* now varies; treat per-fold σ
  accordingly.

One independence result worth quoting: regrouping the folds moved the exact-match retrieval
ceiling by exactly **one case** (75/129 → 76/129), so leakage and label-sparsity really are
independent defects ([12](12-drg-grader.md)).

## Reproducing

Contamination count on the legacy folds:

```python
raw = [l.split(';') for l in open('data/raw/Symptoms-Diagnosis.txt').read().split('\n') if l.strip()]
h2s = {r[0]: r[1] for r in raw}
total = leaked = 0
for f in range(10):
    tr = [l.split('_')[0] for l in open(f'data/folds/Fold{f}/TrainingSet.txt') if l.strip()]
    te = [l.split('_')[0] for l in open(f'data/folds/Fold{f}/TestSet.txt') if l.strip()]
    pool = {h2s[h] for h in tr if h in h2s}
    total += len(te)
    leaked += sum(1 for h in te if h2s.get(h) in pool)
print(leaked, '/', total)          # -> 41 / 129
```

Run the same loop over `data/folds_grouped/` (after `python scripts/make_folds.py --verify`)
and it prints `0 / 129`.

The leaked-vs-clean score comparison parses the per-patient blocks of each committed
`PerformanceIndex.txt`, keyed on the `HADM_ID=` header lines, and averages the TP column within
each subset.

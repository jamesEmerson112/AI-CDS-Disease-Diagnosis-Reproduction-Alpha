# Correctness fixes — the work that must land before any encoder claim

This is the ordered TODO for making the encoder comparison valid. It is separate from
[revival-roadmap.md](revival-roadmap.md), which is about code organisation. Nothing here is a
refactor; every item changes a number.

Ordered by (impact / effort). Metric details live in [metric-redesign.md](metric-redesign.md).

---

## 1. Regroup the folds by patient — `GroupKFold`

**Status:** not started. **Effort:** hours. **Impact:** largest single correction available.

The 10 folds split on `HADM_ID` (admission), but 129 admissions come from only **100 patients**.
One patient has 15 admissions — 11.6% of the dataset by itself.

**41 of 129 test cases (31.8%) have another admission from the same `SUBJECT_ID` sitting in their
own retrieval pool.** Per-fold contamination ranges 15%–54%. Measured inflation at threshold 1.0:
leaked cases score **+0.11 to +0.26** above clean cases.

For scale: the differences *between encoders* at that threshold are 0.015–0.046. The contamination
is roughly an order of magnitude larger than the effect being studied.

The tell that this is real: on contaminated cases all three encoders score **identically**
(0.293 at MAX, 0.415 at TOP-10) while on clean cases they diverge (0.080 / 0.114 / 0.125).
When a patient's own prior chart is in the pool, retrieval finds it regardless of encoder.

**Fix:** regenerate the fold files grouping on `SUBJECT_ID` (`sklearn.model_selection.GroupKFold`).
The folds are static committed files, so this is a data regeneration, not a code change.

**Watch out:** `cython_utils.py:184` does `line[line.index("_") + 1: len(line) - 1]` — it drops the
last character unconditionally, assuming a trailing newline. The committed files end `...NOS.\n`
so this currently strips the newline correctly. **Any regenerated fold file must end with a
trailing newline** or the last symptom of the last line silently loses a character.

**Also note:** with one patient holding 15 admissions, grouped folds will be noticeably uneven in
size. That is correct and expected — report the per-fold n.

See [../findings/05-patient-leakage.md](../findings/05-patient-leakage.md).

---

## 2. Replace the cosine grader with a DRG-based one

**Status:** not started. **Effort:** days. **Impact:** makes cross-encoder comparison valid at all.

Today the same encoder both retrieves candidates and judges whether the retrieved diagnosis is
correct (`cython_utils.py:291-307`). A model with a more compressed embedding space therefore
grades itself more leniently — which is exactly why BiomedBERT, the most compressed of the three,
"wins" at threshold 0.9.

The information needed for an independent grader is already in the data: the DIAGNOSIS field
carries DRG group names. String comparison needs no embedding.

**Do not use exact match alone** — it has a hard ceiling of **58.1%** (see below). Use graded
relevance: full credit for an exact DRG match, partial credit for a same-family match.

See [metric-redesign.md](metric-redesign.md) option C.

---

## 3. Report rank-aware metrics — MRR, Recall@K, Precision@K

**Status:** not started. **Effort:** days. **Impact:** removes the TOP-K artifact.

Rank *is* computed (`bert_models.py:170` sorts by similarity) and then discarded: the scorer
returns true if **any** of the K retrieved clears the threshold, so a hit at rank 1 and a hit at
rank 50 count identically.

Consequence: `score(TOP-50) >= score(TOP-10)` is guaranteed by construction, since the top-50 set
contains the top-10 set. Verified monotonic in 18/18 model × threshold combinations, zero
violations. The TOP-K curve in the README is arithmetic, not a finding.

Rank-aware metrics also happen to be **scale-free** — they never look at the absolute cosine
value, so they are immune to both the saturation and the per-model calibration problems.

---

## 4. Fix the preprocessing defects

**Status:** not started. **Effort:** hours. **Impact:** moderate, but two are clinically severe.

### 4a. `w/o` collapses to `w` — negation is destroyed

`preprocess_sentence` (`cython_utils.py:251-260`) pads `/` with spaces, so `w/o` tokenises to
`["w", "/", "o"]`. The `/` is dropped as punctuation and **`o` is an NLTK English stopword**, so it
is dropped too. Verified:

```
"Tracheostomy w/o Extensive Procedure"  ->  'tracheostomy w extensive procedure'
"Tracheostomy w   Extensive Procedure"  ->  'tracheostomy w extensive procedure'
```

Two clinically opposite DRG groups become the same string. `Dvrtcli colon w/o hmrhg`
(diverticulitis *without* haemorrhage) becomes `dvrtcli colon w hmrhg`.

Affects **symptoms in both arms**, and diagnosis text in the baseline arm only (the BERT path
skips `preprocess_sentence` for diagnoses — see 4c).

**Fix:** protect `w/o` before the slash-padding step, or drop `o` from the stopword set for this
corpus. Any change here moves every number, so it needs its own commit.

### 4b. Comma-split fragments

The SYMPTOMS field is comma-delimited, but some ICD-9 short titles contain commas, so
`cython_utils.py:185` shreds them. **80 of 1,805 tokens (4.4%)** are orphan fragments.

```
"Pneumonia, organism NOS"      ->  "Pneumonia" + " organism NOS"
"Pressure ulcer, stage I"      ->  "Pressure ulcer" + " stage I"
"Pressure ulcer, stage IV"     ->  "Pressure ulcer" + " stage IV"
```

Severity is destroyed (stage I and stage IV collapse to the same token), and the junk fragments get
embedded as if they were symptoms. `" organism NOS"` appears in 26 admissions — since patient
similarity is mean-of-max, any two of those get a spurious **1.0** contribution from a token
carrying no clinical content.

**Fix — deterministic, no model needed.** All 80 fragments begin with a leading space, because the
separator is `,` while intra-label commas are `, `. So: *if a token starts with a space, rejoin it
to the previous token.* Verified to recover every fragment (573 → 564 unique symptom strings,
producing real titles like `Ac kidny fail, tubr necr` and `Dysphagia, oropharyngeal`).

### 4c. The two arms preprocess diagnosis text differently

The baseline calls `preprocess_sentence(diagnosis_description)` before embedding
(`cython_utils.py:226`). The BERT path does **not** (`bert_models.py:318-332`) — it embeds the raw
string. 119 of 145 (82.1%) unique diagnosis descriptions differ between the two paths.

This breaks the project's central design constraint (both arms share everything except the
embedding model), so any baseline-vs-BERT number is confounded by preprocessing, not just encoder.

---

## 5. Centre the embeddings before computing cosine

**Status:** not started. **Effort:** one line. **Impact:** unknown, worth measuring.

BERT sentence embeddings are anisotropic — they occupy a narrow cone rather than spreading over
the sphere, so arbitrary pairs land at high cosine. This is a documented property of the models
(Ethayarajh 2019; the representation-degeneration literature), not of MIMIC-III.

Demonstrated with random vectors: adding a shared offset to otherwise-unrelated 768D vectors moves
mean pairwise cosine from 0.001 to 0.94, reproducing the three models' measured similarity profiles
with no medical content whatsoever.

**Fix:** subtract the mean embedding from every vector before computing cosine (the core of
"BERT-whitening"). If it works, the scales become comparable across models and **a single shared
threshold becomes meaningful again** — which preserves the clean design the original protocol
wanted.

Note this is a genuinely different experiment from the `normalize_embeddings=False` change at
`bert_models.py:337-343`, which cannot affect results at all: L2-normalising before a cosine that
already divides by the norms is algebraically a no-op.

---

## The 58.1% ceiling — read before designing any exact-match metric

The method predicts by copying an answer from a retrieved patient. **The answer must therefore
exist in the pool to be copied.**

145 unique diagnosis descriptions across 129 admissions; 105 of them (72.4%) occur exactly once in
the entire dataset. If a test patient's diagnosis is one of those, no other patient has it, so
nothing in the pool can supply the right answer.

Measured across the 10 folds: **only 75 of 129 test cases (58.1%) have their correct DRG present
anywhere in their fold's training pool.** The other 41.9% are unwinnable under exact matching —
not because retrieval is bad, but because the answer is not in the library.

A perfect retriever therefore scores at most 58.1% under exact match. This is why graded relevance
is preferred, and why any exact-match number must be reported against the ceiling rather than
against 1.0.

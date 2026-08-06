# 07 — Is the BioSentVec-vs-BERT comparison valid yet?

**Short answer: no, and for five separable reasons.** This document exists because the project's
headline deliverable is a comparison — one 700D non-contextual baseline against three 768D
contextual biomedical encoders — and each reason below independently invalidates it. Fixing one
does not fix the others. Findings 01–06 each cover a single defect; this one is the synthesis that
says what the comparison would need in order to mean anything.

Numbers marked **(verified 2026-08-05)** were re-measured on the Windows workstation during the
Phase 2 refactor. The rest are carried from findings 01–06 and are cited to them.

---

## 1. There is no BioSentVec side

The baseline has **never run in this checkout**. The published 0.489 / 0.512 / 0.521 come from the
pre-reorganisation `CS2V.py`, not from any code currently in the repository
([01-baseline-reproduction.md](01-baseline-reproduction.md)). Four independent blockers, all
confirmed 2026-08-05:

| Blocker | State |
|---|---|
| `baseline_sent2vec.py:236` reads `Symptoms-Diagnosis.txt` from the repo root; it lives in `data/raw/` | `FileNotFoundError` at import time |
| `:244-247` reference an unbound name `entity`; line 16 binds `entity_module` | `NameError` at import time |
| `sent2vec` not installed. Two of three manifests named PyPI `sent2vec` — an **unrelated project with no `Sent2vecModel` class**, which is what produced the archived `AttributeError` in `archive/baseline_debug.txt`. `config/requirements.txt` alone had the correct `git+https://github.com/epfml/sent2vec.git` | Corrected in `pyproject.toml` 2026-08-05. **Windows build now confirmed impossible as-is (verified 2026-08-05)** — see below |
| The 20.93 GiB model loads fully resident. It **is** present, at the repo root (not `data/models/`, which holds only a README) | 31.94 GiB total RAM, 8.66 GiB free at measurement **(verified 2026-08-05)** |

### The `sent2vec` build, resolved (2026-08-05)

Attempted directly, `pip install "sent2vec @ git+https://github.com/epfml/sent2vec.git"` against the
repo venv (Python 3.9.13) with MSVC 14.44.35207 present. It resolves to commit `9efbc2d`, gets as
far as building the wheel, and then fails:

```
cl : Command line error D8021 : invalid numeric argument '/Wno-cpp'
```

`setup.py` passes the GCC warning flag `-Wno-cpp`; MSVC reads it as `/Wno-cpp` and rejects it. This
is the *first* GCC-only flag, not the only one — `-std=c++0x`, `-pthread`, `-march=native` and
`-funroll-loops` sit behind it, and MSVC has no `-march=native` equivalent at all. Patching one flag
would simply surface the next. Nothing was left installed; the failure is clean.

**Conclusion: the baseline arm cannot be built on Windows/MSVC without patching upstream's flag
list, and is therefore a Linux-only proposition.** Two routes, in order of preference:

1. **WSL2.** Already installed here (Ubuntu 24.04.3 LTS), and the repo plus the 20.93 GiB model are
   visible at `/mnt/c/...`. Costs nothing and **no data leaves the machine**, which avoids the
   PhysioNet DUA question entirely. Three prerequisites, all needing elevation and therefore the
   owner's hands: `build-essential` is absent, the system Python is 3.12.3 rather than the required
   3.9, and WSL2 sees only 15 GiB of the host's 31.94 GiB (its default ~50%), short of the model's
   20.93 GiB — raising it needs a `.wslconfig` `memory=` entry and `wsl --shutdown`.
2. **A rented Linux box** (RunPod or similar) if WSL2's memory ceiling proves impractical. Choose a
   **CPU- and RAM-heavy instance, not a GPU one**: the fold loop is single-threaded pure-Python
   `cosine_similarity` at 99%+ of runtime, so GPU spend buys nothing here. Note this route does put
   DUA-covered data on third-party infrastructure, which is the owner's call.

If neither route is taken, the framing must change — for example three BERT variants against a
TF-IDF or averaged-word-vector baseline, explicitly labelled a substitute rather than a reproduction
of Comito et al.

## 2. The two arms do not preprocess the same text

This is the defect specific to *cross-arm* comparison, and the cheapest to fix.

- Baseline: `cython_utils.py:244` — `model.embed_sentence(preprocess_sentence(diagnosis_description))`
- BERT: `bert_models.py:283-285` — takes the raw slice after `':'` with **no `preprocess_sentence` call**

**(verified 2026-08-05.)** Consequence: **119 of 145 (82.1%) diagnosis descriptions differ between
the arms** ([06-preprocessing-defects.md](06-preprocessing-defects.md)). Any baseline-versus-BERT
delta measured today is substantially measuring the preprocessing difference, not the encoder. The
project's stated central constraint is that both arms share everything except the embedding model;
that constraint is already broken at this line.

## 3. Every reported "F1" is accuracy

Across **all 12,600 rows** of committed BERT results, precision == recall == F-score, with zero
exceptions ([04-metric-degeneracy.md](04-metric-degeneracy.md)). The mechanism: every test case
increments exactly one of TP or FP, so `tp + fp == nrow`; precision reduces to `tp/nrow`, which
*is* the recall definition in use; and the harmonic mean of two equal numbers is that number.

This is threshold-independent — it is not a tuning artifact. It does **not** extend to the
baseline: the archived TOP-10 baseline run reports P=0.621 / R=0.412 / F1=0.489, i.e. P≠R, meaning
that run abstained on cases with no candidate above `PRUNING_SIMILARITY`. So degeneracy is
plausibly a *consequence* of BERT's compact embedding space rather than a structural property of
the code — but no artifact survives to confirm it.

## 4. All three encoders saturate

Every BERT model scores 1.000 at threshold 0.6 ([03-metric-saturation.md](03-metric-saturation.md)).
Two causes compound: biomedical embeddings are compact — mean pairwise cosine 0.72–0.93 **even
between unrelated diagnoses** — and the aggregator takes a MAX over the Cartesian product of
predicted × true diagnoses, which amplifies that until ~100% of patient pairs clear 0.6.

A corollary worth stating separately: TOP-K scores rise monotonically with K because a single hit
suffices and there is no penalty for the other K−1 predictions. That curve is an artifact of the
metric, not a property of the encoders.

## 5. The folds leak patients, by roughly ten times the effect under study

The folds split on `HADM_ID`, but the 129 admissions come from only **100 distinct patients** — one
patient contributes 15. **41 of 129 test cases (31.8%) have another admission from the same
`SUBJECT_ID` in their own retrieval pool** ([05-patient-leakage.md](05-patient-leakage.md)).

Measured inflation at threshold 1.0 is **+0.11 to +0.26**, against encoder differences of
**0.015–0.046** and per-fold σ of 0.071–0.124. The contamination is about an order of magnitude
larger than the signal being measured. The diagnostic tell: on leaked cases all three encoders
score *identically* (0.293 MAX, 0.415 TOP-10); on clean cases they diverge.

Because the fold files are shared static artifacts, **this affects both arms** — the published
0.489/0.512/0.521 carry it too.

---

## The ceiling nobody set out to impose

Before designing any exact-match metric: only **75 of 129 test cases (58.1%)** have their correct
DRG present anywhere in their fold's training pool, because 105 of the 145 unique diagnoses occur
exactly once in the dataset. **A perfect retriever caps at 58.1% under exact matching.** Any metric
that rewards only exact hits is therefore measuring dataset sparsity as much as model quality,
which is the argument for graded relevance over exact match.

**Re-measured 2026-08-06 on the grouped folds: 76/129 = 58.9%.** Regrouping the folds by
`SUBJECT_ID` moved retrievability by exactly one case, so leakage and sparsity are independent
defects — fixing the split does not make more diagnoses findable. Both figures are now pinned by
`tests/test_drg_grader.py`. Per fold the ceiling ranges from **3/12 (25%) to 13/15 (87%)** on the
grouped split, which is itself a source of the per-fold variance reported elsewhere.

The "105" is document frequency over the deduped labels, not occurrence count over raw entries
(which is 85). It is the correct one for a *retrieval* bound, and the distinction is explained in
[12-drg-grader.md](12-drg-grader.md) — an attempted "correction" to 85 was abandoned.

## What would make the comparison valid, in order

1. **Unify diagnosis preprocessing across arms.** Without this the arms are not comparable at all.
   Cheapest fix, highest leverage on the specific question.
2. **Re-split the folds with `GroupKFold` on `SUBJECT_ID`.** The largest single correction
   available, and it applies to both arms.
3. **Replace the metric** with a genuine set-level P/R/F1 over diagnosis sets, or a rank-aware
   family (Recall@K, MRR, nDCG) against graded DRG relevance. Then "F1" means F1.
4. **Get the baseline to actually run** under the corrected pipeline. Note the dependency: once
   steps 2 and 3 land, the published numbers are no longer a valid reference, because they were
   produced under the leaky folds and the degenerate metric. A fair comparison *requires* re-running
   BioSentVec, which is why the `sent2vec` build sits on the critical path rather than beside it.

Steps 1–3 deliberately change every number the pipeline emits. That is why they are tracked
separately from the refactor in [../plans/correctness-fixes.md](../plans/correctness-fixes.md) and
[../plans/metric-redesign.md](../plans/metric-redesign.md): the refactor's only safety mechanism is
that the numbers must *not* move, so the two kinds of work cannot run concurrently.

## What can be claimed today, defensibly

Not "BiomedBERT beats BioSentVec." The honest, and considerably more interesting, claims are:

- The published metric does not measure what its column headers say. In the BERT results it is
  accuracy wearing an F1 label, provably, in all 12,600 rows.
- The evaluation saturates because of a structural interaction between compact biomedical embedding
  spaces and a MAX aggregator — a reusable lesson about evaluating retrieval over any domain-tuned
  embedding space, not a quirk of this dataset.
- The published result carries patient leakage of roughly ten times the magnitude of the encoder
  differences it reports, so the reported encoder ranking is not supported by the experiment.
- The dataset imposes a 58.1% ceiling on any exact-match retrieval metric, which no model choice
  can lift.

Each of these is a reproduction finding: the value is in showing the original result is not
measuring the claimed quantity, and in quantifying by how much.

---

*Companion documents:* [01](01-baseline-reproduction.md) baseline state ·
[02](02-encoder-comparison.md) encoder comparison · [03](03-metric-saturation.md) saturation ·
[04](04-metric-degeneracy.md) degeneracy · [05](05-patient-leakage.md) leakage ·
[06](06-preprocessing-defects.md) preprocessing

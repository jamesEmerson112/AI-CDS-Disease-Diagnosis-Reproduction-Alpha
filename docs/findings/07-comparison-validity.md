# 07 — Is the BioSentVec-vs-BERT comparison valid yet?

> **In plain words.** This is the document that ties the individual findings together. When it
> was written the answer to its own title was a flat *no*: the old model had never even run
> here, the two sides were fed differently-cleaned text, the reported "F1" was really accuracy,
> the scoring bar was set below everyone's feet, and a third of the test patients could be
> "predicted" by finding their own earlier hospital visit. Since then, four of the five problems
> have been fixed and the fifth has been fenced off. **The comparison is valid now — and its
> answer is a tie.** No encoder measurably beats any other; the modern models just do it with
> 1/51 the parameters and 1/17 the disk.

**Short answer, as originally written (2026-08-05): no, and for five separable reasons.**
**Status, 2026-08-08: reasons 1, 2, 3 and 5 are resolved; reason 4 is open but confined to
cosine grading, which the `drg` pipeline no longer uses.** Each reason below keeps its original
text — they document why the *published* comparison was invalid, which remains true — with its
resolution attached. Fixing one never fixed the others; they really were five independent
defects, which is why they took five separate fixes.

Numbers marked **(verified 2026-08-05)** were re-measured on the Windows workstation during the
Phase 2 refactor. The rest are carried from findings 01–06 and are cited to them.

---

## 1. There is no BioSentVec side — RESOLVED 2026-08-05

**Resolution: the baseline ran end-to-end on rented Linux
([09-baseline-first-run.md](09-baseline-first-run.md)), reproducing the published TOP-10 F1 to
within 0.007 (0.4824 vs 0.489).** The text below records the state that had to be fixed.

The baseline had **never run in this checkout**. The published 0.489 / 0.512 / 0.521 come from
the pre-reorganisation `CS2V.py`, not from any code currently in the repository
([01-baseline-reproduction.md](01-baseline-reproduction.md)). Four independent blockers, all
confirmed 2026-08-05:

| Blocker | State then | State now |
|---|---|---|
| `baseline_sent2vec.py:236` read `Symptoms-Diagnosis.txt` from the repo root; it lives in `data/raw/` | `FileNotFoundError` at import time | Fixed in `c2fee6e` |
| `:244-247` referenced an unbound name `entity`; line 16 binds `entity_module` | `NameError` at import time | Fixed in `c2fee6e` |
| `sent2vec` not installed. Two of three manifests named PyPI `sent2vec` — an **unrelated project with no `Sent2vecModel` class**, which is what produced the archived `AttributeError` in `archive/baseline_debug.txt`. `config/requirements.txt` alone had the correct `git+https://github.com/epfml/sent2vec.git` | Corrected in `pyproject.toml` 2026-08-05 | Working on Linux; **Windows remains impossible** — see below |
| The 20.93 GiB model loads fully resident | 31.94 GiB total RAM, 8.66 GiB free at measurement **(verified 2026-08-05)** | Run on a 64 GB pod |

### The `sent2vec` build, resolved (2026-08-05)

Attempted directly, `pip install "sent2vec @ git+https://github.com/epfml/sent2vec.git"` against
the repo venv (Python 3.9.13) with MSVC 14.44.35207 present. It resolves to commit `9efbc2d`,
gets as far as building the wheel, and then fails:

```
cl : Command line error D8021 : invalid numeric argument '/Wno-cpp'
```

`setup.py` passes the GCC warning flag `-Wno-cpp`; MSVC reads it as `/Wno-cpp` and rejects it.
This is the *first* GCC-only flag, not the only one — `-std=c++0x`, `-pthread`, `-march=native`
and `-funroll-loops` sit behind it, and MSVC has no `-march=native` equivalent at all. Patching
one flag would simply surface the next. Nothing was left installed; the failure is clean.

**Conclusion: the baseline arm cannot be built on Windows/MSVC without patching upstream's flag
list, and is therefore a Linux-only proposition.** The route actually taken was the second of
the two identified: a rented CPU-heavy Linux box (RunPod), chosen CPU-heavy rather than GPU
because the fold loop is single-threaded pure-Python `cosine_similarity` at 99%+ of runtime
([08-runtime-and-cost.md](08-runtime-and-cost.md)), so GPU spend buys nothing. Note this route
does put DUA-covered data on third-party infrastructure, which was the owner's call to make.

## 2. The two arms do not preprocess the same text — RESOLVED 2026-08-05 (`c2115ba`)

**Resolution: both arms now preprocess diagnosis text identically under the `corrected` and
`drg` configs — 145/145 descriptions match. `legacy` keeps the divergence on purpose, so the
golden regression never moves. A `legacy` cross-arm delta is therefore still confounded; a
`corrected` one is not.**

This was the defect specific to *cross-arm* comparison, and the cheapest to fix.

- Baseline: `cython_utils.py` — `model.embed_sentence(preprocess_sentence(diagnosis_description))`
- BERT: `bert_models.py` — took the raw slice after `':'` with **no `preprocess_sentence` call**

**(verified 2026-08-05.)** Consequence: **119 of 145 (82.1%) diagnosis descriptions differed
between the arms** ([06-preprocessing-defects.md](06-preprocessing-defects.md)). Any
baseline-versus-BERT delta measured under `legacy` is substantially measuring the preprocessing
difference, not the encoder. The project's stated central constraint is that both arms share
everything except the embedding model; that constraint was broken at this line.

## 3. Every reported "F1" is accuracy — EXPLAINED, and the columns relabelled

**Resolution: not a code fix but a naming one, and it took two more findings to land.** The
2026-08-05 baseline run ([09](09-baseline-first-run.md)) proved the collapse is a property of
the compact BERT embedding space, not of the code — the baseline abstains on 23.3% of cases and
its P ≠ R in every row. Then [13](13-rank-aware-metrics.md) reproduced the legacy `P`, `R` and
`PR` columns bit-exactly from an independent implementation and identified what they really are:
`R` is the all-cases hit rate, `P` the answered-cases hit rate, `PR` the coverage. **The columns
were never wrong; they were unlabelled.**

Across **all 12,600 rows** of committed BERT results, precision == recall == F-score, with zero
exceptions ([04-metric-degeneracy.md](04-metric-degeneracy.md)). The mechanism: every test case
increments exactly one of TP or FP, so `tp + fp == nrow`; precision reduces to `tp/nrow`, which
*is* the recall definition in use; and the harmonic mean of two equal numbers is that number.
This is threshold-independent — it is not a tuning artifact.

## 4. All three encoders saturate — STILL OPEN, but confined

**Status: unfixed and unfixable at the paper's threshold — but it is now fenced off. Under
`--pipeline drg` there is no cosine threshold at all, so saturation cannot arise there; it
bounds only the cosine-graded tables below threshold 1.0.**

Every BERT model scores 1.000 at threshold 0.6
([03-metric-saturation.md](03-metric-saturation.md)). Two causes compound: biomedical embeddings
are compact — mean pairwise cosine 0.72–0.93 **even between unrelated diagnoses** — and the
aggregator takes a MAX over the Cartesian product of predicted × true diagnoses, which amplifies
that until ~100% of patient pairs clear 0.6.

A corollary worth stating separately: TOP-K scores rise monotonically with K because a single
hit suffices and there is no penalty for the other K−1 predictions. That curve is an artifact of
the metric, not a property of the encoders. (Later verified monotonic in 18 of 18
model × threshold combinations, and removed as a knob by MRR —
[13](13-rank-aware-metrics.md).)

## 5. The folds leak patients — RESOLVED 2026-08-05 (`c2115ba`), 41 → 0

**Resolution: `scripts/make_folds.py` regroups the folds by `SUBJECT_ID` with `GroupKFold`;
leaked test cases went from 41 to 0, recounted independently. The measured cost of removing the
leak is in [11](11-corrected-pipeline-first-results.md) — every arm loses ground, the baseline
most (−0.0638 at TOP-10/1.0).**

The folds split on `HADM_ID`, but the 129 admissions come from only **100 distinct patients** —
one patient contributes 15. **41 of 129 test cases (31.8%) had another admission from the same
`SUBJECT_ID` in their own retrieval pool** ([05-patient-leakage.md](05-patient-leakage.md)).

Measured inflation at threshold 1.0 was **+0.11 to +0.26**, against encoder differences of
**0.015–0.046** and per-fold σ of 0.071–0.124. The contamination was about an order of magnitude
larger than the signal being measured. The diagnostic tell: on leaked cases all three encoders
score *identically* (0.293 MAX, 0.415 TOP-10); on clean cases they diverge.

Because the fold files were shared static artifacts, **this affected both arms** — the published
0.489/0.512/0.521 carry it too.

---

## The ceiling nobody set out to impose

Before designing any exact-match metric: only **75 of 129 test cases (58.1%)** have their
correct DRG present anywhere in their fold's training pool, because 105 of the 145 unique
diagnoses occur exactly once in the dataset. **A perfect retriever caps at 58.1% under exact
matching.** Any metric that rewards only exact hits is therefore measuring dataset sparsity as
much as model quality — which was the argument for graded relevance over exact match. (The
graded scheme was then designed, measured, and rejected on its merits — see
[12-drg-grader.md](12-drg-grader.md) before treating it as an easy win.)

**Re-measured 2026-08-06 on the grouped folds: 76/129 = 58.9%.** Regrouping the folds by
`SUBJECT_ID` moved retrievability by exactly one case, so leakage and sparsity are independent
defects — fixing the split does not make more diagnoses findable. Both figures are now pinned by
`tests/test_drg_grader.py`. Per fold the ceiling ranges from **3/12 (25%) to 13/15 (87%)** on
the grouped split, which is itself a source of the per-fold variance reported elsewhere.

The "105" is document frequency over the deduped labels, not occurrence count over raw entries
(which is 85). It is the correct one for a *retrieval* bound, and the distinction is explained
in [12-drg-grader.md](12-drg-grader.md) — an attempted "correction" to 85 was abandoned.

## What would make the comparison valid, in order — now the record of what was done

1. ~~**Unify diagnosis preprocessing across arms.**~~ **DONE** (`c2115ba`,
   [06](06-preprocessing-defects.md)). Cheapest fix, highest leverage — exactly as predicted.
2. ~~**Re-split the folds with `GroupKFold` on `SUBJECT_ID`.**~~ **DONE** (`c2115ba`,
   [05](05-patient-leakage.md)). The largest single correction available, and it applied to
   both arms.
3. ~~**Replace the metric** with a rank-aware family against an encoder-independent relevance
   label.~~ **DONE** in two steps: the `drg-exact` grader ([12](12-drg-grader.md)) removed the
   threshold, MRR/Precision@K ([13](13-rank-aware-metrics.md)) removed the K. The set-level
   soft P/R/F1 variant remains open as **P6** in
   [../plans/correctness-fixes.md](../plans/correctness-fixes.md).
4. ~~**Get the baseline to actually run** under the corrected pipeline.~~ **DONE** — first under
   `legacy` ([09](09-baseline-first-run.md)), then under `corrected` and `drg`
   ([11](11-corrected-pipeline-first-results.md)). The dependency stated here was honoured: the
   published numbers stopped being a valid reference once steps 2–3 landed, so BioSentVec was
   re-run under every config rather than compared against its published figure.

Steps 1–3 deliberately changed every number the pipeline emits. That is why they were tracked
separately from the refactor in [../plans/correctness-fixes.md](../plans/correctness-fixes.md)
and [../plans/metric-redesign.md](../plans/metric-redesign.md): the refactor's only safety
mechanism is that the numbers must *not* move, so the two kinds of work cannot run concurrently.

## What can be claimed, defensibly — updated for the corrected results

Not "BiomedBERT beats BioSentVec," and now also not "Bio_ClinicalBERT beats BioSentVec." The
honest, and considerably more interesting, claims:

- The published metric does not measure what its column headers say. In the BERT results it is
  accuracy wearing an F1 label, provably, in all 12,600 rows — and the columns' true names are
  hit rate and coverage ([13](13-rank-aware-metrics.md)).
- The evaluation saturates because of a structural interaction between compact biomedical
  embedding spaces and a MAX aggregator — a reusable lesson about evaluating retrieval over any
  domain-tuned embedding space, not a quirk of this dataset.
- The published result carried patient leakage of roughly ten times the magnitude of the encoder
  differences it reports. Removing it cost the baseline's headline figure 18.4% (0.4824 →
  0.3922).
- The dataset imposes a 58.9% ceiling on any exact-match retrieval metric, which no model choice
  can lift.
- **With every knob removed — no threshold, no K, all three abstention conventions — no pair of
  encoders separates** (max paired |t| = 1.718 against the 2.262 needed at 9 df). The null
  result is the finding, and it now rests on four removed confounds rather than one favourable
  setting.
- The three transformers match a ~5.6-billion-parameter n-gram table with ~1/51 the parameters
  and ~1/17 the disk — a hardware fact independent of the metric, and the one asymmetry that
  survives every correction.

Each of these is a reproduction finding: the value is in showing the original result is not
measuring the claimed quantity, and in quantifying by how much.

---

*Companion documents:* [01](01-baseline-reproduction.md) baseline state ·
[02](02-encoder-comparison.md) encoder comparison · [03](03-metric-saturation.md) saturation ·
[04](04-metric-degeneracy.md) degeneracy · [05](05-patient-leakage.md) leakage ·
[06](06-preprocessing-defects.md) preprocessing · [09](09-baseline-first-run.md) the baseline's
first run · [11](11-corrected-pipeline-first-results.md) the corrected results ·
[12](12-drg-grader.md) · [13](13-rank-aware-metrics.md) the two knob removals

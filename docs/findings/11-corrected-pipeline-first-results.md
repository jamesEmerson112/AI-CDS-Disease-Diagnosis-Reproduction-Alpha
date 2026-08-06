# 11. The first uncontaminated four-arm results

**Date:** 2026-08-06 · **Host:** RunPod Linux (x86) · **Commits:** `c2115ba`, `31bea66`

All four arms re-run under `AICDS_PIPELINE=corrected`: `GroupKFold` folds on `SUBJECT_ID`
(zero patient leakage, down from 41 of 129 test cases) and unified, fixed preprocessing
(145/145 diagnosis descriptions now identical across arms, up from 26/145).

These are the first numbers this project has produced that are not contaminated by patient
leakage. **They are still self-graded and still rank-blind** — see the limits section. They
are defensible as a reproduction; they are *not* an encoder ranking.

Runs live in `results_corrected/<model>/<timestamp>/`, gitignored under the same rule as
`results/`. Only aggregates appear here; they carry no `HADM_ID`s.

---

## The headline: the encoder gap was mostly leakage

`MAX` aggregator at threshold 1.0 — the only setting where **no** model saturates, and
therefore the only one where the four arms can be compared at all.

| Encoder | Legacy | Corrected | Δ |
|---|---|---|---|
| BioSentVec (700D) | 0.1843 | **0.0877** | −0.0966 |
| Bio_ClinicalBERT | 0.1462 | **0.0862** | −0.0600 |
| BiomedBERT | 0.1692 | **0.0855** | −0.0837 |
| BlueBERT | 0.1776 | **0.0785** | −0.0991 |

**Spread across the four encoders collapses from 0.038 to 0.009.** Removing patient leakage
cut the apparent between-encoder difference by roughly four times, leaving what is
effectively a four-way tie. The single most defensible statement available from this project
is therefore not that one encoder wins — it is that **most of the measured difference between
encoders was contamination, not capability.**

That is exactly what `docs/findings/05-patient-leakage.md` predicted: inflation of +0.11 to
+0.26 against encoder differences of 0.015–0.046, i.e. contamination ~10× the effect under
study. The prediction was made before these runs existed.

Note also that BioSentVec, the 700-dimensional baseline, now sits at the *top* of a
statistically meaningless ordering. Nothing here supports "transformers beat sent2vec."

## Every arm drops, and the baseline drops most

`TOP-10` at threshold 0.6, the setting Comito et al. report:

| Encoder | Legacy | Corrected | Δ |
|---|---|---|---|
| BioSentVec | 0.4824 | 0.3922 | −0.0902 |
| Bio_ClinicalBERT | 1.0000 | 1.0000 | 0.0000 (saturated) |
| BiomedBERT | 1.0000 | 1.0000 | 0.0000 (saturated) |
| BlueBERT | 1.0000 | 1.0000 | 0.0000 (saturated) |

The published figure is 0.489. This checkout reproduced it at 0.4824 under `legacy` and gets
**0.3922** once leakage and preprocessing are fixed — about a fifth of the headline number was
contamination. The three BERT arms cannot move because they are already pinned at 1.000.

## What the correction did *not* fix

Both remaining defects survive untouched, which is the expected result — they are metric
design, not data splitting.

- **Saturation persists.** BiomedBERT is still 1.000 at every threshold from 0.6 to 0.9 on
  `TOP-10`; Bio_ClinicalBERT at 0.6–0.8; BlueBERT at 0.6–0.7. Compact biomedical embedding
  spaces plus a MAX-over-Cartesian-product aggregator still clear any threshold below ~0.9.
- **Degeneracy persists for all three BERT arms.** Prediction rate is exactly 1.0000, so
  `tp + fp == nrow`, precision reduces to `tp/nrow` — which *is* recall — and P == R == F in
  every row. **Every "F1" in the BERT columns above is accuracy.**
- **The baseline still abstains**, at PR = 0.7558 (24.4% of cases produce no candidate above
  `PRUNING_SIMILARITY`), so it alone has P ≠ R. This is the same asymmetry recorded in
  finding 04, and it survives the correction — confirming degeneracy is a consequence of
  BERT's compact space rather than a structural property of the code.

## Limits — read before quoting any number here

1. **Self-grading is untouched (TODO P4).** The same embedding space both retrieves and
   judges, so a more compressed space grades itself more leniently. Every number above
   inherits that.
2. **Rank is still discarded (TODO P5).** `containGreaterOrEqualsValue` returns true if *any*
   of the K candidates clears threshold, so a hit at rank 1 and a hit at rank 50 count
   identically, and TOP-K rises with K by construction.
3. **The 58.1% exact-match ceiling has not been re-measured** on the grouped folds. The old
   figure was computed on the leaky ones.
4. **The correction bundles two changes.** `corrected` moves folds *and* preprocessing at
   once, so the deltas above cannot be split into a leakage part and a preprocessing part.
   The `folds-only` and `preprocess-only` configs exist for that (TODO P29) but have not been
   run.
5. **Folds are uneven** (114/15 through 117/12) because subject 41976 holds 15 admissions and
   whole patients must stay together. Per-fold *n* varies; treat per-fold σ accordingly.

## Reproducing

```bash
python scripts/make_folds.py --verify          # regenerates data/folds_grouped/
AICDS_PIPELINE=corrected python scripts/run_baseline.py            # Linux + 21 GB model
python scripts/run_bert_analysis.py --model all --pipeline corrected
python scripts/compare_models.py --results-dir results_corrected
```

Runtime on the RunPod box: baseline ~13 min, each BERT arm ~11.5 min, ~48 min for all four.

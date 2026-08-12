# 11. The first uncontaminated four-arm results

**Date:** 2026-08-06 · **Host:** RunPod Linux (x86) · **Commits:** `c2115ba`, `31bea66`

> **In plain words.** This is the first time all four models were scored on a playing field
> with the two known data problems fixed: the data split no longer lets a patient's own earlier
> hospital visit be used as an "answer" for their later one, and both sides now see identically
> cleaned text. Two results. First, every model's score *drops* — the old scores were partly
> free wins, and the baseline had the most free wins to lose (its headline figure fell 18%,
> from 0.482 to 0.392). Second, the clean scores refuse to produce a winner: change one
> arbitrary dial (how many guesses each model gets) and first place changes hands — the 2019
> baseline can be made to finish 1st or 4th at will. The dial-turning is the finding: a ranking
> that depends on the dial is not a ranking of the models.

All four arms re-run under `AICDS_PIPELINE=corrected`: `GroupKFold` folds on `SUBJECT_ID`
(zero patient leakage, down from 41 of 129 test cases) and unified, fixed preprocessing
(145/145 diagnosis descriptions now identical across arms, up from 26/145).

These are the first numbers this project produced that are not contaminated by patient
leakage. **At the time they were still self-graded and still rank-blind** — both limits have
since been closed (see the limits section). They are defensible as a reproduction; they are
*not* an encoder ranking.

Runs live in `results_corrected/<model>/<timestamp>/`, gitignored under the same rule as
`results/`. Only aggregates appear here; they carry no `HADM_ID`s.

---

## The headline: every arm drops, and the ranking stops meaning anything

`MAX` aggregator at threshold 1.0 — one of the settings where **no** model saturates.

| Encoder | Legacy | Corrected | Δ |
|---|---|---|---|
| BioSentVec (700D) | 0.1843 | **0.0877** | −0.0966 |
| Bio_ClinicalBERT | 0.1462 | **0.0862** | −0.0600 |
| BiomedBERT | 0.1692 | **0.0855** | −0.0837 |
| BlueBERT | 0.1776 | **0.0785** | −0.0991 |

Every arm loses ground, which is the expected direction: the leaked cases were free wins.
That much is exactly what `docs/findings/05-patient-leakage.md` predicted — inflation of
+0.11 to +0.26 against encoder differences of 0.015–0.046, i.e. contamination ~10× the effect
under study — and the prediction was made before these runs existed.

### CORRECTION 2026-08-06: the spread does not collapse in general

**This section originally read "spread collapses from 0.038 to 0.009 … effectively a four-way
tie" and generalised it. That was wrong, and it is corrected here rather than quietly edited
away, because the error is instructive: it came from reading one aggregator and assuming the
others agreed.** They do not. Under `MAX` the spread shrinks; under every `TOP-K` it *widens*:

| Aggregator @ threshold 1.0 | Legacy spread | Corrected spread | Direction |
|---|---:|---:|---|
| MAX | 0.0381 | 0.0092 | ↓ 4.1× |
| TOP-10 | 0.0462 | 0.0671 | ↑ 1.45× |
| TOP-20 | 0.0506 | 0.0819 | ↑ 1.62× |
| TOP-30 | 0.0744 | 0.1383 | ↑ 1.86× |

Normalising by the leading value does not rescue the original claim either (MAX 20.7% →
10.5%, TOP-10 16.2% → 26.9%): the *direction* differs by aggregator. So "the encoders
converged once leakage was removed" is not a finding — it is an artifact of picking `MAX`.

### What actually replaces it: the ranking inverts on an arbitrary knob

Corrected pipeline, threshold 1.0 fixed, changing only `K` (how many candidate patients each
arm may draw predictions from):

| Encoder | MAX | TOP-10 | TOP-20 | TOP-30 |
|---|---|---|---|---|
| BioSentVec (700D) | **0.0877 — 1st** | 0.2163 — 2nd | 0.2229 — 4th | 0.2296 — **4th** |
| Bio_ClinicalBERT | 0.0862 — 2nd | **0.2491 — 1st** | **0.3049 — 1st** | 0.3513 — 2nd |
| BiomedBERT | 0.0855 — 3rd | 0.1981 — 3rd | 0.2888 — 3rd | 0.3353 — 3rd |
| BlueBERT | 0.0785 — **4th** | 0.1821 — 4th | 0.3038 — 2nd | **0.3679 — 1st** |

**BioSentVec goes 1st → 4th; BlueBERT goes 4th → 1st, on the same data at the same
threshold.** And there is a mechanism rather than noise: the baseline abstains on 24.4% of
cases (`PR` 0.7558) while every BERT arm predicts on 100% (`PR` 1.0000). Widening `K` cannot
help the baseline on a case where it declined to predict, but it hands each BERT arm another
free guess — and one hit inside `K` suffices with no penalty for the other `K−1`. **TOP-K
structurally rewards not abstaining**, which is a property of the metric, not of the encoders.

The defensible statement is therefore not "the encoders converged" and certainly not that one
wins. It is that **the ranking is decided by two arbitrary knobs — aggregator and threshold —
so this experiment does not support any encoder ranking.** Note in particular that BioSentVec,
the 700-dimensional 2019 baseline, can be made to finish first or last at will. Nothing here
supports "transformers beat sent2vec," and nothing supports the reverse either. (Both knobs
were subsequently removed — [12](12-drg-grader.md) killed the threshold, [13](13-rank-aware-metrics.md)
killed the K — and the no-ranking conclusion survived that too.)

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
contamination. The three BERT arms cannot move because they are already pinned at 1.000 —
which is also the cleanest demonstration that saturation and leakage are independent defects:
fixing the folds cannot rescue a metric that has no headroom.

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

## Limits — read before quoting any number here (with what became of each)

1. ~~**Self-grading is untouched (TODO P4).**~~ **Closed 2026-08-06**
   ([12](12-drg-grader.md)): the DRG-string grader reproduces the threshold-1.0 numbers above
   bit-exactly, so they were never self-grading-inflated — but that was only knowable after
   the fix.
2. ~~**Rank is still discarded (TODO P5).**~~ **Closed 2026-08-06**
   ([13](13-rank-aware-metrics.md)): MRR removed the K knob; no pair of encoders separates.
3. ~~**The 58.1% exact-match ceiling has not been re-measured on the grouped folds.**~~
   **Done**: 76/129 = 58.9% on `folds_grouped`, a change of exactly one case, pinned in
   `tests/test_drg_grader.py`.
4. ~~**The correction bundles two changes.**~~ **Closed 2026-08-12**
   ([15](15-leakage-preprocessing-attribution.md)): the `folds-only` and `preprocess-only` runs
   happened, and the deltas above *are* now split. `corrected` still moves folds *and*
   preprocessing at once — that has not changed — but the attribution no longer has to be inferred
   from it. **Leakage dominates every arm** (+0.027 to +0.070 at threshold 1.0), preprocessing is
   small with an arm-dependent sign, and the 0.0902 drop at the published cell is 54.3% leakage /
   43.8% preprocessing / 1.9% interaction. Read the `preprocess-only` column there as an
   attribution input only — that tree still leaks patients.
5. **Folds are uneven — still true** (114/15 through 117/12) because subject 41976 holds 15
   admissions and whole patients must stay together. Per-fold *n* varies; treat per-fold σ
   accordingly.

## Reproducing

```bash
python scripts/make_folds.py --verify          # regenerates data/folds_grouped/
AICDS_PIPELINE=corrected python scripts/run_baseline.py            # Linux + 21 GB model
python scripts/run_bert_analysis.py --model all --pipeline corrected
python scripts/compare_models.py --results-dir results_corrected
```

Runtime on the RunPod box: baseline ~13 min, each BERT arm ~11.5 min, ~48 min for all four.

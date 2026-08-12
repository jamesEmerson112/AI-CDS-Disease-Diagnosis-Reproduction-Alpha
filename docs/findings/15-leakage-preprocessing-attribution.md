# 15 — Splitting the legacy→corrected drop: leakage, preprocessing, and the interaction

**Date:** 2026-08-12 · **Host:** RunPod Linux, 16 vCPU, 251 GB RAM, torn down after harvest ·
**Commits:** runs `da93b96` (dirty flag set — see Provenance), analysis `bc0f002` + `c5508ef` ·
**Problem number:** P29 (closes it)

Artifacts: `results_folds_only/<arm>/<stamp>/` and `results_preprocess_only/<arm>/<stamp>/` at the
repository root, gitignored by `results*/`, read alongside the existing `results/` (legacy) and
`results_corrected/`. Only 10-fold aggregates appear below — the per-case files in those trees are
keyed by `HADM_ID` and never leave the tree.

> **In plain words.** The corrected pipeline fixed two things at once — how patients were split
> into folds, and how text was cleaned — so the score drop it produced could not be blamed on
> either one. This run rebuilt the experiment twice more, each time fixing exactly one of the two.
> The answer is the same for all four models: **the fold split was the problem.** Closing the leak
> costs every model between 0.027 and 0.070 of its score. Fixing the text moves each model by less
> than 0.013 — and for two of the four it moves the score *up*, not down, which is the opposite of
> what "we fixed a defect" usually implies. The published 0.0902 drop in the baseline's headline
> figure splits roughly **54% leakage, 44% preprocessing, 2% the two fixes interacting.**

This is the answer to the question finding [11](11-corrected-pipeline-first-results.md) left open as
its fourth limit: *"the correction bundles two changes … so the deltas cannot be split into a
leakage part and a preprocessing part."* They can now.

---

## The design: four trees, each differing on exactly one axis

| tree (pipeline name) | fold split | text handling | what it is for |
|---|---|---|---|
| `results/` (`legacy`) | committed `data/folds` — leaky | legacy | the published pipeline; the reference row |
| `results_folds_only/` (`folds-only`) | `folds_grouped` — GroupKFold on `SUBJECT_ID` | legacy | isolates **leakage** |
| `results_preprocess_only/` (`preprocess-only`) | committed `data/folds` — **still leaky** | corrected | isolates **preprocessing** — *instrument only* |
| `results_corrected/` (`corrected`) | `folds_grouped` | corrected | both fixes, as `CORRECTED` ships them |

The two new trees are **4 arms × 2 configs, produced in one pod session** — BioSentVec,
Bio_ClinicalBERT, BiomedBERT, BlueBERT under each of `folds-only` and `preprocess-only`, eight runs
in four driver steps. `legacy` and `corrected` were not re-run; their trees are the committed ones
from 2026-08-05 and 2026-08-06.

**Sign convention, stated once.** Every part is `legacy − fixed`, so a **positive** number means the
fix **removed** score — i.e. the legacy figure was inflated by that much.

```
leakage        legacy − folds-only        (grouped folds alone)
preprocessing  legacy − preprocess-only   (fixed text handling alone)
bundled drop   legacy − corrected         (both, as CORRECTED ships them)
residual       drop − leakage − preprocessing
```

### The rule that governs half of this table

> **PREPROCESS-ONLY NUMBERS ARE ATTRIBUTION INPUTS, NEVER RESULTS.**
> That pipeline keeps `data/folds`, so it **still leaks patients** — 41 of 129 test cases retrieve
> another admission of their own `SUBJECT_ID` (`src/aicds/config.py:143-145`,
> [05](05-patient-leakage.md)). It exists to isolate one effect. A number quoted out of that
> column, **or out of any difference built on it**, is a leaked number. Report `corrected` or
> `drg`. Never this.

`scripts/attribute_effects.py` prints that banner twice, above and below its own output, so a pasted
excerpt carries it. It is repeated here for the same reason. The `preprocessing` column below *is* a
difference built on a leaked tree; it is legitimate as an **attribution term** — the leak is present
on both sides of the subtraction and is what the subtraction cancels — and illegitimate as a score.

---

## The table to read: threshold 1.0, TOP-10, all four arms

Threshold 1.0 is the only cosine setting at which **no arm sits on the ceiling**, so all four rows
carry information. This is the block to quote.

**F-score grid** (10-FOLD TOP-10 aggregate at threshold 1.0)

| arm | legacy | folds-only | preprocess-only | corrected |
|---|---:|---:|---:|---:|
| BioSentVec | 0.280121 | 0.210562 | *0.292554* | 0.216278 |
| Bio_ClinicalBERT | 0.285256 | 0.258077 | *0.277564* | 0.249103 |
| BiomedBERT | 0.254487 | 0.212692 | *0.254487* | 0.198077 |
| BlueBERT | 0.239103 | 0.191026 | *0.246795* | 0.182051 |

*(The italicised column is the instrument column. It is printed so the arithmetic below can be
checked, not so it can be quoted.)*

**Attribution** — every part is `legacy − fixed`; positive = the fix removed score

| arm | legacy FS | leakage | preprocessing | bundled drop | residual |
|---|---:|---:|---:|---:|---:|
| BioSentVec | 0.280121 | **+0.069559** | −0.012433 | +0.063843 | +0.006717 |
| Bio_ClinicalBERT | 0.285256 | **+0.027179** | +0.007692 | +0.036154 | +0.001282 |
| BiomedBERT | 0.254487 | **+0.041795** | +0.000000 | +0.056410 | +0.014615 |
| BlueBERT | 0.239103 | **+0.048077** | −0.007692 | +0.057051 | +0.016667 |

### What it says

**1. Leakage dominates every arm.** The leakage term runs **+0.027 to +0.070** across the four arms
and is the larger term in all four rows. In magnitude it exceeds the preprocessing term by 3.5×
(Bio_ClinicalBERT) to 6.3× (BlueBERT), and unboundedly for BiomedBERT, whose preprocessing term is
exactly zero. This is the direction finding [05](05-patient-leakage.md) predicted before any of
these trees existed — the leaked cases were free wins — and it is now measured per arm rather than
inferred from the bundle.

**2. Preprocessing is small, and its sign varies by arm.** Two arms *gain* from the text fix at this
cell — **BioSentVec −0.0124** and **BlueBERT −0.0077**, where the negative sign means the fix
**raised** the score — one loses a comparable amount (**Bio_ClinicalBERT +0.0077**), and one does not
move at all (**BiomedBERT, exactly +0.000000**; see Measured coincidences). No mechanism is asserted
for the sign pattern. What the pattern rules out is the tidy story that a defect fix must cost score:
here it costs one arm, pays two, and is invisible to a fourth, all on the same 129 admissions.

**3. The interaction is real, positive, and small — and it is a reportable number, not an error.**
The residual runs **+0.0013 to +0.0167**. The two one-change configs are separate pipelines and their
effects have no obligation to add up; a residual is what it looks like when the fixes interact. Note
its size *relative to the preprocessing term*: for BiomedBERT (+0.014615 vs 0.000000) and BlueBERT
(+0.016667 vs −0.007692) the interaction is **larger than one of the main effects**. So
"leakage + preprocessing" is not a decomposition that closes to the bundle, and the honest report has
three terms, not two.

**4. The baseline's preprocessing term changes sign between thresholds.** +0.039507 at 0.6 (below)
against −0.012433 at 1.0. Measurement only; both cells are reported and neither is preferred.

---

## The published cell: threshold 0.6, and the 0.0902 finally split

**F-score grid** (10-FOLD TOP-10 aggregate at threshold 0.6)

| arm | legacy | folds-only | preprocess-only | corrected |
|---|---:|---:|---:|---:|
| BioSentVec | 0.482443 | 0.433464 | *0.442936* | 0.392217 |
| Bio_ClinicalBERT | 1.000000 | 1.000000 | *1.000000* | 1.000000 |
| BiomedBERT | 1.000000 | 1.000000 | *1.000000* | 1.000000 |
| BlueBERT | 1.000000 | 1.000000 | *1.000000* | 1.000000 |

**Attribution at 0.6 — only one row is attributable**

| arm | legacy FS | leakage | preprocessing | bundled drop | residual |
|---|---:|---:|---:|---:|---:|
| BioSentVec | 0.482443 | **+0.048979** | **+0.039507** | **+0.090226** | +0.001740 |
| Bio_ClinicalBERT | 1.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| BiomedBERT | 1.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| BlueBERT | 1.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

**Read the three BERT rows as empty, not as zero effect.** All three scored 1.000000 in every root at
this threshold, so their deltas are differences between two ceilings and measure the *threshold*, not
the fixes — see [03](03-metric-saturation.md). `attribute_effects.py` derives that note from the
data rather than asserting it: an arm is flagged when it hits the ceiling in two or more roots.

**The baseline row is the point of this section.** `+0.090226` is the same number finding
[11](11-corrected-pipeline-first-results.md) reports as the corrected-pipeline drop at the cell
Comito et al. publish (0.4824 → 0.3922, Δ −0.0902). It is now split:

| part | value | share of the bundled drop |
|---|---:|---:|
| leakage (grouped folds alone) | +0.048979 | 54.3% |
| preprocessing (fixed text alone) | +0.039507 | 43.8% |
| interaction (residual) | +0.001740 | 1.9% |
| **bundled drop** | **+0.090226** | 100% |

Both defects are first-order at the published cell, leakage slightly the larger, with the two
interacting almost not at all. That is a *different* shape from the 1.0 cell, where leakage is 5.6×
preprocessing for this same arm and the residual is nearly four times larger. One arm, two cells, two
attributions — which is exactly why the tool prints both and offers no `--threshold` flag to pick
one.

---

## Provenance, and what makes these two trees different from every earlier one

- **Host and environment of record.** RunPod Linux, 16 vCPU (`nproc` captured live), 251 GB host RAM,
  torn down 2026-08-12 after harvest. Python 3.9.23, numpy 2.0.2, torch 2.8.0+cpu, sklearn 1.6.1.
  The architecture is not asserted from a label: **every run's `run_metadata.json` records
  `platform: Linux-6.8.0-45-generic-x86_64`**, which is the citable form. *(Corrected 2026-08-12: the
  header of this page carried a bare "(x86)" token with no source. It was right, and it was still an
  unsourced token in a file whose entire subject is attributing numbers to configurations.)*
- **Code state.** `da93b96`, `dirty: true` — the dirt is **two untracked leftovers** of the retired
  2026-08-06 staged script (`_done_drg/`, `attribution_runs.sh`) and **zero tracked modifications**.
  The delta from `da93b96` to the commit that produced this report is docs, tests, an environment pin
  and the analysis tooling; the pipeline is identical.
- **Session shape.** Ten sequential verify-gated driver steps, 22:57:10 → 02:28:29 UTC (3h31m19s).
  *(Corrected 2026-08-12: this read "5h31m", which the two timestamps printed beside it refute —
  the elapsed time is 3h31m19s, and the ten step timings sum to 211.3 min, i.e. the same 3h31m.)* The
  four P29 steps occupy 23:09:36 → 00:50:56 (1h41m20s): baseline `folds-only` 12m22s, baseline
  `preprocess-only` 11m32s, BERT `--model all` `folds-only` 40m08s, BERT `--model all`
  `preprocess-only` 37m18s. The session's first step re-ran the **`legacy` baseline** into
  `results_verify/` and byte-matched the 2026-08-05 reference — the C4/C5 verification — which
  corroborates the baseline cell of this table's reference row on the same box that produced the two
  new trees. The three BERT cells of that row were not re-verified.
- **These come from the first session to produce metadata-bearing trees.** Every run in both carries
  `run_metadata.json` (P14/C8), so the `folds-only` and `preprocess-only` columns are attributed by
  the run's own record. The `legacy` and `corrected` columns are not — `attribute_effects.py` emits
  one `[WARN] … taken on the directory's word (pre-P14 runs)` line for each, because those trees
  predate the field. Two of four columns still rest on a directory name; that is now visible in the
  output rather than assumed. *(Softened 2026-08-12 from "the first metadata-bearing trees in the
  project": the session's own **first** step — the `legacy` baseline re-verification into
  `results_verify/` — also carries `run_metadata.json` and predates these two trees by minutes. The
  claim worth making is about the **session**, not about which of its trees was first.)*
- **The fold split is pinned by content, not by name.** Every `folds-only` run records the canonical
  `folds_grouped` digest `b36f7216…a6ec5084f` in `fold_dir_sha256` — the split of record from
  [14](14-fold-split-environment-dependence.md). The `preprocess-only` runs record the digest of the
  committed legacy `data/folds` instead, which is the correct value for that config and the fastest
  way to confirm no run was crossed with the wrong split.

---

## Measured coincidences — documented, not "fixed"

**BiomedBERT's `preprocess-only` cell at threshold 1.0 equals its `legacy` cell exactly:**
`0.2544871794871795` in both, hence the exactly `+0.000000` preprocessing term. This is a
**bit-identical cell**, recorded as measurement. No mechanism is asserted, and three things it does
**not** mean:

- It does not mean preprocessing has no effect on BiomedBERT in general. One aggregator (TOP-10) at
  one threshold (1.0) is what was compared; nothing wider is claimed here.
- It does not mean the two runs are the same run. They are separate runs in separate trees, six days
  and many commits apart. What makes the repeat notable is precisely
  that they share a fold directory — `legacy` and `preprocess-only` both use the committed
  `data/folds` — so **text handling is the only axis between them**, and this cell did not notice it.
- It is not a rounding artifact to be cleaned up. Quote it as `0.2544871794871795` — Python's
  shortest round-trip repr. A 17-significant-figure spelling of the same double,
  `0.25448717948717953`, is the *same number*, not a discrepancy.

Both cells sit in `compare_models.py`'s sanity table — pinned to four decimals under `legacy` and
under `preprocess-only`, with a comment recording that the equality was **read off both
`PerformanceIndex.txt` files at all 17 significant figures rather than eyeballed**, so it is not a
copy-paste error. The other three arms all move between those two columns, which is what makes the
repeat worth flagging in the first place.

---

## Limits — read before quoting any number here

1. **The `preprocess-only` column is not a result, and neither is any ranking read off it.** Stated
   twice above; it is the single most likely misuse of this page. The arm ordering is *not* stable
   across the four columns of the 1.0 grid — the grid exists to be differenced row-wise, not to be
   ranked column-wise. For an encoder comparison use `corrected` or `drg`, and see
   [13](13-rank-aware-metrics.md) for why even those support no ranking.
2. **Winnable counts differ between the trees, so cross-tree case-set comparisons are not
   like-for-like.** The `preprocess-only` tree's winnable population is **75 of 129** (it uses the
   leaky committed split) against **76 of 129** on the grouped-fold trees. That one-case difference
   is the same one finding [11](11-corrected-pipeline-first-results.md) records (58.1% → 58.9%); it
   means the denominators behind any per-population figure are not identical across columns. The
   F-score attribution above is unaffected — it compares 10-fold TOP-10 aggregates over all 129 test
   cases in every column — but a comparison built on `winnable` across trees is not valid without
   saying which denominator it used.
3. **Both cells are reported because neither is sufficient.** 0.6 alone attributes one arm and leaves
   three rows of zeros to be misread as "the fixes did not affect the BERT arms". 1.0 alone abandons
   the figure the paper and every earlier finding quote. There is deliberately no `--threshold` flag.
4. **The three BERT rows at 0.6 carry no information at all.** They are `1.000000 − 1.000000` four
   times over. Do not quote them, even as "no effect".
5. **`legacy` and `corrected` were not re-run for this table.** Their columns come from the committed
   2026-08-05 / 2026-08-06 trees, whose runs carry no metadata; the pipeline attribution for those
   two columns rests on the directory name plus the session's byte-exact `legacy` re-verification.
   In the `results/` tree three arms hold two runs each and the tool uses the most recent, printing
   an `[INFO]` line for each.
6. **This is the cosine grader.** The attribution is measured on `PerformanceIndex.txt` F-scores, the
   self-graded metric; [12](12-drg-grader.md) shows the DRG grader reproduces the corrected
   threshold-1.0 numbers bit-exactly, but the attribution itself has not been re-derived under
   `drg-exact`.
7. **Out of scope here: `corrected2`.** The P27 comma-fragment variant is a *fifth* tree and belongs
   to the finding [06](06-preprocessing-defects.md) addendum. The `corrected` column above is
   `CORRECTED` as it shipped and as every earlier finding describes it.

---

## Reproducing

```bash
python scripts/make_folds.py --verify     # must print the canonical digest b36f7216…a6ec5084f
                                          # (finding 14); a different one is not comparable
python scripts/run_baseline.py      --pipeline folds-only      --out results_folds_only
python scripts/run_bert_analysis.py --model all --pipeline folds-only      --out results_folds_only
python scripts/run_baseline.py      --pipeline preprocess-only --out results_preprocess_only
python scripts/run_bert_analysis.py --model all --pipeline preprocess-only --out results_preprocess_only

python scripts/attribute_effects.py       # reads all four roots; prints both threshold blocks
```

The baseline arm is Linux-only (`sent2vec` will not build under MSVC) and needs the 21 GB BioSentVec
model. `attribute_effects.py` is read-only, needs no model, and runs anywhere the four trees exist;
its four parser anchors must print `[SUCCESS]` or the table is not trustworthy. Default roots are
`results results_folds_only results_preprocess_only results_corrected`, overridable with `--roots`.

---

*Companion documents:* [05](05-patient-leakage.md) the leakage defect ·
[06](06-preprocessing-defects.md) the preprocessing defects ·
[11](11-corrected-pipeline-first-results.md) the bundled correction this page splits, whose fourth
limit this closes · [03](03-metric-saturation.md) why three rows at 0.6 are empty ·
[14](14-fold-split-environment-dependence.md) the canonical split these runs pin by digest

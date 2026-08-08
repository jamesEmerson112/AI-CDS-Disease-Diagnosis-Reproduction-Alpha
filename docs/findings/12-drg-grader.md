# 12. The encoder-independent grader, and the partial-credit scheme that was rejected

**Date:** 2026-08-06 · **TODO:** P4 · **Status:** done — code landed and the four-arm results
are in (see "The results" below)

> **In plain words.** Until this change, every model graded its own homework. A prediction
> counted as "correct" if it was *close* to the truth — but "close" was measured inside the same
> model's own embedding space that produced the prediction. A model with a cramped space, where
> everything looks similar to everything, could therefore mark itself generously. The fix is a
> grader with no model in it at all: a prediction is correct only if its DRG label — the
> category a hospital files the stay under for billing, e.g. "septicemia & disseminated
> infections" — is an *exact text match* for the true label. Two side effects were predicted in
> writing before the runs, and both happened. The surprise came third: at the strictest
> threshold (1.0), the new grader reproduces the old cosine numbers digit for digit, which
> proves the old headline column was never actually inflated by self-grading. A fancier
> partial-credit grader was also designed, measured, and deliberately thrown away — it would
> have re-introduced exactly the arbitrariness this fix exists to remove.

Every number this project had ever reported was **self-graded**. The published grader,
`get_diagnosis_similarity_by_description_max` (`src/aicds/utils/cython_utils.py`), scores a
prediction by cosine similarity *in the same embedding space that produced the retrieval*. Each
arm therefore marks its own work with its own ruler, and a more compressed space marks itself
more leniently — which is why BiomedBERT, the most compact of the three, holds 1.000 through
threshold 0.9.

All **four** arms self-grade, not just the transformers. At the time this was written, the
defect made the cross-encoder comparison unfalsifiable — the README then recorded that every
remaining defect biased the same way. (That framing has since been retired: the results section
below shows the bias was possible but never actual, and the README now says so.) This document
covers the replacement, and the more elaborate scheme that was designed, measured, and thrown
away.

---

## What landed: `drg-exact`, a grader with zero free parameters

Selected via `--pipeline drg` (or `AICDS_PIPELINE=drg`). Score is **1.0 if any true DRG label
is exactly the predicted one, else 0.0.** No embeddings are consulted.

The diagnosis field carries DRG *group descriptions* under three coding systems
(`APR:` 168 entries, `HCFA:` 57, `MS:` 72 — three competing schemes hospitals use to classify a
stay). There is **no numeric DRG code anywhere in this dataset** — only description text — so
"DRG matching" necessarily means matching that text.

Three measured facts justify the specific implementation:

1. **Compare descriptions, not whole labels.** `preprocess_diagnosis` reconstructs the system
   prefix with a *substring* test, so **19 of the 224 emitted labels carry a wrong prefix set**
   — `hadm 103770` emits `hcfa,apr:intracranial hemorrhage` when the truth is `{apr}`. The
   prefix is also built from an unordered `set`, so its spelling varies *between and within*
   processes. Comparing whole labels would be both wrong and nondeterministic.
2. **Dropping the prefix costs nothing.** Restricting matches to the same coding system yields
   *identical* counts — 75/75 on `folds`, 76/76 on `folds_grouped` — because the three systems'
   vocabularies are disjoint in practice. A cross-system string match essentially never fires.
3. **Slicing at the first `:` is safe.** Verified: **0 of the 145 descriptions contain a
   colon.**

The grader is therefore deterministic even though `preprocess_diagnosis` is not — the
instability lives in the prefix, which is stripped, and the description *set* is
content-stable. Pinned by a subprocess test across `PYTHONHASHSEED` 0/1/12345/99991.

### The ceiling, which must be quoted with every number

`drg-exact` cannot reach 1.0, because the right answer is often not there to be found. Test
cases whose correct label appears anywhere in their own fold's training pool:

| Fold set | Reachable | Ceiling |
|---|---|---|
| `data/folds` | 75 / 129 | **58.1%** |
| `data/folds_grouped` | 76 / 129 | **58.9%** |

**Fixing the leakage moved this by one case.** It is tempting to assume regrouping the folds
also improved what is findable; it did not. The two defects are independent, and a test now
pins that.

Per fold the ceiling varies enormously on the grouped split — **3/12 (25%) in fold 7 up to
13/15 (87%) in fold 0** — so it is a fresh source of per-fold variance and must be reported per
fold.

### Two consequences to expect, written down before the runs

- **The five threshold rows will become identical.** A binary grader returns only 0.0 or 1.0.
  A 1.0 clears every threshold from 0.6 to 1.0; a 0.0 clears none. So the threshold sweep
  collapses to a single number per aggregator. That is not a bug — it is what an honest binary
  relevance judgement does, and it **eliminates the threshold-inversion artifact** documented
  in the README. The TOP-K knob remains, so the TOP-K artifact does not go away.
- **Degeneracy will survive.** `P == R == F` comes from the *retrieval-side* pruning gate: the
  prediction list descends from symptom similarity against `PRUNING_SIMILARITY`, entirely
  upstream of the grader. Do not claim degeneracy is resolved until `TP+FP` stops summing to
  the fold test size. Saturation, by contrast, should disappear for the BERT arms — a lexical
  grader has no compact-space leniency to exploit.

---

## What was rejected: a partial-credit ladder

The obvious objection to exact matching is the 58.9% ceiling, so a graded scheme — partial
credit for a near-miss label — was designed properly rather than dismissed. Three candidate
similarity functions were built independently and measured: token-set overlap (Jaccard / Dice /
overlap coefficient), character n-gram and `SequenceMatcher` fuzzy matching, and a discrete
tier ladder keyed to clinical head terms.

The evaluation used a real signal rather than intuition: the **72 label pairs that co-occur on
a single admission**. Those describe one hospital episode, so they are a usable proxy for
"should score high." The tier ladder won on recall — 54 of 72 cleared 0.6, versus 33 and 25,
and its set was a strict superset of both competitors'. It correctly unified the septicemia
family (the two most frequent labels in the dataset) and the stroke, tracheostomy and ECMO
families.

**It was still rejected.** The reasons are worth recording, because they are the reasons *any*
hand-built lexical grader fails here:

- **It needs ~156 hand-written lexicon strings** — mined from the same 146 diagnosis titles it
  then scores, several from a single occurrence, with **no held-out set** on which to show it
  had not simply memorised them.
- **Its two lowest rungs are mostly false credit.** Against non-co-occurring pairs, precision
  was **0.13 at threshold 0.6 and 0.08 at 0.7** — roughly seven of every eight admits are not
  same-episode pairs.
- **It was the *worst* of the three on discrimination.** Bootstrapped AUC over the 72
  positives: 0.902 / 0.893 / **0.868**. The differences are not significant, which is precisely
  the point — there was no measured basis for preferring it beyond judgement.
- **It scored "discharged alive" against "expired" at 0.900.** Patient survived versus patient
  died, graded as the same DRG modulo boilerplate. Two of the three designs had this defect.
- **It is structurally blind to numbers.** Its junk filter drops non-alphabetic tokens, so
  `ventilator support 96+ hours` and `< 96 hours` — genuinely different DRGs — sit at 0.900
  and cannot be separated by any lexicon edit.
- **A hard floor no lexicon reaches:** 18 of the 72 same-episode pairs score 0.000 under *all
  three* designs. `cirrhosis & alcoholic hepatitis` versus `other disorders of the liver` is
  pure cross-vocabulary paraphrase.

**The decisive argument is the purpose of P4.** It exists to remove an arbitrary ruler.
Replacing cosine with a more elaborate arbitrary ruler — 156 tuned strings, unfalsifiable rung
boundaries, fitted to the evaluation set — would concede the argument while appearing to answer
it. `drg-exact` has zero free parameters, and that is its entire value. There is consequently
**no `drg-graded` config**, and `from_name` refuses the name.

If partial credit is ever revisited, the honest route is an *external* DRG hierarchy — a
published code-to-family mapping, not a lexicon derived from these 146 strings.

---

## A correction that was *not* made

An earlier draft of the P4 plan proposed to change "**105** of the 145 unique diagnoses occur
exactly once" to "85" across `CLAUDE.md` and `docs/findings/07`. **That correction was wrong
and was abandoned.** Both numbers are real and they count different things:

- **85** — occurrence-level singletons over the 297 raw diagnosis entries.
- **105** — admission-level singletons (document frequency = 1) over the 224 entries that
  survive `preprocess_diagnosis`'s per-admission dedup. The 85 are a strict subset; the
  20-label gap is labels written **twice inside one admission**.

**105 is the correct figure for the sentence it appears in**, because that sentence bounds
exact-match *retrieval*: a label written twice in a single admission gives a retriever no
second admission to retrieve it from. It is also the pipeline-true count, since the dedup
happens before any label reaches the grader. Substituting 85 would have replaced a
pipeline-true number with a pre-dedup raw-file number and broken the 58.1% argument it
supports.

Root cause of the duplicates, confirmed: **71 of 129 admissions list their APR label twice,
byte-identically** — an upstream DRGCODES extraction artifact.

---

## Related defects found while wiring this up

Recorded because each would silently produce a wrong or unnoticed result:

- **A config field with no reader is worse than no field.** For a window during development
  `--pipeline drg` was selectable, printed a plausible banner, ran a full 10-fold pass and
  emitted *cosine* numbers labelled as DRG ones. Nothing failed. `SUPPORTED_GRADERS` plus
  `require_supported_grader` now make that a loud error at startup, before the model load.
- **`bert_models.py` keeps its own copy of `containGreaterOrEqualsValue`**, and no test
  exercises it — the tests import the `cython_utils` original or define a third. Change the
  threshold semantics and the baseline gets the new rule while the BERT arm keeps the old one.
- **`scripts/analyze_score_distributions.py` re-implements the grader** against its own cosine
  (a fourth in the repo) and imports no config, so it will keep generating saturation evidence
  for a retired ruler. Its self-check guards only the cosine kernel, not the aggregator.
- **A future mean-based aggregator would reintroduce nondeterminism.** Three tier constants
  summed in hash-varying order differ in the last ULP for 7 of 56 combinations, and the one
  admission with three descriptions (`HADM 178513`) is a *test* case in both fold sets. `MAX`
  is order-invariant, so today's path is safe; sort the labels before reducing if that ever
  changes.
- ~~**`scripts/compare_models.py` will refuse to build a PDF** for these runs — its 16 sanity
  assertions pin legacy values (TODO P34).~~ **Fixed since (P34, `c429d1b`):** the assertions
  are now keyed by pipeline, and the script refuses loudly on a directory whose pipeline it
  cannot identify rather than mislabeling a chart.

---

## The results: both predictions held, and a third thing nobody predicted

Run 2026-08-06 on RunPod Linux, `--pipeline drg`, `PYTHONHASHSEED=0`, HEAD `538d368`.

**Prediction 1 — the five threshold rows became identical. CONFIRMED.** Every aggregate block
prints the same `TP FP P R FS PR` at 0.6, 0.7, 0.8, 0.9 and 1.0. A binary grader returns only
0.0 or 1.0, a 1.0 clears every threshold and a 0.0 clears none, so the sweep collapses to one
number per aggregator. **The threshold knob is gone** — one of the two arbitrary dials that
decided the encoder ordering.

**Prediction 2 — degeneracy survived. CONFIRMED.** Bio_ClinicalBERT still prints
`P == R == FS == 0.2491025641025641` with `PR` exactly `1.0` and `TP+FP = 3.3 + 9.6 = 12.9`,
the mean fold test size. The baseline alone has `P = 0.2512 != R = 0.1923`, `PR = 0.7558`,
`TP+FP = 9.8`. As predicted, this comes from the retrieval-side `PRUNING_SIMILARITY` gate,
entirely upstream of the grader, so no grader change could touch it. Saturation, by contrast,
is gone — with the threshold sweep collapsed there is nothing left to saturate.

### The unpredicted result: `drg-exact` reproduces corrected cosine at threshold 1.0 EXACTLY

Not approximately. **72 numbers — 6 aggregators × 6 columns × 2 arms — bit-identical.**

| aggregator | baseline cos@1.0 | baseline drg | Bio_ClinicalBERT cos@1.0 | drg |
|---|---|---|---|---|
| MAX | 0.087652 | 0.087652 | 0.086154 | 0.086154 |
| TOP-10 | 0.216278 | 0.216278 | 0.249103 | 0.249103 |
| TOP-20 | 0.222944 | 0.222944 | 0.304872 | 0.304872 |
| TOP-30 | 0.229611 | 0.229611 | 0.351282 | 0.351282 |
| TOP-40 | 0.238307 | 0.238307 | 0.391026 | 0.391026 |
| TOP-50 | 0.238307 | 0.238307 | 0.406410 | 0.406410 |

**Why, measured rather than assumed.** Cosine reaches exactly 1.0 only when two diagnosis
*embeddings* are parallel, and `embending_diagnosis` keys the dict by the **raw** description
while embedding the **preprocessed** string. So the two graders can differ only where distinct
raw descriptions preprocess to the same text. Counted over all 145 descriptions that reach the
grader:

| preprocessing | distinct forms | colliding groups | spurious 1.0 pairs |
|---|---:|---:|---:|
| `legacy` | 143 | **2** | 2 |
| `corrected` | 144 | **1** | 1 |

The single surviving collision under `corrected` is `respiratory system diagnosis w/ ventilator
support 96+ hours` against `... w ventilator support 96+ hours` — semantically **the same
thing**, since `w/` and `w` both mean "with". It never occurs as a (truth, prediction) pair in
the 129 test cases, which is why the agreement is exact rather than merely close.

**The collision that `legacy` had and `corrected` does not is the `w/o` defect itself:**
`tracheostomy w long term mechanical ventilation **w** extensive procedure` and
`... **w/o** extensive procedure` collapse to one string under legacy preprocessing. So **on
the legacy pipeline the two graders would have disagreed, and `drg-exact` would have been the
correct one.** P7's `w/o` fix is what makes them equivalent — the two corrections compose.

### What this does and does not buy

**Does:** the README's threshold-1.0 table — the only column it reports, chosen because it is
the only setting where the encoders separate at all — turns out to have been an
encoder-independent exact-match metric all along. That is now proven numerically instead of
argued. The self-grading defect is real at thresholds 0.6–0.9, which is exactly where the three
BERT arms saturate, and structurally absent at 1.0.

The honest way to state the outcome: **P4 did not move the numbers, it removed the reason to
doubt them.** A defect that turns out not to have contaminated the reported column is still
worth finding, because "it didn't matter here" is a measurement, not an assumption — and it
*did* matter under legacy.

**Does not:** rescue the encoder comparison. `drg-exact` kills the *threshold* knob but not the
*K* knob, and the ordering still inverts on K alone (BioSentVec 1st under MAX, 4th under
TOP-30). The conclusion stands unchanged: **no encoder ranking is supported by this
experiment.** Removing the last knob was P5's job, and MRR is the metric with no K at all —
see [13](13-rank-aware-metrics.md).

---

*Companion documents:* [04](04-metric-degeneracy.md) degeneracy ·
[05](05-patient-leakage.md) leakage · [07](07-comparison-validity.md) the synthesis ·
[11](11-corrected-pipeline-first-results.md) the corrected four-arm results ·
[13](13-rank-aware-metrics.md) the rank-aware comparison

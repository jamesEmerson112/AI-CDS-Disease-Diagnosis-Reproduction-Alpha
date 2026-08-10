# Metric degeneracy: precision, recall, and F-score are the same number in every row

> **In plain words.** The result files print three separate grades per row — precision, recall,
> and F-score — and in every one of the 12,600 BERT rows the three grades are the *same number*.
> That is not a coincidence; it is arithmetic. Precision asks "of the cases you answered, how
> many were right?" and recall asks "of all the cases, how many did you get right?" Those are
> different questions **only if you skip some cases**. The BERT models never skip — they answer
> all 129 — so "the cases you answered" and "all the cases" are the same pile, the two fractions
> have the same denominator, and F-score (a blend of the two) collapses onto them. Every BERT
> "F1" in this repo is therefore plain accuracy wearing three labels. The story got its ending
> later: the baseline finally ran ([09](09-baseline-first-run.md)) and its three columns *do*
> differ, because it skips 23% of cases — and [13](13-rank-aware-metrics.md) proved the columns
> were never broken, just mislabelled: `P` is the hit rate over answered cases, `R` the hit rate
> over all cases, and `PR` the fraction answered.

## Conclusion

In every committed result file — all three BERT arms, all 10 folds, all 6 retrieval strategies
(MAX and TOP-10 through TOP-50), all 5 thresholds, both per-patient and fold-aggregated rows —
**precision, recall, and F-score are numerically identical**. Not close: identical to at least
1e-12, in 12,600 out of 12,600 metric rows checked. A representative row, straight from the file:

```
0.9     1       0       1.0     1.0     1.0      0.07692307692307693
```
(`TP FP P R FS PR`, HADM_ID=124073, fold 0, MAX strategy, from
[`docs/Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/PerformanceIndex.txt`](../Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/PerformanceIndex.txt), line 3-9.)

`P`, `R`, and `FS` are printed as three separate columns, but they carry one bit of information
between them: whether this test case's retrieved diagnosis cleared the threshold. There is no
row anywhere in the committed data where they diverge. **Every "F1" in the committed BERT results
— including the headline F1 = 1.000 — is arithmetically identical to accuracy**, not to
precision, recall, or a genuine harmonic mean of the two.

The published BioSentVec figures are a separate case and this claim does **not** extend to them;
see [The baseline may not be degenerate](#the-baseline-may-not-be-degenerate) below — a question
that has since been settled by running it.

**This is a different, independent problem from
[03-metric-saturation.md](03-metric-saturation.md).** Saturation says the 0.6 threshold is too
lenient for BERT embeddings, so almost every score clears it — a problem that improves if you
raise the threshold. Degeneracy says P, R, and F1 collapse to the same value **at every
threshold**, 0.6 or 1.0, saturated or not, because of how a "test case" is scored, not how
similar two diagnoses are judged to be. Raising the threshold shrinks the number; it does not
restore three independent measurements. The two problems have to be fixed separately.

## Verification

The claim was checked directly against the three committed result files, not assumed from reading
the code. Every line matching a `threshold TP FP P R FS PR` data row (7 whitespace-separated
tokens, first one a threshold in `{0.6, 0.7, 0.8, 0.9, 1.0}`) was parsed and tested for
`abs(P - R) > 1e-12 or abs(R - FS) > 1e-12`.

| File | Rows parsed | P≠R or R≠FS | Aggregate rows with `PR`≠1.0 |
|---|---:|---:|---:|
| `Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/PerformanceIndex.txt` | 4,200 | 0 | 0 |
| `Prediction_Output_BiomedBERT_15022026_12-03-36/PerformanceIndex.txt` | 4,200 | 0 | 0 |
| `Prediction_Output_BlueBERT_15022026_12-24-38/PerformanceIndex.txt` | 4,200 | 0 | 0 |
| **Total** | **12,600** | **0** | **0** |

4,200 rows per file splits as 3,870 per-patient rows (129 test cases × 6 strategies × 5
thresholds) + 330 aggregate rows (10 folds × 6 strategies × 5 thresholds, plus the final
"10-FOLD" summary block, 6 strategies × 5 thresholds). Both counts were verified directly, not
assumed — see the script output below.

### Script

```python
#!/usr/bin/env python3
"""Verify that reported P, R, and F-score are identical in every PerformanceIndex.txt row.
Run from the repository root: python3 verify_metric_degeneracy.py
"""
import glob, os

EPS = 1e-12
FILES = sorted(glob.glob("docs/Prediction_Output_*/PerformanceIndex.txt"))

def parse_rows(path):
    """Yield (threshold, tp, fp, p, r, fs, pr, is_aggregate) for every metric row."""
    is_aggregate = False
    with open(path) as fh:
        for line in fh:
            if "PERFORMANCE INDEX of" in line:
                is_aggregate = "HADM_ID" not in line   # fold/10-fold block vs per-patient block
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            try:
                threshold, tp, fp, p, r, fs, pr = (float(x) for x in parts)
            except ValueError:
                continue  # header row or prose
            if threshold not in (0.6, 0.7, 0.8, 0.9, 1.0):
                continue
            yield threshold, tp, fp, p, r, fs, pr, is_aggregate

total, violations, pr_violations = 0, 0, 0
for path in FILES:
    rows = list(parse_rows(path))
    v = sum(1 for _, _, _, p, r, fs, _, _ in rows if abs(p - r) > EPS or abs(r - fs) > EPS)
    agg_pr_bad = sum(1 for *_, pr, agg in rows if agg and abs(pr - 1.0) > EPS)
    total += len(rows); violations += v; pr_violations += agg_pr_bad
    print(f"{path}: {len(rows)} rows, {v} P/R/FS mismatches, {agg_pr_bad} bad aggregate PR")

print(f"\nTOTAL: {total} rows, {violations} P/R/FS mismatches, {pr_violations} bad aggregate PR")
```

Running it against the three committed files reproduces the table above exactly: 12,600 rows
parsed, 0 mismatches, 0 bad aggregate `PR` values.

## Mechanism

There are two code paths that produce this result, and they are subtly different — worth being
precise about, since only one of them matches the mechanism visible from the file format alone.
(Path note: this document predates the package move; the files named below now live under
`src/aicds/`, and the line numbers have drifted — go by function name.)

### Aggregate rows: `precision` and `recall` are different formulas that coincide

The 330 aggregate rows per file (per-fold totals and the final 10-fold mean) are computed by
`compute_performance_index()` and `compute_aggregated_performance_index()` in
`cython_utils.py`:

```python
if (tp + fp) != 0:
    precision = tp / (tp + fp)
else:
    precision = 0

recall = tp / nrow

if recall + precision != 0:
    f_score = (2 * recall * precision) / (recall + precision)
else:
    f_score = 0

prediction_rate = (tp + fp) / nrow
```
(`cython_utils.py:381-393` at the time of writing, and identically in the aggregated variant.)

`precision` and `recall` are genuinely different expressions here — `tp/(tp+fp)` vs. `tp/nrow` —
so their equality is not definitional; it depends on an invariant holding across the whole fold:
**`tp + fp == nrow`**, i.e. every one of the `nrow` test cases in a fold ends up counted as
either a TP or an FP, never both, never neither.

That invariant comes from the confusion-matrix update code in the fold loop (`bert_models.py`,
MAX strategy and each TOP-K strategy — the baseline's `cython_utils.predictS2V()` follows the
identical pattern):

```python
if diagnosis_similarity_max >= b:
    values[TP] += 1
else:
    values[FP] += 1
```

Every test case hits exactly one branch of this `if/else` for every threshold — there is no
third outcome. Sum that over the `nrow` test cases in a fold and `TP_total + FP_total == nrow`
follows directly. Substituting into the formulas above:

```
precision = TP / (TP + FP) = TP / nrow
recall    = TP / nrow                        (defined this way directly)
    ⇒ precision ≡ recall, exactly, for every threshold, every fold, every model

F1 = 2·P·R / (P + R), and with P = R = x this is 2x² / 2x = x
    ⇒ F1 ≡ precision ≡ recall ≡ TP / nrow  (accuracy)

prediction_rate = (TP + FP) / nrow = nrow / nrow = 1.0
```

Both are confirmed directly in the committed data. `prediction_rate` reads exactly `1.0` in all
990 aggregate rows across the three files (330 × 3, 0 exceptions — see the Verification table).
And the final 10-fold summary row for `MAX SIMILARITY`, threshold 0.6, in every file, reads
`TP=12.9, FP=0.0` — `12.9 = 129 / 10`, the mean fold size, because nine folds test on 13
patients and one fold tests on 12 (`FOLD 9: LEN test: 12`, same file). The pattern holds at
every threshold, not just 0.6:

| Model | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | TP+FP (all thresholds) |
|---|---|---|---|---|---|---|
| Bio_ClinicalBERT | 12.9+0.0 | 12.9+0.0 | 12.5+0.4 | 3.7+9.2 | 1.9+11.0 | 12.9 |
| BiomedBERT | 12.9+0.0 | 12.9+0.0 | 12.9+0.0 | 12.9+0.0 | 2.2+10.7 | 12.9 |
| BlueBERT | 12.9+0.0 | 11.7+1.2 | 5.1+7.8 | 2.5+10.4 | 2.3+10.6 | 12.9 |

(All 15 cells sum to 12.9. Source: `10-FOLD PERFORMANCE INDEX of MAX SIMILARITY by MAX` block in
each file, around line 5921-5927. **This `TP+FP` column is still the fastest degeneracy
diagnostic available** — 12.9 means the arm never abstained; the baseline's later run sums to
9.9 instead, which is the tell that it did.)

The final summary row is itself an arithmetic mean of the 10 per-fold rows
(`util_cy.print_performance_index()` divides accumulated per-fold sums by `K_FOLD = 10`). Since
every per-fold row already satisfies `P = R = FS` before the averaging, the mean of the P column
and the mean of the R column are means of two sequences that were identical
element-for-element — so the summary row inherits the equality trivially.

### Per-patient rows: `precision` and `recall` are the *same* formula, by construction

The remaining 3,870 rows per file (one block per test patient per strategy) do not go through
`cython_utils` at all. `bert_models.py` computes them inline, and here the equality isn't even
an emergent consequence of an invariant — it's definitional:

```python
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fp) if (tp + fp) > 0 else 0
```
(`bert_models.py`, MAX strategy; identically for every TOP-K strategy.)

`tp` and `fp` here are local to a single test case (always `(1, 0)` or `(0, 1)` — one patient,
one verdict), and `recall` is written with the exact same right-hand side as `precision`, not
`tp / nrow`. So for these rows, `P == R` isn't a consequence of the counting protocol; the two
lines of code are copies of each other. This is a blunter version of the same underlying defect:
nothing in a per-patient row's arithmetic can ever separate "precision" from "recall" from "was
this one prediction right or wrong," because only one binary outcome was ever recorded for it.

### The invariant is empirical, not absolute, but it holds throughout the BERT data

`TP + FP == nrow` requires every test case to have at least one retrieval candidate that clears
`PRUNING_SIMILARITY = 0.5` (`Constants.py`) — a test case with zero candidates skips the TP/FP
update entirely (guarded by `max_index != -1` in both arms). It is not a law of the code; it is
a fact about the embeddings. Given the compactness finding in
[03-metric-saturation.md](03-metric-saturation.md) — virtually every patient pair clears far
higher thresholds than 0.5 — it holds without exception for the three BERT arms, which is
exactly what the zero-violation `prediction_rate` count above confirms.

### The baseline may not be degenerate

**Status update, 2026-08-08: this section's question has been answered — the baseline is NOT
degenerate. It was settled by running it** ([09-baseline-first-run.md](09-baseline-first-run.md),
2026-08-05): P ≠ R in every row, prediction rate 0.7679 throughout, because 30 of 129 test cases
(23.3%) abstain when nothing clears `PRUNING_SIMILARITY`. The reasoning below is kept as
written, because its prediction — that degeneracy is a downstream consequence of embedding
compactness, not a structural property of the code — is exactly what the run confirmed.

The invariant is verified across the 12,600 committed BERT rows. At the time of writing it had
**not** been shown for the BioSentVec arm, and the only surviving record of a baseline run
suggested it did *not* hold there.
[`archive/stale-docs/Reproduce_w_transformers.md`](../../archive/stale-docs/Reproduce_w_transformers.md)
lines 134-143 report, at threshold 0.6:

| Method | F1 | Precision | Recall |
|---|---:|---:|---:|
| TOP-10 | 0.489 | 0.621 | 0.412 |
| TOP-20 | 0.512 | 0.598 | 0.448 |
| TOP-30 | 0.521 | 0.587 | 0.467 |

Precision and recall diverge, and the F1 column is a genuine harmonic mean of them — TOP-20 and
TOP-30 reproduce to three decimals. If those numbers are real then `tp + fp < nrow` for that
run: some test cases had *no* candidate clearing `PRUNING_SIMILARITY = 0.5` and abstained
entirely. That is what one would expect from sent2vec's less compressed similarity distribution,
and it would make degeneracy a **downstream consequence of embedding compactness** rather than
an independent structural property of the code.

The one suspicious detail in that archived table — the implied prediction rate
(`recall / precision`) drifting 0.663 → 0.749 → 0.795 as K increases, when the pruning gate is
independent of K — remains unexplained; the 2026-08-05 run shows a constant prediction rate at
every K, as the code requires, so the archived table was likely assembled from more than one run
or edited by hand. Its headline number was nonetheless reproduced to within 0.007.

**And [13-rank-aware-metrics.md](13-rank-aware-metrics.md) put the final label on the whole
finding**: `RankMetrics.txt` reproduces legacy's `P`, `R`, and `PR` columns bit-exactly from an
independent code path, identifying `R` as the all-cases hit rate, `P` as the answered-cases hit
rate, and `PR` as coverage. So `P == R` was never the metric collapsing — it is `PR == 1.0`,
i.e. the answered and all-cases populations being the *same set* for arms that never abstain.
**The columns were never wrong; they were unlabelled.**

## Why this matters

- **Every score in the committed BERT results is accuracy, not F1.** The BERT arm's F1 = 1.000
  is, by this arithmetic, "fraction of test patients for whom the retrieved diagnosis (or
  diagnoses, for TOP-K) cleared the threshold." That is a real, computable, and honest quantity
  — but it is not a precision/recall trade-off, and comparing it directly to the paper's own
  reported F1 assumes a measurement that was never actually taken.

- **The metric cannot trade precision against recall**, because there is no mechanism by which
  retrieving more candidates can cost anything. A TOP-K test case is scored TP the moment *any
  one* of its K retrieved diagnoses clears the threshold — the other K-1 are free. Retrieving 50
  candidates instead of 10 can only ever help, never hurt, which is exactly why TOP-K scores
  rise monotonically with K in the committed data — an artifact of the counting rule, not a
  finding about the models:

  | K | 10 | 20 | 30 | 40 | 50 |
  |---|---|---|---|---|---|
  | Bio_ClinicalBERT F1 @ threshold 0.9 | 0.797 | 0.906 | 0.930 | 0.946 | 0.953 |

  (10-fold means, `MAX SIMILARITY by MAX` is the K=1 case at 0.285; source:
  [`Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/PerformanceIndex.txt`](../Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/PerformanceIndex.txt),
  the `10-FOLD PERFORMANCE INDEX of TOP-K SIMILARITY by MAX` blocks.) A metric that can only go
  up as you retrieve more is not measuring retrieval quality; it is measuring how many chances
  the model was given. (This is the defect [13](13-rank-aware-metrics.md) later closed with MRR
  and Precision@K.)

- **It cannot distinguish "predicted the right diagnosis" from "predicted many diagnoses, one of
  which happened to match."** A TOP-50 prediction that is 49 wrong guesses and 1 lucky hit
  scores identically to a TOP-1 prediction that is exactly right. There is no column in the
  output where that difference could show up.

- **This is independent of saturation** ([03-metric-saturation.md](03-metric-saturation.md)).
  Raising the threshold makes the reported number smaller — Bio_ClinicalBERT's MAX-strategy F1
  drops from 1.000 at 0.6 to 0.146 at 1.0 — but it stays exactly one-dimensional at every
  threshold along the way (all 5 threshold rows in the mechanism table above satisfy
  `TP + FP = 12.9`). Fixing the threshold does not fix this.

- **To be fair to the metric as reported: accuracy@K is a legitimate quantity.** "Fraction of
  test patients for whom at least one of the K nearest neighbors carried a matching diagnosis"
  is a real, interpretable, defensible number to report. The problem is not that it was computed
  — it's that it is printed under three column headers (`P`, `R`, `FS`) that imply three
  independent measurements were made, inviting a reader (including the paper comparison in
  [`README.md`](../../README.md)) to treat it as a precision/recall/F1 result when only one
  quantity was ever produced.

## The fix

A genuine set-level precision/recall/F1 would restore the trade-off that's currently absent. For
each test patient, define the ground-truth diagnosis set `GT` (already available per-admission —
`test_admission.diagnosis` is a list, since `preprocess_diagnosis()` can return multiple
DRG-coded strings per admission) and the predicted diagnosis set `Pred` (the union of diagnoses
across the K retrieved candidates, rather than a single MAX-similarity scalar). Then, using some
match criterion between individual diagnoses:

```
Precision = |GT ∩ Pred| / |Pred|
Recall    = |GT ∩ Pred| / |GT|
F1        = 2 · Precision · Recall / (Precision + Recall)
```

This is a real fix, not a relabeling, because it removes the property that makes the current
metric degenerate: growing `K` grows `|Pred|`, and every diagnosis added to `Pred` that isn't in
`GT` now *lowers* precision. TOP-K expansion stops being free. A model that pads its predictions
with irrelevant diagnoses to inflate its odds of a lucky hit — rewarded under the current scheme
— is penalized under this one.

This is the same proposal as **Strategy D (Set-Level Jaccard/F1)** in
[`docs/score_distribution_analysis/next_steps.md`](../score_distribution_analysis/next_steps.md),
and it is tracked as **P6** in `docs/plans/correctness-fixes.md` — still open as of 2026-08-08.
This finding changes the original prioritization: Strategies A and C (MEAN aggregation,
Hungarian matching) still collapse each test case to a single scalar scored against a single
threshold, so they inherit the one-TP-or-FP-per-test-case counting rule and remain degenerate in
exactly the way described above — they would change *which* threshold saturates, not whether P,
R, and F1 stay locked together. Only a set-level formulation like Strategy D changes what a
"test case" is scored against, which is the actual prerequisite for P, R, and F1 to mean three
different things. Strategy B (DRG matching) is not a competing option here — it **shipped** as
the `drg-exact` grader ([12](12-drg-grader.md)) and is the natural match criterion to plug into
`GT ∩ Pred` above, making "pluggable match criterion, pluggable metric" the right frame going
forward rather than "pick one of four independent strategies."

## Reproducing this analysis

Save the script above as `verify_metric_degeneracy.py` in the repository root and run:

```bash
conda activate disease-diagnosis
python3 verify_metric_degeneracy.py
```

It only reads the three committed `Prediction_Output_*/PerformanceIndex.txt` files — it does not
touch the prediction pipeline, require a GPU, or write anything.

## Summary

| Question | Answer |
|---|---|
| Do P, R, and F-score ever differ in the committed BERT results? | No — 0 of 12,600 rows, across all 3 models, all 10 folds, all 6 strategies, all 5 thresholds. |
| Is the reported "F1" actually a harmonic mean of two independent measurements? | No — it is `TP/nrow`, i.e. accuracy, at every row. |
| Does this extend to the baseline? | **No — settled 2026-08-05 by running it** ([09](09-baseline-first-run.md)): the baseline abstains on 23.3% of cases, so its P ≠ R in every row. |
| Is this the same problem as metric saturation ([03](03-metric-saturation.md))? | No — saturation is about the 0.6 threshold being too lenient; degeneracy holds at every threshold. |
| Does raising the threshold fix it? | No — it changes the accuracy value, not the one-dimensionality. |
| Is `accuracy@K` a legitimate metric to report on its own? | Yes — the defect is labeling it as three separate columns (P, R, FS), not computing it. `P`/`R`/`PR` are really answered-hit-rate / all-cases-hit-rate / coverage ([13](13-rank-aware-metrics.md)). |
| Does the fix require new data? | No — ground-truth diagnosis sets are already parsed per admission; the fix is a scoring-logic change (set overlap instead of single-scalar threshold), tracked as P6. |

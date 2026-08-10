# 09 — The baseline runs, and it settles the degeneracy question

> **In plain words.** After months of the original authors' model being unable to run at all —
> crash bugs, a dependency that will not compile on Windows, a 21 GB model file — it finally
> executed, on a rented Linux machine. Two things came out. First, it reproduced the paper's
> headline score almost exactly (0.482 vs the published 0.489), which is the first proof that
> this repository computes what the paper computed. Second, it answered a question no amount of
> code-reading could: the strange all-three-grades-identical pattern in the BERT results
> ([04](04-metric-degeneracy.md)) is caused by the BERT models' cramped embedding space, not by
> a bug in the shared code — because the baseline, with its roomier space, produces three
> *different* grades. The tell is one column: the baseline sometimes says "I don't know"
> (23% of cases), and the BERT models never do.

**On 2026-08-05 the BioSentVec arm executed end-to-end for the first time in this checkout.** It
reproduces the published F1 to within 0.007, and — more valuable — it resolves the open question
that [04-metric-degeneracy.md](04-metric-degeneracy.md) could not answer for lack of an
artifact.

Run environment: RunPod CPU pod, AMD Ryzen Threadripper 7960X, 32 vCPU / 64 GB, Ubuntu 20.04,
Python 3.9 (conda). Exit code 0. Model load 25.28 s, ten folds in 737.2 s, total ~12.7 min. Raw
output preserved at `results/Prediction Output_05082026 18-55-32/` (gitignored — it carries
`HADM_ID`s).

---

## 1. It reproduces the published number

Comito et al.'s headline baseline, and this run, both at TOP-10 / threshold 0.6:

| | Precision | Recall | F1 |
|---|---|---|---|
| Published | 0.621 | 0.412 | **0.489** |
| This run | 0.562 | 0.426 | **0.482** |

**F1 differs by 0.007 (1.4% relative).** Precision comes in 0.059 low and recall 0.014 high,
which is the signature of a slightly different abstention rate rather than a different retrieval
quality — the two errors partly cancel in the harmonic mean.

This is the first evidence that the pipeline in this repository computes what the paper
computed. Every previous number in the project came from either the deleted pre-reorganisation
`CS2V.py` or the BERT arm.

**Caveat, and it is not small:** this run used the **original leaky folds and the original
metric**. It is a reproduction of the published baseline, not a corrected result. Both arms'
numbers here are still subject to the patient leakage of [05](05-patient-leakage.md) and the
ceiling of [07](07-comparison-validity.md). Do not quote 0.482 as a corrected figure — the
corrected equivalent is 0.3922 ([11](11-corrected-pipeline-first-results.md)), and the 18.4%
drop between them is itself a finding.

## 2. The degeneracy is BERT-specific, not structural

[04-metric-degeneracy.md](04-metric-degeneracy.md) established that precision == recall ==
F-score in all 12,600 rows of committed BERT results, and flagged as unresolved whether this was
a property of the *code* or of the *BERT embedding space*. No baseline artifact survived to test
it.

**It is the embedding space.** In this run precision and recall are different numbers in every
row:

```
10-FOLD PERFORMANCE INDEX of TOP-10 SIMILARITY by MAX
        TP      FP       P       R       FS      PR
0.6     5.5     4.4     0.5624  0.4256  0.4824  0.7679
0.7     4.2     5.7     0.4217  0.3250  0.3658  0.7679
0.8     3.7     6.2     0.3710  0.2865  0.3222  0.7679
0.9     3.2     6.7     0.3254  0.2474  0.2801  0.7679
1       3.2     6.7     0.3254  0.2474  0.2801  0.7679
```

### The mechanism, confirmed arithmetically

Note the `PR` column: **0.7679 in every single row**, across every TOP-K and every threshold.
That is the *prediction rate* — the fraction of test cases on which the system commits to an
answer at all. The baseline **abstains on 23.2% of cases** because no training patient clears
`PRUNING_SIMILARITY = 0.5`.

Check it directly: TP + FP = 5.5 + 4.4 = 9.9, against ~12.9 test cases per fold.
9.9 / 12.9 = 0.767. The missing 3.0 cases per fold produced no prediction, so they increment
neither TP nor FP.

Counted exactly: **30 of 129 test cases (23.3%) abstain**, between 1 and 5 per fold. The
condition in code is `max_index == -1` in `cython_utils.py` — no training patient cleared
`PRUNING_SIMILARITY = 0.5`.

### The same column, across all four arms

The `TP + FP` sum is the whole story, and it separates the arms cleanly (TOP-10, threshold 1.0):

| Arm | TP | FP | **TP+FP** | P | R | F | P == R? |
|---|---|---|---|---|---|---|---|
| **BioSentVec** | 3.2 | 6.7 | **9.9** | 0.3254 | 0.2474 | 0.2801 | **No** |
| Bio_ClinicalBERT | 3.7 | 9.2 | **12.9** | 0.2853 | 0.2853 | 0.2853 | Yes |
| BiomedBERT | 3.3 | 9.6 | **12.9** | 0.2545 | 0.2545 | 0.2545 | Yes |
| BlueBERT | 3.1 | 9.8 | **12.9** | 0.2391 | 0.2391 | 0.2391 | Yes |

Every BERT arm sums to exactly 12.9 — the mean fold test size — so `tp + fp == nrow` and the
collapse is forced. BioSentVec sums to 9.9. **That single column is the whole degeneracy
finding, visible at a glance** — and it remains the fastest diagnostic available for any future
run.

### The one honest comparison available at the time

*(Status note: the ranking in this subsection is from the LEGACY pipeline and is superseded —
the corrected ranking, and the statistics showing no ranking survives at all, are in
[11](11-corrected-pipeline-first-results.md) and [13](13-rank-aware-metrics.md). Kept because
the conclusion it reached is the one that held.)*

Threshold 1.0 is the only setting where the three BERT encoders separate at all (at 0.6 all
three report a saturated 1.000). At that setting:

**Bio_ClinicalBERT 0.285 > BioSentVec 0.280 > BiomedBERT 0.254 > BlueBERT 0.239.**

The baseline sits *inside* the BERT range, not below it. The spread between the best BERT model
and the baseline is **0.005** — far smaller than the +0.11 to +0.26 leakage inflation of
[05](05-patient-leakage.md) and smaller than the per-fold σ of 0.071–0.124. **No encoder ranking
is supported by this experiment.** That is the defensible claim, and it is a more interesting
one than a spurious win.

That is exactly the condition finding 04 identified as necessary for non-degeneracy. Restating
its mechanism: degeneracy requires `tp + fp == nrow`, which forces `precision = tp/nrow`, which
*is* the recall definition in use, and the harmonic mean of two equal numbers is that number.
The baseline breaks the premise — `tp + fp < nrow` — so precision and recall stay independent.

## 3. Why this unifies findings 03 and 04

Saturation and degeneracy had looked like two separate metric defects. They are **one cause
expressed at two different gates.**

Biomedical BERT embeddings are compact — mean pairwise cosine 0.72–0.93 even between unrelated
diagnoses ([03](03-metric-saturation.md)). That single fact drives both:

| Gate | Constant | What compactness does | Finding |
|---|---|---|---|
| Pruning gate | `PRUNING_SIMILARITY = 0.5` | Nothing ever falls below it, so the BERT arm **never abstains** → `PR = 1.0` → `tp+fp == nrow` → **degeneracy** | [04](04-metric-degeneracy.md) |
| Scoring threshold | `0.6` | Nearly every pair clears it → **saturation at 1.000** | [03](03-metric-saturation.md) |

The baseline, whose 700D sent2vec space is far less compact, trips neither: it abstains 23.2% of
the time and it scores 0.482 rather than 1.000.

**This is the sharpest single statement the project can make.** It is not "BERT is worse." It is
that a compact embedding space silently disables *two independent safety valves* in an
evaluation harness designed around a non-compact one — a transferable lesson about porting a
retrieval evaluation to any domain-tuned encoder, and one that only became provable once both
arms ran.

## 4. TOP-K monotonicity, confirmed on the baseline too

| Strategy @ thr 0.6 | F1 |
|---|---|
| MAX | 0.2469 |
| TOP-10 | 0.4824 |
| TOP-20 | 0.4824 |
| TOP-30 | 0.4920 |
| TOP-40 | 0.4920 |
| TOP-50 | 0.4920 |

Monotonically non-decreasing, and flat past TOP-30 — the pruning gate stops supplying new
candidates, so larger K adds nothing. This is the artifact described in
[07](07-comparison-validity.md): one hit suffices and there is no penalty for the other K−1
predictions. It reproduces in the baseline arm, confirming it is a property of the metric rather
than of any encoder. ([13](13-rank-aware-metrics.md) later removed the K knob entirely.)

## 5. Provenance

| | |
|---|---|
| Repo state | `f56f64e` + a two-hunk patch equal to fixes 1–2 of `c2fee6e` |
| Not applied | fix 3 of `c2fee6e` (the discarded `line.replace`), **verified inert** in that commit's own message |
| Equivalent to | `main` @ `7da5901`, modulo the package move (proven byte-exact by `pytest -m golden`) and the inert fix |

The honest position: **reproducible from `7da5901` in principle, not yet reproduced from it in
fact.** Closing that gap costs 13 minutes and ~$0.25 — re-run from a clean checkout of `main`
and diff the `PerformanceIndex.txt`. Worth doing, because the same run would double as the
project's **first baseline-arm golden**, which currently does not exist. (The later `corrected`
and `drg` runs of [11](11-corrected-pipeline-first-results.md) were made from clean checkouts,
so the gap is closed for those pipelines; it remains open for this specific legacy run.)

---

*Companion documents:* [01](01-baseline-reproduction.md) the method walkthrough and provenance ·
[03](03-metric-saturation.md) saturation · [04](04-metric-degeneracy.md) degeneracy ·
[07](07-comparison-validity.md) comparison validity · [08](08-runtime-and-cost.md) runtime ·
[11](11-corrected-pipeline-first-results.md) the corrected results

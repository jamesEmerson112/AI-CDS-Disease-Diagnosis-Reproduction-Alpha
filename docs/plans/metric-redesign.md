# Metric redesign

> **In plain words.** The original scoring rule had three separate flaws: it forgave every
> wrong guess as long as one guess was close (MAX), it let each model grade its own homework
> (the model's similarity score decided whether the model's answer was "right"), and it ignored
> *where* in the list the right answer appeared. This document weighed the replacement options.
> Option C — rank-aware metrics against a model-independent answer key — was implemented
> ([12](../findings/12-drg-grader.md), [13](../findings/13-rank-aware-metrics.md)) and its
> measurements overturned two of this document's own assumptions along the way, both preserved
> below with their corrections. Option B (a real precision/recall) was **retired on 2026-08-11**
> — the degeneracy it was designed to kill turned out to be a labelling artifact, and its
> method would have undone option C's main gain. *(Updated 2026-08-11 by the TODO audit.)*

**Status: option C implemented 2026-08-06** (`drg-exact` grader + `RankMetrics.txt`); **option B
RETIRED 2026-08-11 as P6** (§B below carries the reasoning; the math is kept for reference);
options A and D remain unimplemented, with A's payoff diminished now that the
threshold knob is gone. Originally: the substance of the "pluggable metrics" line in
[revival-roadmap.md](revival-roadmap.md) Phase 5.

The legacy metric has three independent defects. Fixing one does not fix the others, so they
are listed separately with separate remedies.

## What is wrong now

The pipeline scores a test case as correct when

```
MAX over (ground_truth_diagnoses x predicted_diagnoses) of cosine(gt, pred)  >=  threshold
```

with `threshold` swept over {0.6, 0.7, 0.8, 0.9, 1.0}.

### Defect 1 — MAX is maximally lenient

Patients carry 1.74 diagnoses on average, so the Cartesian product holds ~3 pairs. Taking the
MAX means *if any true diagnosis is close to any predicted diagnosis, the whole case counts as
correct.* Nothing penalises the predictions that were wrong.

Measured: Bio_ClinicalBERT's mean pairwise diagnosis similarity is 0.8253, but the per-patient
MAX mean is 0.8502 — the MAX operator alone moves the distribution up
(`docs/score_distribution_analysis/score_distribution_summary.txt`). *Re-measured 2026-08-12 under
`corrected` and against the shipped grader (TODO P37); the `legacy` pair was 0.8348 → 0.8586, so
the gap the argument rests on is 0.0249 rather than 0.0238.*

### Defect 2 — the grader is the model being graded

The same encoder both retrieves the candidate patients and decides whether the retrieved
diagnosis counts as a match. A model with a more compressed embedding space therefore marks its
own work more leniently. This is why BiomedBERT — the most compressed of the three, mean
pairwise 0.9267 — "wins" at threshold 0.9 with a perfect 1.000 while 99.06% of its patient
pairs clear 0.9 *regardless of whether the diagnosis is right*. (`legacy`: 0.9282 and 99.71%;
re-measured 2026-08-12, P37.)

A fixed absolute threshold cannot be fair across encoders whose similarity distributions differ
this much:

| Model | min pairwise | median | % of pairs >= 0.6 |
|---|---:|---:|---:|
| BiomedBERT | 0.7246 | 0.9341 | 100.00% |
| Bio_ClinicalBERT | 0.6454 | 0.8371 | 100.00% |
| BlueBERT | 0.4810 | 0.7176 | 96.35% |

For two of the three models, *no pair of diagnoses in the dataset can score below 0.6*.

*(Postscript, from [12](../findings/12-drg-grader.md): this defect was real at thresholds
0.6–0.9 and structurally absent at 1.0 — the DRG grader reproduces the threshold-1.0 column
bit-exactly, so the one column the README reports was never self-grading-inflated. The fix
removed the reason to doubt it, which is worth as much as moving it.)*

### Defect 3 — rank is computed and then discarded

The pipeline sorts candidates by similarity and takes the top K. The scorer
(`containGreaterOrEqualsValue`) then returns true if **any** of the K clears the threshold. A
hit at rank 1 and a hit at rank 50 count identically.

Consequence: score(TOP-50) >= score(TOP-10) is guaranteed by construction, because the top-50
set contains the top-10 set. Verified in the committed results: monotonic in 18/18
model x threshold combinations, zero violations. The TOP-K curve in the README is arithmetic,
not a finding.

## Options

### A. MEAN instead of MAX — not implemented

Replaces the MAX aggregator with the mean over the Cartesian product.

- **Fixes:** defect 1. Wrong predictions now drag the score down.
- **Does not fix:** defects 2 and 3.
- **New problem:** dilution. A patient with 3 true diagnoses and 1 correct prediction averages
  the correct match against 2 irrelevant comparisons, so patients with more diagnoses are
  penalised for having more diagnoses.
- **New problem, found later:** a mean over hash-ordered labels reintroduces last-ULP
  nondeterminism — see the warning in [12](../findings/12-drg-grader.md) (sort before reducing).

Worth implementing as a registered aggregator alongside MAX for comparison, but it is not the
end state.

### B. Set-level soft precision / recall / F1 — RETIRED 2026-08-11 as P6

**Not implemented, and deliberately not deferred.** The math below is kept for reference, and
because the retirement is easier to argue with when the proposal is in front of you.

Treat ground truth and prediction as *sets* of diagnoses and match them:

```
soft_precision = mean over predicted p of ( max over gt g of cos(g, p) )
soft_recall    = mean over gt g       of ( max over predicted p of cos(g, p) )
soft_f1        = harmonic mean of the two
```

- **Was claimed to fix:** defects 1 and, critically, the degeneracy documented in
  [../findings/04-metric-degeneracy.md](../findings/04-metric-degeneracy.md) — precision and
  recall become genuinely different quantities that can trade off, instead of collapsing to the
  same number.
- **Does not fix:** defect 2 (still cosine-based, still model-dependent) or defect 3.

> **Why it was retired rather than scheduled, 2026-08-11.** Two independent reasons, either
> sufficient on its own.
>
> 1. **The degeneracy it targets does not exist as described.** This document's own footnote at
>    §"The population question" below says so: `P == R` for the BERT arms is `PR == 1.0` — the
>    answered-only and all-cases populations being *the same set*, because those arms never
>    abstain. The baseline's `P != R` is the identical formula on a genuinely smaller answered
>    population. Finding [13](../findings/13-rank-aware-metrics.md) then reproduced legacy
>    `P`/`R`/`PR` **bit-exactly** from an independent code path, which settles it. The columns
>    were unlabelled, not collapsing. **This section and that footnote had disagreed with each
>    other since 2026-08-06**; the footnote was right.
> 2. **It would undo option C's main gain.** Every term above is `cos(g, p)` over the diagnosis
>    embeddings — the arm that produced the retrieval marking its own retrieval, which is
>    precisely the self-grading that the `drg-exact` grader removed. A compact space would once
>    again mark its own work leniently, and no amount of set-level structure changes that.
>
> **What survives:** the intuition that a prediction *set* should be scored against a truth
> *set*, so that missing a second true diagnosis costs something. Nothing in the project measures
> that today — MRR and Hit@K both stop at the *first* relevant candidate. If it is ever wanted,
> build it on the encoder-independent DRG labels, and expect the 58.9% ceiling to bind it.

### C. Rank-aware retrieval metrics — Recall@K, MRR, nDCG — IMPLEMENTED 2026-08-06

Score the ranked candidate list directly, using an *encoder-independent* relevance label.
**Shipped as `grader="drg-exact"` + `RankMetrics.txt`**
([12](../findings/12-drg-grader.md), [13](../findings/13-rank-aware-metrics.md)); the pure
metric implementations live in `src/aicds/analysis/rank_metrics.py`, differentially validated
against three independent implementations.

- **Fixes:** all three defects at once. Rank-aware metrics penalise burying the right answer,
  cannot be gamed by expanding K, and are scale-free — so they are immune to both the
  saturation and the per-model calibration problem.

  > **"Cannot be gamed by expanding K" is FALSE for nDCG, measured 2026-08-06.** It holds for
  > Precision@K and MRR only. `IDCG@k` sums `min(R, k)` slots, so once `k >= R` the ideal stops
  > growing while `DCG@k` keeps accumulating — therefore `nDCG@k` is **non-decreasing in k** for
  > every case with `R <= k`. Against the measured relevant-count distribution on `folds_grouped`
  > under `drg-exact` (`R>10` in 5 of 129 cases, `R>20` in **zero**), that means
  > `nDCG(TOP-50) >= nDCG(TOP-10)` still holds by construction on **124 of 129** cases at K=10 and
  > on **all 129** at K≥20. Worked example, one relevant candidate at rank 15:
  > `nDCG@10 = 0.0000 → nDCG@20 = 0.2500 → nDCG@50 = 0.2500` — flat once `k >= 15`.
  >
  > nDCG shrinks the *size* of the free gain (a rank-50 hit contributes `1/log2(51) = 0.176`
  > rather than `1.000`) but not its *direction*. Keep nDCG for the discount — it is the only one
  > of the four that rewards ranking the answer higher — but do not claim it removes the K
  > artifact. Only **Precision@K** (8,407 strictly-decreasing steps measured) and **MRR**
  > (invariant to anything appended past the first hit, 30,540 checks, bit-exact) actually break it.
  >
  > Established by three independently written implementations of the four metrics, cross-checked
  > over 771,539 comparisons plus an exhaustive sweep of every binary vector of length ≤ 12.
  > No author had run this particular probe; it was found by asking the question they did not.
- **Cost:** requires deciding what "relevant" means without reference to the embedding. DRG
  equality is the obvious candidate and is already in the data.

**Exact DRG matching has a hard ceiling of 58.1% on the legacy folds, 58.9% (76/129) on the
grouped ones** — the leakage fix moved retrievability by exactly one case. For the other ~41%
of cases the correct DRG description appears nowhere in that fold's training pool, because 105
of the 145 unique diagnosis descriptions occur exactly once in the entire dataset. Those cases
are unwinnable regardless of retrieval quality. Three ways to handle it were listed here, and
**the outcome inverted the original preference order**:

1. ~~**Graded relevance** — partial credit for a same-family DRG. *(Was ranked first.)*~~
   **Tried and REJECTED on measurement** ([12](../findings/12-drg-grader.md)): the ladder
   needed ~156 hand-tuned strings fitted to the evaluation set with no held-out data, its low
   rungs were 87% false credit, and it scored "discharged alive" vs "expired" at 0.900.
   Replacing an arbitrary cosine ruler with an arbitrary lexicon ruler would concede the
   argument P4 exists to win. If ever revisited: an external DRG hierarchy, not a lexicon.
2. **Report against the ceiling** — "X of a possible 58.9%." **Adopted.**
3. **Restrict evaluation to the winnable cases** and say so. **Adopted as well** — this is
   `RankMetrics.txt`'s *winnable* population (n=76), reported alongside all-cases and answered.

### D. Per-model threshold calibration — not implemented, likely moot

If absolute thresholds were kept at all, set each model's threshold at a fixed *percentile* of
its own similarity distribution rather than hardcoding 0.6 for everyone. Roughly
equal-selectivity thresholds from the measured medians: BiomedBERT ~0.93, Bio_ClinicalBERT
~0.84, BlueBERT ~0.72.

This was always a patch, not a fix — and with the `drg` grader collapsing the threshold sweep
entirely, there is no longer a threshold to calibrate on the headline path.

## Recommendation — and what actually happened

The original order was: C first, then B, then A/D as cheap continuity options, keeping every
metric side by side rather than replacing. That order held:

1. **C (rank-aware)** — **done.** MRR and Precision@K against DRG relevance, all three
   populations, additively (new `RankMetrics.txt`, golden untouched). The result: no pair of
   encoders separates, max paired |t| = 1.718 vs the 2.262 needed.
2. **B (set-level soft F1)** — **RETIRED 2026-08-11 as P6**, and the order line above is why the
   retirement is worth stating rather than quietly dropping: B sat here as "next" for five days
   after the measurements that removed its reason to exist. It was going to be the fix that made
   P, R and F1 three genuinely different numbers — but they already are three different numbers
   on a genuinely smaller answered population (the baseline), and identical on arms that never
   abstain because those two populations are the same set. See §B.
3. **A (MEAN)** and **D (calibrated thresholds)** — unimplemented, and **demoted 2026-08-11** to
   optional/off-the-publication-path (P12 and P11). D is likely moot post-`drg` — there is no
   threshold left to calibrate on the headline path. A is worth having as a *registered
   aggregator* beside MAX, with the sort-before-reduce ULP hazard from
   [12](../findings/12-drg-grader.md) handled in the same commit.

The existing MAX-at-0.6 number is still reported alongside, clearly labelled as the legacy
metric, so the new results remain comparable to Comito et al. and to this project's own
history.

## The population question, and why it outranks the choice of metric

*Added 2026-08-06, from measurement. This turned out to be exactly right — it became the
abstention-asymmetry finding, the third knob, and the one that cannot be closed
([13](../findings/13-rank-aware-metrics.md)). Its self-selection half was then measured in
[16](../findings/16-self-selection.md): the answered population is genuinely easier, by nearly the
same margin for every arm, and the matched-case ordering inverts.*

The baseline **abstains** on ~24% of cases; every BERT arm answers all of them. So every metric
has to declare which denominator it uses, and there are three honest choices:

| Population | Question it answers |
|---|---|
| **answered-only** | when it does answer, how good is it? |
| **all-cases** (abstention scores 0) | how much does it deliver over the whole workload? |
| **coverage** | how often does it answer at all? |

**These are not interchangeable, and the choice alone reorders the encoders.** On the committed
corrected numbers — baseline 25 hits / 98 answered / 31 abstained, BiomedBERT 26 / 129 / 0:

- answered-only: baseline **0.2551** > BiomedBERT **0.2016**
- all-cases: BiomedBERT **0.2016** > baseline **0.1938**

Second and third place swap on the convention, with the retrieval identical. This is a *third*
arbitrary knob alongside threshold and K, and it was about to be introduced silently.

**The trap that was nearly built.** The natural design — `precision_at_k` returns `None` on an
empty candidate list, everything else returns `0.0` — puts the P column on the answered-only
population and the Hit / nDCG / MRR columns on all-cases, **in the same row, unlabelled**. A
reader comparing across that row is comparing two different denominators.

**Two further measurements that killed the `min(K, len)` denominator as a *protection*
argument:**

- It buys almost nothing here: on the measured Bio_ClinicalBERT list lengths only 3 of 129
  lists are shorter than 10, so at K=10 `min(k,len)` gives 0.026689 against plain `k`'s
  0.026357 — a **1.26%** difference. It is the `None` channel, not the denominator, that
  handles abstention.
- It is an *unbounded* gaming lever. A retriever that simply truncates its own output —
  identical ranking, fewer candidates reported — delivers strictly less (hit@10
  0.2636 → 0.0930) while its reported P@10 **rises 3.53×**. An oracle-gated abstainer reaches
  **37.9×** inflation at 9.3% coverage while hitting on 12 of 129 cases against the guesser's
  34. Even a *non-oracle* noisy confidence gate buys +134% to +237% reported precision while
  true hit@10 falls up to 29.4%.

**So: report all three populations for every metric, always, in labelled columns. Never one
population per column.** *(This is what `RankMetrics.txt` shipped with, and
[13](../findings/13-rank-aware-metrics.md) validated the shape bit-exactly against legacy.)*

### The legacy pipeline already does this, and nobody noticed

`cython_utils.py` — an empty candidate list increments **neither** TP nor FP:

```python
if containGreaterOrEqualsValue(topk, top_similarities_max, b):
    values[TP] += 1
else:
    if len(top_similarities_max) > 0:      # <-- an abstention lands here and is DROPPED
        values[FP] += 1
```

`bert_models.py` does the same. So `compute_performance_index`'s three columns already *are*
the three populations:

| Legacy column | Formula | What it actually is |
|---|---|---|
| `P` | `tp/(tp+fp)` | **answered-only** hit rate |
| `R` | `tp/nrow` | **all-cases** hit rate |
| `PR` | `(tp+fp)/nrow` | **coverage** |

Verified numerically on a 129/31/25 population: mean Hit@K over all 129 cases
(abstention = 0.0) equals legacy `R` **exactly** (0.193798), and over the 98 answered cases
equals legacy `P` **exactly** (0.255102). *(Finding 13 later reproduced all three columns
bit-exactly from the real run — the prediction written here held to the last digit.)*

Two consequences worth stating plainly:

1. **`RankMetrics.txt` kept this three-column shape** rather than inventing one. It was already
   the right answer, and matching it makes the two artifacts directly comparable.
2. **The "degeneracy" framing got its footnote.** `P == R` for the BERT arms is not the metric
   collapsing — it is `PR == 1.0`, i.e. the answered-only and all-cases populations being *the
   same set* because those arms never abstain. The baseline's `P != R` is the same formula on a
   genuinely smaller answered population. The columns were never wrong; they were unlabelled.

## Prerequisite — satisfied 2026-08-05

None of this was worth measuring until the fold leakage was fixed — 41 of 129 test cases had
another admission from the same patient in their own retrieval pool, worth **+0.11 to +0.26**
at threshold 1.0, against a between-encoder spread of 0.015–0.046 and per-fold standard
deviation of 0.071–0.124: contamination roughly an order of magnitude larger than the effect
being studied.

The fix — regrouping the folds by `SUBJECT_ID` with `GroupKFold` — landed in `c2115ba` before
any rank-aware number was produced, exactly as this prerequisite demanded. The write-up is
[../findings/05-patient-leakage.md](../findings/05-patient-leakage.md), and what removing the
leakage cost each arm — separated from the preprocessing fix it shipped with — is
[../findings/15-leakage-preprocessing-attribution.md](../findings/15-leakage-preprocessing-attribution.md).

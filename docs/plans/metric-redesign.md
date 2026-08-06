# Metric redesign

**Status:** decided in principle, not implemented. This is the substance of the "pluggable
metrics" line in [revival-roadmap.md](revival-roadmap.md) Phase 5.

The current metric has three independent defects. Fixing one does not fix the others, so they
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

Measured: Bio_ClinicalBERT's mean pairwise diagnosis similarity is 0.8348, but the per-patient
MAX mean is 0.8586 — the MAX operator alone moves the distribution up
(`docs/score_distribution_analysis/score_distribution_summary.txt`).

### Defect 2 — the grader is the model being graded

The same encoder both retrieves the candidate patients and decides whether the retrieved
diagnosis counts as a match. A model with a more compressed embedding space therefore marks its
own work more leniently. This is why BiomedBERT — the most compressed of the three, mean
pairwise 0.9282 — "wins" at threshold 0.9 with a perfect 1.000 while 99.71% of its patient pairs
clear 0.9 *regardless of whether the diagnosis is right*.

A fixed absolute threshold cannot be fair across encoders whose similarity distributions differ
this much:

| Model | min pairwise | median | % of pairs >= 0.6 |
|---|---:|---:|---:|
| BiomedBERT | 0.7246 | 0.9341 | 100.00% |
| Bio_ClinicalBERT | 0.6454 | 0.8371 | 100.00% |
| BlueBERT | 0.4810 | 0.7176 | 96.35% |

For two of the three models, *no pair of diagnoses in the dataset can score below 0.6*.

### Defect 3 — rank is computed and then discarded

`bert_models.py:170` sorts candidates by similarity and takes the top K. The scorer
(`containGreaterOrEqualsValue`, `cython_utils.py:366-370`) then returns true if **any** of the K
clears the threshold. A hit at rank 1 and a hit at rank 50 count identically.

Consequence: score(TOP-50) >= score(TOP-10) is guaranteed by construction, because the top-50 set
contains the top-10 set. Verified in the committed results: monotonic in 18/18 model x threshold
combinations, zero violations. The TOP-K curve in the README is arithmetic, not a finding.

## Options

### A. MEAN instead of MAX

Replaces the MAX aggregator with the mean over the Cartesian product.

- **Fixes:** defect 1. Wrong predictions now drag the score down.
- **Does not fix:** defects 2 and 3.
- **New problem:** dilution. A patient with 3 true diagnoses and 1 correct prediction averages
  the correct match against 2 irrelevant comparisons, so patients with more diagnoses are
  penalised for having more diagnoses.

Worth implementing as a registered aggregator alongside MAX for comparison, but it is not the
end state.

### B. Set-level soft precision / recall / F1

Treat ground truth and prediction as *sets* of diagnoses and match them:

```
soft_precision = mean over predicted p of ( max over gt g of cos(g, p) )
soft_recall    = mean over gt g       of ( max over predicted p of cos(g, p) )
soft_f1        = harmonic mean of the two
```

- **Fixes:** defects 1 and, critically, the degeneracy documented in
  [../findings/04-metric-degeneracy.md](../findings/04-metric-degeneracy.md) — precision and
  recall become genuinely different quantities that can trade off, instead of collapsing to the
  same number.
- **Does not fix:** defect 2 (still cosine-based, still model-dependent) or defect 3.

### C. Rank-aware retrieval metrics — Recall@K, MRR, nDCG

Score the ranked candidate list directly, using an *encoder-independent* relevance label
(exact DRG-code match, or DRG family match).

- **Fixes:** all three defects at once. Rank-aware metrics penalise burying the right answer,
  cannot be gamed by expanding K, and are scale-free — so they are immune to both the saturation
  and the per-model calibration problem.

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

**Exact DRG matching has a hard ceiling of 58.1%.** Measured: for only 75 of 129 test cases does
the correct DRG description appear *anywhere* in that fold's training pool. The other 42% are
unwinnable regardless of retrieval quality, because 105 of the 145 unique diagnosis descriptions
occur exactly once in the entire dataset. Three ways to handle this, in order of preference:

1. **Graded relevance** — partial credit for a same-family DRG rather than exact string equality.
   Raises the ceiling and is more clinically sensible than all-or-nothing.
2. **Report against the ceiling** — "X of a possible 58.1%." Honest and standard, but invites the
   question every time.
3. **Restrict evaluation to the 75 winnable cases** and say so. Cleanest signal, smallest n.

### D. Per-model threshold calibration

If absolute thresholds are kept at all, set each model's threshold at a fixed *percentile* of
its own similarity distribution rather than hardcoding 0.6 for everyone. Roughly equal-selectivity
thresholds from the measured medians: BiomedBERT ~0.93, Bio_ClinicalBERT ~0.84, BlueBERT ~0.72.

This is a patch, not a fix — it makes cross-model comparison meaningful without addressing
defects 1 or 3.

## Recommendation

Implement in this order, keeping every metric side by side rather than replacing:

1. **C (rank-aware)** — highest value, and the only option that makes the encoder comparison
   defensible. Report Recall@K and MRR against DRG-code relevance.
2. **B (set-level soft F1)** — makes precision and recall independent, killing the degeneracy.
3. **A (MEAN)** and **D (calibrated thresholds)** — cheap, useful for continuity with the
   published numbers, but neither is sufficient alone.

Keep the existing MAX-at-0.6 number reported alongside, clearly labelled as the legacy metric, so
the new results remain comparable to Comito et al. and to this project's own history.

## The population question, and why it outranks the choice of metric

*Added 2026-08-06, from measurement. This is the part that will decide whether the rank-aware
numbers mean anything, and it is not about which metric you pick.*

The baseline **abstains** on ~24% of cases; every BERT arm answers all of them. So every metric has
to declare which denominator it uses, and there are three honest choices:

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
population and the Hit / nDCG / MRR columns on all-cases, **in the same row, unlabelled**. A reader
comparing across that row is comparing two different denominators.

**Two further measurements that killed the `min(K, len)` denominator as a *protection* argument:**

- It buys almost nothing here: on the measured Bio_ClinicalBERT list lengths only 3 of 129 lists
  are shorter than 10, so at K=10 `min(k,len)` gives 0.026689 against plain `k`'s 0.026357 — a
  **1.26%** difference. It is the `None` channel, not the denominator, that handles abstention.
- It is an *unbounded* gaming lever. A retriever that simply truncates its own output — identical
  ranking, fewer candidates reported — delivers strictly less (hit@10 0.2636 → 0.0930) while its
  reported P@10 **rises 3.53×**. An oracle-gated abstainer reaches **37.9×** inflation at 9.3%
  coverage while hitting on 12 of 129 cases against the guesser's 34. Even a *non-oracle* noisy
  confidence gate buys +134% to +237% reported precision while true hit@10 falls up to 29.4%.

**So: report all three populations for every metric, always, in labelled columns. Never one
population per column.**

### The legacy pipeline already does this, and nobody noticed

`cython_utils.py:299-303` — an empty candidate list increments **neither** TP nor FP:

```python
if containGreaterOrEqualsValue(topk, top_similarities_max, b):
    values[TP] += 1
else:
    if len(top_similarities_max) > 0:      # <-- an abstention lands here and is DROPPED
        values[FP] += 1
```

`bert_models.py` does the same. So `compute_performance_index`'s three columns already *are* the
three populations:

| Legacy column | Formula | What it actually is |
|---|---|---|
| `P` | `tp/(tp+fp)` | **answered-only** hit rate |
| `R` | `tp/nrow` | **all-cases** hit rate |
| `PR` | `(tp+fp)/nrow` | **coverage** |

Verified numerically on a 129/31/25 population: mean Hit@K over all 129 cases (abstention = 0.0)
equals legacy `R` **exactly** (0.193798), and over the 98 answered cases equals legacy `P`
**exactly** (0.255102).

Two consequences worth stating plainly:

1. **`RankMetrics.txt` should keep this three-column shape** rather than inventing one. It is
   already the right answer, and matching it makes the two artifacts directly comparable.
2. **The "degeneracy" framing needs a footnote.** `P == R` for the BERT arms is not the metric
   collapsing — it is `PR == 1.0`, i.e. the answered-only and all-cases populations being *the same
   set* because those arms never abstain. The baseline's `P != R` is the same formula on a
   genuinely smaller answered population. The columns were never wrong; they were unlabelled.

## Prerequisite

None of this is worth measuring until the fold leakage is fixed — 41 of 129 test cases currently
have another admission from the same patient in their own retrieval pool, worth **+0.11 to +0.26**
at threshold 1.0. Compare that to the spread *between* encoders at the same threshold
(0.015-0.046) and the per-fold standard deviation (0.071-0.124): the contamination is roughly an
order of magnitude larger than the effect being studied, and the encoder differences do not clear
the noise floor at all.

The fix is to regroup the folds by `SUBJECT_ID` rather than `HADM_ID` (`GroupKFold`). The folds
are static committed files, so this is a data regeneration, not a code change. *(A dedicated
`findings/05-patient-leakage.md` write-up is not yet written.)*

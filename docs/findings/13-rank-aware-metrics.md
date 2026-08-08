# 13 — Rank-aware metrics: the encoders still do not separate, and abstention is the third knob

**Run 2026-08-06, RunPod Linux, `--pipeline drg`, `PYTHONHASHSEED=0`, HEAD `5393cab`.**
All four arms. Artifacts in `results_p5/<arm>/<timestamp>/RankMetrics.txt` (gitignored).

> **In plain words.** Until this change, a prediction scored the same whether the right answer
> was the system's first guess or its fiftieth. This run switched to a score that cares about
> *position* — MRR, which pays 1.0 for first place, 1/4 for fourth place, 1/50 for fiftieth.
> Two things came out of it. First, the four models still tie: none is measurably better, just
> as under every earlier metric. Second, a problem nobody had been scoring surfaced: the old
> baseline sometimes refuses to answer ("I don't know"), and the three BERT models never do.
> Count a refusal as a failure and the BERT models rank ahead; drop the refused cases from the
> scoring and the baseline ranks first. There is no neutral way to score a refusal, so the
> numbers are reported all three ways — and the tie holds under all three.

## Why P5 existed

A "knob" here means a setting the experimenter picks that changes the answer. Every number this
project had reported was decided by one. P4 removed the first knob: under `drg-exact` the five
threshold rows collapse to a single value, so there is no threshold left to choose.

**K — how many candidate patients the system may offer — was the other knob**, and it was the
one that inverted the answer. BioSentVec ranks 1st under MAX and 4th under TOP-30 *on identical
data*. The cause is the old scoring function, `containGreaterOrEqualsValue`: it asks only
whether *any* of the top K cleared the bar. A hit at rank 1 and a hit at rank 50 count the same,
so widening K can only ever raise the score.

MRR has no K. Under `drg-exact` there is no threshold either. It should therefore have been the
first number this project produced with no knob in it.

## The headline: no pair separates, under any treatment

Three populations — three answers to the question "which test cases count?":

- **winnable** (n=76) — cases whose correct label exists somewhere in their own fold's training
  pool, so a hit is possible at all;
- **all-cases** (n=129) — every test case, winnable or not;
- **answered** — only the cases the arm actually offered a prediction for.

The test is a paired *t*-test on per-fold MRR@50: compare two arms fold by fold, on the same 10
folds, and ask whether the mean difference is larger than its own noise. With 9 degrees of
freedom, a difference is statistically significant (p < 0.05) only if `|t| > 2.262`.

| population | max \|t\| across all six pairs | verdict |
|---|---:|---|
| winnable (n=76) | 1.718 | no pair separates |
| all-cases (n=129) | 1.651 | no pair separates |
| answered | 1.174 | no pair separates |

**This was predicted in writing before the run.** The reasoning: nothing in P5 improves
retrieval — it only stops rewarding guessing. The prediction held.

The result is stronger than the earlier null, because it survives with **the threshold knob
gone, the K knob gone, and across every abstention convention** — not because one favourable
setting was chosen.

## The finding: abstention is a third knob, and it flips the ranking

| population | ranking by MRR@50 |
|---|---|
| **winnable** / all-cases (abstention scores 0) | Bio_ClinicalBERT > BiomedBERT > BlueBERT > **BioSentVec** |
| **answered** (abstentions excluded) | **BioSentVec** > Bio_ClinicalBERT > BiomedBERT > BlueBERT |

| arm | winnable | all-cases | answered |
|---|---:|---:|---:|
| BioSentVec | 0.202856 | 0.110817 | **0.147361** |
| Bio_ClinicalBERT | **0.246218** | **0.132127** | 0.132127 |
| BiomedBERT | 0.243230 | 0.130187 | 0.130187 |
| BlueBERT | 0.231420 | 0.122870 | 0.122870 |

**Every sign flips.** BioSentVec's mean differences against the three BERT arms go from
−0.043/−0.040/−0.029 to +0.015/+0.017/+0.024. Last place to first, on the same data, at the
same threshold, with the same K.

**The mechanism.** MRR is *not* abstention-neutral. An abstained case scores RR = 0 (RR is the
reciprocal rank — the 1/rank credit a single case earns), while an arm that always offers 50
candidates has some chance of a hit on every case. The baseline abstains — coverage 0.848 on
winnable cases, 0.756 overall — and all three BERT arms abstain on nothing (coverage exactly
1.0). So how you treat abstention maps directly onto the comparison you were trying to make.

Note the BERT columns are identical across `all-cases` and `answered`: with coverage 1.0 those
two populations are the *same set*. Only the baseline moves between the columns. That is the
whole effect.

## Neither treatment is neutral, and one confound is NOT testable here

- `winnable` / `all-cases` **penalise** abstention: declining to answer scores the same as
  answering wrongly. This is the same bias that made TOP-K meaningless, in a quieter form.
- `answered` **may flatter** the baseline, because those 98 cases are *self-selected* — the arm
  itself chose which cases to answer, and an arm that only answers when confident is grading
  itself on its easiest material. It is also not case-matched: baseline n=98 against BERT n=129.

**Do not run the obvious shortcut test for that self-selection — it is a trap, and it was fallen
into once already.** Dividing `MRR_all-cases` by `coverage` and comparing the result against
`MRR_answered` returns ≈1.0 and *looks* like it bounds the self-selection effect at 0.5%. It
bounds nothing. The two quantities are **algebraically identical** —

```
MRR_all      = (Σ RR over answered) / 129      # abstentions contribute 0
coverage     = 98 / 129
MRR_all / coverage = [(Σ RR)/129] · [129/98] = (Σ RR)/98 = MRR_answered
```

— so the ratio is 1.0 by construction and carries no information about selectivity at all. The
0.995 actually observed is just mean-of-folds aggregation interacting with per-fold coverage.
It was computed, believed for a minute, and withdrawn.

Testing self-selection properly requires the BERT arms' MRR **restricted to the baseline's 98
answered cases**, compared against their MRR on all 129. That needs per-case relevance vectors.
`RankMetrics.txt` carries only aggregates, and **every per-case output file in this repo is
empty** — a documented defect (`cython_utils.py:65-66` opens the handles, `:170-171` closes
them, nothing is written in between). So it is not testable from current artifacts. Fixing the
per-case output is the prerequisite, and that is the concrete next step this finding generates.

**The honest statement: there is no abstention-neutral comparison available, because abstention
is a property of the arm rather than of the metric.** The ranking depends on how you treat it,
and no treatment is defensibly neutral. What rescues the conclusion is that *no pair separates
under any of them*.

## What Precision@K delivered, as designed

Precision@K asks: of the K candidates the arm offered, what fraction were actually relevant? It
was the other metric expected to break the TOP-K artifact, and it did — measurably, on the
winnable population:

| arm | P@1 | P@10 | P@50 | fall |
|---|---:|---:|---:|---:|
| BioSentVec | 0.1484 | 0.0868 | 0.0857 | 1.7× |
| Bio_ClinicalBERT | 0.1749 | 0.0712 | 0.0290 | **6.0×** |
| BiomedBERT | 0.1693 | 0.0579 | 0.0280 | **6.0×** |
| BlueBERT | 0.1549 | 0.0570 | 0.0316 | **4.9×** |

**Precision@K falls with K for every arm** — the exact inversion of the README's TOP-K curve,
which rises with K for every arm. That inversion was the predicted reportable finding, and it
landed.

**It is also not monotone**, which is the point: BioSentVec goes
0.0868 → 0.0845 → 0.0855 → 0.0858 → 0.0857 across K = 10…50. Nothing forces the direction, so
the number carries information. Hit@K cannot do that — it is non-decreasing by construction.

One warning: this is a *within-arm* diagnostic only. An arm that truncates its own candidate
list inflates its reported P@K 3.53×, so the column above is **not** a cross-arm ranking. See
`rank_metrics.py` N3.

Hit@K behaves exactly as pinned. BioSentVec flattens at **0.358 from K=40** because it has run
out of candidates to offer; the BERT arms climb to 0.666–0.730 because they always have fifty.
Note BlueBERT's 0.730 is the highest of the four — and BlueBERT is 4th on MRR. **Neither figure
breaches the 58.9% exact-match ceiling**, despite exceeding it numerically: that ceiling is an
*all-cases* figure, and within the winnable population the label is reachable by construction,
so the bound there is 1.0. nDCG@K rises monotonically for all four arms, confirming N4 on real
data.

## The validation that makes all of the above trustworthy

**`RankMetrics.txt`'s three populations reproduce legacy's `P`, `R` and `PR` columns exactly**,
on real data, from an independent code path:

| legacy column | value | RankMetrics equivalent | value |
|---|---|---|---|
| `R` | 0.19230769230769232 | all-cases Hit@10 | 0.19230769230769232 |
| `P` | 0.25124098124098121 | answered Hit@10 | 0.25124098124098121 |
| `PR` | 0.75576923076923075 | coverage | 0.75576923076923075 |

Exact to the last bit. This was written down as a prediction in `rank_metrics.py`'s N2 note
before the code existed. Two independent implementations landing on the same 17 digits is what
makes the new numbers trustworthy — and it settles the **degeneracy** framing permanently.
`P == R` for the BERT arms was never the metric collapsing. It is `PR == 1.0` — the answered
and all-cases populations being the *same set*, because those arms never abstain. The columns
were never wrong; they were unlabelled.

**P5 is additive, verified twice.** `pytest -m golden` byte-exact in 43:28 (the BERT path, via
StubEncoder), and all four arms' `PerformanceIndex.txt` identical to the pre-P5 `drg` run once
the wall-clock trailer is stripped (the real-encoder path, both arms).

## Open, generated by this finding

- **Per-case output** (`docs/findings/10`) blocks the self-selection test above. Highest-value
  item this finding produces.
- **P39** — the two arms break score ties differently (`list.sort` is stable, `np.argsort` is
  not). Ties are common (83 pairs, 81 of 129 cases) but only one pair differs in relevance,
  bounding the MRR shift at 0.0022. Filed, not fixed.
- **nDCG@K is reported but is *not* a fix**: `IDCG@k` sums `min(R,k)` slots, so for `k >= R` it
  is non-decreasing in k exactly like Hit@K. `R > 20` in zero of 129 cases. See
  `metric-redesign.md`.

---

*Companion documents:* [11](11-corrected-pipeline-first-results.md) the corrected four-arm results ·
[12](12-drg-grader.md) the encoder-independent grader · [07](07-comparison-validity.md) the synthesis

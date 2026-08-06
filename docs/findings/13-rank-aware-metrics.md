# 13 — Rank-aware metrics: the encoders still do not separate, and abstention is the third knob

**Run 2026-08-06, RunPod Linux, `--pipeline drg`, `PYTHONHASHSEED=0`, HEAD `5393cab`.**
All four arms. Artifacts in `results_p5/<arm>/<timestamp>/RankMetrics.txt` (gitignored).

## Why P5 existed

Every number this project reports had been decided by a knob the experimenter picked. P4 removed
one: under `drg-exact` the five threshold rows collapse to a single value. **K was the other**, and
it was the one that inverted the answer — BioSentVec ranks 1st under MAX and 4th under TOP-30 on
identical data, because `containGreaterOrEqualsValue` asks only whether *any* of the top K cleared
the bar, so a hit at rank 1 and a hit at rank 50 count the same.

MRR has no K. Under `drg-exact` there is no threshold either. It should therefore have been the
first number here with no knob in it.

## The headline: no pair separates, under any treatment

Paired *t*-test on per-fold MRR@50, 9 df, so `|t| > 2.262` is p < 0.05.

| population | max \|t\| across all six pairs | verdict |
|---|---:|---|
| winnable (n=76) | 1.718 | no pair separates |
| all-cases (n=129) | 1.651 | no pair separates |
| answered | 1.174 | no pair separates |

**This was predicted in writing before the run**, on the reasoning that nothing in P5 improves
retrieval — it only stops rewarding guessing. The prediction held.

The result is stronger than the earlier null, because it survives with **the threshold knob gone,
the K knob gone, and across every abstention convention** — not because one favourable setting was
chosen.

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
−0.043/−0.040/−0.029 to +0.015/+0.017/+0.024. Last place to first, same data, same threshold,
same K.

**The mechanism.** MRR is *not* abstention-neutral. An abstained case scores RR = 0, while an arm
that always offers 50 candidates has some chance of a hit. The baseline abstains — coverage 0.848 on
winnable cases, 0.756 overall — and all three BERT arms abstain on nothing (coverage exactly 1.0).
So the treatment of abstention maps directly onto the comparison of interest.

Note the BERT columns are identical across `all-cases` and `answered`: with coverage 1.0 those two
populations are the *same set*. Only the baseline moves. That is the whole effect.

## Neither treatment is neutral, and one confound is NOT testable here

- `winnable` / `all-cases` **penalise** abstention: declining to answer scores the same as answering
  wrongly. This is the same bias that made TOP-K meaningless, in a quieter form.
- `answered` **may flatter** the baseline, because those 98 cases are *self-selected* — the arm chose
  which to answer. It is also not case-matched: baseline n=98 against BERT n=129.

**The obvious quick test of that self-selection is vacuous, and it is worth recording why, because it
looks convincing.** Dividing `MRR_all-cases` by `coverage` and comparing against `MRR_answered`
returns ≈1.0 and appears to bound the effect at 0.5%. It bounds nothing: the two are **algebraically
identical** —

```
MRR_all      = (Σ RR over answered) / 129      # abstentions contribute 0
coverage     = 98 / 129
MRR_all / coverage = [(Σ RR)/129] · [129/98] = (Σ RR)/98 = MRR_answered
```

so the ratio is 1.0 by construction and carries no information about selectivity. The 0.995 observed
is just mean-of-folds aggregation interacting with per-fold coverage.

Testing it properly requires the BERT arms' MRR **restricted to the baseline's 98 answered cases**,
compared against their MRR on all 129. That needs per-case relevance vectors. `RankMetrics.txt`
carries only aggregates, and **every per-case output file in this repo is empty** — a documented
defect (`cython_utils.py:65-66` opens the handles, `:170-171` closes them, nothing is written
between). So it is not testable from current artifacts. Fixing the per-case output is the
prerequisite, and that is the concrete next step this finding generates.

**So the honest statement is: there is no abstention-neutral comparison available, because abstention
is a property of the arm rather than of the metric.** The ranking depends on how you treat it, and no
treatment is defensibly neutral. What rescues the conclusion is that *no pair separates under any of
them*.

## What Precision@K delivered, as designed

Precision@K was the other metric expected to break the artifact, and it did — measurably, on the
winnable population:

| arm | P@1 | P@10 | P@50 | fall |
|---|---:|---:|---:|---:|
| BioSentVec | 0.1484 | 0.0868 | 0.0857 | 1.7× |
| Bio_ClinicalBERT | 0.1749 | 0.0712 | 0.0290 | **6.0×** |
| BiomedBERT | 0.1693 | 0.0579 | 0.0280 | **6.0×** |
| BlueBERT | 0.1549 | 0.0570 | 0.0316 | **4.9×** |

**Precision@K falls with K for every arm** — the exact inversion of the README's TOP-K curve, which
rises with K for every arm. That inversion was the predicted reportable finding and it landed.

**It is also not monotone**, which is the point: BioSentVec goes
0.0868 → 0.0845 → 0.0855 → 0.0858 → 0.0857 across K = 10…50. Nothing forces the direction, so the
number carries information. Hit@K cannot do that — it is non-decreasing by construction.

This is a *within-arm* diagnostic only. An arm that truncates its own candidate list inflates
reported P@K 3.53×, so the column above is **not** a cross-arm ranking. See `rank_metrics.py` N3.

Hit@K behaves exactly as pinned: BioSentVec flattens at **0.358 from K=40** because it has run out of
candidates to offer, while the BERT arms climb to 0.666–0.730 because they always have fifty. Note
BlueBERT's 0.730 is the highest of the four — and it is 4th on MRR. **Neither figure breaches the
58.9% exact-match ceiling**, despite exceeding it numerically: that ceiling is an *all-cases* figure,
and within the winnable population the label is reachable by construction, so the bound there is 1.0.
nDCG@K rises monotonically for all four arms, confirming N4 on real data.

## The validation that makes all of the above trustworthy

**`RankMetrics.txt`'s three populations reproduce legacy's `P`, `R` and `PR` columns exactly**, on
real data, from an independent code path:

| legacy column | value | RankMetrics equivalent | value |
|---|---|---|---|
| `R` | 0.19230769230769232 | all-cases Hit@10 | 0.19230769230769232 |
| `P` | 0.25124098124098121 | answered Hit@10 | 0.25124098124098121 |
| `PR` | 0.75576923076923075 | coverage | 0.75576923076923075 |

Exact to the last bit. This was written down as a prediction in `rank_metrics.py`'s N2 note before
the code existed.

It also settles the **degeneracy** framing permanently: `P == R` for the BERT arms was never the
metric collapsing. It is `PR == 1.0` — the answered and all-cases populations being the *same set*,
because those arms never abstain. The columns were never wrong; they were unlabelled.

**P5 is additive, verified twice.** `pytest -m golden` byte-exact in 43:28 (the BERT path, via
StubEncoder), and all four arms' `PerformanceIndex.txt` identical to the pre-P5 `drg` run once the
wall-clock trailer is stripped (the real-encoder path, both arms).

## Open, generated by this finding

- **Per-case output** (`docs/findings/10`) blocks the self-selection test above. Highest-value item
  this finding produces.
- **P39** — the two arms break score ties differently (`list.sort` stable, `np.argsort` not). Ties
  are common (83 pairs, 81 of 129 cases) but only one pair differs in relevance, bounding the MRR
  shift at 0.0022. Filed, not fixed.
- **nDCG@K** is reported but is **not** a fix: `IDCG@k` sums `min(R,k)` slots, so for `k >= R` it is
  non-decreasing in k exactly like Hit@K. `R > 20` in zero of 129 cases. See `metric-redesign.md`.

---

*Companion documents:* [11](11-corrected-pipeline-first-results.md) the corrected four-arm results ·
[12](12-drg-grader.md) the encoder-independent grader · [07](07-comparison-validity.md) the synthesis

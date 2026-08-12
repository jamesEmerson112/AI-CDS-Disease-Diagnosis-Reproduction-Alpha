# 16 — Self-selection: the baseline's answered cases are easier, and equally so for every arm

**Date:** 2026-08-12 · **Host:** RunPod Linux, 16 vCPU (Python 3.9.23, numpy 2.0.2,
torch 2.8.0+cpu, sklearn 1.6.1; pod torn down after harvest) · **Commits:** `efa3794` (P40 —
runs record their per-case relevance), `da93b96` (the pod tree of record), `c5508ef` (analysis
as written) · **Problem number:** P40, closed by this finding.

Artifacts: `results_p40/<arm>/<timestamp>/RankCases.txt` and `RankMetrics.txt`, plus
`results_p40/model_comparison.pdf`, all **gitignored at the repo root** under the `results*/`
glob. `RankCases.txt` rows are keyed by `HADM_ID` and are DUA-covered — **only aggregates appear
here**, and no row of it may be quoted anywhere.

> **In plain words.** The 2019 baseline sometimes refuses to answer; the three transformer models
> never do. Finding 13 showed that if you simply drop the refused cases before scoring, the
> baseline jumps from last place to first — and warned this might be because the cases it chooses
> to answer are its easy ones. This run tested that directly: score all four models on exactly the
> same 98 cases the baseline chose. Two results. Those 98 cases really are easier — every model
> scores about 0.032–0.038 higher on them, *including the three that had no say in picking them*,
> so the worry was justified. But the boost is nearly the same size for all four, so it is not a
> hidden advantage for the baseline. Once every arm is scored on the same case set, the baseline
> finishes last again, and its earlier first place turns out to have come from comparing it on 98
> easy cases against models scored on all 129. No pair of models separates by a statistical test,
> exactly as before.

## The question, and why it stayed open

Finding 13 removed the last two knobs — threshold (`drg-exact`) and `K` (MRR) — and then exposed a
third that no code change can remove. The baseline abstains; the three BERT arms abstain on
nothing. So *how you score an abstained case* reorders all four arms, and neither convention is
neutral:

- `winnable` / `all-cases` **penalise** abstention: declining to answer scores the same as
  answering wrongly.
- `answered` **excludes** abstentions, and may flatter the abstaining arm — those 98 cases are
  **self-selected**, and the comparison is not case-matched (baseline n=98 against BERT n=129).

That second clause is the open question this finding answers: **are the 98 simply the easy ones?**

### The trap, restated deliberately

This is the repo's canonical example of a convincing non-measurement, and it is repeated here
because it was computed, believed for a minute, and withdrawn once already.

**Do not bound self-selection with `MRR_all-cases / coverage`.** The quotient returns ≈1.0 and
*looks* like it bounds the effect at half a percent. It bounds nothing, because the two quantities
are algebraically identical:

```
MRR_all      = (Σ RR over answered) / 129      # abstentions contribute exactly 0
coverage     = 98 / 129
MRR_all / coverage = [(Σ RR)/129] · [129/98] = (Σ RR)/98 = MRR_answered
```

The ratio is 1.0 by construction and carries no information about selectivity whatsoever. The
0.995 that was actually observed is nothing but mean-of-folds aggregation interacting with
per-fold coverage. `scripts/analyze_rank_metrics.py` prints a standing warning against it at the
foot of the self-selection block, and `_self_selection`'s docstring carries the same note.

The real test is a **matched comparison**, not a correction factor: score every arm on exactly the
cases the baseline answered, and read that column against its all-cases one. That needs per-case
relevance vectors, and every per-case output file in this repo was empty
([finding 10](10-output-path-fragmentation.md)). **P40 wrote a new sibling — `RankCases.txt`,
one row per test case: fold, `HADM_ID`, status, candidate count, first-relevant rank — rather than
repairing the dead handles**, the same additive shape that let P4 and P5 land without re-minting
the golden. This run is the first that could be asked the question at all.

## The answer: the 98 are easier — for everyone, by nearly the same margin

Pooled MRR@50, `--pipeline drg`, canonical grouped folds. Restriction: the 98 cases BioSentVec
answered, of the 129 it was scored on.

| arm | n (all) | MRR (all 129) | MRR (restricted, 98) | Δ |
|---|---:|---:|---:|---:|
| BioSentVec \* | 129 | 0.111321 | 0.146535 | **+0.035214** |
| Bio_ClinicalBERT | 129 | 0.132798 | 0.171072 | **+0.038274** |
| BiomedBERT | 129 | 0.131040 | 0.167828 | **+0.036788** |
| BlueBERT | 129 | 0.123301 | 0.155362 | **+0.032061** |

\* *the baseline's restricted set **is** its own answered set, so its two columns are the
all-cases and answered populations of the blocks above — the anchor the other three rows are read
against, not a fifth comparison.* This is what makes the table interpretable: the anchor is fixed
by construction, so every other row's Δ is a measurement of the case set rather than of the arm's
own selectivity.

**Both halves of that table matter.**

- **Self-selection is real.** Three arms that had no hand in choosing the 98 all score materially
  higher on them. The baseline's answered population is genuinely easier material, and `answered`
  therefore does overstate it — finding 13's warning was correct, not merely cautious.
- **The confound is arm-neutral.** The four gains span **+0.032061 to +0.038274**, a range of
  0.006213, and the baseline's own +0.035214 sits *inside* it, third of four. There is no arm for
  which the restriction is worth appreciably more than for the others. Whatever makes these cases
  easier is a property of the cases, not an advantage the baseline's pruning rule confers on
  itself.

## The inversion: the baseline's answered-only lead was an artifact of mismatched case sets

Read the two comparisons side by side, both in pooled MRR@50 so nothing rides on aggregation:

| comparison | BioSentVec | Bio_ClinicalBERT | BiomedBERT | BlueBERT | 1st |
|---|---:|---:|---:|---:|---|
| `answered` block — **mismatched n** (baseline on 98, BERT on 129) | **0.146535** | 0.132798 | 0.131040 | 0.123301 | BioSentVec |
| restricted — **matched n** (all four on the same 98) | 0.146535 | **0.171072** | 0.167828 | 0.155362 | Bio_ClinicalBERT |

**The baseline's number is byte-identical in both rows.** Nothing about the baseline changed
between them — the only thing that moved is which cases the three BERT arms were scored on. The
ordering on the fixed 98-case set is

**Bio_ClinicalBERT 0.171072 > BiomedBERT 0.167828 > BlueBERT 0.155362 > BioSentVec 0.146535**

— first place to last for the baseline, purely from matching the denominators.

The same inversion holds in the mean-of-folds spelling `RankMetrics.txt` prints as primary
(BioSentVec 0.147361 > Bio_ClinicalBERT 0.132127 > BiomedBERT 0.130187 > BlueBERT 0.122870 in the
mismatched `answered` block), so this is not a pooled-versus-macro effect: the baseline's answered
figure is 0.147361 macro against 0.146535 pooled, a difference of 0.0008, while the BERT arms move
by 0.032–0.038 when the case set is matched.

**The tidy consequence.** Restricted ordering, `all-cases` ordering and `winnable` ordering are
now the *same* ordering — Bio_ClinicalBERT > BiomedBERT > BlueBERT > BioSentVec in all three. The
`answered` population was the only one of finding 13's three that put the baseline first, and it
was also the only one comparing different case sets. That does not make it wrong to report — it
remains a real convention with a real answer — but it is no longer unexplained.

**This is descriptive, and the caveat is load-bearing.** The restricted column is a **pooled**
MRR over a fixed case set; `analyze_rank_metrics.py` prints **no paired t-test on it**, because a
per-fold restriction would score 10 unequal, arm-independent subsets and reintroduce exactly the
fold-composition variance a fixed case set exists to remove. The gap between first and last on the
restricted set is 0.024537, which is the same order as the mean per-fold differences that failed
to separate in the tested blocks. So there is no reason to expect a separation here — but that is
an expectation, not a measurement.

**And no pair separates anywhere that *was* tested.** The paired t-tests on per-fold MRR@50 in
`results_p40` come back with max |t| = **1.718 / 1.651 / 1.174** on winnable / all-cases /
answered, against the 2.262 needed at 9 df. Those are **identical to the committed `results_p5`
figures reported in finding 13** — a fresh tree, five days and a numpy pin later, reproducing them
to the digit. The knobless headline stands: **no encoder ranking is supported by this
experiment**, and it now survives a matched-case-set comparison as well as three abstention
conventions.

## Zero `[WARN]` rows: the join is clean

`_self_selection` emits a loud `[WARN]` for any arm whose record misses even one of the restricted
cases, because a missing case means the arms were not evaluated on the same admissions — a
different fold split, or a run from another pipeline harvested into the tree — and every number in
that row would then be a comparison between two different experiments.

**All four arms produced zero `[WARN]` lines**: every one of the 98 `(fold, HADM_ID)` keys is
present in all four `RankCases.txt` files. The discovery pass also reported no
`no RankCases.txt (pre-P40 run?)` line, so all four runs in this tree are post-P40. The join key
is the `(fold, hadm)` pair rather than the bare ID — redundant today, since an admission appears
as a test case in exactly one fold, and correct if that ever stops being true.

That clean join is what licenses the matched comparison, and it is corroborated independently:
all four runs' `run_metadata.json` carry `fold_dir: folds_grouped` and the **canonical**
`fold_dir_sha256` `b36f7216…a6ec5084f` — the field C8/P14 added, and the one that makes "which
split did this run use" answerable at all (see
[finding 14](14-fold-split-environment-dependence.md)).

## Cross-validation: the instrumented P39 run reproduces this tree exactly

A separate, *instrumented* baseline run — the P39 tie census, carrying an additive dump inside
`predictS2V`, patched and reverted pod-side — was scored independently and returned the baseline's
all-cases MRR as **macro 0.110817, pooled 0.111321**. Those are the same numbers
`analyze_rank_metrics.py` reports for `results_p40`'s baseline, to the digit.

Two things follow, and both are prerequisites for trusting anything above. **The instrumentation
was inert** — it observed the pipeline without perturbing it. And **the pod's baseline run is
repeat-deterministic**: two independent executions of the same configuration produced identical
per-case outcomes, so the 98-case restriction set is a stable property of the configuration rather
than of one execution.

P39's own verdict — **measured-moot**, envelope span 0.000000 on all three populations, for MRR
and Hit@10, macro and pooled — rests on the tie census taken from that same instrumented run, which
is its artifact. Writing that verdict into `docs/plans/TODO.txt`'s P39 block is still pending; the
block there carries the older, pre-measurement impact *bound*, not this measurement, so do not read
it as the record. It is not duplicated here either; what this finding takes from P39 is only the
cross-validation above.

## Limits — read before quoting any number here

1. **The restricted comparison is descriptive.** Pooled MRR over a fixed case set, no per-fold
   paired t-test. Do not report the restricted ordering as a *significant* ordering; nothing on
   this tree separates, and the restricted set was not tested at all.
2. **Two aggregations are in play and they are different numbers.** `RankMetrics.txt`'s primary is
   **mean-of-folds**; the self-selection table is **pooled per case**, deliberately, because a
   fixed case set is the point and per-fold weighting would let fold composition move the quantity
   under test. On `results_p40` the two spellings of the all-cases MRR differ by **0.0004–0.0009**
   per arm (0.00011–0.00244 across all four measured trees) and do not reorder the arms *there* —
   **but the ordering is not aggregation-proof in general.** On `results_preprocess_only` the
   all-cases block puts BioSentVec above BiomedBERT mean-of-folds (BioSentVec 0.191770 > BiomedBERT
   0.191703) and below it pooled (BiomedBERT 0.192521 > BioSentVec 0.192396) — two arms swapping on
   nothing but the aggregation. The gap is also far wider on the abstaining arm's winnable
   block, where the baseline is **0.202856** mean-of-folds against **0.188953** pooled, a
   difference of 0.0139. Never mix a figure from one spelling with a figure from the other, and
   scope any "aggregation does not matter here" claim to the tree it was checked on.
3. **Coverage and abstention also have two spellings.** 98 of 129 answered is 0.7597 pooled and
   **0.7558** as the mean-of-folds `coverage` column; 31 of 129 abstained is 24.0% pooled and
   24.4% mean-of-folds. Finding 13's "24.4%" and this file's "31 of 129" are the same fact.
4. **The gain is not decomposed.** This finding does not split the +0.032…+0.038 into "the
   excluded cases were unwinnable" versus "the excluded cases were winnable but hard". Directional
   evidence only: the baseline's coverage is **0.8477** on the winnable population against
   **0.7558** overall, so it abstains less often where a hit is possible at all — abstention is at
   least partly aligned with winnability. **That is a direction, not a split**, and no percentage
   of the effect should be attributed to either cause from what is measured here.
5. **98 is a property of this tree, not a constant.** It is `results_p40`: `drg-exact` grader,
   canonical grouped folds, corrected preprocessing. Folds and preprocessing both move pruning,
   and pruning decides who abstains, so a different pipeline gives a different restriction set.
   `_self_selection` refuses to hard-code the count for exactly this reason and measures it from
   the file in front of it; read the restriction line, not this paragraph.
6. **This does not make abstention neutral, and finding 13's conclusion is unchanged.** Abstention
   is a property of the arm rather than of the metric, so no scoring convention is defensibly
   neutral and all three populations are still reported. What changed is narrower and worth
   stating precisely: the specific flattery the `answered` convention pays the baseline is now
   **measured and explained**, rather than merely suspected.
7. **Non-canonical fold splits give a different 98.** A local split that does not match digest
   `b36f7216…` is internally sound but not comparable with anything here; `make_folds.py --verify`
   warns, and `tests/test_populations.py` fails with an explanation rather than a mystery number.
8. **DUA.** Every figure above is an aggregate. `RankCases.txt` carries an `HADM_ID` per row and
   is gitignored under `results*/`; it must not be committed, pasted, or quoted row-wise.

## Open, generated by this finding

- **A per-fold restricted t-test.** Restricting fold by fold and running the paired test on the
  matched subsets would convert the restricted ordering from descriptive to tested. It was not
  implemented, and the fold-composition variance it reintroduces is the reason — the trade is real
  and should be decided deliberately, not by default.
- **Decomposing the restriction gain** into winnability and difficulty (limit 4). `RankCases.txt`
  plus `analysis/populations.py` now carry everything needed; nothing else does.

## Reproducing

```bash
python scripts/make_folds.py --verify            # must print digest b36f7216… ; 0 leaked
python scripts/run_baseline.py --pipeline drg --out results_p40          # Linux only
python scripts/run_bert_analysis.py --model all --pipeline drg --out results_p40
python scripts/analyze_rank_metrics.py results_p40
```

The last command is read-only and re-derives every number in this finding from the harvested tree,
including the three population blocks, the six paired t-tests per block, and the self-selection
table. Pod runtimes for the two run steps: baseline 11m30s, `--model all` 37m26s
(`d5-timings-and-env.txt`).

---

*Companion documents:* [13](13-rank-aware-metrics.md) the rank-aware comparison that opened this
question · [10](10-output-path-fragmentation.md) the empty per-case files P40 worked around ·
[14](14-fold-split-environment-dependence.md) the canonical split every number here depends on ·
[07](07-comparison-validity.md) the synthesis

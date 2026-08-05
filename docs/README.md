# Documentation index

Start here. Documents are grouped by *kind*, because they answer different questions and go
stale at different rates.

- **findings/** — the science. Why this project exists and what it actually showed. Written by
  hand, changes rarely, and is the part worth reading carefully.
- **guides/** — how to operate it. Setup, environment, data handling. Changes when tooling changes.
- **reference/** — what the system is. Module structure and data flow.
- **plans/** — where it's going. Sequenced work, with what's done and what's next.
- Generated artifacts and raw run outputs, described at the bottom. Never hand-edit these.

## If you are coming back to this project after a break

Read in this order:

If you want one document, read **[findings/07-comparison-validity.md](findings/07-comparison-validity.md)** —
it is the synthesis, and it says what the comparison would need in order to mean anything. Otherwise:

1. [findings/01-baseline-reproduction.md](findings/01-baseline-reproduction.md) — what the
   original paper (Comito et al., 2022) did, and the status of reproducing it. **Superseded by 09:
   the baseline now runs.**
2. [findings/02-encoder-comparison.md](findings/02-encoder-comparison.md) — the four-encoder
   comparison (BioSentVec plus the three BERT models), which is the original contribution here.
   All four have now been measured, on one machine.
3. [findings/03-metric-saturation.md](findings/03-metric-saturation.md) — why all three encoders
   score a perfect 1.000 at threshold 0.6.
4. [findings/04-metric-degeneracy.md](findings/04-metric-degeneracy.md) — the reported precision,
   recall, and F-score are the same number in all 12,600 committed BERT rows, so every "F1" there
   is accuracy. **Resolved 2026-08-05** — it is the embedding space, not the code; see 09.
5. [findings/05-patient-leakage.md](findings/05-patient-leakage.md) — **read this one.** 41 of 129
   test cases can retrieve the same patient's own other admission. Worth +0.11 to +0.26, roughly
   ten times the difference between the encoders being compared.
6. [findings/06-preprocessing-defects.md](findings/06-preprocessing-defects.md) — `w/o` becomes
   `w`, so negation is destroyed; and 4.4% of symptom tokens are fragments of shredded labels.
7. [findings/07-comparison-validity.md](findings/07-comparison-validity.md) — the synthesis of
   01–06: five separable reasons the comparison is not valid yet, and what would fix each.
8. [findings/08-runtime-and-cost.md](findings/08-runtime-and-cost.md) — the encoder costs almost
   nothing. Embedding is 0.17–0.45% of wall-clock; 93%+ is a single-threaded pure-Python cosine
   loop. **A GPU buys nothing for these arms.** Also: results are platform-independent, verified
   bit-for-bit across macOS/ARM and x86 Linux.
9. [findings/09-baseline-first-run.md](findings/09-baseline-first-run.md) — the baseline finally
   ran, reproducing the published F1 to within 0.007 (0.482 vs 0.489), and in doing so settled the
   open question in 04.
10. [findings/10-output-path-fragmentation.md](findings/10-output-path-fragmentation.md) — nothing
    in the repo can find the baseline's output: it writes `Prediction Output_` with a space while
    all five discovery sites glob `Prediction_Output_*`. Also, every per-case output file is empty.

## If you are setting up on a new machine

[guides/setup.md](guides/setup.md). It covers the one failure that will otherwise cost you an
afternoon: `import torch` dies with `OMP: Error #15` because conda and pip each supply an
OpenMP runtime.

[guides/data-use.md](guides/data-use.md) explains why this repository must not receive new
clinical data, and how the pre-commit hook enforces that.

## The four problems, and how they relate

These are easy to conflate, and conflating them leads to fixing the wrong thing. The first two are
metric design; the last two are plain bugs that nobody would defend.

| Problem | What goes wrong | Fix |
|---|---|---|
| **Saturation** (03) | Nearly every diagnosis pair clears threshold 0.6, because the embedding space is compact and MAX over the Cartesian product amplifies it | A different aggregator, a stricter threshold, or centring the embeddings |
| **Degeneracy** (04) | P, R, and F1 are the same number in all 12,600 committed BERT rows — every case increments exactly one of TP or FP | A genuine set-level P/R/F1, or rank-aware metrics |
| **Leakage** (05) | 41 of 129 test cases can retrieve the same patient's own other admission. Worth +0.11 to +0.26, ~10× the encoder differences | `GroupKFold` on `SUBJECT_ID`; regenerate the folds |
| **Preprocessing** (06) | `w/o` → `w` destroys negation; 4.4% of symptom tokens are fragments of comma-shredded labels | Protect `w/o`; rejoin space-prefixed tokens |

Fixing any one of these does not fix the others — with one exception, now settled.

**Saturation and degeneracy turned out to be the same root cause at two different gates**
(confirmed 2026-08-05, when the baseline finally ran and supplied the missing artifact). A compact
biomedical embedding space means nothing ever falls below `PRUNING_SIMILARITY = 0.5`, so the BERT
arm **never abstains**, so `tp + fp == nrow`, so precision collapses into recall — that is
degeneracy. The same compactness means nearly every pair clears 0.6 — that is saturation. The
baseline's looser 700D space trips neither: it abstains on 30 of 129 cases (23.3%) and scores 0.482
rather than 1.000. The tell is a single column: `TP+FP` sums to exactly 12.9, the mean fold test
size, for all three BERT arms, and to 9.9 for the baseline.

Leakage and preprocessing remain wholly independent of both, and of each other.

**Order of attack** is in [plans/correctness-fixes.md](plans/correctness-fixes.md). Leakage first:
it is the largest correction and the cheapest, since the folds are data rather than code.

## Reference

- [reference/architecture.md](reference/architecture.md) — module layout, data flow, and the
  design constraint that keeps the two arms comparable: both share preprocessing, fold loading,
  and evaluation, so the embedding model is the only intended variable.

## What happens next

[plans/revival-roadmap.md](plans/revival-roadmap.md) — the sequenced plan for reviving and
extending the project. Phases 0 and 4 (environment repair, test suite, documentation) are done.
Phase 1 is next: characterization tests and a deterministic stub encoder, so the pipeline gets a
regression net *before* any code moves. Then the `src/aicds/` package layout and a single
`main.py` CLI replacing the eight ad-hoc scripts.

The roadmap also records the three decisions that shaped it and the reasoning behind the risky
parts — notably why the two evaluation loops must **not** be merged, and why the committed
`Prediction_Output_*` directories are treated as a read-only regression oracle.

## Generated and raw material

Do not hand-edit anything below; regenerate it instead.

| Path | What | Regenerate with |
|---|---|---|
| `readme_plots/*.svg` | Charts embedded in the root README | `python scripts/build_readme_plots.py` (**currently broken** — globs the repo root, but the run directories are under `docs/`) |
| `score_distribution_analysis/` | Saturation study: plots and summary statistics | `python scripts/analyze_score_distributions.py` |
| `Prediction_Output_*/` | Raw output of the three committed BERT runs, 15 Feb 2026 | `python scripts/run_bert_analysis.py --model all` |
| `bert_model_comparison.md` | The original comparison write-up | superseded by [findings/02](findings/02-encoder-comparison.md); kept for provenance |

The three `Prediction_Output_*` directories are the project's **regression oracle** — the only
record of a working pipeline's exact output. Treat them as read-only.

## Archived

Documents describing the pre-reorganization layout (a `CS2V.py` at the repository root, which no
longer exists) were moved to [`../archive/stale-docs/`](../archive/stale-docs/) rather than
deleted. They are historically interesting and factually wrong; do not follow their instructions.

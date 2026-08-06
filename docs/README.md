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

**Read the [root README](../README.md) first.** As of 2026-08-06 it leads with the corrected,
leakage-free four-arm result and states what can and cannot be claimed from it. This index is the
map of how that conclusion was reached; the root README is the conclusion.

Then, if you want one document from here, read
**[findings/07-comparison-validity.md](findings/07-comparison-validity.md)** — it is the synthesis,
and it says what the comparison would need in order to mean anything. Otherwise, in order:

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
   test cases could retrieve the same patient's own other admission. Worth +0.11 to +0.26, roughly
   ten times the difference between the encoders being compared. **FIXED 2026-08-05** (`c2115ba`,
   `GroupKFold` on `SUBJECT_ID`); 11 measures what removing it cost.
6. [findings/06-preprocessing-defects.md](findings/06-preprocessing-defects.md) — `w/o` becomes
   `w`, so negation is destroyed; and 4.4% of symptom tokens are fragments of shredded labels.
   **FIXED 2026-08-05** (`c2115ba`) except nine fragment occurrences; see P27 in the TODO.
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
11. [findings/11-corrected-pipeline-first-results.md](findings/11-corrected-pipeline-first-results.md)
    — **the first uncontaminated four-arm numbers**, with leakage removed and preprocessing unified.
    Roughly a fifth of the published headline was those two defects. Its own original headline
    ("the spread collapses") was an overclaim, walked back in `19e602a` and preserved in place: the
    collapse is MAX-only, and under every TOP-K the spread *widens*. What replaced it is stronger —
    the encoder ordering inverts on the choice of K alone, so no ranking is supported.
12. [findings/12-drg-grader.md](findings/12-drg-grader.md) — every arm was grading its own
    predictions by cosine *in the space that produced the retrieval*, so a compact space marked
    itself leniently. Replaced with an exact DRG-label match: one ruler for all four arms. Records
    the 58.9% ceiling this imposes, the partial-credit ladder that was designed and rejected, and
    two consequences written down *before* the runs so they could be wrong.

## If you are setting up on a new machine

[guides/setup.md](guides/setup.md). It covers the one failure that will otherwise cost you an
afternoon: `import torch` dies with `OMP: Error #15` because conda and pip each supply an
OpenMP runtime.

[guides/data-use.md](guides/data-use.md) explains why this repository must not receive new
clinical data, and how the pre-commit hook enforces that.

## The problems, and how they relate

These are easy to conflate, and conflating them leads to fixing the wrong thing. Saturation and
degeneracy are metric *design*; leakage and preprocessing were plain bugs nobody would defend;
self-grading sits in between — defensible in the original paper, indefensible once the point is a
comparison *between* embedding spaces.

The list was four items long until 2026-08-06. Self-grading is the fifth, found while asking why
every remaining defect happened to bias the same direction.

| Problem | What goes wrong | Fix | Status |
|---|---|---|---|
| **Saturation** (03) | Nearly every diagnosis pair clears threshold 0.6, because the embedding space is compact and MAX over the Cartesian product amplifies it | A different aggregator, a stricter threshold, or centring the embeddings | **Survives** the corrected pipeline — all three BERT arms still sit at 1.000. Addressed by the DRG grader (12), which is scale-free |
| **Degeneracy** (04) | P, R, and F1 are the same number in all 12,600 committed BERT rows — every case increments exactly one of TP or FP | A genuine set-level P/R/F1, or rank-aware metrics | **Survives**, and was *predicted* to: it comes from the retrieval-side pruning gate, upstream of any grader. Open (P5, P6) |
| **Leakage** (05) | 41 of 129 test cases can retrieve the same patient's own other admission. Worth +0.11 to +0.26, ~10× the encoder differences | `GroupKFold` on `SUBJECT_ID`; regenerate the folds | **Fixed** 2026-08-05 (`c2115ba`). 41 → 0, recounted independently |
| **Preprocessing** (06) | `w/o` → `w` destroys negation; 4.4% of symptom tokens are fragments of comma-shredded labels | Protect `w/o`; rejoin space-prefixed tokens | **Fixed** 2026-08-05 (`c2115ba`), 80 of 89 fragments; nine remain (P27) |
| **Self-grading** (12) | Each arm scores its own predictions by cosine in the space that produced them, so a compact space marks itself leniently | Grade on an exact DRG label match instead — one ruler for all arms | **Fixed** 2026-08-06 (`75b6530`), behind `--pipeline drg` |

Fixing any one of these does not fix the others — with one exception, now settled.

**Saturation and degeneracy turned out to be the same root cause at two different gates**
(confirmed 2026-08-05, when the baseline finally ran and supplied the missing artifact). A compact
biomedical embedding space means nothing ever falls below `PRUNING_SIMILARITY = 0.5`, so the BERT
arm **never abstains**, so `tp + fp == nrow`, so precision collapses into recall — that is
degeneracy. The same compactness means nearly every pair clears 0.6 — that is saturation. The
baseline's looser 700D space trips neither: it abstains on 30 of 129 cases (23.3%) and scores 0.482
rather than 1.000. The tell is a single column: `TP+FP` sums to exactly 12.9, the mean fold test
size, for all three BERT arms, and to 9.9 for the baseline.

Leakage and preprocessing remain wholly independent of both, and of each other. Fixing them
confirmed it: the corrected pipeline moved the numbers substantially and left saturation and
degeneracy exactly where they were, with all three BERT arms still pinned at 1.000 and `PR` still
exactly 1.0000.

**Order of attack** is in [plans/correctness-fixes.md](plans/correctness-fixes.md), with the ranked
digest in [plans/TODO.txt](plans/TODO.txt). Leakage went first, as planned — it was the largest
correction and the cheapest, since the folds are data rather than code. Self-grading followed. What
remains is rank-awareness (P5) and set-level metrics (P6): the metric still cannot tell a hit at
rank 1 from a hit at rank 50, which is why TOP-K rises with K no matter what the encoder does.

## Reference

- [reference/architecture.md](reference/architecture.md) — module layout, data flow, and the
  design constraint that keeps the two arms comparable: both share preprocessing, fold loading,
  and evaluation, so the embedding model is the only intended variable.

## What happens next

[plans/revival-roadmap.md](plans/revival-roadmap.md) is the refactor sequence;
[plans/TODO.txt](plans/TODO.txt) is the ranked digest and carries current status.

**Phases 0, 1, 2 and 4 are done** — environment repair, the data-use guard, this documentation, the
byte-exact golden regression net, and the `src/aicds/` package move. **Phase 3 is next and is the
ship point**: a `SentenceEncoder` protocol and encoder registry, a `main.py` CLI (`bert_models.py`
still calls `input()` interactively), one `PerformanceIndex` parser replacing three, one
run-discovery rule replacing five, and the dashboard fix.

Running alongside it, and now ahead of it in priority: P5 (rank-aware metrics) and P38 (a
code-only public repository, which blocks publication because this one carries DUA-covered data).

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

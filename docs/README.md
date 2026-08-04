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

1. [findings/01-baseline-reproduction.md](findings/01-baseline-reproduction.md) — what the
   original paper (Comito et al., 2022) did, and the status of reproducing it.
2. [findings/02-encoder-comparison.md](findings/02-encoder-comparison.md) — the three-encoder
   BERT comparison, which is the original contribution here.
3. [findings/03-metric-saturation.md](findings/03-metric-saturation.md) — why all three encoders
   score a perfect 1.000 at threshold 0.6.
4. [findings/04-metric-degeneracy.md](findings/04-metric-degeneracy.md) — **read this one.** The
   reported precision, recall, and F-score are the same number in all 12,600 result rows. Every
   "F1" in this project is accuracy.

## If you are setting up on a new machine

[guides/setup.md](guides/setup.md). It covers the one failure that will otherwise cost you an
afternoon: `import torch` dies with `OMP: Error #15` because conda and pip each supply an
OpenMP runtime.

[guides/data-use.md](guides/data-use.md) explains why this repository must not receive new
clinical data, and how the pre-commit hook enforces that.

## The two problems, and why they are separate

These are easy to conflate, and conflating them leads to fixing the wrong thing.

| | Saturation (doc 03) | Degeneracy (doc 04) |
|---|---|---|
| **What** | Nearly every diagnosis pair clears threshold 0.6 | P, R, and F1 are always the same number |
| **Cause** | Compact embedding space + MAX over the Cartesian product | `tp + fp == nrow` by construction, so precision reduces to `tp/nrow`, which is recall |
| **Depends on threshold?** | Yes — raising it helps | No — holds at every threshold |
| **Fix** | A different aggregator, or a stricter threshold | A genuine set-level P/R/F1 over the diagnosis sets |

Fixing saturation alone would not fix degeneracy. The scores would come down from 1.000 and
still be one-dimensional.

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

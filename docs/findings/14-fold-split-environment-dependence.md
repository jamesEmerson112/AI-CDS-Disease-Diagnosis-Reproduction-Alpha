# Finding 14 — the grouped fold split depends on the numpy major version

**Date:** 2026-08-11 · **Status:** mitigated same day (canonical digest pinned) · **Problem number:** P42

## What was found

`scripts/make_folds.py` produces a **different `data/folds_grouped/` split on numpy 1.x than on
numpy 2.x**, from identical committed inputs, with the same sklearn (1.6.1) and the same script.
Each environment is internally deterministic — regeneration reproduces its own digest bit-exactly,
every time, hash seed irrelevant — but the two environments disagree:

| environment | numpy | `fold_dir_digest("folds_grouped")` |
|---|---|---|
| RunPod Linux (`dd` env) — **the split of record** | 2.0.2 | `b36f7216…a6ec5084f` |
| Local Windows (`disease-diagnosis` env) | 1.26.4 | `a703c33f…ee062317b` |

Fold **sizes** are identical ({15, 13×6, 12×3}); fold 0's membership is identical; folds 1–9 have
the same sizes but different members. The winnable **total is split-invariant** — 76/129 = 58.9%
on both — so the retrievability ceiling and every headline claim stand. What differs is the
**per-fold composition**, and with it every per-fold quantity: the per-fold winnable range is
**4/13 (30.8%) – 13/15 (86.7%)** on the canonical split, not the 3/12 (25%) – 13/15 (87%) that
finding 12 reported (measured, it turns out, on the Windows split — see "Casualties").

## Mechanism

`GroupKFold` in its unshuffled form assigns groups to folds greedily after sorting groups by
size via `np.argsort`. 85 of the 100 subjects have exactly one admission, so the sort is almost
entirely ties — and `np.argsort`'s default introsort is **unstable**, with tie ordering that
changed between numpy 1.x and 2.x. Same code, same data, different tie resolution, different
(equally valid, still leak-free — `--verify` passes 0 leaked on both) split.

Ruled out empirically before numpy was found: `PYTHONHASHSEED` (local regeneration is stable
with and without it), raw-file line endings (`make_folds` reads with universal newlines and
writes `newline="\n"`; the CRLF working-tree copy parses identically), sklearn version
(1.6.1 both sides).

## How it was caught

The first-ever run of the fast suite on Linux (459 tests, D5 pod session) failed
`test_the_range_is_as_wide_as_finding_12_reports`: min rate 4/13 where the pin said 3/12. The
digest comparison that settled which side had "drifted" — neither had; they never agreed — used
`aicds.runs.fold_dir_digest`, the field **C8/P14 added to `run_metadata.json` five days after
the last pre-C8 run**. This finding is the argument for that field made flesh: without a content
digest, "which split did this run use" is unanswerable for every pre-C8 tree, and the question
turned out to be live, not hypothetical.

## Resolution

**The pod's split is canonical**, because every committed results tree (`results_corrected/`,
`results_drg/`, `results_p5/`) was produced on it, and the pod regenerates it bit-exactly to
this day. Recorded as `CANONICAL_FOLDS_GROUPED_DIGEST` (a hash — the fold files themselves are
DUA-covered and gitignored; their digest is not) in two places kept in sync:

- `scripts/make_folds.py` — `--verify` now prints the digest and WARNS (not fails: the
  environment's own split is internally sound, it just is not comparable with the committed
  trees) when it differs from canonical.
- `tests/test_populations.py` — the per-fold range test asserts the digest first, so a
  non-canonical local split fails with an explanation instead of a mystery number.

A Windows machine gets the canonical split by copying it from the pod or regenerating under
numpy 2.0.x. `config/environment.yml` floats `numpy>=1.19.0`, which is what allowed the two
environments to diverge; pinning it is deliberately deferred until the local env's torch
compatibility with numpy 2.x is tested — the digest guards make the divergence loud in the
meantime.

## Casualties corrected

- Finding 12's per-fold ceiling range "3/12 (25%) to 13/15 (87%)" described the Windows split,
  which no committed result ever used. Canonical: **4/13 (30.8%) to 13/15 (86.7%)**. The
  finding's argument (per-fold ceiling variance is a real variance source) survives unchanged —
  the range is nearly as wide either way.
- `CLAUDE.md`'s copy of the same sentence, corrected in the same commit.
- The test pin, renamed to say what it now checks
  (`test_the_range_is_as_wide_as_finding_14_corrects`).

## What this does NOT affect

The legacy `data/folds/` (committed files, never regenerated — the golden's input). The winnable
total 76/129 and both retrievability ceilings. All four arms' committed numbers — they share the
canonical split. The D5 runs — the pod regenerated the canonical split before any run launched,
and every D5 run's `run_metadata.json` now records the digest it ran on.

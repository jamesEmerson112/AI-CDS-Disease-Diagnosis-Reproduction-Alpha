# Finding 14 — the grouped fold split depends on the numpy major version

**Date:** 2026-08-11 · **Status:** mitigated same day (canonical digest pinned); **CLOSED 2026-08-12**
— `config/environment.yml` pinned to `numpy>=2.0,<3` · **Problem number:** P42

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

**Regenerating under the pinned environment is the normal way to get the canonical split on any
machine, Windows included** — `python scripts/make_folds.py --verify` is the procedure, and copying
the pod's files is a fallback, not the route. `config/environment.yml` used to float
`numpy>=1.19.0`, which is what allowed the two environments to diverge; **it is pinned to
`numpy>=2.0,<3` as of 2026-08-12 (P42 closed).**

The deferral this section used to record — pin "until the local env's torch compatibility with
numpy 2.x is tested" — **was tested, and it passed.** *(Corrected 2026-08-12: this paragraph first
claimed the deferral's premise was "vacuous: the local venv carries no torch at all". That was
false and checkable — `venv/Lib/site-packages/torch-2.8.0.dist-info` is dated 2025-12-31, months
before the numpy upgrade, so torch was never absent. The conclusion is unaffected and strictly
stronger: the gate was passed, not bypassed.)* What was measured, 2026-08-12:

- **The headline.** `scripts/make_folds.py`, run under a numpy-2.0.2 venv **on Windows**,
  reproduced the canonical digest `b36f7216…a6ec5084f` — `VERIFY: PASS`, 0 leaked. So the
  divergence is explained by the **numpy major version alone**, with no platform, BLAS or
  hash-seed residue left over. The 1.26.4 → 2.0.2 upgrade was done in place on the local venv.
- Fast suite under numpy 2.0.2: **485 passed, 3 deselected** — both on a throwaway testbed and
  again after the real venv was upgraded.
- `pytest -m golden` under numpy 2.0.2: **byte-exact, 1 passed in 2052.84s (34:12)**. The
  committed golden is unchanged under numpy 2. This is also the **eighth** golden runtime
  measurement and the fastest of the eight; the documented range moves from 42–53 to **34–53 min**.
- **The torch gate the deferral named was run locally and passed.** In the repo venv
  (`venv/Scripts/python.exe` — the only working local env), **torch 2.8.0+cpu imports green under
  numpy 2.0.2**, `torch.from_numpy` round-trips an array unchanged, and both hold in the same
  interpreter that ran the 485-passed suite above. 2.8.0 is well past **torch >= 2.3**, the
  numpy-2-compatible line, which stays the requirement for any environment that adds torch. The
  RunPod `dd` env corroborates from production: it ran **every D5 BERT run on torch + numpy 2.0.2**.

The digest guards stay exactly as they are, pin or no pin: `--verify` still **WARNS** on a digest
mismatch and `tests/test_populations.py` still fails with an explanation. A pin tells you what to
install; the digest is what catches an environment that ignored it.

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

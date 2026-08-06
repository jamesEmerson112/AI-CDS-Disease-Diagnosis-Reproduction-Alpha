# Making this repository publishable: a code-only public release

**Status: investigation complete, NOTHING EXECUTED.** This document exists so the decision can be
made with the facts in hand. Every destructive step below needs explicit sign-off, and one of them
needs GitHub Support.

## Why

The repository is **public** and contains committed MIMIC-III records under a PhysioNet DUA that
prohibits redistribution. That conflict blocks publication: a paper citing this repo points
reviewers — and PhysioNet — straight at it. It is also simply the wrong thing to leave in place.

## What is actually exposed

29 tracked files carry clinical identifiers. 25 are substantive; 4 are incidental prose mentions.

| Path | Files | Identifiers each | Content |
|---|---:|---:|---|
| `data/raw/Symptoms-Diagnosis.txt` | 1 | 129 | **The source extract.** HADM_ID, SUBJECT_ID, admit/discharge timestamps, ICD-9 short titles, DRG labels |
| `data/folds/Fold*/TrainingSet.txt` | 10 | 116–117 | HADM_ID + symptom text |
| `data/folds/Fold*/TestSet.txt` | 10 | 12–13 | HADM_ID + symptom text |
| `docs/Prediction_Output_*/PerformanceIndex.txt` | 3 | 129 | Per-case rows keyed by HADM_ID |
| `tests/golden/stub768/PerformanceIndex.txt` | 1 | 129 | Same shape — **this is the regression net** |
| `archive/conversation_with_LLM*.txt`, `.system-design-visualization.md`, `docs/findings/04` | 4 | 1–4 | Incidental mentions in prose |

`docs/score_distribution_analysis/` came back clean of identifiers, but its PNGs should be eyeballed
before release — plots derived from per-patient data can leak structure even without labels.

## History, and the fact that changes the recommendation

Good news first: **the data entered at commit 2 of 53 (`b23657b`, 2026-01-01) and was never
modified.** Each clinical path was touched by exactly one commit. A `git filter-repo` path removal
would be mechanically clean.

**But a history rewrite does not remove the data from GitHub.** Once pushed, objects stay reachable
by direct SHA through the web UI and API until GitHub Support garbage-collects them on request; any
fork holds an independent full copy; and every existing clone keeps it. Rewriting in place produces
the *appearance* of remediation without the substance.

**Therefore the recommendation is not "rewrite history." It is "publish a new clean repository and
retire this one."**

## The recommended shape

### 1. A new public repo, clean from its first commit

Code, tests, docs, dashboard. No `data/`, no committed run outputs, no unmasked golden. Fresh
history, so there is nothing to purge and nothing to explain.

### 2. Retire this repository

Make it **private** (preserves history and the 53-commit record for your own reference), or delete
it. If neither, the data stays public and the exercise is pointless. Even after going private,
consider asking GitHub Support to purge cached views, since public forks and cached blobs can
outlive the visibility change.

### 3. The golden can be kept, and that is the important discovery

`tests/golden/stub768/PerformanceIndex.txt` is the project's only regression net, and it carries all
129 HADM_IDs. It does **not** have to be sacrificed.

`tests/test_golden.py:155` already has a `canonicalise()` that strips the timing trailer and
normalises run-specific paths before comparing. **Extend it to mask the HADM_ID token on both sides,
and commit a masked golden.** The identifiers are labels, not measurements: every `TP FP P R FS PR`
value, every formatting quirk the golden exists to catch — threshold `1` versus `1.0`, the bare `0`
in 1597 rows — is preserved untouched. The regression net keeps its full numerical power and stops
being clinical data.

This requires no change to pipeline output, so it is a test-only change and the existing golden
proves it did not move anything.

### 4. Fold definitions can ship without identifiers

**Measured and confirmed: all 1290 fold rows are exactly `raw_symptoms + "."`** — a uniform rule with
zero exceptions across both fold sets. The fold files therefore add no information beyond *membership*
plus that trailing period.

So ship **fold membership as index lists** into a canonical ordering of `data/raw`, plus the
documented `+ "."` rule. A credentialed user who rebuilds `data/raw` reconstructs the exact legacy
splits byte-for-byte. `load_dataset` needs a small change to resolve symptoms from the raw file, and
the golden gates it.

### 5. `data/raw` cannot ship, and there is an honest route

The extract maps onto MIMIC-III as: `ADMISSIONS` (HADM_ID, SUBJECT_ID, ADMITTIME, DISCHTIME) ·
`DIAGNOSES_ICD` → `D_ICD_DIAGNOSES.SHORT_TITLE` for the symptom list · `DRGCODES`
(DRG_TYPE, DESCRIPTION) for the target. **Verify this against the schema before publishing it as a
recipe** — it is inferred from the data's shape, not from the original authors' code.

**The unavoidable gap: reconstruction needs to know *which* 129 admissions, and that list is itself
identifying.** The selection was made upstream and no criterion is recorded anywhere in the repo, so
it cannot be derived. Two honest options:

- **Preferred — publish the derived subset on PhysioNet** as a credentialed-access supplementary
  project, and keep GitHub code-only. This is the sanctioned route and reviewers recognise it.
- **Fallback** — state plainly that the public repo reproduces the *method* but not the exact
  numbers, and share the ID list with credentialed researchers on request.

### 6. The committed run outputs

`docs/Prediction_Output_*/` are described in `CLAUDE.md` as the project's regression oracle. Their
`10-FOLD` aggregate blocks carry **no** identifiers; only the per-case rows do. Ship them ID-masked
under the same rule as the golden, and they keep their oracle role.

## Order of work

1. Extend `canonicalise()` to mask identifiers; re-mint the masked golden. Golden must pass. *This
   is the only step that is useful on its own and safe to do today.*
2. Convert `data/folds` to index lists; adjust `load_dataset`; golden must pass byte-exact.
3. Mask the three committed `Prediction_Output_*` files.
4. Write the MIMIC-III reconstruction recipe and verify it against the schema.
5. Scrub the four incidental mentions.
6. Build the new repo from the resulting tree with a fresh initial commit. Confirm zero identifiers
   with the same scanner used for the table above, and confirm `.githooks/pre-commit` passes with no
   `--no-verify` anywhere.
7. Only then: retire this repository.

Steps 1–5 are ordinary work in this repo and reversible. Step 6 is new-repo creation. **Step 7 is the
irreversible one and needs explicit instruction.**

## What I am not doing without being told

- Rewriting this repository's history.
- Changing its visibility, or deleting it.
- Creating the new repository or pushing anything to GitHub.
- Contacting GitHub Support.

## A prerequisite that sits outside the code

Publishing MIMIC-derived work requires being a credentialed PhysioNet user with the DUA signed and
the CITI training completed. If that is not already in place it is step zero, ahead of everything
above.

# Data use and redistribution

> **In plain words.** The patient data in this repo comes from MIMIC-III, a hospital database
> you may only access after training and signing an agreement — and that agreement forbids
> passing the data on to anyone else. A public repo *is* passing it on. The data is anonymised,
> so this is not a patient-privacy breach; it is a licence violation, and the distinction
> matters legally. The rules: never add new clinical data here (a pre-commit hook enforces it),
> and the long-term exit is a fresh code-only repository
> ([../plans/public-release.md](../plans/public-release.md), P38).

## The constraint

This project's dataset is derived from **MIMIC-III v1.4**, distributed by PhysioNet under a
credentialed **Data Use Agreement (DUA)**. Access requires completing CITI human-subjects
research training and signing the DUA. The agreement prohibits redistributing the data —
including sharing it with people who have not themselves been credentialed.

**A public Git repository is redistribution.**

## Current status of this repository

The repository is **public**, and the following MIMIC-III derived files are **committed to
version control**:

| Path | Contents |
|---|---|
| `data/raw/Symptoms-Diagnosis.txt` | 129 admissions: `HADM_ID`, `SUBJECT_ID`, admit/discharge timestamps, symptom list, diagnosis list |
| `data/folds/Fold0..9/{TrainingSet,TestSet}.txt` | The same 129 records, partitioned 10 ways |

To be precise about the risk, because the two are often conflated:

- **This is not a HIPAA breach.** MIMIC-III is de-identified under HIPAA Safe Harbor. Dates are
  shifted into the future (hence admit dates in the 2100s–2200s), and direct identifiers are
  removed. The `SUBJECT_ID` and `HADM_ID` values are surrogate keys, not medical record numbers.
- **It does appear to conflict with the PhysioNet DUA**, which restricts redistribution
  regardless of de-identification status.

Resolving the existing committed data requires rewriting Git history. That is destructive — it
breaks every existing clone and fork — and is a decision for the repository owner. It has
deliberately not been done automatically. The investigated exit is a **new, code-only public
repository** rather than a rewrite (a rewrite does not actually purge data from GitHub);
see [../plans/public-release.md](../plans/public-release.md), tracked as P38.

## The rule going forward

**Do not commit new clinical data to this repository.**

A pre-commit hook (`.githooks/pre-commit`) enforces this with **two rules**: it rejects any
*newly added* file under `data/raw/` or any `data/folds*/` directory, **and** it rejects any
new file — wherever it lives — carrying 20 or more distinct `HADM_ID`s. The second rule exists
because the first proved insufficient: a generated test file under `tests/` once carried all
129 IDs and sailed past the path-only check. Install the hook once per clone:

```bash
git config core.hooksPath .githooks
```

The hook blocks additions only. It does not touch the already-committed files, and it can be
bypassed with `git commit --no-verify` when you genuinely intend to — any such bypass deserves
written reasoning in the commit, as the one existing precedent has (the golden reference file,
committed on the reasoning that its 129 IDs were already public in `docs/`). It is a tripwire,
not a security control.

## If you are adding more clinical data

This matters especially for planned work that expands the dataset. Before adding any new source:

1. **Confirm its de-identification status.** If the data is not de-identified under Safe Harbor
   or an expert determination, it is PHI, and committing it to *any* repository — public or
   private — is the wrong move. Private repositories are not an approved control for PHI.
2. **Read that source's own DUA.** Terms differ between MIMIC-III, MIMIC-IV, eICU, and
   institutional datasets.
3. **Keep data out of Git.** Reference it by path and checksum. Store the data itself outside
   the repository — a local directory, an access-controlled share, or a credentialed download
   performed at setup time.
4. **Commit the recipe, not the records:** the loader, the checksum manifest, and the download
   instructions.

## Citing MIMIC-III

Johnson, A., Pollard, T., Shen, L. et al. *MIMIC-III, a freely accessible critical care
database.* Sci Data 3, 160035 (2016). https://doi.org/10.1038/sdata.2016.35

"""Which test cases a rank-aware number should be averaged over.

This is the least obvious part of P5 and the part most likely to be "simplified"
into a wrong answer later, so the reasoning lives here rather than in a commit
message.

Three candidate populations exist, and **they put the encoders in different
orders**, so the choice is a third arbitrary knob alongside threshold and K:

* **all cases** (129) -- abstention scores 0. Reproduces the legacy ``R`` column.
* **answered-only** -- abstention excluded. Reproduces the legacy ``P`` column.
  Arm-dependent: 98 cases for the baseline, 129 for every BERT arm.
* **winnable** (76) -- cases whose own DRG label appears anywhere in their fold's
  training pool. **Arm-invariant**, because it depends only on the folds and the
  labels, never on the encoder.

Measured on the committed corrected run: the baseline scores 0.2551 answered-only
against 0.1938 all-cases, while BiomedBERT scores 0.2016 in both -- so the
baseline is ahead under one convention and behind under the other, on identical
retrieval.

``winnable`` is the headline population, for two measured reasons rather than a
preference:

1. Under ``drg-exact`` **every arm scores exactly 0 on an unwinnable case** -- no
   candidate can exact-match a label that is absent from the pool. Including the
   other 53 cases therefore multiplies every arm by the same 76/129 factor, which
   is pure dilution: it compresses precisely the differences under test.
2. It is **not** symmetric for precision. On an unwinnable case an arm that
   answers scores ``0.0`` while an arm that abstains returns ``None`` and is
   dropped from the average. So the all-129 population differentially penalises
   the arm that answers -- reintroducing the exact bias P5 exists to remove.

Reporting all three anyway is deliberate: the first two are what a reader
expects and what ties this artifact to ``PerformanceIndex.txt``, and disagreement
between them is informative rather than embarrassing.
"""

from __future__ import annotations

import os

from aicds.config import LEGACY
from aicds.utils.Constants import CH_DIR
from aicds.utils.cython_utils import drg_descriptions

__all__ = ["admission_labels", "winnable_test_cases", "winnable_by_fold"]


def admission_labels():
    """``{HADM_ID: {drg description, ...}}`` for all 129 admissions.

    Reads the raw extract directly rather than going through ``load_dataset``,
    because the fold files carry symptoms only -- the DRG labels live in the
    last ``;``-delimited field of the raw file. ``drg_descriptions`` strips the
    ``apr:`` / ``hcfa:`` / ``ms:`` prefixes and lowercases, so the keys here are
    exactly what the ``drg-exact`` grader compares.
    """
    path = os.path.join(CH_DIR, "data", "raw", "Symptoms-Diagnosis.txt")
    labels = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split(";")
            labels[fields[0]] = drg_descriptions(fields[-1].split("--"))
    return labels


def _fold_ids(fold_dir, fold, name):
    """HADM_IDs listed in one fold file.

    The rows are ``<HADM_ID>_<symptoms>``, so the ID is everything before the
    first underscore. A row without one is skipped rather than guessed at.
    """
    path = os.path.join(CH_DIR, "data", fold_dir, "Fold%d" % fold, name)
    with open(path, encoding="utf-8") as handle:
        return [line[: line.index("_")] for line in handle if "_" in line]


def winnable_by_fold(config=LEGACY):
    """``{fold: (winnable HADM_IDs, all test HADM_IDs)}``.

    Per-fold rather than pooled because the ceiling varies enormously across
    folds -- 3/12 (25%) up to 13/15 (87%) on ``folds_grouped`` -- which makes it
    a source of per-fold variance in its own right, and the per-fold values are
    what the paired *t*-test consumes.
    """
    labels = admission_labels()
    out = {}
    for fold in range(10):
        pool = set()
        for hadm in _fold_ids(config.fold_dir, fold, "TrainingSet.txt"):
            pool |= labels.get(hadm, set())

        winnable, tested = set(), set()
        for hadm in _fold_ids(config.fold_dir, fold, "TestSet.txt"):
            mine = labels.get(hadm)
            if not mine:
                continue
            tested.add(hadm)
            if mine & pool:
                winnable.add(hadm)
        out[fold] = (winnable, tested)
    return out


def winnable_test_cases(config=LEGACY):
    """``(winnable HADM_IDs, all test HADM_IDs)`` pooled over the ten folds.

    Pinned by ``tests/test_drg_grader.py``: 75/129 on ``folds`` and 76/129 on
    ``folds_grouped``. **Fixing the leakage moved retrievability by exactly one
    case** -- it is tempting to assume regrouping the folds also improved what is
    findable, and it did not.

    A HADM_ID can appear as a test case in only one fold, so the pooled union is
    a plain count with no double-counting.
    """
    winnable, tested = set(), set()
    for fold_winnable, fold_tested in winnable_by_fold(config).values():
        winnable |= fold_winnable
        tested |= fold_tested
    return winnable, tested

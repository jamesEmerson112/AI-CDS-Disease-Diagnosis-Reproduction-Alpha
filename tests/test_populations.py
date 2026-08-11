"""Pin the winnable-case population that every drg-exact number is read against.

The logic here was promoted out of ``tests/test_drg_grader.py`` (where it existed
only as a ceiling assertion) into ``aicds.analysis.populations``, because P5 needs
it at *run* time to pick a denominator. These tests keep the promoted version
honest against the values the original pinned.

Why the population matters at all: the choice between all-cases, answered-only
and winnable **reorders the encoders**. Measured on the committed corrected run,
the baseline scores 0.2551 answered-only against 0.1938 all-cases while
BiomedBERT scores 0.2016 in both -- ahead under one convention, behind under the
other, on identical retrieval.
"""

import pytest

from aicds.analysis.populations import (
    admission_labels,
    winnable_by_fold,
    winnable_test_cases,
)
from aicds.config import CORRECTED, LEGACY

# The grouped split every committed results tree was produced on (finding 14).
# Recorded as a hash, not data: the fold files themselves are DUA-covered and
# gitignored, but their digest is not, and it is what makes "which split did
# this run use" answerable at all. scripts/make_folds.py --verify checks the
# same constant.
CANONICAL_FOLDS_GROUPED_DIGEST = (
    "b36f721638987fe0280393a28983a90c65127a04ffe21bcd2341024a6ec5084f"
)


class TestCeilingMatchesTheOriginalPin:
    """The promoted function must agree with ``test_drg_grader``'s numbers.

    If these drift, one of the two is wrong and the drg-exact denominator is not
    what any document claims it is.
    """

    def test_legacy_folds(self):
        winnable, tested = winnable_test_cases(LEGACY)
        assert (len(winnable), len(tested)) == (75, 129)

    def test_grouped_folds(self):
        winnable, tested = winnable_test_cases(CORRECTED)
        assert (len(winnable), len(tested)) == (76, 129)

    def test_fixing_the_leakage_barely_moved_the_ceiling(self):
        """One case. It is tempting to assume otherwise, so assert it.

        Regrouping the folds by SUBJECT_ID removed all 41 leaked cases but moved
        retrievability from 75/129 to 76/129. The two defects are independent.
        """
        legacy, _ = winnable_test_cases(LEGACY)
        grouped, _ = winnable_test_cases(CORRECTED)
        assert len(grouped) - len(legacy) == 1


class TestPopulationIsArmInvariant:
    """The property that makes ``winnable`` the right headline denominator."""

    def test_it_depends_only_on_folds_and_labels(self):
        """No encoder, no embedding, no threshold reaches this computation.

        Called twice with the same config it must give an identical *set*, not
        merely an identical count -- a count could match while membership drifted
        with dict ordering, and membership is what selects the cases to average.
        """
        first, _ = winnable_test_cases(CORRECTED)
        second, _ = winnable_test_cases(CORRECTED)
        assert first == second
        assert isinstance(first, set)

    def test_winnable_is_a_subset_of_tested(self):
        winnable, tested = winnable_test_cases(CORRECTED)
        assert winnable <= tested


class TestPerFoldCeiling:
    """Per-fold, because it varies enough to be its own variance source."""

    def test_every_fold_is_covered_and_folds_partition_the_test_cases(self):
        by_fold = winnable_by_fold(CORRECTED)
        assert sorted(by_fold) == list(range(10))

        # A HADM_ID is a test case in exactly one fold, so the per-fold test sets
        # must be disjoint. If they were not, the pooled count would double-count
        # and every average would be silently wrong.
        seen = set()
        for _, tested in by_fold.values():
            assert not (seen & tested), "a HADM_ID is a test case in two folds"
            seen |= tested
        assert len(seen) == 129

    def test_the_range_is_as_wide_as_finding_14_corrects(self):
        """4/13 (30.8%) to 13/15 (86.7%) on the CANONICAL split.

        This pin said 3/12 (25%) until 2026-08-11, quoting finding 12 -- and
        both described a split no committed result ever used. GroupKFold's
        tie-break among the ~85 single-admission subjects goes through
        np.argsort, whose unstable-sort behaviour changed between numpy 1.x
        and 2.x, so Windows/numpy-1.26 and the pod/numpy-2.0 deterministically
        produce DIFFERENT grouped splits from identical inputs (finding 14,
        P42). Every committed results tree was produced on the pod's split,
        so that split is canonical; the digest guard below is what turns a
        non-canonical local regeneration into an explanation instead of a
        mystery failure. The winnable TOTAL (76/129) is split-invariant.
        """
        from aicds.runs import fold_dir_digest

        digest = fold_dir_digest(CORRECTED.fold_dir)
        assert digest == CANONICAL_FOLDS_GROUPED_DIGEST, (
            "data/folds_grouped is not the canonical split every committed "
            "result used (finding 14): regenerating with numpy 1.x produces a "
            "different-but-equally-valid GroupKFold assignment. Copy the "
            "canonical folds from the pod or regenerate under numpy 2.0.x. "
            "Got digest %s" % digest
        )
        rates = [
            len(w) / len(t) for w, t in winnable_by_fold(CORRECTED).values()
        ]
        assert min(rates) == pytest.approx(4 / 13)
        assert max(rates) == pytest.approx(13 / 15)

    def test_grouped_folds_are_uneven_by_design(self):
        """GroupKFold cannot split subject 41976's 15 admissions, so sizes vary.

        Pinned because an even 13/13 split would mean the grouped folds had been
        regenerated without the grouping -- i.e. the leakage fix silently undone.
        """
        sizes = sorted(len(t) for _, t in winnable_by_fold(CORRECTED).values())
        assert sizes == [12, 12, 12, 13, 13, 13, 13, 13, 13, 15]


class TestAdmissionLabels:
    def test_every_admission_has_at_least_one_drg_description(self):
        labels = admission_labels()
        assert len(labels) == 129
        assert all(labels.values()), "an admission with no DRG label cannot be graded"

    def test_prefixes_are_stripped_and_text_lowercased(self):
        """The keys must be exactly what the drg-exact grader compares."""
        for descriptions in admission_labels().values():
            for description in descriptions:
                assert description == description.lower()
                assert not description.startswith(("apr:", "hcfa:", "ms:"))
                assert description == description.strip()

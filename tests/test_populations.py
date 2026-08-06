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

    def test_the_range_is_as_wide_as_finding_12_reports(self):
        """3/12 (25%) to 13/15 (87%). A per-fold sigma has to absorb that."""
        rates = [
            len(w) / len(t) for w, t in winnable_by_fold(CORRECTED).values()
        ]
        assert min(rates) == pytest.approx(3 / 12)
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

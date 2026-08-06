"""Pin the rank-aware metrics, including the things that must NOT be "fixed".

``src/aicds/analysis/rank_metrics.py`` was not hand-written -- it is the merge of
three independently authored implementations, cross-checked over 771,539
comparisons plus an exhaustive sweep of every binary relevance vector of length
<= 12. These tests are the part of that work that ships: the hand-computed
anchors, the properties that are the whole point of the change, and regression
tests for the two mistakes the differential test actually caught.

Two classes here assert **current, intentional, undesirable** behaviour --
``TestArtifactsPreservedDeliberately``. Hit@K rising with K and nDCG@K rising
with K are the artifacts P5 exists to expose, not to silently remove. Pinning
them means a future change shows up as a deliberate edit to this file rather
than as a number that moved for reasons nobody can reconstruct.
"""

import math

import pytest

from aicds.analysis.rank_metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)

# Every K the pipeline reports, plus 1 and 5 which the rank metrics add.
PRODUCTION_KS = [1, 5, 10, 20, 30, 40, 50]


# ---------------------------------------------------------------------------
# Hand-computed anchors -- worked out on paper, not read back off the code
# ---------------------------------------------------------------------------

class TestHandComputedAnchors:
    """Exact expected values, derived independently of the implementation.

    Constants used: 1/log2(2)=1, 1/log2(3)=0.63092975357145753,
    1/log2(4)=0.5, 1/log2(5)=0.43067655807339306, 1/log2(6)=0.38685280723454163.
    """

    VECTOR = [0, 0, 1, 0, 1]

    def test_mrr_reads_the_first_hit_at_rank_3(self):
        # first relevant entry is at 0-indexed 2 -> 1/(1+2)
        assert reciprocal_rank(self.VECTOR) == 1.0 / 3.0

    def test_precision_at_5(self):
        # 2 relevant in the first 5, denominator min(5, 5) = 5
        assert precision_at_k(self.VECTOR, 5) == 0.4

    def test_hit_at_5(self):
        assert hit_at_k(self.VECTOR, 5) == 1.0

    def test_ndcg_at_5_to_the_last_bit(self):
        """DCG/IDCG worked by hand, then asserted bit-exactly.

        DCG@5  = 0 + 0 + 1/log2(4) + 0 + 1/log2(6)
               = 0.5 + 0.38685280723454163 = 0.88685280723454163
        IDCG@5: n_relevant over the WHOLE list is 2, so ideal_len = min(2,5) = 2
               = 1/log2(2) + 1/log2(3) = 1 + 0.63092975357145753
               = 1.6309297535714575
        nDCG@5 = 0.88685280723454163 / 1.6309297535714575
        """
        dcg = 1.0 / math.log2(4) + 1.0 / math.log2(6)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert ndcg_at_k(self.VECTOR, 5) == dcg / idcg
        # And the literal, so a change to either side of the division is caught.
        assert ndcg_at_k(self.VECTOR, 5) == 0.54377130915202543

    def test_single_hit_at_rank_1_is_a_perfect_ranking(self):
        vector = [1, 0, 0, 0, 0]
        assert reciprocal_rank(vector) == 1.0
        assert precision_at_k(vector, 5) == 0.2
        assert ndcg_at_k(vector, 5) == 1.0, "an ideal ordering must score exactly 1.0"


# ---------------------------------------------------------------------------
# The abstention boundary -- this distinction IS the convention
# ---------------------------------------------------------------------------

class TestAbstentionBoundary:
    """``[]`` (abstained) and ``[0, 0]`` (answered, missed) must differ.

    The baseline abstains on ~24% of cases and every BERT arm on none, so this
    boundary decides whether a cross-arm table is comparing like with like. See
    N1-N3 in the module docstring.
    """

    def test_empty_list_is_no_observation_for_precision(self):
        assert precision_at_k([], 10) is None

    def test_answered_and_missed_is_a_measured_zero(self):
        assert precision_at_k([0.0, 0.0], 10) == 0.0

    def test_the_other_three_score_abstention_as_a_failure(self):
        # Deliberately NOT None -- but see N2: averaging these over all cases
        # reproduces legacy R, and over answered cases legacy P. Neither alone
        # is "the legacy number".
        assert hit_at_k([], 10) == 0.0
        assert reciprocal_rank([]) == 0.0
        assert ndcg_at_k([], 10) == 0.0

    @pytest.mark.parametrize("k", PRODUCTION_KS)
    def test_short_lists_do_not_crash_or_silently_zero(self, k):
        """The abstention path at every production K."""
        assert precision_at_k([1.0], k) == 1.0  # denominator is min(k, 1) = 1
        assert hit_at_k([1.0], k) == 1.0
        assert reciprocal_rank([1.0]) == 1.0


# ---------------------------------------------------------------------------
# The properties that are the point of the change
# ---------------------------------------------------------------------------

class TestPrecisionBreaksTheArtifact:
    """Precision@K must FALL as K grows. If it does not, P5 did not work."""

    def test_precision_falls_with_k_when_the_hit_is_front_loaded(self):
        # One relevant candidate at rank 5, list of 50 -- the canonical shape.
        vector = [0.0] * 4 + [1.0] + [0.0] * 45
        scores = [precision_at_k(vector, k) for k in PRODUCTION_KS]
        assert scores == [0.0, 0.2, 0.1, 0.05, 1 / 30, 0.025, 0.02]
        # Strictly decreasing from K=5 on: expanding K now costs something.
        tail = scores[1:]
        assert all(a > b for a, b in zip(tail, tail[1:])), scores

    def test_precision_is_flat_once_k_exceeds_the_list_length(self):
        """A measured caveat, not an oversight.

        The denominator is ``min(k, len)``, so for every ``k >= len`` the score
        stops moving -- K ceases to be a knob for an arm with short candidate
        lists. Measured on 6,227 of 6,328 sampled vectors.
        """
        vector = [0.0, 1.0, 0.0]
        assert precision_at_k(vector, 3) == precision_at_k(vector, 50)


class TestMrrIsImmuneToListLength:
    """MRR is the headline precisely because nothing past the first hit matters."""

    def test_appending_irrelevant_candidates_changes_nothing(self):
        vector = [0.0, 1.0]
        assert reciprocal_rank(vector) == reciprocal_rank(vector + [0.0] * 48)

    def test_splicing_zeros_immediately_after_the_hit_changes_nothing(self):
        assert reciprocal_rank([0.0, 1.0, 0.0]) == reciprocal_rank(
            [0.0, 1.0] + [0.0] * 30 + [0.0]
        )

    def test_mrr_has_no_k_parameter_at_all(self):
        """The reason MRR is reportable: there is no knob to choose."""
        with pytest.raises(TypeError):
            reciprocal_rank([1.0], 10)

    def test_rank_is_read_directly(self):
        for rank, expected in ((1, 1.0), (2, 0.5), (10, 0.1), (50, 0.02)):
            vector = [0.0] * (rank - 1) + [1.0]
            assert reciprocal_rank(vector) == expected


class TestArtifactsPreservedDeliberately:
    """Two metrics DO rise with K. Pinned so the fact stays visible.

    Neither is a bug to be fixed here. Hit@K is the existing metric, kept for
    continuity so the artifact can be seen by comparison; nDCG's behaviour is the
    standard convention. Both are why MRR and Precision@K -- not these -- carry
    the P5 claim.
    """

    def test_hit_at_k_is_monotonic_in_k(self):
        vector = [0.0] * 49 + [1.0]
        scores = [hit_at_k(vector, k) for k in PRODUCTION_KS]
        assert scores == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        assert all(a <= b for a, b in zip(scores, scores[1:]))

    def test_ndcg_is_also_monotonic_in_k_which_the_plan_denied(self):
        """N4. IDCG@k sums min(R,k) slots, so for k >= R the ideal freezes.

        Worked case: one relevant candidate at rank 15. nDCG@10 = 0 (not seen),
        nDCG@20 = nDCG@50 = 0.25 -- flat once k >= 15, so K is free again, just
        cheaper. R > 20 in ZERO of the 129 real test cases, so this holds for all
        of them at K >= 20.
        """
        vector = [0.0] * 14 + [1.0] + [0.0] * 35
        assert ndcg_at_k(vector, 10) == 0.0
        assert ndcg_at_k(vector, 20) == ndcg_at_k(vector, 50)
        scores = [ndcg_at_k(vector, k) for k in PRODUCTION_KS]
        assert all(a <= b for a, b in zip(scores, scores[1:])), scores

    def test_the_size_of_the_free_gain_is_what_ndcg_shrinks(self):
        """A rank-50 hit is worth 1/log2(51), not 1.0. Direction, not magnitude."""
        buried = [0.0] * 49 + [1.0]
        assert hit_at_k(buried, 50) == 1.0
        assert ndcg_at_k(buried, 50) == pytest.approx(1.0 / math.log2(51), rel=1e-12)


# ---------------------------------------------------------------------------
# Regression tests for the two mistakes the differential test caught
# ---------------------------------------------------------------------------

class TestTheUnitGainIdcgTrap:
    """One of three implementations built IDCG from a constant unit gain.

    Bit-identical on binary input, so no drg-exact number could reveal it -- but
    it breaks both of nDCG's defining invariants. ``metric-redesign.md`` keeps
    graded relevance on the table, so this would have fired silently later.
    """

    def test_ndcg_never_exceeds_one(self):
        assert ndcg_at_k([2.0], 1) == 1.0, "unit-gain IDCG returns 2.0 here"
        assert ndcg_at_k([5.0, 0.0], 2) == 1.0, "unit-gain IDCG returns 5.0 here"

    def test_an_already_ideal_ordering_scores_exactly_one(self):
        assert ndcg_at_k([0.5, 0.5, 0.5], 3) == 1.0, "unit-gain IDCG returns 0.5"

    def test_graded_relevance_matches_the_hand_worked_value(self):
        # [0.5, 1.0] at k=2: DCG = 0.5/1 + 1.0/log2(3); ideal = [1.0, 0.5]
        dcg = 0.5 + 1.0 / math.log2(3)
        idcg = 1.0 + 0.5 / math.log2(3)
        assert ndcg_at_k([0.5, 1.0], 2) == dcg / idcg
        assert ndcg_at_k([0.5, 1.0], 2) == pytest.approx(0.85971869985219718, rel=1e-15)


class TestKCoercion:
    """``operator.index``, not ``isinstance(k, int)`` -- load-bearing on Windows."""

    def test_numpy_integer_k_is_accepted(self):
        numpy = pytest.importorskip("numpy")
        # numpy.int64 is NOT an int subclass on Windows, so an isinstance check
        # would spuriously reject the type the pipeline actually produces.
        assert hit_at_k([1.0], numpy.int64(10)) == 1.0
        assert precision_at_k([1.0], numpy.int32(10)) == 1.0

    def test_integral_float_k_is_accepted(self):
        assert precision_at_k([1.0, 0.0], 2.0) == 0.5

    def test_fractional_k_raises_rather_than_truncating(self):
        """Truncation is a silent wrong answer in the abstention direction.

        ``k=0.5`` would truncate to 0 and make ``precision_at_k`` return None --
        i.e. report "no observation" for a positive k, quietly exempting the case
        from the average. This repo's standing rule is that a silent wrong answer
        is worse than a crash.
        """
        for bad in (0.5, 9.999, "10"):
            with pytest.raises(TypeError):
                precision_at_k([1.0], bad)

    def test_non_positive_k(self):
        assert hit_at_k([1.0], 0) == 0.0
        assert precision_at_k([1.0], 0) is None
        assert ndcg_at_k([1.0], -5) == 0.0


class TestInputStrictness:
    """Leniency here is a liability: abstention is the privileged outcome."""

    def test_none_relevance_raises_and_is_not_abstention(self):
        # Mapping None -> [] would hand a plumbing bug the abstention exemption.
        for call in (lambda: hit_at_k(None, 10),
                     lambda: precision_at_k(None, 10),
                     lambda: reciprocal_rank(None),
                     lambda: ndcg_at_k(None, 10)):
            with pytest.raises(TypeError):
                call()

    def test_one_shot_iterables_raise(self):
        """A generator scores the first call and abstains on the rest.

        Measured: one implementation accepted generators, so the same case was
        scored as a hit by ``hit_at_k`` and as an abstention by
        ``precision_at_k`` -- silently splitting the population.
        """
        with pytest.raises(TypeError):
            precision_at_k((v for v in [1.0, 0.0]), 2)

    def test_inputs_are_not_mutated(self):
        vector = [0.0, 1.0, 0.0]
        snapshot = list(vector)
        for k in PRODUCTION_KS:
            hit_at_k(vector, k)
            precision_at_k(vector, k)
            ndcg_at_k(vector, k)
        reciprocal_rank(vector)
        assert vector == snapshot

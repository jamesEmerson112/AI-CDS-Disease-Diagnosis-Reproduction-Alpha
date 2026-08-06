"""Pin the TOP-K threshold gate that the BERT arm actually calls.

``containGreaterOrEqualsValue`` decides, for each test case and each K, whether
any of the top-K candidates cleared the threshold -- so it decides every TP and
FP in every TOP-K row of every ``PerformanceIndex.txt``. It is the highest-
leverage five lines in the repository.

Until P36 there were **three** copies of it: the shared one in ``cython_utils``,
a private one in ``bert_models``, and a third defined inside
``tests/test_bert_symptom_pairwise.py`` and injected. The existing tests
exercised the first and the third. **Nothing touched the copy the BERT arm
actually called in production**, which is the only one that could move a
committed number.

These tests exist so that stays true after the copy is gone:

1. ``bert_models`` must not define its own -- if a copy reappears, that is the
   drift this file exists to catch.
2. The shared function must still agree with the deleted copy's semantics
   everywhere it can be reached, so the deletion is provably inert rather than
   merely believed to be.

Both are cheap and neither needs a model, so they run in the default suite.
The golden is the backstop; this is the part that names the cause.
"""

import inspect

import pytest

from aicds.utils.Constants import TOP_K_INCR, TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND
from aicds.utils.cython_utils import containGreaterOrEqualsValue


def _deleted_bert_copy(topk, similarity_list, threshold):
    """Verbatim body of the copy P36 removed from ``bert_models.py``.

    Kept only as the reference side of the equivalence check below. It is not
    imported by anything and must never be called by production code.
    """
    for sim in similarity_list[:topk]:
        if sim >= threshold:
            return True
    return False


PRODUCTION_TOPKS = list(range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR))

THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 1.0]


class TestNoPrivateCopy:
    """The BERT arm must reach the shared gate, not a local re-implementation."""

    def test_bert_models_defines_no_copy(self):
        from aicds.models import bert_models

        assert not hasattr(bert_models, "containGreaterOrEqualsValue"), (
            "bert_models has grown its own containGreaterOrEqualsValue again. "
            "The two arms must share this function: it decides every TP and FP "
            "in every TOP-K row, so a private copy is a silent divergence "
            "between the baseline and BERT results."
        )

    def test_call_site_goes_through_util_cy(self):
        """Guard the *spelling* at the call site, not just the module namespace.

        A bare ``containGreaterOrEqualsValue(...)`` call would raise NameError
        now, so this is belt-and-braces -- but a future ``from ... import`` at
        the top of bert_models would resurrect the unqualified spelling without
        resurrecting the copy, and then the test above would still pass.
        """
        from aicds.models import bert_models

        source = inspect.getsource(bert_models)
        calls = [
            line.strip()
            for line in source.splitlines()
            if "containGreaterOrEqualsValue(" in line and not line.strip().startswith("#")
        ]
        assert calls, "the TOP-K threshold gate call site has disappeared entirely"
        for call in calls:
            assert "util_cy.containGreaterOrEqualsValue(" in call, (
                "call site must be qualified as util_cy.containGreaterOrEqualsValue "
                "so it provably reaches the shared implementation: %r" % call
            )


class TestEquivalenceWithDeletedCopy:
    """The shared function must behave exactly as the deleted copy did."""

    @pytest.mark.parametrize("topk", PRODUCTION_TOPKS)
    @pytest.mark.parametrize("threshold", THRESHOLDS)
    def test_agrees_on_production_shapes(self, topk, threshold):
        # Candidate lists shorter than, equal to and longer than topk, since the
        # two implementations differed precisely in how they bounded the scan:
        # an ``i < len()`` guard versus a slice.
        lengths = [0, 1, topk - 1, topk, topk + 1, topk * 2]
        for length in lengths:
            if length < 0:
                continue
            for pattern in _patterns(length, threshold):
                assert containGreaterOrEqualsValue(
                    topk, pattern, threshold
                ) is _deleted_bert_copy(topk, pattern, threshold), (
                    "divergence at topk=%d threshold=%s list=%r"
                    % (topk, threshold, pattern)
                )

    def test_only_the_first_topk_entries_count(self):
        """A hit *past* K must not register. This is what makes K meaningful."""
        similarities = [0.1] * 10 + [1.0]
        assert containGreaterOrEqualsValue(10, similarities, 0.9) is False
        assert containGreaterOrEqualsValue(11, similarities, 0.9) is True

    def test_rank_blindness_is_the_current_behaviour(self):
        """Rank 1 and rank K score identically -- pinned, not endorsed.

        This is the defect P5 exists to fix. Pinning it means the change will
        show up here as a deliberate edit rather than as a silently moved
        number, which is the whole point of characterising before changing.
        """
        front = [1.0] + [0.1] * 9
        back = [0.1] * 9 + [1.0]
        assert containGreaterOrEqualsValue(10, front, 1.0) is True
        assert containGreaterOrEqualsValue(10, back, 1.0) is True

    def test_empty_candidate_list_is_false_not_an_error(self):
        """The abstention path. The baseline reaches it on ~24% of cases."""
        assert containGreaterOrEqualsValue(10, [], 0.6) is False
        assert _deleted_bert_copy(10, [], 0.6) is False


def _patterns(length, threshold):
    """Similarity lists that straddle ``threshold`` in the ways that matter."""
    if length == 0:
        return [[]]
    below = threshold - 0.05
    at = threshold
    above = min(threshold + 0.05, 1.0)
    return [
        [below] * length,                        # nothing clears
        [at] + [below] * (length - 1),           # exactly at the bar, at rank 1
        [below] * (length - 1) + [at],           # exactly at the bar, last
        [below] * (length - 1) + [above],        # clears, last
        [above] * length,                        # everything clears
    ]

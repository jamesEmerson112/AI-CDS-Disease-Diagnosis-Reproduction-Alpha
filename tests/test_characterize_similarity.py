"""
Characterization tests for the similarity / pruning / retrieval logic shared by
both arms of the pipeline.

These tests pin CURRENT behaviour only. They exist as a safety net for later
refactors (see docs/reference/architecture.md and the CLAUDE.md "Two independent
problems with the headline metric" section) — they do not assert the logic is
"correct" in any normative sense, and known quirks (e.g. the max(len_test,
len_train) denominator, or the >= vs > pruning boundary) are pinned exactly as
found, not as they "should" be.

Targets:
- src/utils/cython_utils.py: cosine_similarity, containGreaterOrEqualsValue
- src/models/bert_models.py: compute_patient_similarity_pairwise,
  predict_topk_diagnoses_pure
- src/utils/Constants.py: PRUNING_SIMILARITY, TOP_K_* bounds

Derivation note: every expected numeric value below was computed by actually
running the target function (see the derivation commands in the task history)
and then hard-coded here — none are hand-derived from reading the source only.
"""

import math

import pytest

from aicds.utils.cython_utils import cosine_similarity, containGreaterOrEqualsValue
from aicds.utils.Constants import (
    PRUNING_SIMILARITY,
    MIN_SIMILARITY,
    TOP_K_LOWER_BOUND,
    TOP_K_UPPER_BOUND,
    TOP_K_INCR,
)
from aicds.entity.SymptomsDiagnosis import SymptomsDiagnosis

# Importing bert_models is expensive (sentence-transformers, and torch
# transitively through it) but the task brief targets its functions directly, so
# we pay the cost once at module scope like the rest of the suite does.
from aicds.models.bert_models import (
    compute_patient_similarity_pairwise,
    predict_topk_diagnoses_pure,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Pin the literal constant values that drive pruning and TOP-K ranges.

    An accidental edit to any of these silently changes every reported number
    in the project without touching a single line of similarity logic.
    """

    def test_pruning_similarity_is_exactly_point_five(self):
        assert PRUNING_SIMILARITY == 0.5

    def test_min_similarity_is_zero(self):
        assert MIN_SIMILARITY == 0

    def test_topk_bounds(self):
        assert TOP_K_LOWER_BOUND == 10
        assert TOP_K_UPPER_BOUND == 60
        assert TOP_K_INCR == 10

    def test_topk_range_is_10_20_30_40_50(self):
        # This is the actual iteration used at every TOP-K call site
        # (range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR)). An
        # off-by-one on either bound would silently drop or add a K value
        # from every reported TOP-K performance block.
        assert list(range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR)) == [
            10,
            20,
            30,
            40,
            50,
        ]


# ---------------------------------------------------------------------------
# cosine_similarity (src/utils/cython_utils.py)
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_mismatched_lengths_raise_assertion_error(self):
        with pytest.raises(AssertionError):
            cosine_similarity([1, 2], [1, 2, 3])

    def test_zero_vector_guard_returns_zero(self):
        # uu == 0 -> the `if uu != 0 and vv != 0` guard short-circuits to the
        # initialized cos_theta = 0.0, not a ZeroDivisionError / NaN.
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_both_zero_vectors_returns_zero(self):
        assert cosine_similarity([0, 0], [0, 0]) == 0.0

    def test_known_pair_returns_known_value(self):
        # u=[1,2,3], v=[4,5,6]: uv=32, uu=14, vv=77, cos = 32/sqrt(14*77)
        u = [1, 2, 3]
        v = [4, 5, 6]
        result = cosine_similarity(u, v)
        expected = 32 / math.sqrt(14 * 77)
        assert result == pytest.approx(expected)
        assert result == pytest.approx(0.9746318461970762)

    def test_identical_vector_returns_one(self):
        # cos_theta = uu / sqrt(uu*uu). Pinned separately from the general
        # case because it is used below to construct exact-0.5 fixtures for
        # the pruning-boundary tests without depending on irrational sqrt
        # rounding.
        assert cosine_similarity([1, 2, 3], [1, 2, 3]) == 1.0


# ---------------------------------------------------------------------------
# containGreaterOrEqualsValue (src/utils/cython_utils.py)
# ---------------------------------------------------------------------------


class TestContainGreaterOrEqualsValue:
    def test_any_value_clearing_threshold_returns_true(self):
        assert containGreaterOrEqualsValue(2, [0.4, 0.9], 0.6) is True

    def test_exact_boundary_value_counts_as_clearing(self):
        # >= , not >.
        assert containGreaterOrEqualsValue(1, [0.6], 0.6) is True

    def test_empty_input_returns_false(self):
        assert containGreaterOrEqualsValue(5, [], 0.6) is False

    def test_value_just_below_threshold_returns_false(self):
        assert containGreaterOrEqualsValue(1, [0.599999], 0.6) is False

    def test_only_first_topk_elements_are_examined(self):
        # The qualifying value (0.9) sits at index 1. With topK=1 only index
        # 0 (0.4) is inspected, so it must NOT be found even though it is
        # present later in the list -- this pins that the function is a
        # bounded prefix scan, not an unbounded `any(...)`.
        assert containGreaterOrEqualsValue(1, [0.4, 0.9], 0.6) is False
        # With topK=2 the same list now finds it.
        assert containGreaterOrEqualsValue(2, [0.4, 0.9], 0.6) is True


# ---------------------------------------------------------------------------
# compute_patient_similarity_pairwise (src/models/bert_models.py)
# ---------------------------------------------------------------------------


class TestComputePatientSimilarityPairwise:
    """Embedding dicts here follow the project-wide contract: keyed by
    preprocessed symptom TEXT (not HADM_ID), each value a ONE-ELEMENT LIST so
    call sites can index emb[0]. Diverging from that contract changes results
    silently (wrong numbers) rather than raising, EXCEPT when the stored
    value is not indexable at all, which is pinned below as a TypeError.
    """

    def test_denominator_is_max_of_test_and_train_lengths_not_test_count(self):
        # test has 1 symptom, train has 2. s1 matches t1 exactly (cos=1.0);
        # t2 is a zero vector so cos=0.0 and loses the max. The mean is
        # therefore 1.0 / max(1, 2) = 0.5 -- NOT 1.0 / len(test_symptoms) = 1.0.
        # This is the discriminating case: if the implementation used the
        # test-symptom count as the denominator instead, this test would
        # observe 1.0 here rather than 0.5.
        embeddings = {
            "s1": [[1, 2, 3]],
            "t1": [[1, 2, 3]],  # identical to s1 -> cosine 1.0
            "t2": [[0, 0, 0]],  # zero vector -> cosine 0.0
        }
        result = compute_patient_similarity_pairwise(["s1"], ["t1", "t2"], embeddings)
        assert result == 0.5

    def test_denominator_uses_max_when_train_is_shorter_than_test(self):
        # Same max(...) denominator, mirrored: now test has 2 symptoms,
        # train has 1. s2 finds no better match than the same t1 (cosine
        # 0.0 for the zero-vector test symptom), s1 matches t1 exactly
        # (1.0). Sum of maxes = 1.0, denominator = max(2, 1) = 2 -> 0.5.
        embeddings = {
            "s1": [[1, 2, 3]],
            "s2": [[0, 0, 0]],
            "t1": [[1, 2, 3]],
        }
        result = compute_patient_similarity_pairwise(["s1", "s2"], ["t1"], embeddings)
        assert result == 0.5

    def test_perfect_single_symptom_match_returns_one(self):
        embeddings = {
            "s1": [[1, 2, 3]],
            "t1": [[1, 2, 3]],
        }
        result = compute_patient_similarity_pairwise(["s1"], ["t1"], embeddings)
        assert result == 1.0

    def test_one_element_list_contract_violation_raises_typeerror(self):
        # Storing the raw vector instead of a one-element list wrapping it
        # (the documented embedding-dict contract) breaks emb[0] indexing:
        # indexing a raw numeric vector by [0] yields a scalar, and that
        # scalar is then passed into cosine_similarity's `for i in
        # range(len(u))`, which fails on a plain int. This is pinned as a
        # TypeError, NOT a silent wrong answer -- but note the contract
        # violation only surfaces this loudly because the vector elements
        # here are ints; the project docs warn other violations of this
        # contract can change results silently instead of raising.
        bad_embeddings = {
            "s1": [1, 2, 3],  # missing the outer one-element-list wrapper
            "t1": [1, 2, 3],
        }
        with pytest.raises(TypeError):
            compute_patient_similarity_pairwise(["s1"], ["t1"], bad_embeddings)


# ---------------------------------------------------------------------------
# predict_topk_diagnoses_pure (src/models/bert_models.py)
# ---------------------------------------------------------------------------


def _mk_admission(hadm_id, diagnosis):
    return SymptomsDiagnosis(hadm_id, "subj", "admit", "disch", "sym", diagnosis)


class TestPredictTopkDiagnosesPure:
    """Fixture layout shared by all tests in this class:

    test_symptoms = ['s1'], embedded identically to train symptom 't1'
    (cosine 1.0), with 't2'/'t3' as zero vectors (cosine 0.0, never chosen
    as the max). Three synthetic training patients give pairwise
    similarities of exactly 1.0, exactly 0.5 (the PRUNING_SIMILARITY
    boundary), and 1/3 (clearly below it):

      p_high     train_symptoms=['t1']            -> similarity 1.0
      p_boundary train_symptoms=['t1', 't2']       -> similarity 0.5
      p_low      train_symptoms=['t1', 't2', 't3'] -> similarity 1/3
    """

    embeddings = {
        "s1": [[1, 2, 3]],
        "t1": [[1, 2, 3]],
        "t2": [[0, 0, 0]],
        "t3": [[0, 0, 0]],
    }

    admissions = {
        "p_high": _mk_admission("p_high", ["apr:HIGH"]),
        "p_boundary": _mk_admission("p_boundary", ["apr:BOUNDARY"]),
        "p_low": _mk_admission("p_low", ["apr:LOW"]),
    }

    x_train = [
        {"p_high": ["t1"]},
        {"p_boundary": ["t1", "t2"]},
        {"p_low": ["t1", "t2", "t3"]},
    ]

    def test_precondition_similarities_are_1_point5_and_third(self):
        # Sanity-check the fixture itself against the directly-tested
        # function, independent of predict_topk_diagnoses_pure.
        assert compute_patient_similarity_pairwise(["s1"], ["t1"], self.embeddings) == 1.0
        assert (
            compute_patient_similarity_pairwise(["s1"], ["t1", "t2"], self.embeddings)
            == 0.5
        )
        assert compute_patient_similarity_pairwise(
            ["s1"], ["t1", "t2", "t3"], self.embeddings
        ) == pytest.approx(1 / 3)

    def test_pruning_boundary_exactly_point_five_is_included(self):
        # k=2 admits both p_high (1.0) and p_boundary (0.5, the exact
        # PRUNING_SIMILARITY boundary). p_low (1/3) must be pruned out.
        diags, sims, ids = predict_topk_diagnoses_pure(
            None, ["s1"], self.x_train, self.embeddings, {}, self.admissions, k=2
        )
        assert ids == ["p_high", "p_boundary"]
        assert sims == [1.0, 0.5]
        assert diags == [["apr:HIGH"], ["apr:BOUNDARY"]]

    def test_below_pruning_threshold_is_excluded_even_with_room_in_k(self):
        # k=10 gives plenty of room, but only 2 of the 3 training patients
        # ever clear the >= 0.5 pruning threshold, so p_low never appears
        # regardless of how large k is.
        diags, sims, ids = predict_topk_diagnoses_pure(
            None, ["s1"], self.x_train, self.embeddings, {}, self.admissions, k=10
        )
        assert ids == ["p_high", "p_boundary"]
        assert sims == [1.0, 0.5]

    def test_max_mode_k_none_returns_single_best_match(self):
        diags, sims, ids = predict_topk_diagnoses_pure(
            None, ["s1"], self.x_train, self.embeddings, {}, self.admissions, k=None
        )
        assert ids == ["p_high"]
        assert sims == [1.0]
        assert diags == [["apr:HIGH"]]

    def test_topk_1_returns_only_the_single_best_above_threshold(self):
        diags, sims, ids = predict_topk_diagnoses_pure(
            None, ["s1"], self.x_train, self.embeddings, {}, self.admissions, k=1
        )
        assert ids == ["p_high"]
        assert sims == [1.0]

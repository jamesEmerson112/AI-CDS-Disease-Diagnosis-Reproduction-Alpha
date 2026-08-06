"""Pin RankMetrics.txt's aggregation, populations and output format.

Four of these tests exist because an adversarial review found four defects in the
first draft of ``rank_report.py``, every one of which produced a wrong number
while looking correct. They are regression tests, not hypotheticals:

  D1  ``%.16f`` is not round-trip safe. 25.1% of ``a/b`` values with ``b <= 129``
      -- exactly this project's value space -- read back as a different float.
  D2  the aggregate divided by the number of POPULATED folds instead of
      ``K_FOLD``, so an arm abstaining through a whole fold would stop
      reproducing legacy P.
  D3  the module docstring asserted an exact identity against
      PerformanceIndex.txt that nothing computed. That identity is the strongest
      correctness test available in this project; it is now ``TestLegacyIdentity``.
  D4  both arms skip a test case with ``continue``, and accumulation happened
      AFTER the skip -- so a skipped case would shrink the all-cases denominator
      while legacy still counted it in ``nrow``.

D1, D2 and D4 were all LATENT on today's data. That is why they are pinned: a
latent numeric divergence surfaces later as a fourth-decimal disagreement after
some unrelated edit, and reads as a metric bug rather than as this.
"""

import re

import pytest

from aicds.analysis.populations import winnable_by_fold
from aicds.analysis.rank_report import RankAccumulator, REPORT_KS, binarise
from aicds.config import CORRECTED, from_name
from aicds.utils.Constants import K_FOLD

DRG = from_name("drg")


def _hit_at_rank(rank, length=50):
    """A relevance vector whose only relevant candidate sits at 1-indexed ``rank``."""
    return [0.0] * (rank - 1) + [1.0] + [0.0] * (length - rank)


def _filled(config=DRG, winnable_vector=None, other_vector=None):
    """An accumulator covering all 129 real test cases."""
    acc = RankAccumulator(config)
    for fold, (winnable, tested) in winnable_by_fold(config).items():
        acc.begin_fold(fold, len(tested))
        for hadm in sorted(tested):
            vector = winnable_vector if hadm in winnable else other_vector
            acc.add(fold, hadm, list(vector))
    return acc


class TestBinarise:
    def test_drg_values_pass_through_unchanged(self):
        """Under drg-exact the grader already emits 0.0/1.0, so this is identity."""
        assert binarise([1.0, 0.0, 1.0]) == [1.0, 0.0, 1.0]

    def test_cosine_below_one_is_not_relevant(self):
        """Reproduces threshold 1.0, which P4 proved equals drg-exact."""
        assert binarise([0.9999999999999998, 1.0]) == [0.0, 1.0]

    def test_it_does_not_mutate_its_input(self):
        """The BERT arm scores all_diag_sims AFTER handing it to the accumulator.

        An in-place mutation here would move every TOP-K number in
        PerformanceIndex.txt -- the single most expensive bug this file could
        have. Pinned rather than assumed.
        """
        original = [1.0, 0.0, 0.5]
        snapshot = list(original)
        binarise(original)
        assert original == snapshot

    def test_accumulator_add_does_not_mutate_either(self):
        acc = RankAccumulator(DRG)
        vector = [1.0, 0.0, 0.5]
        snapshot = list(vector)
        acc.add(0, "100001", vector)
        assert vector == snapshot


class TestRoundTripFormatting:
    """D1. Every emitted float must read back as the same float."""

    def test_no_emitted_float_loses_precision(self, tmp_path):
        acc = _filled(winnable_vector=_hit_at_rank(3), other_vector=[0.0] * 50)
        path = acc.write(str(tmp_path))
        text = open(path, encoding="utf-8").read()
        tokens = re.findall(r"(?<![\w.])-?\d+\.\d+(?:e[-+]?\d+)?", text)
        assert tokens, "no numeric output found -- the format changed"
        bad = [t for t in tokens if repr(float(t)) != t]
        assert not bad, "these do not round-trip: %s" % bad[:5]

    def test_the_format_matches_what_cython_utils_emits(self):
        """``str(x)`` is ``repr(x)`` for floats, so the two artifacts compare textually."""
        value = 0.24743589743589745
        assert str(value) == repr(value)
        assert "%.16f" % value != repr(value), (
            "if these ever agree, this test has stopped proving anything"
        )


class TestAggregationDivisor:
    """D2. The divisor is K_FOLD, always -- matching cython_utils.py:744-746."""

    def test_an_empty_fold_still_divides_by_k_fold(self):
        # Fold 0 abstains on everything; every other fold hits at rank 1.
        acc = RankAccumulator(DRG)
        for fold, (_, tested) in winnable_by_fold(DRG).items():
            acc.begin_fold(fold, len(tested))
            for hadm in sorted(tested):
                acc.add(fold, hadm, [] if fold == 0 else _hit_at_rank(1))

        aggregate = acc.aggregate("answered")
        populated = [
            acc.fold_stats(f, "answered") for f in range(K_FOLD)
        ]
        populated = [s for s in populated if s["n"]]

        assert len(populated) == 9
        assert aggregate["empty_folds"] == 1
        expected = sum(s["mrr"] for s in populated) / K_FOLD
        assert aggregate["mrr"] == expected
        # The old, wrong divisor would have given 1.0 here.
        assert aggregate["mrr"] != sum(s["mrr"] for s in populated) / len(populated)

    def test_empty_fold_count_is_reported_not_hidden(self):
        acc = _filled(winnable_vector=_hit_at_rank(1), other_vector=[0.0] * 50)
        assert acc.aggregate("winnable")["empty_folds"] == 0


class TestCompletenessCheck:
    """D4. A case that is skipped rather than recorded must raise."""

    def test_omitting_a_case_raises(self):
        acc = RankAccumulator(DRG)
        tested = sorted(winnable_by_fold(DRG)[0][1])
        acc.begin_fold(0, len(tested))
        for hadm in tested[:-1]:            # one short
            acc.add(0, hadm, _hit_at_rank(1))
        with pytest.raises(ValueError, match="expected"):
            acc.fold_stats(0, "all-cases")

    def test_recording_the_skip_as_an_abstention_satisfies_it(self):
        acc = RankAccumulator(DRG)
        tested = sorted(winnable_by_fold(DRG)[0][1])
        acc.begin_fold(0, len(tested))
        for hadm in tested[:-1]:
            acc.add(0, hadm, _hit_at_rank(1))
        acc.add(0, tested[-1], [])          # the skipped case, as an abstention
        stats = acc.fold_stats(0, "all-cases")
        assert stats["n"] == len(tested)
        assert stats["answered"] == len(tested) - 1

    def test_no_declared_size_means_no_check(self):
        """``add`` without ``begin_fold`` still works -- for unit tests like these."""
        acc = RankAccumulator(DRG)
        acc.add(0, "100001", _hit_at_rank(1))
        assert acc.fold_stats(0, "all-cases")["n"] == 1


class TestPopulations:
    """N1: one denominator per block, for all four metrics."""

    def test_winnable_is_the_arm_invariant_76(self):
        acc = _filled(winnable_vector=_hit_at_rank(3), other_vector=[0.0] * 50)
        assert acc.aggregate("winnable", pooled=True)["n"] == 76
        assert acc.aggregate("all-cases", pooled=True)["n"] == 129

    def test_pooled_winnable_does_not_widen_to_all_cases(self):
        """The bug the ``__new__`` version of ``aggregate`` actually had.

        It rebuilt a throwaway accumulator and re-ran the population filter with
        the population remapped, so ``pooled + winnable`` risked silently
        becoming ``pooled + all-cases``. n=76 vs n=129 is the tell.
        """
        acc = _filled(winnable_vector=_hit_at_rank(3), other_vector=[0.0] * 50)
        winnable = acc.aggregate("winnable", pooled=True)
        assert winnable["n"] == 76
        # Every winnable case hits at rank 3, every other case never hits, so a
        # widened population would drag this well below 1/3.
        assert winnable["mrr"] == pytest.approx(1 / 3, rel=1e-12)
        assert winnable["per_k"][5]["hit"] == 1.0

    def test_precision_shares_the_denominator_in_abstention_inclusive_blocks(self):
        """N1's fix: precision's None becomes 0.0 so all four metrics share n."""
        acc = RankAccumulator(DRG)
        acc.begin_fold(0, 2)
        acc.add(0, "A", _hit_at_rank(1, length=10))
        acc.add(0, "B", [])                  # abstention
        stats = acc.fold_stats(0, "all-cases")
        assert stats["n"] == 2
        # hit averages 1.0 and 0.0 -> 0.5; precision must use the SAME n, so
        # 1.0 and 0.0 -> 0.5, not 1.0 (which is what dropping the None gives).
        assert stats["per_k"][1]["hit"] == 0.5
        assert stats["per_k"][1]["precision"] == 0.5

    def test_answered_excludes_abstentions_entirely(self):
        acc = RankAccumulator(DRG)
        acc.begin_fold(0, 2)
        acc.add(0, "A", _hit_at_rank(1, length=10))
        acc.add(0, "B", [])
        stats = acc.fold_stats(0, "answered")
        assert stats["n"] == 1
        assert stats["per_k"][1]["hit"] == 1.0
        assert stats["coverage"] == 1.0


class TestLegacyIdentity:
    """D3. The identity the docstring claimed but nothing computed.

    Legacy's three columns ARE the three populations (cython_utils.py:302 drops
    abstentions from both counters), so:

      pooled  Hit@K over all-cases  ==  sum(TP) / 129
      pooled  Hit@K over answered   ==  sum(TP) / answered

    Checked here against a hand-built population rather than a real run, so it is
    exact and needs no model. The real-run version is step 8 of the plan.
    """

    def test_pooled_all_cases_hit_equals_tp_over_129(self):
        acc = _filled(winnable_vector=_hit_at_rank(3), other_vector=[0.0] * 50)
        # 76 winnable cases hit within K>=3; 53 never hit.
        assert acc.aggregate("all-cases", pooled=True)["per_k"][10]["hit"] == 76 / 129

    def test_pooled_answered_hit_equals_tp_over_answered(self):
        # 40 cases answer and hit, 20 answer and miss, 69 abstain.
        acc = RankAccumulator(DRG)
        acc.begin_fold(0, 129)
        for i in range(40):
            acc.add(0, "hit%d" % i, _hit_at_rank(1, length=10))
        for i in range(20):
            acc.add(0, "miss%d" % i, [0.0] * 10)
        for i in range(69):
            acc.add(0, "abstain%d" % i, [])

        answered = acc.aggregate("answered", pooled=True)
        all_cases = acc.aggregate("all-cases", pooled=True)
        assert answered["n"] == 60
        assert all_cases["n"] == 129
        assert answered["per_k"][1]["hit"] == 40 / 60      # legacy P
        assert all_cases["per_k"][1]["hit"] == 40 / 129    # legacy R
        assert all_cases["coverage"] == 60 / 129           # legacy PR

    def test_the_two_populations_genuinely_disagree(self):
        """If they ever coincide, the three-population ceremony is pointless."""
        acc = RankAccumulator(DRG)
        acc.begin_fold(0, 4)
        acc.add(0, "A", _hit_at_rank(1, length=10))
        acc.add(0, "B", [0.0] * 10)
        acc.add(0, "C", [])
        acc.add(0, "D", [])
        answered = acc.fold_stats(0, "answered")["per_k"][1]["hit"]
        all_cases = acc.fold_stats(0, "all-cases")["per_k"][1]["hit"]
        assert answered == 0.5
        assert all_cases == 0.25
        assert answered != all_cases


class TestOutputFile:
    def test_it_names_its_population_in_every_block(self, tmp_path):
        acc = _filled(winnable_vector=_hit_at_rank(3), other_vector=[0.0] * 50)
        text = open(acc.write(str(tmp_path)), encoding="utf-8").read()
        for population in ("winnable", "all-cases", "answered"):
            assert "population=%s" % population in text

    def test_it_carries_the_per_fold_blocks_the_t_test_needs(self, tmp_path):
        acc = _filled(winnable_vector=_hit_at_rank(3), other_vector=[0.0] * 50)
        text = open(acc.write(str(tmp_path)), encoding="utf-8").read()
        for fold in range(K_FOLD):
            assert "FOLD %d" % fold in text
        assert "mean of per-fold rates" in text
        assert "pooled over cases" in text

    def test_it_labels_mrr_with_its_cap(self, tmp_path):
        """MRR@50, not MRR -- a hit at rank 51+ is invisible and scores 0."""
        acc = _filled(winnable_vector=_hit_at_rank(3), other_vector=[0.0] * 50)
        text = open(acc.write(str(tmp_path)), encoding="utf-8").read()
        assert "MRR@50" in text

    def test_it_warns_against_cross_population_comparison(self, tmp_path):
        acc = _filled(winnable_vector=_hit_at_rank(3), other_vector=[0.0] * 50)
        text = open(acc.write(str(tmp_path)), encoding="utf-8").read()
        assert "NOT comparable" in text
        assert "WITHIN-ARM" in text

    def test_every_reported_k_appears(self, tmp_path):
        acc = _filled(winnable_vector=_hit_at_rank(3), other_vector=[0.0] * 50)
        lines = open(acc.write(str(tmp_path)), encoding="utf-8").read().splitlines()
        ks = {int(l.split("\t")[0].strip()) for l in lines
              if l.startswith("  ") and "\t" in l and l.split("\t")[0].strip().isdigit()}
        assert ks == set(REPORT_KS)


class TestDeterminism:
    """populations.py builds sets; no reported number may depend on their order."""

    def test_repeated_construction_gives_identical_output(self, tmp_path):
        outputs = []
        for name in ("a", "b"):
            directory = tmp_path / name
            directory.mkdir()
            accumulator = _filled(
                winnable_vector=_hit_at_rank(3), other_vector=[0.0] * 50
            )
            path = accumulator.write(str(directory))
            outputs.append(open(path, encoding="utf-8").read())
        assert outputs[0] == outputs[1]

    def test_legacy_config_also_works(self):
        """The 75/129 fold set, so the report is not silently drg-only."""
        acc = _filled(config=CORRECTED, winnable_vector=_hit_at_rank(1),
                      other_vector=[0.0] * 50)
        assert acc.aggregate("winnable", pooled=True)["n"] == 76

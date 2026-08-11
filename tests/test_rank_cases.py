"""Pin RankCases.txt (P40): the writer, its format, and the consumer's math.

RankCases.txt exists to answer ONE question that no aggregate can: are the 98
cases the baseline chose to answer simply the easy ones? Answering it means
scoring the BERT arms on exactly that case set, so two things have to hold and
both are pinned here.

  * The file must identify cases well enough to intersect two arms' records --
    fold, HADM_ID, and enough outcome to recompute a reciprocal rank. A dropped
    or mis-parsed row shrinks a denominator by one and moves an MRR in the fourth
    decimal, with nothing on screen to say so.
  * The restriction must be a restriction, not a rescaling. The withdrawn
    ``MRR_all-cases / coverage`` idea returned ~1.0 by construction and looked
    like evidence; the fixture below is built so a *rescaling* and a
    *restriction* give visibly different answers -- the BERT arm's best case is
    one the baseline abstained on, so restricting must LOWER its MRR.

EVERY IDENTIFIER HERE IS SYNTHETIC (999xxx), and there are five of them. Real
HADM_IDs are DUA-covered and may only ever appear inside a gitignored run
directory; the pre-commit hook additionally blocks any new file carrying 20+
distinct IDs, which a per-case test could trip without noticing.
"""

from __future__ import annotations

import os

import pytest

from aicds.analysis.populations import winnable_test_cases
from aicds.analysis.rank_metrics import reciprocal_rank
from aicds.analysis.rank_report import (
    RANK_CASES_FILENAME,
    RankAccumulator,
    first_relevant_rank,
)
from aicds.config import from_name

import scripts.analyze_rank_metrics as consumer

DRG = from_name("drg")

# Five synthetic admissions, reused by every fixture below.
IDS = ["999001", "999002", "999003", "999004", "999005"]


def _vector(rank, length=50):
    """A candidate vector of ``length`` whose only relevant entry sits at ``rank``.

    ``rank=None`` is an answer that missed -- ``length`` candidates, none
    relevant. That is NOT an abstention: an abstention is the empty list, and
    telling those two apart is what the candidate-count column is for.
    """
    if rank is None:
        return [0.0] * length
    return [0.0] * (rank - 1) + [1.0] + [0.0] * (length - rank)


def _accumulator(rows, fold=0):
    """``rows`` is ``[(hadm, vector), ...]`` recorded into one fold, in order."""
    acc = RankAccumulator(DRG)
    acc.begin_fold(fold, len(rows))
    for hadm, vector in rows:
        acc.add(fold, hadm, vector)
    return acc


# The fixture the consumer test computes by hand. Ranks are chosen so every
# reciprocal rank is an exact binary fraction (1, 1/2, 1/4) and the sums land on
# 1.75 and 2.75, which divide exactly by 4.
BASELINE_ROWS = [
    ("999001", _vector(1)),        # RR 1.0
    ("999002", _vector(2)),        # RR 0.5
    ("999003", _vector(4)),        # RR 0.25
    ("999004", _vector(None)),     # answered, missed  -> RR 0.0
    ("999005", []),                # ABSTAINED         -> RR 0.0
]

# The same five admissions, all answered -- the BERT shape. Three properties are
# load-bearing, and none of them is decorative:
#
#   * A rank-1 sits on 999005, the case the baseline ABSTAINED on, so restricting
#     to the baseline's answered set must REMOVE a win and LOWER this arm's MRR.
#     A rescaling by coverage would have raised it.
#   * The row ORDER differs from BASELINE_ROWS. With the ids in matching order the
#     baseline's four answered cases ARE the first four rows, so a positional
#     ``rows[:len(answered)]`` slice returns exactly what the (fold, HADM_ID) join
#     returns and the join is unfalsifiable.
#   * The sums are chosen so the BERT arm's restricted MRR (2.25/4) equals NEITHER
#     the baseline's restricted MRR (1.75/4) NOR the positional slice's (2.75/4).
#     Otherwise an assertion on the BERT restricted column is satisfied by the
#     baseline's printed row and never inspects the BERT one at all.
#
# Both mutations -- the positional slice, and computing every arm's restricted
# column from the baseline's rows -- were applied to the production join and
# passed this whole file before the fixture was rebuilt. They fail it now.
BERT_ROWS = [
    ("999003", _vector(1)),        # RR 1.0
    ("999005", _vector(1)),        # RR 1.0   <- the case the baseline abstained on
    ("999001", _vector(2)),        # RR 0.5
    ("999004", _vector(4)),        # RR 0.25
    ("999002", _vector(2)),        # RR 0.5
]


class TestFirstRelevantRank:
    """The rank column, and its identity with the metric module's RR."""

    def test_it_is_one_based(self):
        assert first_relevant_rank(_vector(1)) == 1
        assert first_relevant_rank(_vector(7)) == 7

    def test_no_relevant_candidate_gives_none(self):
        assert first_relevant_rank(_vector(None)) is None

    def test_an_abstention_gives_none_too(self):
        assert first_relevant_rank([]) is None

    def test_it_agrees_with_reciprocal_rank_exactly(self):
        """The identity that keeps the two files consistent.

        RankMetrics.txt reports 1/rank and RankCases.txt reports rank. If these
        ever disagree, one of the two artifacts is describing a different run.
        Checked over the whole reportable range rather than a sample.
        """
        for rank in range(1, 51):
            vector = _vector(rank)
            assert reciprocal_rank(vector) == 1.0 / first_relevant_rank(vector)

    def test_only_the_first_relevant_entry_counts(self):
        assert first_relevant_rank([0.0, 1.0, 1.0, 1.0]) == 2


class TestWriterRoundTrip:
    """Write with the real writer, read back with the real consumer parser."""

    def test_every_field_survives(self, tmp_path):
        acc = _accumulator(BASELINE_ROWS)
        path = acc.write_cases(str(tmp_path))
        assert os.path.basename(path) == RANK_CASES_FILENAME

        cases = consumer.parse_rank_cases(path)
        assert [c.hadm for c in cases] == IDS

        by_id = {c.hadm: c for c in cases}

        hit = by_id["999001"]
        assert (hit.fold, hit.status, hit.candidates, hit.rank) == (0, "answered", 50, 1)

        mid = by_id["999003"]
        assert (mid.status, mid.candidates, mid.rank) == ("answered", 50, 4)

        # Answered but nothing relevant: 50 candidates, no rank. The row that
        # would be indistinguishable from an abstention without the count column.
        missed = by_id["999004"]
        assert (missed.status, missed.candidates, missed.rank) == ("answered", 50, None)

        abstained = by_id["999005"]
        assert (abstained.status, abstained.candidates, abstained.rank) == \
            ("abstained", 0, None)

    def test_abstained_iff_the_vector_is_empty(self, tmp_path):
        """The status is derived, never asserted by the caller.

        A zero-length candidate list is the ONLY thing that makes a case an
        abstention; an all-zero vector of length 50 is an arm that answered and
        was wrong, and conflating the two hands the abstaining arm the exemption
        the whole population argument turns on.
        """
        acc = _accumulator([("999001", []), ("999002", _vector(None))])
        cases = consumer.parse_rank_cases(acc.write_cases(str(tmp_path)))
        statuses = {c.hadm: c.status for c in cases}
        assert statuses == {"999001": "abstained", "999002": "answered"}

    def test_cases_appear_in_stored_order_within_a_fold(self, tmp_path):
        reversed_rows = list(reversed(BASELINE_ROWS))
        acc = _accumulator(reversed_rows)
        cases = consumer.parse_rank_cases(acc.write_cases(str(tmp_path)))
        assert [c.hadm for c in cases] == [hadm for hadm, _ in reversed_rows]

    def test_folds_are_written_in_ascending_order(self, tmp_path):
        acc = RankAccumulator(DRG)
        for fold in (3, 0, 1):
            acc.begin_fold(fold, 1)
            acc.add(fold, "99900%d" % (fold + 1), _vector(1))
        cases = consumer.parse_rank_cases(acc.write_cases(str(tmp_path)))
        assert [c.fold for c in cases] == [0, 1, 3]

    def test_the_header_describes_the_format(self, tmp_path):
        acc = _accumulator(BASELINE_ROWS)
        text = open(acc.write_cases(str(tmp_path)), encoding="utf-8").read()
        header = [l for l in text.splitlines() if l.startswith("#")]
        assert any("format" in l and "first_relevant_rank" in l for l in header)
        assert any("k_fold" in l for l in header)
        assert any("DUA" in l for l in header)
        # Every non-comment line is a case row with five fields.
        rows = [l for l in text.splitlines() if l and not l.startswith("#")]
        assert len(rows) == len(BASELINE_ROWS)
        assert all(len(l.split("\t")) == 5 for l in rows)

    def test_a_malformed_row_raises_rather_than_being_skipped(self, tmp_path):
        path = tmp_path / RANK_CASES_FILENAME
        path.write_text("# header\n0\t999001\tanswered\t50\n", encoding="utf-8")
        with pytest.raises(ValueError, match="5 tab-separated fields"):
            consumer.parse_rank_cases(str(path))


class TestDeterminism:
    def test_two_writes_are_byte_identical(self, tmp_path):
        outputs = []
        for name in ("a", "b"):
            directory = tmp_path / name
            directory.mkdir()
            acc = _accumulator(BASELINE_ROWS)
            with open(acc.write_cases(str(directory)), "rb") as handle:
                outputs.append(handle.read())
        assert outputs[0] == outputs[1]

    def test_it_writes_unix_newlines_on_every_platform(self, tmp_path):
        """So a pod-produced record and a Windows-produced one are comparable."""
        acc = _accumulator(BASELINE_ROWS)
        with open(acc.write_cases(str(tmp_path)), "rb") as handle:
            assert b"\r\n" not in handle.read()


class TestRestrictedMRR:
    """The consumer's arithmetic, on a fixture computable by hand.

    Baseline RRs, in ITS stored order (999001 .. 999005):
        1.0, 0.5, 0.25, 0.0, 0.0(abstained)   -> sum 1.75, MRR 1.75/5 = 0.350000
    BERT RRs, in ITS stored order (999003, 999005, 999001, 999004, 999002):
        1.0, 1.0, 0.5, 0.25, 0.5              -> sum 3.25, MRR 3.25/5 = 0.650000

    The baseline answered four of the five; the fifth (999005) is a BERT rank-1.
    Restricted to those four the BERT arm loses it: 1.0 + 0.5 + 0.25 + 0.5 = 2.25,
    MRR 2.25/4 = 0.562500.

    Three numbers have to stay mutually DISTINCT or these tests assert nothing:
        2.25/4 = 0.562500   the answer -- BERT joined on (fold, HADM_ID)
        1.75/4 = 0.437500   the baseline's own restricted column
        2.75/4 = 0.687500   the first four rows of BERT_ROWS, i.e. a positional join
    """

    def _parse(self, tmp_path, rows, name):
        directory = tmp_path / name
        directory.mkdir()
        return consumer.parse_rank_cases(_accumulator(rows).write_cases(str(directory)))

    def test_unrestricted_mrr(self, tmp_path):
        baseline = self._parse(tmp_path, BASELINE_ROWS, "baseline")
        bert = self._parse(tmp_path, BERT_ROWS, "bert")
        assert consumer.mrr(baseline) == 1.75 / 5
        assert consumer.mrr(bert) == 3.25 / 5

    def test_restricting_to_the_baselines_answered_set(self, tmp_path):
        """The arithmetic, stated independently of the consumer's join.

        This test re-implements the restriction rather than calling it, so it
        pins what the answer IS, not that the shipped code computes it. The
        production join is exercised only by
        ``test_the_reported_section_prints_both_columns`` below -- which is why
        that test, not this one, carries the anti-mutation assertions.
        """
        baseline = self._parse(tmp_path, BASELINE_ROWS, "baseline")
        bert = self._parse(tmp_path, BERT_ROWS, "bert")

        answered = {(c.fold, c.hadm) for c in baseline if c.status == "answered"}
        assert len(answered) == 4

        restricted_baseline = [c for c in baseline if (c.fold, c.hadm) in answered]
        restricted_bert = [c for c in bert if (c.fold, c.hadm) in answered]
        assert len(restricted_bert) == 4

        # The baseline's restriction is its own answered population: same
        # numerator, denominator 4 instead of 5.
        assert consumer.mrr(restricted_baseline) == 1.75 / 4

        # The BERT arm LOSES its rank-1 on the case the baseline declined, so
        # restricting lowers it -- 2.25/4 against 3.25/5. A rescaling by coverage
        # would have raised it instead, which is the whole point of the fixture.
        assert consumer.mrr(restricted_bert) == 2.25 / 4
        assert consumer.mrr(restricted_bert) < consumer.mrr(bert)

        # The two arms' restricted columns are different numbers, so a reader --
        # or an assertion on stdout -- can tell whose row it is looking at.
        assert consumer.mrr(restricted_bert) != consumer.mrr(restricted_baseline)

        # A positional join would have taken BERT_ROWS[:4] instead. Distinct
        # from the real answer, which is what makes the join testable at all.
        assert consumer.mrr(bert[:len(answered)]) == 2.75 / 4

    def test_the_join_is_keyed_on_the_fold_too_not_the_hadm_id_alone(
            self, tmp_path, capsys):
        """Same admission, different fold -- the pair key must not match it.

        A bare HADM_ID key is indistinguishable from the pair on today's data,
        because an admission is a test case in exactly one fold; the consumer
        says so where it builds the key, and keeps the pair anyway. This is the
        fixture for the day that stops being true: an arm that scored 999001 in
        fold 1 must NOT be credited against a baseline that answered 999001 in
        fold 0, and the empty restriction has to surface rather than be absorbed
        into a plausible-looking number.
        """
        baseline = self._parse(tmp_path, [("999001", _vector(1))], "b-fold0")

        directory = tmp_path / "bert-fold1"
        directory.mkdir()
        acc = RankAccumulator(DRG)
        acc.begin_fold(1, 1)
        acc.add(1, "999001", _vector(1))
        bert = consumer.parse_rank_cases(acc.write_cases(str(directory)))

        answered = {(c.fold, c.hadm) for c in baseline if c.status == "answered"}
        assert [c for c in bert if (c.fold, c.hadm) in answered] == []

        consumer._self_selection({"baseline": baseline, "biomedbert": bert})
        out = capsys.readouterr().out
        assert "(no cases in one of the two sets)" in out

    def test_mrr_of_no_cases_is_none_not_zero(self):
        """An empty set is no measurement; 0.0 would assert it scored nothing."""
        assert consumer.mrr([]) is None

    def test_an_abstention_scores_zero_not_none(self, tmp_path):
        cases = self._parse(tmp_path, [("999005", [])], "only-abstention")
        assert consumer.mrr(cases) == 0.0

    def test_the_reported_section_prints_both_columns(self, tmp_path, capsys):
        """The ONLY test that runs the production join, so it is the picky one.

        ``_self_selection`` builds the restriction itself; every other test in
        this class re-implements it and therefore cannot see it. The four MRRs
        below are mutually distinct by construction, so the BiomedBERT restricted
        assertion cannot be satisfied by the baseline's printed row, and the
        two wrong joins each print a number that must NOT appear.
        """
        baseline = self._parse(tmp_path, BASELINE_ROWS, "baseline")
        bert = self._parse(tmp_path, BERT_ROWS, "bert")

        consumer._self_selection({"baseline": baseline, "biomedbert": bert})
        out = capsys.readouterr().out

        assert "THE SELF-SELECTION TEST" in out
        assert "%.6f" % (1.75 / 5) in out        # BioSentVec, all cases
        assert "%.6f" % (1.75 / 4) in out        # BioSentVec, restricted (the anchor)
        assert "%.6f" % (3.25 / 5) in out        # BiomedBERT, all cases
        assert "%.6f" % (2.25 / 4) in out        # BiomedBERT, restricted

        # A positional join prints 2.75/4 for BiomedBERT; joining every arm
        # against the BASELINE's rows prints 1.75/4 for it, which would leave
        # 2.25/4 absent. Neither wrong answer can hide behind a right one.
        assert "%.6f" % (2.75 / 4) not in out
        assert out.count("%.6f" % (1.75 / 4)) == 1

        # The delta is signed and negative: restricting COSTS the BERT arm its
        # rank-1 on the case the baseline declined.
        assert "%.6f" % (2.25 / 4 - 3.25 / 5) in out

        assert "the 4 cases BioSentVec ANSWERED" in out
        # The withdrawn quotient must stay withdrawn, and say so.
        assert "MRR_all-cases / coverage" in out


class TestConsumerDegradesOnPreP40Runs:
    """A run with no RankCases.txt warns and drops out of the new section only."""

    def _run_dir(self, root, key="baseline", stamp="01012026_00-00-00", cases=None):
        path = os.path.join(str(root), key, stamp)
        os.makedirs(path)
        for name in ("PerformanceIndex.txt", "RankMetrics.txt"):
            with open(os.path.join(path, name), "w", encoding="utf-8") as handle:
                handle.write("marker\n")
        if cases is not None:
            _accumulator(cases).write_cases(path)
        return path

    def test_missing_rank_cases_warns_and_keeps_the_aggregates(self, tmp_path, capsys):
        self._run_dir(tmp_path)
        found, case_files = consumer.discover(str(tmp_path))
        out = capsys.readouterr().out

        assert "[WARN] baseline: no RankCases.txt (pre-P40 run?)" in out
        assert case_files == {}
        # The existing analysis is untouched: RankMetrics.txt is still found.
        assert set(found) == {"baseline"}

    def test_a_p40_run_is_picked_up_without_a_warning(self, tmp_path, capsys):
        self._run_dir(tmp_path, cases=BASELINE_ROWS)
        found, case_files = consumer.discover(str(tmp_path))
        out = capsys.readouterr().out

        assert "no RankCases.txt" not in out
        assert set(case_files) == {"baseline"}
        assert set(found) == {"baseline"}

    def test_no_baseline_record_skips_the_section_without_crashing(self, tmp_path, capsys):
        bert = consumer.parse_rank_cases(
            _accumulator(BERT_ROWS).write_cases(str(tmp_path))
        )
        consumer._self_selection({"biomedbert": bert})
        out = capsys.readouterr().out
        assert "no answered set" in out.replace("\n", " ") or "Skipped" in out

    def test_no_records_at_all_skips_the_section(self, capsys):
        consumer._self_selection({})
        out = capsys.readouterr().out
        assert "predates P40" in out


class TestPopulationsUntouched:
    """P40 is additive: the winnable population is exactly what it was."""

    def test_the_seventy_six_pin_still_holds(self):
        winnable, tested = winnable_test_cases(DRG)
        assert (len(winnable), len(tested)) == (76, 129)

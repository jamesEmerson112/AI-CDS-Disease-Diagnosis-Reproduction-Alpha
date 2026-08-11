"""``aicds.analysis.performance_index`` vs the four parsers it replaces.

The four generations coexist for exactly one commit -- this one. Nothing in
``scripts/`` is touched here; the point is to prove, on the real files, that the
new parser reads the same numbers out of them as each of the four old ones does,
*before* the old ones are deleted. If a difference exists, it should surface
while both are still runnable side by side rather than as a mysteriously moved
number in a report a week later.

Read this file as three claims:

1. **Equivalence on real fixtures.** The three committed BERT runs under
   ``docs/Prediction_Output_*`` plus the trailer-stripped
   ``tests/golden/stub768`` golden. Every comparison is ``==``, never
   ``pytest.approx``: both sides call ``float()`` on the identical token, so the
   results are bit-identical, and a tolerance would hide precisely the failure
   worth catching -- a column read one position off, where P and R happen to be
   equal anyway (they are, in all 12,600 committed BERT rows; see
   ``docs/findings/04-metric-degeneracy.md``). Approximate equality would make
   that invisible.

2. **The defects are captured as facts, not fixed.** Two tests assert that
   ``scripts/analyze_performance.py`` behaves badly:
   :func:`test_analyze_performance_globals_merge_across_files` on its
   module-level accumulating state, and
   :func:`test_per_case_block_after_fold_aggregate_wins_in_old_parser` on its
   per-case mis-attribution. Pinning the old behaviour is what makes the
   replacement a decision rather than an accident.

3. **The baseline arm is covered without DUA data.** The baseline's
   ``PerformanceIndex.txt`` differs from the BERT arm's in ways a BERT-only
   fixture cannot exercise (tab-separated per-case rows carrying the
   *cumulative* confusion matrix, a fold-aggregate MAX header with no preceding
   blank line, a ``*`` banner between the k-FOLD blocks). No baseline output may
   be committed here, so :func:`_write_baseline_shaped_fixture` *generates* one
   at test time by driving the real writers in ``cython_utils``. Generated and
   not committed on purpose: a writer format change then breaks this test
   instead of breaking production.
"""

from __future__ import annotations

import importlib
import io
import os

import pytest

from aicds.analysis import performance_index as pindex
from aicds.utils import cython_utils as util_cy
from aicds.utils.Constants import (
    FP,
    FS,
    K_FOLD,
    P,
    PERFORMANCE_INDEX_HEADER,
    PR,
    R,
    TOP_K_INCR,
    TOP_K_LOWER_BOUND,
    TOP_K_UPPER_BOUND,
    TP,
)

import scripts.analyze_performance
import scripts.build_dashboard_data as build_dashboard_data
import scripts.build_readme_plots as build_readme_plots
import scripts.compare_models as compare_models


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The three committed BERT runs are the project's regression oracle, and the
# golden is the refactor safety net. All four are read-only here.
REAL_FIXTURES = [
    os.path.join(
        REPO_ROOT, "docs", "Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48",
        "PerformanceIndex.txt",
    ),
    os.path.join(
        REPO_ROOT, "docs", "Prediction_Output_BiomedBERT_15022026_12-03-36",
        "PerformanceIndex.txt",
    ),
    os.path.join(
        REPO_ROOT, "docs", "Prediction_Output_BlueBERT_15022026_12-24-38",
        "PerformanceIndex.txt",
    ),
    os.path.join(REPO_ROOT, "tests", "golden", "stub768", "PerformanceIndex.txt"),
]

FIXTURE_IDS = ["bio_clinical_bert", "biomedbert", "bluebert", "golden_stub768"]

# The threshold rows are emitted in *set-iteration* order from the literal
# {1, 0.9, 0.8, 0.7, 0.6}: 0.9, 1, 0.6, 0.8, 0.7. Floats are not hash-randomised,
# so this order is stable, and several assertions below depend on it being the
# write order rather than the sorted order.
WRITE_ORDER = [0.9, 1, 0.6, 0.8, 0.7]

STRATEGY_KEYS = ["MAX"] + [
    "TOP-%d" % k
    for k in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_analyze_performance():
    """Return ``scripts.analyze_performance`` with its module globals reset.

    Its parser does not return a fresh structure -- it accumulates into
    ``fold_data`` and ``aggregate_data``, two module-level ``defaultdict``s
    created at import. Two files parsed in one interpreter therefore merge.
    ``importlib.reload`` re-executes the module body, which rebinds both to empty
    defaultdicts, and is the only way to compare it per file.
    :func:`test_analyze_performance_globals_merge_across_files` asserts that the
    merge is real rather than taking it on trust.
    """
    return importlib.reload(scripts.analyze_performance)


def _as_legacy(aggregate):
    """``{strategy: {threshold: MetricRow}}`` -> the six-key legacy row dicts."""
    return {
        strategy: {t: row.as_dict() for t, row in rows.items()}
        for strategy, rows in aggregate.items()
    }


def _drop_pr(rows):
    """``analyze_performance`` never reads column 7, so compare without it."""
    return {key: value for key, value in rows.items() if key != "PR"}


def _rename_fs_to_f1(rows):
    """``build_dashboard_data`` calls the F-score column ``F1``.

    It is not an F1 -- across all 12,600 committed BERT rows precision == recall
    == F-score, so the column is accuracy (finding 04). The rename is asserted,
    not endorsed.
    """
    renamed = dict(rows)
    renamed["F1"] = renamed.pop("FS")
    return renamed


# ---------------------------------------------------------------------------
# 1. Structural facts about the real fixtures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", REAL_FIXTURES, ids=FIXTURE_IDS)
def test_real_fixture_parses_and_validates(path):
    index = pindex.read(path)
    pindex.validate(index)

    assert index.k_fold == 10
    assert sorted(index.folds) == list(range(10))
    # 129 admissions x 6 strategies. Counted, values discarded -- see trap 5 in
    # the module docstring.
    assert index.per_case_blocks == 774
    assert list(index.aggregate) == STRATEGY_KEYS
    for strategy in STRATEGY_KEYS:
        # Insertion order is the file's write order, not sorted order. If this
        # ever fails, the writers' set literal changed.
        assert list(index.aggregate[strategy]) == [float(t) for t in WRITE_ORDER]


def test_golden_has_no_trailer_and_that_is_legal():
    """``tests/golden/stub768`` is trailer-stripped by design (test_golden.py)."""
    index = pindex.read(REAL_FIXTURES[3])
    assert index.model is None
    assert index.timing == {}
    assert index.fold_times == []


def test_real_bert_fixtures_carry_a_ten_fold_trailer():
    for path in REAL_FIXTURES[:3]:
        index = pindex.read(path)
        assert index.model is not None
        assert len(index.fold_times) == 10
        # "Fold 0: 141.53 seconds (1.34 minutes)" matches the seconds pattern as
        # well as the named timings; fold_times is derived from those keys.
        assert index.timing["Fold 0"] == index.fold_times[0]
        assert "TOTAL EXECUTION TIME" in index.timing


# ---------------------------------------------------------------------------
# 2. Equivalence, one old parser at a time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", REAL_FIXTURES, ids=FIXTURE_IDS)
def test_matches_compare_models(path):
    """``compare_models`` reads only the six 10-FOLD blocks, strictly."""
    old = compare_models.parse_performance_index(path)
    new = pindex.read(path)

    # It normalises the header descriptor through _strategy_from_header before
    # keying; run the same normalisation on our keys rather than assuming the two
    # spellings coincide (they do, for MAX and TOP-K).
    normalised = {
        compare_models._strategy_from_header(strategy): rows
        for strategy, rows in _as_legacy(new.aggregate).items()
    }
    assert None not in normalised
    assert old == normalised

    # And its own validator agrees with ours about completeness.
    compare_models.validate_parse("equivalence", old)


@pytest.mark.parametrize("path", REAL_FIXTURES, ids=FIXTURE_IDS)
def test_matches_build_readme_plots(path):
    """``build_readme_plots`` adds the timing trailer and the model name."""
    old = build_readme_plots.parse_performance_file(path)
    new = pindex.read(path)

    assert old["metrics"] == _as_legacy(new.aggregate)
    assert old["timing"] == new.timing

    # Both clones fall back to the directory name when there is no MODEL: line,
    # which is the only reason the trailer-stripped golden yields a name at all.
    fallback = os.path.basename(os.path.dirname(path)).replace(
        "Prediction_Output_", ""
    )
    assert old["model"] == (new.model or fallback)
    assert old["path"] == new.path == path


@pytest.mark.parametrize("path", REAL_FIXTURES, ids=FIXTURE_IDS)
def test_matches_build_dashboard_data(path):
    """Same clone as build_readme_plots, but it calls the F-score ``F1``."""
    old = build_dashboard_data.parse_performance_file(path)
    new = pindex.read(path)

    expected = {
        strategy: {t: _rename_fs_to_f1(row) for t, row in rows.items()}
        for strategy, rows in _as_legacy(new.aggregate).items()
    }
    assert old["metrics"] == expected
    assert old["timing"] == new.timing

    fallback = os.path.basename(os.path.dirname(path)).replace(
        "Prediction_Output_", ""
    )
    assert old["model"] == (new.model or fallback)


@pytest.mark.parametrize("path", REAL_FIXTURES, ids=FIXTURE_IDS)
def test_matches_analyze_performance(path):
    """The only old parser that reads per-fold blocks -- and it agrees by luck.

    It attributes per-case rows to ``fold_data[fold][method]`` exactly as it
    attributes the fold aggregate, and the two agree on these files only because
    the aggregate happens to be written *after* all 129x6 per-case blocks and so
    overwrites them.
    :func:`test_per_case_block_after_fold_aggregate_wins_in_old_parser` shows
    what happens when that ordering does not hold.
    """
    module = _fresh_analyze_performance()
    old_folds, old_aggregate = module.parse_performance_index(path)
    new = pindex.read(path)

    expected_aggregate = {
        strategy: {t: _drop_pr(row) for t, row in rows.items()}
        for strategy, rows in _as_legacy(new.aggregate).items()
    }
    assert old_aggregate == expected_aggregate

    expected_folds = {
        fold: {
            strategy: {t: _drop_pr(row.as_dict()) for t, row in rows.items()}
            for strategy, rows in strategies.items()
        }
        for fold, strategies in new.folds.items()
    }
    assert old_folds == expected_folds


# ---------------------------------------------------------------------------
# 3. The defects, pinned as facts
# ---------------------------------------------------------------------------

def _minimal_file(fold_rows, aggregate_rows, fold=0, strategy="MAX", per_case_rows=None):
    """Build the smallest file that parses: one FOLD, one strategy, one 10-FOLD.

    ``per_case_rows``, when given, is written *after* the fold aggregate -- which
    is the inverted ordering the real writers never produce and which
    ``analyze_performance`` cannot survive.
    """
    out = io.StringIO()
    out.write("\n FOLD %d: LEN train: 8, LEN test: 2 \n" % fold)
    out.write("PERFORMANCE INDEX of %s SIMILARITY by MAX\n" % strategy)
    out.write(PERFORMANCE_INDEX_HEADER)
    for row in fold_rows:
        out.write("\t".join(str(v) for v in row) + "\n")
    if per_case_rows is not None:
        out.write(
            "0 - HADM_ID=999001: PERFORMANCE INDEX of %s SIMILARITY by MAX\n" % strategy
        )
        out.write(PERFORMANCE_INDEX_HEADER)
        for row in per_case_rows:
            out.write("\t".join(str(v) for v in row) + "\n")
    out.write("*" * 108 + "\n")
    # Labelled 10-FOLD even though one fold is present: both old clones hard-code
    # the literal "10-FOLD", so any other k would make them find nothing and the
    # comparison below would be vacuous.
    out.write("\n10-FOLD PERFORMANCE INDEX of %s SIMILARITY by MAX\n" % strategy)
    out.write(PERFORMANCE_INDEX_HEADER)
    for row in aggregate_rows:
        out.write("\t".join(str(v) for v in row) + "\n")
    return out.getvalue()


def _rows(tp, fp, value):
    """Five rows in write order, all carrying the same distinctive values."""
    return [[t, tp, fp, value, value, value, 1.0] for t in WRITE_ORDER]


def test_per_case_block_after_fold_aggregate_wins_in_old_parser(tmp_path):
    """THE non-cosmetic difference: per-case rows overwrite the fold aggregate.

    ``analyze_performance`` has no notion of a per-case block. Its header test is
    a substring check (``'PERFORMANCE INDEX of MAX SIMILARITY by MAX' in line``),
    which the per-case header satisfies, and the rows that follow land in
    ``fold_data[fold][method]`` on top of the fold aggregate. On the real files
    the aggregate is written last and wins, so nobody noticed. Reverse the order
    and the reported per-fold performance becomes one test case's single trial.

    The new parser counts the per-case block and discards its values, so the
    fold aggregate survives regardless of where the block sits.
    """
    text = _minimal_file(
        fold_rows=_rows(tp=7, fp=5, value=0.583333),
        aggregate_rows=_rows(tp=7.0, fp=5.0, value=0.583333),
        per_case_rows=_rows(tp=1, fp=0, value=1.0),
    )
    path = tmp_path / "PerformanceIndex.txt"
    path.write_text(text, encoding="utf-8")

    module = _fresh_analyze_performance()
    old_folds, _ = module.parse_performance_index(str(path))

    # The defect, asserted: the fold's TP is the per-case 1, not the aggregate 7.
    assert old_folds[0]["MAX"][0.9]["TP"] == 1.0
    assert old_folds[0]["MAX"][0.9]["FS"] == 1.0

    new = pindex.read(str(path))
    assert new.folds[0]["MAX"][0.9].tp == 7.0
    assert new.folds[0]["MAX"][0.9].f_score == 0.583333
    # Recognised and counted, never mixed into the numbers.
    assert new.per_case_blocks == 1


def test_analyze_performance_globals_merge_across_files(tmp_path):
    """Its module-level state merges two files; the new parser cannot.

    ``fold_data`` and ``aggregate_data`` are created once at import. The proof
    below is deliberately sharp: file B contains no ``MAX`` section at all, yet
    parsing B returns file A's ``MAX`` numbers, and reports a fold B never
    mentioned.
    """
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text(
        _minimal_file(_rows(3, 1, 0.75), _rows(3.0, 1.0, 0.75), fold=0, strategy="MAX"),
        encoding="utf-8",
    )
    file_b.write_text(
        _minimal_file(
            _rows(2, 2, 0.5), _rows(2.0, 2.0, 0.5), fold=5, strategy="TOP-10"
        ),
        encoding="utf-8",
    )

    module = _fresh_analyze_performance()
    module.parse_performance_index(str(file_a))
    folds_b, aggregate_b = module.parse_performance_index(str(file_b))

    # Contamination, asserted as fact: B's result carries A's strategy and fold.
    assert set(aggregate_b) == {"MAX", "TOP-10"}
    assert aggregate_b["MAX"][0.9]["TP"] == 3.0
    assert sorted(folds_b) == [0, 5]

    # The new parser returns a value per call; there is nothing to merge into.
    index_a = pindex.read(str(file_a))
    index_b = pindex.read(str(file_b))
    assert list(index_a.aggregate) == ["MAX"]
    assert list(index_b.aggregate) == ["TOP-10"]
    assert sorted(index_a.folds) == [0]
    assert sorted(index_b.folds) == [5]


# ---------------------------------------------------------------------------
# 4. Baseline-arm coverage, generated rather than committed
# ---------------------------------------------------------------------------

# Four invented admissions. The count matters: .githooks/pre-commit rejects any
# new file containing 20 or more distinct HADM_IDs, wherever it lives, because a
# path-only rule let the 129-ID golden through. Four is far under that gate --
# but the real reason this fixture is generated into tmp_path instead of
# committed is that a change to the writers should break THIS test rather than
# silently diverge from a frozen copy. The IDs are outside MIMIC-III's range and
# carry no clinical data.
SYNTHETIC_HADM_IDS = ["999001", "999002", "999003", "999004"]
SYNTHETIC_FOLDS = 3


def _hit(fold, strategy_index, threshold, case_index):
    """Deterministic TP/FP decision -- no RNG, no hashing, no seed dependence.

    ``MAX`` at threshold 1 is forced to always miss so the fixture exercises the
    zero branches in ``compute_performance_index``: ``tp == 0`` gives
    ``precision == 0`` and ``f_score == 0``, and the writer emits that as a bare
    ``0`` rather than ``0.0``.
    """
    if strategy_index == 0 and threshold == 1:
        return False
    return (fold + strategy_index + case_index + int(round(threshold * 10))) % 3 != 0


def _write_baseline_shaped_fixture(path):
    """Drive the REAL writers to produce a baseline-shaped PerformanceIndex.txt.

    Mirrors ``baseline_sent2vec.run_analysis`` exactly where the format is
    concerned:

    * the fold delimiter with its leading and trailing space (``:378``);
    * per-case blocks emitted through ``compute_performance_index`` on the
      *accumulating* confusion matrix (``cython_utils.py:312-318``), so row *i*
      is the running total after *i+1* cases -- unlike the BERT arm's
      single-trial rows;
    * the fold-aggregate ``MAX`` header at column 0 with **no** preceding blank
      line (``:429``), and each ``TOP-K`` header with one leading space (``:434``);
    * a ``*`` banner *between* the k-FOLD MAX block and the TOP-K blocks
      (``:450-458``), which the BERT arm does not write.

    Returns the expected values, computed independently of the parser.
    """
    nrow = len(SYNTHETIC_HADM_IDS)
    # Per (fold, strategy, threshold) final confusion counts, accumulated as the
    # writers accumulate them.
    final_counts = {}
    # The performance matrix the writers accumulate across folds; mirrored here
    # so the k-FOLD expectation is computed from arithmetic, not from the file.
    accumulated = {
        strategy: {t: [0.0] * 6 for t in WRITE_ORDER} for strategy in STRATEGY_KEYS
    }

    with open(path, "w", encoding="utf-8") as handle:
        matrices_by_fold = {}
        performance_matrices = {
            "MAX": util_cy.init_performance_matrix(),
        }
        for k in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
            performance_matrices["TOP-%d" % k] = util_cy.init_performance_matrix()

        for fold in range(SYNTHETIC_FOLDS):
            handle.write(
                "\n FOLD %s: LEN train: %s, LEN test: %s \n"
                % (fold, 12 - nrow, nrow)
            )
            confusion = {
                strategy: util_cy.init_confusion_matrix() for strategy in STRATEGY_KEYS
            }
            matrices_by_fold[fold] = confusion

            for case_index, hadm_id in enumerate(SYNTHETIC_HADM_IDS):
                for strategy_index, strategy in enumerate(STRATEGY_KEYS):
                    for threshold in WRITE_ORDER:
                        values = confusion[strategy].get(threshold)
                        if _hit(fold, strategy_index, threshold, case_index):
                            values[TP] += 1
                        else:
                            values[FP] += 1
                    handle.write(
                        "%d - HADM_ID=%s: PERFORMANCE INDEX of %s SIMILARITY by MAX\n"
                        % (case_index, hadm_id, strategy)
                    )
                    util_cy.compute_performance_index(
                        confusion[strategy], nrow, handle
                    )

            handle.write("PERFORMANCE INDEX of MAX SIMILARITY by MAX\n")
            util_cy.compute_aggregated_performance_index(
                confusion["MAX"], performance_matrices["MAX"], nrow, handle
            )
            for k in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
                strategy = "TOP-%d" % k
                handle.write(
                    "\n PERFORMANCE INDEX of TOP-%d SIMILARITY by MAX\n" % k
                )
                util_cy.compute_aggregated_performance_index(
                    confusion[strategy], performance_matrices[strategy], nrow, handle
                )

            for strategy in STRATEGY_KEYS:
                for threshold in WRITE_ORDER:
                    values = confusion[strategy].get(threshold)
                    final_counts[(fold, strategy, threshold)] = (
                        values[TP],
                        values[FP],
                    )

        handle.write("*" * 108 + "\n")
        handle.write(
            "\n%d-FOLD PERFORMANCE INDEX of MAX SIMILARITY by MAX\n" % SYNTHETIC_FOLDS
        )
        util_cy.print_performance_index(performance_matrices["MAX"], handle)
        handle.write("*" * 108 + "\n")
        for k in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR):
            handle.write(
                "\n%d-FOLD PERFORMANCE INDEX of TOP-%d SIMILARITY by MAX\n"
                % (SYNTHETIC_FOLDS, k)
            )
            util_cy.print_performance_index(
                performance_matrices["TOP-%d" % k], handle
            )
        handle.write("*" * 108 + "\n")

    # Independent expectation: the same formulas as compute_performance_index,
    # written out here rather than called, over the counts chosen by _hit.
    expected_folds = {}
    for fold in range(SYNTHETIC_FOLDS):
        expected_folds[fold] = {}
        for strategy in STRATEGY_KEYS:
            expected_folds[fold][strategy] = {}
            for threshold in WRITE_ORDER:
                tp, fp = final_counts[(fold, strategy, threshold)]
                precision = tp / (tp + fp) if (tp + fp) != 0 else 0
                recall = tp / nrow
                f_score = (
                    (2 * recall * precision) / (recall + precision)
                    if recall + precision != 0
                    else 0
                )
                prediction_rate = (tp + fp) / nrow
                expected_folds[fold][strategy][float(threshold)] = (
                    tp,
                    fp,
                    precision,
                    recall,
                    f_score,
                    prediction_rate,
                )
                slot = accumulated[strategy][threshold]
                slot[TP] += tp
                slot[FP] += fp
                slot[P] += precision
                slot[R] += recall
                slot[FS] += f_score
                slot[PR] += prediction_rate

    # print_performance_index divides by the K_FOLD *constant*, not by the number
    # of folds that ran. For a 3-fold file that makes the k-FOLD block
    # arithmetically wrong (each value is a tenth of the sum, not a third). That
    # is a writer defect; the parser reports it faithfully and this expectation
    # reproduces it rather than correcting it.
    expected_aggregate = {
        strategy: {
            float(threshold): tuple(v / K_FOLD for v in accumulated[strategy][threshold])
            for threshold in WRITE_ORDER
        }
        for strategy in STRATEGY_KEYS
    }
    return expected_folds, expected_aggregate


def test_baseline_shaped_fixture_round_trips(tmp_path):
    path = str(tmp_path / "PerformanceIndex.txt")
    expected_folds, expected_aggregate = _write_baseline_shaped_fixture(path)

    index = pindex.read(path)
    pindex.validate(
        index, strategies=STRATEGY_KEYS, k_fold=SYNTHETIC_FOLDS
    )

    # k_fold comes from the header, not from an assumption of 10.
    assert index.k_fold == SYNTHETIC_FOLDS
    assert sorted(index.folds) == list(range(SYNTHETIC_FOLDS))
    # 3 folds x 4 cases x 6 strategies, counted and discarded.
    assert index.per_case_blocks == SYNTHETIC_FOLDS * len(SYNTHETIC_HADM_IDS) * len(
        STRATEGY_KEYS
    )
    assert index.model is None and index.timing == {}

    for fold in range(SYNTHETIC_FOLDS):
        for strategy in STRATEGY_KEYS:
            for threshold, expected in expected_folds[fold][strategy].items():
                row = index.folds[fold][strategy][threshold]
                assert (
                    row.tp,
                    row.fp,
                    row.precision,
                    row.recall,
                    row.f_score,
                    row.prediction_rate,
                ) == expected

    for strategy in STRATEGY_KEYS:
        for threshold, expected in expected_aggregate[strategy].items():
            row = index.aggregate[strategy][threshold]
            assert (
                row.tp,
                row.fp,
                row.precision,
                row.recall,
                row.f_score,
                row.prediction_rate,
            ) == expected


def test_baseline_shaped_fixture_exercises_the_zero_branches(tmp_path):
    """MAX at threshold 1 never hits, so P, R and FS are all zero there.

    The writer prints that F-score as a bare ``0``; ``float("0")`` and
    ``float("0.0")`` are the same value, which is why the parser can normalise it
    while the golden comparison must not.
    """
    path = str(tmp_path / "PerformanceIndex.txt")
    _write_baseline_shaped_fixture(path)
    text = open(path, encoding="utf-8").read()
    # The fold-aggregate MAX row at threshold 1: tp=0, fp=4 over nrow=4. Note the
    # threshold prints as a bare "1" (it comes from the set literal) and the
    # F-score as a bare "0" while P and R print "0.0" -- str() of an int 0 versus
    # a float division. That mixture is exactly what the golden comparison is
    # sensitive to and what this parser is allowed to normalise away.
    assert "\n1\t0\t4\t0.0\t0.0\t0\t1.0\n" in text

    index = pindex.read(path)
    row = index.folds[0]["MAX"][1.0]
    assert (row.tp, row.fp, row.precision, row.recall, row.f_score) == (
        0.0, 4.0, 0.0, 0.0, 0.0,
    )
    # The bare 1 in the aggregate rows and the 1.0 the BERT arm writes per case
    # are the same key.
    assert 1.0 in index.aggregate["MAX"] and 1 in index.aggregate["MAX"]


def test_old_parsers_find_nothing_in_a_three_fold_file(tmp_path):
    """All four old parsers hard-code the literal ``10-FOLD``.

    Not a formatting nit: a run with a different K -- which is what any future
    fold-count experiment produces -- reads as an empty file to every one of
    them, silently. The new parser takes K from the header.
    """
    path = str(tmp_path / "PerformanceIndex.txt")
    _write_baseline_shaped_fixture(path)

    assert compare_models.parse_performance_index(path) == {}
    assert build_readme_plots.parse_performance_file(path)["metrics"] == {}
    assert build_dashboard_data.parse_performance_file(path)["metrics"] == {}

    module = _fresh_analyze_performance()
    _, old_aggregate = module.parse_performance_index(path)
    assert old_aggregate == {}

    assert pindex.read(path).k_fold == 3


# ---------------------------------------------------------------------------
# 5. Failure modes -- every one of them is a whitespace failure
# ---------------------------------------------------------------------------

def _error(text):
    with pytest.raises(pindex.PerformanceIndexError) as caught:
        pindex.parse(io.StringIO(text), path="fixture.txt")
    return caught.value


def test_fold_aggregate_without_a_fold_header_raises():
    error = _error(
        "PERFORMANCE INDEX of MAX SIMILARITY by MAX\n" + PERFORMANCE_INDEX_HEADER
    )
    assert error.lineno == 1
    assert error.path == "fixture.txt"
    assert "no preceding FOLD header" in str(error)


def test_fold_delimiter_needs_its_trailing_space():
    """Drop the trailing space and the delimiter stops matching.

    The consequence is not a parse of nine folds instead of ten -- it is the
    ``no preceding FOLD header`` error above, which is the whole reason that
    error exists.
    """
    text = (
        " FOLD 0: LEN train: 8, LEN test: 2\n"  # trailing space removed
        "PERFORMANCE INDEX of MAX SIMILARITY by MAX\n"
    )
    assert "no preceding FOLD header" in str(_error(text))


def test_topk_fold_aggregate_leading_space_is_accepted_and_optional():
    """Both spellings are fold aggregates; neither may be privileged."""
    text = (
        " FOLD 0: LEN train: 8, LEN test: 2 \n"
        "PERFORMANCE INDEX of MAX SIMILARITY by MAX\n"
        + PERFORMANCE_INDEX_HEADER
        + "0.9\t1\t3\t0.25\t0.25\t0.25\t1.0\n"
        "\n PERFORMANCE INDEX of TOP-10 SIMILARITY by MAX\n"
        + PERFORMANCE_INDEX_HEADER
        + "0.9\t2\t2\t0.5\t0.5\t0.5\t1.0\n"
        "\n1-FOLD PERFORMANCE INDEX of MAX SIMILARITY by MAX\n"
        + PERFORMANCE_INDEX_HEADER
        + "0.9\t1\t3\t0.25\t0.25\t0.25\t1.0\n"
    )
    index = pindex.parse(io.StringIO(text))
    assert sorted(index.folds[0]) == ["MAX", "TOP-10"]
    assert index.k_fold == 1


def test_eight_token_row_raises_rather_than_being_truncated():
    text = (
        " FOLD 0: LEN train: 8, LEN test: 2 \n"
        "PERFORMANCE INDEX of MAX SIMILARITY by MAX\n"
        + PERFORMANCE_INDEX_HEADER
        + "0.9\t1\t3\t0.25\t0.25\t0.25\t1.0\t9.9\n"
    )
    error = _error(text)
    assert "got 8" in str(error)
    assert repr("0.9\t1\t3\t0.25\t0.25\t0.25\t1.0\t9.9") in str(error)


def test_unknown_performance_index_shape_raises():
    error = _error(
        " FOLD 0: LEN train: 8, LEN test: 2 \n"
        "  PERFORMANCE INDEX of MAX SIMILARITY by MEAN\n"
    )
    assert "matches none of the three header shapes" in str(error)


def test_truncated_file_with_no_aggregate_still_parses():
    """The k-FOLD block is written *last*, so requiring it rejects live runs.

    In the committed BiomedBERT file the ``10-FOLD PERFORMANCE INDEX of MAX``
    header sits at line 5921 of 6000. A run that died in fold 6 has every fold
    block it managed to write and no aggregate at all -- which is exactly the
    file you want to read. ``parse`` must return it; only :func:`pindex.validate`
    is allowed to call it incomplete.
    """
    text = (
        " FOLD 0: LEN train: 8, LEN test: 2 \n"
        "PERFORMANCE INDEX of MAX SIMILARITY by MAX\n"
        + PERFORMANCE_INDEX_HEADER
        + "0.9\t1\t3\t0.25\t0.25\t0.25\t1.0\n"
    )
    index = pindex.parse(io.StringIO(text), path="truncated.txt")
    assert sorted(index.folds) == [0]
    assert index.aggregate == {}
    assert index.k_fold == 0

    with pytest.raises(pindex.PerformanceIndexError) as caught:
        pindex.validate(index)
    assert caught.value.lineno is None


@pytest.mark.parametrize(
    "kind, text",
    [
        ("garbage", "hello\nworld\n"),
        (
            "per-case only",
            " FOLD 0: LEN train: 116, LEN test: 13 \n"
            "0 - HADM_ID=124073: PERFORMANCE INDEX of MAX SIMILARITY by MAX\n"
            + PERFORMANCE_INDEX_HEADER
            + "1.0\t1\t0\t1.0\t1.0\t1.0\t1.0\n",
        ),
    ],
    ids=["garbage", "per_case_only"],
)
def test_file_with_neither_aggregate_nor_fold_block_raises(kind, text):
    """The floor is one *aggregate* block of either kind.

    Per-case rows are counted and dropped (trap 5), so a per-case-only prefix
    leaves ``folds`` empty and is rejected alongside outright garbage. Reported
    at end of parse, hence no line number.
    """
    error = _error(text)
    assert error.lineno is None
    assert "no k-FOLD aggregate and no fold-aggregate section" in str(error)


def test_repeated_ten_fold_strategy_raises():
    block = (
        "10-FOLD PERFORMANCE INDEX of MAX SIMILARITY by MAX\n"
        + PERFORMANCE_INDEX_HEADER
        + "0.9\t1\t3\t0.25\t0.25\t0.25\t1.0\n\n"
    )
    assert "repeated 10-FOLD section" in str(_error(block + block))


def test_validate_is_separate_from_parse():
    """A file that parses may still be incomplete; only validate() says so."""
    text = (
        " FOLD 0: LEN train: 8, LEN test: 2 \n"
        "PERFORMANCE INDEX of MAX SIMILARITY by MAX\n"
        + PERFORMANCE_INDEX_HEADER
        + "0.9\t1\t3\t0.25\t0.25\t0.25\t1.0\n"
        "\n10-FOLD PERFORMANCE INDEX of MAX SIMILARITY by MAX\n"
        + PERFORMANCE_INDEX_HEADER
        + "0.9\t1\t3\t0.25\t0.25\t0.25\t1.0\n"
    )
    index = pindex.parse(io.StringIO(text), path="truncated.txt")
    assert list(index.aggregate) == ["MAX"]

    with pytest.raises(pindex.PerformanceIndexError) as caught:
        pindex.validate(index)
    assert "missing strategies" in str(caught.value)
    assert caught.value.path == "truncated.txt"
    assert caught.value.lineno is None


def test_crlf_and_lf_parse_identically():
    path = REAL_FIXTURES[3]
    crlf = open(path, "rb").read().decode("utf-8")
    assert "\r\n" in crlf
    assert pindex.parse(io.StringIO(crlf)).aggregate == pindex.parse(
        io.StringIO(crlf.replace("\r\n", "\n"))
    ).aggregate


def test_metric_row_is_frozen_and_hashable():
    row = pindex.MetricRow(1.0, 2.0, 3.0, 0.4, 0.5, 0.6, 0.7)
    with pytest.raises(Exception):
        row.tp = 9.0
    assert row.as_dict() == {
        "TP": 2.0, "FP": 3.0, "P": 0.4, "R": 0.5, "FS": 0.6, "PR": 0.7,
    }
    assert "threshold" not in row.as_dict()

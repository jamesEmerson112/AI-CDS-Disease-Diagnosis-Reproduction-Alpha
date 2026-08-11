"""``aicds.analysis.performance_index``, pinned against the committed runs.

This file replaces ``tests/test_performance_index_equivalence.py``, which compared
the parser against the four hand-rolled ones in ``scripts/`` it was written to
replace. That comparison was a **migration property**: worth proving exactly once,
at the commit where five parsers coexisted (C6, ``1f69e11``), and unprovable
afterwards because the four are now deleted. Deleting the test with them would
have thrown away the coverage too, so the numbers it derived are pinned here as
literals instead. The claim changes from "the new parser agrees with the old ones"
to "the parser reads *these numbers* out of *these files*", which is the claim that
still has teeth once the old code is gone.

The literals are not self-certifying, so they carry a second, independent check.
**Three of them cross-check ``compare_models.SANITY_CHECKS_BY_PIPELINE["legacy"]``**
-- the TOP-10 threshold-1.0 F-scores 0.2853 / 0.2545 / 0.2391 for
Bio_ClinicalBERT / BiomedBERT / BlueBERT. Those expectations were read off the
*2026-08-05 RunPod Linux* runs under ``results/`` and hand-checked; the fixtures
here are the *2026-02-15 M-series Mac* runs under ``docs/``. Two different machines,
two different transcription paths, same four decimals -- which is also the
bit-for-bit cross-platform reproduction the README claims, asserted rather than
asserted-about. :func:`test_three_literals_cross_check_the_sanity_table` is that
check, and it fails loudly if either table is edited alone.

Read the rest as three claims:

1. **Pinned values on real fixtures.** Every comparison is ``==``, never
   ``pytest.approx``. A tolerance would hide the one failure worth catching -- a
   column read one position off, where P and R happen to be equal anyway. They
   are equal, in all 12,600 committed BERT rows
   (``docs/findings/04-metric-degeneracy.md``), which is why
   :func:`test_pinned_rows_are_degenerate` states that as a property of the
   literals rather than leaving it as an accident nobody would notice.

2. **A deleted parser's defect, kept as a shape the new one survives.**
   :func:`test_per_case_block_after_fold_aggregate_does_not_win` builds the file
   ordering that broke ``scripts/analyze_performance.py``: per-case rows written
   *after* the fold aggregate. Only the new parser's behaviour is asserted now,
   because the old one no longer exists to compare against -- but the fixture is
   the interesting artifact and it outlives the parser that failed it.

3. **The baseline arm is covered without DUA data.** The baseline's
   ``PerformanceIndex.txt`` differs from the BERT arm's in ways a BERT-only
   fixture cannot exercise (tab-separated per-case rows carrying the *cumulative*
   confusion matrix, a fold-aggregate MAX header with no preceding blank line, a
   ``*`` banner between the k-FOLD blocks). No baseline output may be committed
   here, so :func:`_write_baseline_shaped_fixture` *generates* one at test time by
   driving the real writers in ``cython_utils``. Generated and not committed on
   purpose: a writer format change then breaks this test instead of breaking
   production.
"""

from __future__ import annotations

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

import scripts.compare_models as compare_models


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The three committed BERT runs are the project's regression oracle, and the
# golden is the refactor safety net. All four are read-only here.
REAL_FIXTURES = {
    "bio_clinical_bert": os.path.join(
        REPO_ROOT, "docs", "Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48",
        "PerformanceIndex.txt",
    ),
    "biomedbert": os.path.join(
        REPO_ROOT, "docs", "Prediction_Output_BiomedBERT_15022026_12-03-36",
        "PerformanceIndex.txt",
    ),
    "bluebert": os.path.join(
        REPO_ROOT, "docs", "Prediction_Output_BlueBERT_15022026_12-24-38",
        "PerformanceIndex.txt",
    ),
    "golden_stub768": os.path.join(
        REPO_ROOT, "tests", "golden", "stub768", "PerformanceIndex.txt"
    ),
}

FIXTURE_IDS = list(REAL_FIXTURES)

# The threshold rows are emitted in *set-iteration* order from the literal
# {1, 0.9, 0.8, 0.7, 0.6}: 0.9, 1, 0.6, 0.8, 0.7. Floats are not hash-randomised,
# so this order is stable, and several assertions below depend on it being the
# write order rather than the sorted order -- including the key order of every
# inner dict in EXPECTED_AGGREGATE.
WRITE_ORDER = [0.9, 1, 0.6, 0.8, 0.7]

STRATEGY_KEYS = ["MAX"] + [
    "TOP-%d" % k
    for k in range(TOP_K_LOWER_BOUND, TOP_K_UPPER_BOUND, TOP_K_INCR)
]

# The three timing keys pinned per fixture, chosen to be three *different shapes*
# rather than three arbitrary lines: a named phase, a per-fold entry that matches
# the seconds pattern only incidentally (trap 7 -- it is where `fold_times` comes
# from), and the all-caps total.
TIMING_KEYS = ("Model Loading", "Fold 0", "TOTAL EXECUTION TIME")

# ---------------------------------------------------------------------------
# The pinned numbers: 6 strategies x 5 thresholds x (P, R, FS, PR) per fixture.
#
# Generated by reading the four fixtures with this parser at C6 (1f69e11), the
# commit at which it was proven equal to all four parsers it replaced, and frozen
# here. Full float repr, not rounded: these are exactly the values float() returns
# for the tokens in the files, so == is the right comparison and any difference is
# a real one. DO NOT regenerate these to make a failing test pass -- a changed
# number means the parser changed, and that is the finding.
# ---------------------------------------------------------------------------

EXPECTED_AGGREGATE = {
    "bio_clinical_bert": {
        "MAX": {
            0.9: (0.2852564102564103, 0.2852564102564103, 0.2852564102564103, 1.0),
            1.0: (0.14615384615384616, 0.14615384615384616, 0.14615384615384616, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (0.9692307692307693, 0.9692307692307693, 0.9692307692307693, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-10": {
            0.9: (0.7974358974358975, 0.7974358974358975, 0.7974358974358975, 1.0),
            1.0: (0.28525641025641024, 0.28525641025641024, 0.28525641025641024, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (1.0, 1.0, 1.0, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-20": {
            0.9: (0.9064102564102564, 0.9064102564102564, 0.9064102564102564, 1.0),
            1.0: (0.33141025641025645, 0.33141025641025645, 0.33141025641025645, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (1.0, 1.0, 1.0, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-30": {
            0.9: (0.9301282051282052, 0.9301282051282052, 0.9301282051282052, 1.0),
            1.0: (0.3628205128205128, 0.3628205128205128, 0.3628205128205128, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (1.0, 1.0, 1.0, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-40": {
            0.9: (0.9455128205128206, 0.9455128205128206, 0.9455128205128206, 1.0),
            1.0: (0.3935897435897436, 0.3935897435897436, 0.3935897435897436, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (1.0, 1.0, 1.0, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-50": {
            0.9: (0.9532051282051283, 0.9532051282051283, 0.9532051282051283, 1.0),
            1.0: (0.4012820512820513, 0.4012820512820513, 0.4012820512820513, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (1.0, 1.0, 1.0, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
    },
    "biomedbert": {
        "MAX": {
            0.9: (1.0, 1.0, 1.0, 1.0),
            1.0: (0.16923076923076924, 0.16923076923076924, 0.16923076923076924, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (1.0, 1.0, 1.0, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-10": {
            0.9: (1.0, 1.0, 1.0, 1.0),
            1.0: (0.2544871794871795, 0.2544871794871795, 0.2544871794871795, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (1.0, 1.0, 1.0, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-20": {
            0.9: (1.0, 1.0, 1.0, 1.0),
            1.0: (0.32371794871794873, 0.32371794871794873, 0.32371794871794873, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (1.0, 1.0, 1.0, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-30": {
            0.9: (1.0, 1.0, 1.0, 1.0),
            1.0: (0.3397435897435897, 0.3397435897435897, 0.3397435897435897, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (1.0, 1.0, 1.0, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-40": {
            0.9: (1.0, 1.0, 1.0, 1.0),
            1.0: (0.37051282051282053, 0.37051282051282053, 0.37051282051282053, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (1.0, 1.0, 1.0, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-50": {
            0.9: (1.0, 1.0, 1.0, 1.0),
            1.0: (0.37820512820512825, 0.37820512820512825, 0.37820512820512825, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (1.0, 1.0, 1.0, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
    },
    "bluebert": {
        "MAX": {
            0.9: (0.19358974358974362, 0.19358974358974362, 0.19358974358974362, 1.0),
            1.0: (0.17756410256410254, 0.17756410256410254, 0.17756410256410254, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (0.3948717948717949, 0.3948717948717949, 0.3948717948717949, 1.0),
            0.7: (0.9070512820512822, 0.9070512820512822, 0.9070512820512822, 1.0),
        },
        "TOP-10": {
            0.9: (0.3403846153846154, 0.3403846153846154, 0.3403846153846154, 1.0),
            1.0: (0.23910256410256409, 0.23910256410256409, 0.23910256410256409, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (0.8134615384615383, 0.8134615384615383, 0.8134615384615383, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-20": {
            0.9: (0.4865384615384616, 0.4865384615384616, 0.4865384615384616, 1.0),
            1.0: (0.3391025641025641, 0.3391025641025641, 0.3391025641025641, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (0.9070512820512819, 0.9070512820512819, 0.9070512820512819, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-30": {
            0.9: (0.5173076923076924, 0.5173076923076924, 0.5173076923076924, 1.0),
            1.0: (0.3467948717948718, 0.3467948717948718, 0.3467948717948718, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (0.9301282051282049, 0.9301282051282049, 0.9301282051282049, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-40": {
            0.9: (0.541025641025641, 0.541025641025641, 0.541025641025641, 1.0),
            1.0: (0.37820512820512825, 0.37820512820512825, 0.37820512820512825, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (0.9455128205128205, 0.9455128205128205, 0.9455128205128205, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
        "TOP-50": {
            0.9: (0.5724358974358974, 0.5724358974358974, 0.5724358974358974, 1.0),
            1.0: (0.4173076923076923, 0.4173076923076923, 0.4173076923076923, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (0.9615384615384617, 0.9615384615384617, 0.9615384615384617, 1.0),
            0.7: (1.0, 1.0, 1.0, 1.0),
        },
    },
    "golden_stub768": {
        "MAX": {
            0.9: (0.13846153846153847, 0.13846153846153847, 0.13846153846153847, 1.0),
            1.0: (0.13846153846153847, 0.13846153846153847, 0.13846153846153847, 1.0),
            0.6: (0.8737179487179487, 0.8737179487179487, 0.8737179487179487, 1.0),
            0.8: (0.15384615384615385, 0.15384615384615385, 0.15384615384615385, 1.0),
            0.7: (0.5871794871794871, 0.5871794871794871, 0.5871794871794871, 1.0),
        },
        "TOP-10": {
            0.9: (0.2621794871794872, 0.2621794871794872, 0.2621794871794872, 1.0),
            1.0: (0.2621794871794872, 0.2621794871794872, 0.2621794871794872, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (0.3083333333333333, 0.3083333333333333, 0.3083333333333333, 1.0),
            0.7: (0.9846153846153847, 0.9846153846153847, 0.9846153846153847, 1.0),
        },
        "TOP-20": {
            0.9: (0.3467948717948718, 0.3467948717948718, 0.3467948717948718, 1.0),
            1.0: (0.3467948717948718, 0.3467948717948718, 0.3467948717948718, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (0.4012820512820513, 0.4012820512820513, 0.4012820512820513, 1.0),
            0.7: (0.9923076923076923, 0.9923076923076923, 0.9923076923076923, 1.0),
        },
        "TOP-30": {
            0.9: (0.36987179487179483, 0.36987179487179483, 0.36987179487179483, 1.0),
            1.0: (0.36987179487179483, 0.36987179487179483, 0.36987179487179483, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (0.4397435897435898, 0.4397435897435898, 0.4397435897435898, 1.0),
            0.7: (0.9923076923076923, 0.9923076923076923, 0.9923076923076923, 1.0),
        },
        "TOP-40": {
            0.9: (0.40064102564102566, 0.40064102564102566, 0.40064102564102566, 1.0),
            1.0: (0.40064102564102566, 0.40064102564102566, 0.40064102564102566, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (0.4782051282051283, 0.4782051282051283, 0.4782051282051283, 1.0),
            0.7: (0.9923076923076923, 0.9923076923076923, 0.9923076923076923, 1.0),
        },
        "TOP-50": {
            0.9: (0.4320512820512821, 0.4320512820512821, 0.4320512820512821, 1.0),
            1.0: (0.4320512820512821, 0.4320512820512821, 0.4320512820512821, 1.0),
            0.6: (1.0, 1.0, 1.0, 1.0),
            0.8: (0.5096153846153846, 0.5096153846153846, 0.5096153846153846, 1.0),
            0.7: (0.9923076923076923, 0.9923076923076923, 0.9923076923076923, 1.0),
        },
    },
}

# (model, (Model Loading, Fold 0, TOTAL EXECUTION TIME)). The golden is
# trailer-stripped by design -- test_golden.py cuts everything below the 80-'='
# banner -- so all four of its entries are None, and that is the pinned fact.
EXPECTED_TRAILER = {
    "bio_clinical_bert": ("Bio_ClinicalBERT", (81.32, 141.53, 1318.82)),
    "biomedbert": ("BiomedBERT", (15.31, 152.21, 1267.33)),
    "bluebert": ("BlueBERT", (11.12, 133.36, 1241.36)),
    "golden_stub768": (None, (None, None, None)),
}


# ---------------------------------------------------------------------------
# 1. The pinned values, and their independent cross-check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURE_IDS)
def test_aggregate_rows_match_the_pinned_literals(fixture):
    index = pindex.read(REAL_FIXTURES[fixture])
    expected = EXPECTED_AGGREGATE[fixture]

    # Strategy order is the file's write order, and so is the threshold order
    # inside each strategy. Comparing the key lists as well as the values is what
    # makes this a check on the parser's *structure*, not just its arithmetic.
    assert list(index.aggregate) == STRATEGY_KEYS == list(expected)

    for strategy in STRATEGY_KEYS:
        rows = index.aggregate[strategy]
        assert list(rows) == [float(t) for t in WRITE_ORDER]
        assert list(rows) == list(expected[strategy])
        for threshold, want in expected[strategy].items():
            row = rows[threshold]
            # == not approx: both sides are float() of the same token.
            assert (
                row.precision, row.recall, row.f_score, row.prediction_rate
            ) == want


def test_three_literals_cross_check_the_sanity_table():
    """The literals above vs ``compare_models.SANITY_CHECKS_BY_PIPELINE["legacy"]``.

    Two independent derivations meet here. That table was transcribed by hand from
    the 2026-08-05 RunPod Linux runs under ``results/``; ``EXPECTED_AGGREGATE`` was
    generated by this parser from the 2026-02-15 M-series Mac runs under ``docs/``.
    Agreement to four decimals is therefore two things at once: evidence the
    literals are right, and the cross-platform bit-for-bit reproduction the README
    claims for the three BERT arms.

    The table's two ``baseline`` entries are skipped because no baseline run is
    committed anywhere in this repository, and none may be -- the arm is covered by
    the generated fixture in section 3 instead. ``checked == 3`` is what stops that
    skip from quietly swallowing the whole test.
    """
    checked = 0
    for key, strategy, threshold, expected in (
        compare_models.SANITY_CHECKS_BY_PIPELINE["legacy"]
    ):
        if key not in EXPECTED_AGGREGATE:
            assert key == "baseline"
            continue
        precision, recall, f_score, _rate = EXPECTED_AGGREGATE[key][strategy][threshold]
        assert round(precision, 4) == expected["P"]
        assert round(recall, 4) == expected["R"]
        assert round(f_score, 4) == expected["FS"]
        checked += 1
    assert checked == 3


@pytest.mark.parametrize("fixture", FIXTURE_IDS)
def test_pinned_rows_are_degenerate(fixture):
    """P == R == FS in all 30 aggregate rows of all four fixtures, exactly.

    Stated as a property of the pinned numbers rather than left implicit, for two
    reasons. It is the finding (``docs/findings/04-metric-degeneracy.md``): these
    arms never abstain, so ``tp + fp == n``, precision reduces to ``tp/n`` -- which
    is recall -- and the harmonic mean is that same value. And it is precisely why
    every comparison above is ``==``: with three columns carrying one value, a
    parser that read them one position off would produce identical output under any
    tolerance.
    """
    for strategy, rows in EXPECTED_AGGREGATE[fixture].items():
        for threshold, (precision, recall, f_score, rate) in rows.items():
            assert precision == recall == f_score, (strategy, threshold)
            assert rate == 1.0, (strategy, threshold)


@pytest.mark.parametrize("fixture", FIXTURE_IDS)
def test_trailer_matches_the_pinned_literals(fixture):
    index = pindex.read(REAL_FIXTURES[fixture])
    model, timings = EXPECTED_TRAILER[fixture]

    assert index.model == model
    assert tuple(index.timing.get(key) for key in TIMING_KEYS) == timings

    if model is None:
        # A file with no trailer at all is legal: tests/golden/stub768 is
        # trailer-stripped by design and the parser must still read it.
        assert index.timing == {}
        assert index.fold_times == []
    else:
        assert len(index.fold_times) == 10
        # "Fold 0: 141.53 seconds (1.34 minutes)" matches the seconds pattern as
        # well as the named timings do; fold_times is derived from those keys.
        assert index.timing["Fold 0"] == index.fold_times[0]


@pytest.mark.parametrize("fixture", FIXTURE_IDS)
def test_real_fixture_structure(fixture):
    index = pindex.read(REAL_FIXTURES[fixture])
    pindex.validate(index)

    assert index.k_fold == 10
    assert sorted(index.folds) == list(range(10))
    # 129 admissions x 6 strategies. Counted, values discarded -- see trap 5 in
    # the parser's module docstring.
    assert index.per_case_blocks == 774


# ---------------------------------------------------------------------------
# 2. The file shape that broke the deleted parser
# ---------------------------------------------------------------------------

def _minimal_file(fold_rows, aggregate_rows, fold=0, strategy="MAX", per_case_rows=None):
    """Build the smallest file that parses: one FOLD, one strategy, one 10-FOLD.

    ``per_case_rows``, when given, is written *after* the fold aggregate -- the
    inverted ordering the real writers never produce, and the one
    ``scripts/analyze_performance.py``'s parser could not survive.
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
    # Labelled 10-FOLD even though one fold is present. The label mattered when
    # the four deleted parsers were still here: every one of them hard-coded the
    # literal "10-FOLD", so any other k made them read the file as empty. This
    # parser takes k from the header -- asserted on the 3-fold fixture below.
    out.write("\n10-FOLD PERFORMANCE INDEX of %s SIMILARITY by MAX\n" % strategy)
    out.write(PERFORMANCE_INDEX_HEADER)
    for row in aggregate_rows:
        out.write("\t".join(str(v) for v in row) + "\n")
    return out.getvalue()


def _rows(tp, fp, value):
    """Five rows in write order, all carrying the same distinctive values."""
    return [[t, tp, fp, value, value, value, 1.0] for t in WRITE_ORDER]


def test_per_case_block_after_fold_aggregate_does_not_win(tmp_path):
    """Per-case rows must never overwrite the fold aggregate they follow.

    This was THE non-cosmetic difference between the parsers.
    ``analyze_performance``'s had no notion of a per-case block: its header test
    was the substring check ``'PERFORMANCE INDEX of MAX SIMILARITY by MAX' in
    line``, which the per-case header satisfies, so the rows that followed landed
    in ``fold_data[fold][method]`` on top of the fold aggregate. On the real files
    the aggregate is written last and wins, which is why nobody noticed for a
    year. Reverse the order -- as this fixture does -- and the reported per-fold
    performance became one test case's single trial: TP 1 instead of 7, F-score
    1.0 instead of 0.583.

    The parser under test recognises the block, counts it, and discards its
    values, so the fold aggregate survives wherever the block sits.
    """
    text = _minimal_file(
        fold_rows=_rows(tp=7, fp=5, value=0.583333),
        aggregate_rows=_rows(tp=7.0, fp=5.0, value=0.583333),
        per_case_rows=_rows(tp=1, fp=0, value=1.0),
    )
    path = tmp_path / "PerformanceIndex.txt"
    path.write_text(text, encoding="utf-8")

    index = pindex.read(str(path))
    assert index.folds[0]["MAX"][0.9].tp == 7.0
    assert index.folds[0]["MAX"][0.9].f_score == 0.583333
    # Recognised and counted, never mixed into the numbers.
    assert index.per_case_blocks == 1


def test_two_files_in_one_process_cannot_merge(tmp_path):
    """Each call returns its own value; there is no module state to accumulate.

    ``analyze_performance`` kept ``fold_data`` and ``aggregate_data`` as
    module-level ``defaultdict``s created at import, so two files parsed in one
    interpreter merged: file B, containing no ``MAX`` section at all, came back
    carrying file A's ``MAX`` numbers and reporting a fold it never mentioned.
    Both globals are gone. This pins the property that replaced them, because the
    cheapest way to reintroduce the bug is a well-meaning cache.
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

    index_a = pindex.read(str(file_a))
    index_b = pindex.read(str(file_b))
    assert list(index_a.aggregate) == ["MAX"]
    assert list(index_b.aggregate) == ["TOP-10"]
    assert sorted(index_a.folds) == [0]
    assert sorted(index_b.folds) == [5]


# ---------------------------------------------------------------------------
# 3. Baseline-arm coverage, generated rather than committed
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

    * the fold delimiter with its leading and trailing space;
    * per-case blocks emitted through ``compute_performance_index`` on the
      *accumulating* confusion matrix (``cython_utils.py:312-318``), so row *i*
      is the running total after *i+1* cases -- unlike the BERT arm's
      single-trial rows;
    * the fold-aggregate ``MAX`` header at column 0 with **no** preceding blank
      line, and each ``TOP-K`` header with one leading space;
    * a ``*`` banner *between* the k-FOLD MAX block and the TOP-K blocks, which
      the BERT arm does not write.

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

    # k_fold comes from the header, not from an assumption of 10. All four deleted
    # parsers hard-coded the literal "10-FOLD" and would read this file as empty,
    # which is what any future fold-count experiment would have run into.
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


# ---------------------------------------------------------------------------
# 4. Failure modes -- every one of them is a whitespace failure
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
    path = REAL_FIXTURES["golden_stub768"]
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

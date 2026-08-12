"""``scripts/attribute_effects.py``: the four-root attribution table.

The script is the only thing in the repository that reads more than one
``results*/`` root, and it exists because ``CORRECTED`` bundles two independent
fixes. Everything worth testing follows from that:

1. **The arithmetic**, over plain dicts. :func:`attribute` is pure and takes one
   threshold's slice, so the grid tests build values by hand and never touch the
   filesystem.
2. **The identity** ``leakage + preprocessing + residual == drop``. It holds by
   construction, which is exactly why it is worth pinning: the residual is
   *defined* as the leftover, so an "improvement" that computed it independently
   would break the identity silently and leave a table that no longer adds up.
3. **Two thresholds, not one.** 0.6 is the published cell and the one where the
   three BERT arms saturate; 1.0 is the only cosine setting where none of them
   does. Which rows carry information differs between the two, so the tests check
   that each block says so *from its own data*.
4. **The refusals.** Roots that are missing, duplicated, malformed, or hold the
   wrong pipeline; and the anchors, which are the only check that the parser read
   the right column.

Fixtures are synthetic trees under ``tmp_path``, built by driving the real
``PerformanceIndex.txt`` shape -- the same minimal-file approach as
``tests/test_performance_index.py``, extended to a *complete* run because the
script calls ``performance_index.validate`` and a truncated tree is supposed to
be refused. **No HADM_ID appears in these fixtures at all, real or synthetic:**
only fold delimiters and aggregate blocks are written, and the per-case section
-- the only part of the format that carries an ID -- is never emitted. That is
stronger than the ``999xxx`` placeholder convention the sibling test modules use,
and it is free here, because nothing this script reads lives in a per-case block.
"""

from __future__ import annotations

import json
import os

import pytest

from aicds import runs
from aicds.utils.Constants import K_FOLD, PERFORMANCE_INDEX_HEADER

import scripts.attribute_effects as attribute_effects


# The threshold rows are emitted in set-iteration order from {1, 0.9, 0.8, 0.7,
# 0.6}, not sorted order. Reproduced here so the fixtures exercise the same
# key-not-position lookup the script has to survive; see trap 4 in
# aicds.analysis.performance_index. Note `1` is the bare int the writers emit
# from the set literal -- float() is what makes it the same key as 1.0.
WRITE_ORDER = [0.9, 1, 0.6, 0.8, 0.7]
STRATEGIES = ["MAX", "TOP-10", "TOP-20", "TOP-30", "TOP-40", "TOP-50"]

STAMP = "12082026_09-00-00"

# Binary fractions, so every difference and every sum below is EXACT in double
# arithmetic and the identity test can use ==. Chosen to make the four
# quantities distinguishable: leakage 0.125, preprocessing 0.0625, drop 0.25,
# residual 0.0625. A set of decimal values would force a tolerance and hide the
# difference between "the identity holds" and "the identity nearly holds".
EXACT = {
    attribute_effects.LEGACY: 0.5,
    attribute_effects.FOLDS_ONLY: 0.375,
    attribute_effects.PREPROCESS_ONLY: 0.4375,
    attribute_effects.CORRECTED: 0.25,
}

# The same idea at threshold 1.0, and DELIBERATELY DISJOINT from EXACT -- no
# value and no derived part is shared between the two sets. That is what makes a
# threshold mix-up visible: a script that read 0.6 where it meant 1.0 would
# return a number from EXACT, and every assertion below would reject it.
# leakage 0.1875, preprocessing 0.03125, drop 0.375, residual 0.15625.
EXACT_10 = {
    attribute_effects.LEGACY: 0.9375,
    attribute_effects.FOLDS_ONLY: 0.75,
    attribute_effects.PREPROCESS_ONLY: 0.90625,
    attribute_effects.CORRECTED: 0.5625,
}

EXACT_BY_THRESHOLD = {0.6: EXACT, 1.0: EXACT_10}

# The BioSentVec row this script actually publishes, read off the four local
# results trees. 10-FOLD AGGREGATE CELLS ONLY -- an aggregate block carries no
# HADM_ID, so nothing here is DUA-covered content.
#
# Four of these eight are ANCHORS and four are the one-change columns the script
# subtracts. They live in one table rather than being retyped per test because
# three separate tests need the same row: the identity to double precision, the
# NEGATIVE preprocessing cell at threshold 1.0, and the end-to-end report. That
# negative cell is the most easily "corrected" number in the whole table -- fixing
# the text handling ADDED score to the baseline at threshold 1.0 -- so it is
# pinned from real data rather than from a binary fraction chosen to be tidy.
MEASURED_BASELINE = {
    0.6: {
        attribute_effects.LEGACY: 0.4824430641821946,
        attribute_effects.FOLDS_ONLY: 0.43346368340095165,
        attribute_effects.PREPROCESS_ONLY: 0.44293619424054204,
        attribute_effects.CORRECTED: 0.3922168068392324,
    },
    1.0: {
        attribute_effects.LEGACY: 0.2801210239036326,
        attribute_effects.FOLDS_ONLY: 0.21056198831548004,
        attribute_effects.PREPROCESS_ONLY: 0.292553547901374,
        attribute_effects.CORRECTED: 0.2162775515864761,
    },
}


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------

def _cells(at_06, at_10=None):
    """``{threshold: f_score}``. One argument fills both reported cells."""
    return {0.6: at_06, 1.0: at_06 if at_10 is None else at_10}


def _write_index(path, cells):
    """A complete, valid ``PerformanceIndex.txt`` whose TOP-10 rows carry ``cells``.

    ``cells`` is ``{threshold: f_score}`` over the two thresholds the script
    reports. Complete rather than minimal on purpose: the script calls
    ``performance_index.validate``, which requires all 6 strategies x 5
    thresholds x 10 folds, so a fixture that only carried the cells under test
    would be refused for the right reason and prove nothing about the reading.
    Every other cell is a filler constant -- if the script ever reads one of them,
    the arithmetic below stops matching.

    No per-case block is written, so no HADM_ID of any kind reaches these files.
    """
    filler = 0.111111
    with open(path, "w", encoding="utf-8") as handle:
        for fold in range(K_FOLD):
            handle.write("\n FOLD %d: LEN train: 116, LEN test: 13 \n" % fold)
            for strategy in STRATEGIES:
                handle.write(" PERFORMANCE INDEX of %s SIMILARITY by MAX\n" % strategy)
                handle.write(PERFORMANCE_INDEX_HEADER)
                for threshold in WRITE_ORDER:
                    handle.write(
                        "%s\t1\t3\t%s\t%s\t%s\t1.0\n"
                        % (threshold, filler, filler, filler)
                    )
        for strategy in STRATEGIES:
            handle.write(
                "\n%d-FOLD PERFORMANCE INDEX of %s SIMILARITY by MAX\n"
                % (K_FOLD, strategy)
            )
            handle.write(PERFORMANCE_INDEX_HEADER)
            for threshold in WRITE_ORDER:
                # Only TOP-10 carries the values under test, and only at the two
                # thresholds the script reports. The bare `1` for threshold 1.0 is
                # what the writers emit from the set literal; float() is what
                # makes it the same key as `1.0`.
                cell = filler
                if strategy == attribute_effects.STRATEGY:
                    cell = cells.get(float(threshold), filler)
                handle.write(
                    "%s\t1\t3\t%s\t%s\t%s\t1.0\n" % (threshold, cell, cell, cell)
                )
    return path


def _make_root(tmp_path, name, by_arm, pipeline=None, stamp=STAMP):
    """One ``ROOT/<arm>/<stamp>/`` tree. ``by_arm`` is ``{arm_key: cells}``.

    ``pipeline`` writes a ``run_metadata.json`` carrying that name, which is what
    the provenance check reads. Omitting it reproduces every pre-P14 tree, which
    warns rather than fails.
    """
    root = os.path.join(str(tmp_path), name)
    for arm, cells in by_arm.items():
        run_dir = os.path.join(root, arm, stamp)
        os.makedirs(run_dir, exist_ok=True)
        _write_index(os.path.join(run_dir, runs.PERFORMANCE_INDEX), cells)
        if pipeline is not None:
            with open(
                os.path.join(run_dir, runs.RUN_METADATA), "w", encoding="utf-8"
            ) as handle:
                json.dump({"pipeline": {"name": pipeline}}, handle)
    return root


def _arm_cells(pipeline, arm):
    """The cells one arm carries in one root.

    Split two ways on purpose. The BASELINE arm carries its real measured row, so
    every fixture satisfies ``check_anchors`` (which refuses any tree that does
    not reproduce the pinned cells) *and* so the report tests render the real
    published row, sign and all. Every other arm stays on exact binary fractions,
    whose differences the arithmetic tests can assert with ``==`` rather than a
    tolerance.
    """
    if arm == "baseline":
        return {
            threshold: MEASURED_BASELINE[threshold][pipeline]
            for threshold in attribute_effects.THRESHOLDS
        }
    return {
        threshold: EXACT_BY_THRESHOLD[threshold][pipeline]
        for threshold in attribute_effects.THRESHOLDS
    }


def _anchored_roots(tmp_path, pipelines=False):
    """All four roots, in ``--roots`` order, satisfying every parser anchor."""
    roots = []
    for spec in attribute_effects.ROOT_SPECS:
        by_arm = {
            arm: _arm_cells(spec.pipeline, arm) for arm in runs.MODEL_KEYS.values()
        }
        roots.append(_make_root(
            tmp_path,
            "results_%s" % spec.pipeline.replace("-", "_"),
            by_arm,
            pipeline=spec.pipeline if pipelines else None,
        ))
    return roots


def _realistic_roots(tmp_path):
    """The real shape: BERT arms pinned to the ceiling at 0.6, off it at 1.0.

    ``_anchored_roots`` deliberately keeps every non-baseline arm on plain binary
    fractions, so nothing in it saturates and the report's two saturation branches
    would never both be exercised. This fixture reproduces what the committed
    trees actually look like -- 1.000000 for all three BERT arms at threshold 0.6
    in all four roots -- which is the case the SATURATION note exists for.
    """
    roots = []
    for spec in attribute_effects.ROOT_SPECS:
        by_arm = {}
        for arm in runs.MODEL_KEYS.values():
            cells = _arm_cells(spec.pipeline, arm)
            if arm != "baseline":
                cells[0.6] = 1.0
            by_arm[arm] = cells
        roots.append(_make_root(
            tmp_path, "results_%s" % spec.pipeline.replace("-", "_"), by_arm,
            pipeline=spec.pipeline,
        ))
    return roots


# ---------------------------------------------------------------------------
# 1. The grid and the arithmetic
# ---------------------------------------------------------------------------

def test_collect_reads_the_top10_cell_at_both_reported_thresholds(tmp_path, capsys):
    collected = attribute_effects.collect(_anchored_roots(tmp_path))
    capsys.readouterr()

    assert collected.arms == list(runs.MODEL_KEYS.values())
    assert sorted(collected.values) == sorted(attribute_effects.THRESHOLDS)
    for (pipeline, arm, threshold), expected in attribute_effects.ANCHORS.items():
        assert collected.values[threshold][pipeline][arm] == expected
    for spec in attribute_effects.ROOT_SPECS:
        assert collected.values[0.6][spec.pipeline]["bluebert"] == EXACT[spec.pipeline]
        assert collected.values[1.0][spec.pipeline]["bluebert"] == EXACT_10[spec.pipeline]


def test_the_two_thresholds_do_not_share_a_cell(tmp_path, capsys):
    """The 0.6 and 1.0 slices are disjoint, so a mix-up cannot pass unnoticed.

    ``EXACT`` and ``EXACT_10`` share no value, which is the property that makes
    every other assertion in this module able to tell the two cells apart. Pinned
    here so a future edit to either table cannot quietly destroy it.
    """
    assert not set(EXACT.values()) & set(EXACT_10.values())
    collected = attribute_effects.collect(_anchored_roots(tmp_path))
    capsys.readouterr()
    for spec in attribute_effects.ROOT_SPECS:
        at_06 = collected.values[0.6][spec.pipeline]
        at_10 = collected.values[1.0][spec.pipeline]
        assert at_06["biomedbert"] != at_10["biomedbert"]


def test_the_threshold_is_a_key_not_a_row_position(tmp_path, capsys):
    """A file whose 0.6 row is not first must still yield the 0.6 value.

    ``WRITE_ORDER`` puts 0.9 in row 0, the bare int ``1`` in row 1 and 0.6 in row
    2, which is what the real writers emit, so a parser or caller indexing by
    position would return the filler constant here rather than the value under
    test. Asserting the filler is *not* what comes back is what makes this a real
    check -- and the 1.0 lookup additionally proves ``1`` and ``1.0`` are one key.
    """
    collected = attribute_effects.collect(_anchored_roots(tmp_path))
    capsys.readouterr()
    assert WRITE_ORDER.index(0.6) == 2
    assert WRITE_ORDER.index(1) == 1
    assert collected.values[0.6][attribute_effects.FOLDS_ONLY]["biomedbert"] == 0.375
    assert collected.values[1.0][attribute_effects.FOLDS_ONLY]["biomedbert"] == 0.75
    for threshold in attribute_effects.THRESHOLDS:
        assert collected.values[threshold][attribute_effects.FOLDS_ONLY][
            "biomedbert"
        ] != 0.111111


def test_attribute_splits_the_drop_into_named_parts():
    values = {pipeline: {"baseline": value} for pipeline, value in EXACT.items()}
    parts = attribute_effects.attribute(values, "baseline")

    assert parts["legacy"] == 0.5
    assert parts["leakage"] == 0.125          # 0.5    - 0.375
    assert parts["preprocessing"] == 0.0625   # 0.5    - 0.4375
    assert parts["drop"] == 0.25              # 0.5    - 0.25
    assert parts["residual"] == 0.0625        # 0.25 - 0.125 - 0.0625


def test_attribute_is_threshold_agnostic():
    """The same pure function over the 1.0 slice, with no threshold argument.

    ``attribute`` takes ``collected.values[threshold]`` and never learns which
    threshold it was. That is the design: a threshold parameter here would be a
    second place to get the cell wrong, and the arithmetic is identical either
    way. This is the evidence it really is identical.
    """
    values = {pipeline: {"baseline": value} for pipeline, value in EXACT_10.items()}
    parts = attribute_effects.attribute(values, "baseline")

    assert parts["legacy"] == 0.9375
    assert parts["leakage"] == 0.1875         # 0.9375 - 0.75
    assert parts["preprocessing"] == 0.03125  # 0.9375 - 0.90625
    assert parts["drop"] == 0.375             # 0.9375 - 0.5625
    assert parts["residual"] == 0.15625       # 0.375 - 0.1875 - 0.03125


def test_the_identity_holds_exactly_on_binary_fractions():
    """leakage + preprocessing + residual == drop, by construction.

    Pinned because the residual is *defined* as the leftover: anything that
    computed it independently -- a second subtraction order, a rounded print fed
    back in -- would leave a published table whose columns no longer sum, and
    nothing else in the script would notice.
    """
    for table in (EXACT, EXACT_10):
        values = {pipeline: {"baseline": value} for pipeline, value in table.items()}
        parts = attribute_effects.attribute(values, "baseline")
        assert parts["leakage"] + parts["preprocessing"] + parts["residual"] == parts["drop"]


def test_the_identity_holds_to_double_precision_on_real_decimals():
    """The same identity on the values this script will actually see.

    Separated from the exact test above, and deliberately not merged into it: the
    real F-scores are not binary fractions, so the sum is subject to ordinary
    rounding and the honest claim is "to double precision", not "exactly". A
    single test using a tolerance would have hidden the fact that the identity IS
    exact when the inputs allow it.
    """
    values = {
        pipeline: {"baseline": value}
        for pipeline, value in MEASURED_BASELINE[0.6].items()
    }
    parts = attribute_effects.attribute(values, "baseline")
    total = parts["leakage"] + parts["preprocessing"] + parts["residual"]
    assert total == pytest.approx(parts["drop"], abs=1e-15)
    # The bundled drop is the published 0.090 figure this script exists to split.
    assert round(parts["drop"], 4) == 0.0902


def test_a_negative_part_is_reported_as_a_sign_not_an_error():
    """At threshold 1.0 the baseline's preprocessing part is NEGATIVE, and real.

    ``preprocess-only`` scores 0.292554 against legacy's 0.280121, so
    ``legacy - fixed`` is below zero: fixing the text handling ADDED score on that
    arm at that threshold. The sign convention is stated once in the module
    docstring and the table prints ``%+.6f``; nothing anywhere clamps it, and this
    pins that nothing ever does. A clamp would turn the most interesting cell in
    the 1.0 block into a zero.
    """
    values = {
        pipeline: {"baseline": value}
        for pipeline, value in MEASURED_BASELINE[1.0].items()
    }
    parts = attribute_effects.attribute(values, "baseline")
    assert parts["preprocessing"] < 0
    assert round(parts["preprocessing"], 6) == -0.012433
    assert round(parts["drop"], 6) == 0.063843
    assert round(parts["leakage"], 6) == 0.069559
    assert parts["leakage"] + parts["preprocessing"] + parts["residual"] == pytest.approx(
        parts["drop"], abs=1e-15
    )


def test_the_measured_baseline_row_carries_every_anchor():
    """``MEASURED_BASELINE`` and ``ANCHORS`` are the same cells, read once.

    The fixtures build the baseline arm from ``MEASURED_BASELINE`` and the script
    checks it against ``ANCHORS``, so the two agreeing is what makes every
    end-to-end test here pass for the right reason rather than because the
    fixture was reverse-engineered from the expectation. If they ever disagree,
    this fails before ``check_anchors`` does, and says which side moved.
    """
    for (pipeline, arm, threshold), expected in attribute_effects.ANCHORS.items():
        assert arm == "baseline", (pipeline, arm, threshold)
        assert MEASURED_BASELINE[threshold][pipeline] == expected


def test_zero_parts_are_reported_not_hidden():
    """A fix with no measured effect prints 0.000000; it is a finding, not a gap."""
    flat = {spec.pipeline: 0.5 for spec in attribute_effects.ROOT_SPECS}
    values = {pipeline: {"baseline": value} for pipeline, value in flat.items()}
    parts = attribute_effects.attribute(values, "baseline")
    assert (parts["leakage"], parts["preprocessing"], parts["drop"], parts["residual"]) == (
        0.0, 0.0, 0.0, 0.0,
    )


def test_saturated_arms_are_derived_from_the_data():
    """Which arms sit on the ceiling is measured, never a hard-coded arm list.

    Hard-coding "the three BERT arms" would keep printing the note after a run in
    which one of them left the ceiling -- and, worse, would print it at threshold
    1.0, where the whole point is that nothing saturates.
    """
    values = {
        spec.pipeline: {"baseline": 0.4, "bluebert": 1.0, "biomedbert": 0.9}
        for spec in attribute_effects.ROOT_SPECS
    }
    assert attribute_effects.saturated_arms(
        values, ["baseline", "bluebert", "biomedbert"]
    ) == ["bluebert"]


def test_thresholds_are_the_published_cell_and_the_unsaturated_one():
    """Two cells, in print order, and every one of them carries anchors.

    The pairing is the invariant: a threshold added to ``THRESHOLDS`` without an
    anchor would print a whole grid and attribution table whose parse nothing
    verified, which is the failure ``check_anchors`` refuses on zero checks to
    prevent -- except it would not fire, because the *other* threshold's anchors
    still passed.
    """
    assert attribute_effects.THRESHOLDS == (0.6, 1.0)
    anchored = {threshold for _, _, threshold in attribute_effects.ANCHORS}
    assert anchored == set(attribute_effects.THRESHOLDS)
    for threshold in attribute_effects.THRESHOLDS:
        assert attribute_effects.THRESHOLD_NOTES[threshold]


def test_report_prints_the_banner_and_a_full_block_per_threshold(tmp_path, capsys):
    collected = attribute_effects.collect(_realistic_roots(tmp_path))
    capsys.readouterr()
    attribute_effects.report(collected)
    out = capsys.readouterr().out

    # The banner is mandatory on every output, top and bottom: a reader who
    # copies only the tail of this report must still be told what
    # preprocess-only is for.
    assert out.count("PREPROCESS-ONLY NUMBERS ARE ATTRIBUTION INPUTS, NEVER RESULTS.") == 2
    assert "src/aicds/config.py:143-145" in out
    # One complete block per threshold -- grid, attribution, saturation note.
    assert "# THRESHOLD 0.6 --" in out and "# THRESHOLD 1.0 --" in out
    assert out.count("F-SCORE GRID") == 2
    assert out.count("ATTRIBUTION   (every part is legacy - fixed") == 2
    # The ROOTS provenance block is printed once, not once per threshold.
    assert out.count("ROOTS\n") == 1
    # The residual is labelled as reportable, so nobody reads it as a failed check.
    assert out.count("THE RESIDUAL IS A REPORTABLE NUMBER, NOT AN ERROR") == 2
    # Display names from aicds.runs.KEY_LABELS, not directory keys.
    assert "BioSentVec" in out and "Bio_ClinicalBERT" in out
    # The published cell and the drop this script exists to split.
    assert "0.482443" in out and "0.392217" in out
    assert "+0.090226" in out


def test_each_threshold_block_states_its_own_saturation_from_its_own_data(tmp_path, capsys):
    """0.6 says three rows are empty; 1.0 says every row carries information.

    This is the whole reason both cells are reported, so it is asserted on a
    fixture shaped like the real trees: the BERT arms pinned to 1.000000 at 0.6 in
    all four roots, and off the ceiling at 1.0. A single shared footnote, or a
    note derived once and reused, would get exactly one of these two wrong.
    """
    collected = attribute_effects.collect(_realistic_roots(tmp_path))
    capsys.readouterr()
    attribute_effects.report(collected)
    out = capsys.readouterr().out

    assert out.count("SATURATION -- read these rows as empty, not as zero effect:") == 1
    assert out.count("NO SATURATION at this threshold") == 1
    # The saturated block points the reader at the one that is not.
    assert "Threshold 1.0 in this same report has no arm on the ceiling" in out
    # And it names the arms it measured, rather than a hard-coded list.
    assert "Bio_ClinicalBERT, BiomedBERT, BlueBERT" in out
    # The 0.6 block leaves three rows of zeros; the 1.0 block is populated for
    # every arm, including the one cell whose sign is negative.
    assert "+0.187500" in out   # a BERT arm's leakage at 1.0 (EXACT_10 fixture)
    assert "-0.012433" in out   # BioSentVec preprocessing at 1.0 -- negative, and real
    assert "+0.069559" in out   # BioSentVec leakage at 1.0


def test_labels_come_from_the_shared_key_table():
    """No parallel naming table. ``runs.KEY_LABELS`` is the one mapping."""
    assert attribute_effects._label("bio_clinical_bert") == "Bio_ClinicalBERT"
    assert runs.KEY_LABELS["baseline"] == "BioSentVec"
    # An unregistered key shows itself rather than being dropped or guessed at.
    assert attribute_effects._label("mystery_encoder") == "mystery_encoder"


# ---------------------------------------------------------------------------
# 2. Refusal: the roots
# ---------------------------------------------------------------------------

def test_missing_roots_name_all_of_them_in_one_exit(tmp_path):
    """Two of four roots do not exist until P29 is harvested, so this IS the UX.

    All gaps are reported in one pass. Fixing them one exit at a time would mean
    a second pod session to discover the second missing tree.
    """
    roots = _anchored_roots(tmp_path)
    roots[1] = os.path.join(str(tmp_path), "results_folds_only_absent")
    roots[2] = os.path.join(str(tmp_path), "results_preprocess_only_absent")

    with pytest.raises(SystemExit) as caught:
        attribute_effects.collect(roots)

    message = str(caught.value)
    assert "results_folds_only_absent" in message
    assert "results_preprocess_only_absent" in message
    assert message.count("no such directory") == 2
    # Instructions, not a diagnosis: the fix is to produce the runs.
    assert "--pipeline folds-only --out" in message
    assert "--pipeline preprocess-only --out" in message
    assert "run_baseline.py" in message and "run_bert_analysis.py" in message
    assert "P29" in message
    # The two roots that ARE fine are shown as such, so it is clear how far it got.
    assert message.count("ok ") == 2


def test_the_missing_root_error_carries_the_never_a_result_banner(tmp_path):
    """This path hands the reader the command to CREATE preprocess-only data.

    Telling somebody how to produce a leaked column without telling them it may
    never be quoted is how the leaked column escapes -- and this is the only
    refusal that prints a ``--pipeline preprocess-only`` command line, so it is the
    one place the banner is load-bearing rather than decorative.
    """
    roots = _anchored_roots(tmp_path)
    roots[2] = os.path.join(str(tmp_path), "results_preprocess_only_absent")

    with pytest.raises(SystemExit) as caught:
        attribute_effects.collect(roots)

    message = str(caught.value)
    assert "python scripts/run_baseline.py      --pipeline preprocess-only" in message
    assert "PREPROCESS-ONLY NUMBERS ARE ATTRIBUTION INPUTS, NEVER RESULTS." in message
    assert "Report 'corrected' or 'drg'. Never this." in message


def test_an_existing_root_with_no_runs_is_refused(tmp_path):
    """An empty directory is not "zero effect" -- it is an unusable column."""
    roots = _anchored_roots(tmp_path)
    empty = os.path.join(str(tmp_path), "results_empty")
    os.makedirs(empty)
    roots[2] = empty

    with pytest.raises(SystemExit) as caught:
        attribute_effects.collect(roots)
    assert "holds no runs" in str(caught.value)


def test_every_malformed_root_is_reported_in_one_pass(tmp_path):
    """Two unreadable layouts, one exit, both named -- not one exit per root.

    Four roots are normally harvested by one person in one session and go wrong
    the same way, so exiting on the first ``RunLayoutError`` costs a second
    session to discover the second. Matched to ``_unusable_roots_error``'s
    one-pass design for missing roots, which had it right already.
    """
    roots = _anchored_roots(tmp_path)
    for index, name in ((1, "results_bad_folds"), (2, "results_bad_preprocess")):
        junk = os.path.join(str(tmp_path), name, "not_a_run_directory")
        os.makedirs(junk)
        _write_index(os.path.join(junk, runs.PERFORMANCE_INDEX), _cells(0.4))
        roots[index] = os.path.join(str(tmp_path), name)

    with pytest.raises(SystemExit) as caught:
        attribute_effects.collect(roots)

    message = str(caught.value)
    assert message.count("unreadable run layout") == 2
    assert "results_bad_folds" in message and "results_bad_preprocess" in message
    # The underlying RunLayoutError text is carried through verbatim, per root.
    assert message.count("fits neither run layout") == 2
    assert "folds-only" in message and "preprocess-only" in message
    # A tree that is already there at the wrong depth is not fixed by re-running
    # the pipeline, so no command line is offered for these two.
    assert "--pipeline folds-only --out" not in message


def test_an_arm_missing_from_one_root_is_refused(tmp_path):
    """A difference across four roots cannot be taken over three of them."""
    roots = _anchored_roots(tmp_path)
    partial = _make_root(
        tmp_path, "results_partial",
        {"baseline": _cells(0.4), "bluebert": _cells(0.4)},
    )
    roots[1] = partial

    with pytest.raises(SystemExit) as caught:
        attribute_effects.collect(roots)

    message = str(caught.value)
    assert "every arm must be present in every root" in message
    assert "bio_clinical_bert" in message and "biomedbert" in message
    assert "results_partial" in message


def test_an_unregistered_arm_is_required_everywhere_rather_than_dropped(tmp_path):
    """A stranger arm found in one root must appear in all four, or it refuses.

    Silently dropping it is the failure ``aicds.runs.discover`` refuses upstream,
    and it is worse here: the columns are differences, so an arm present in three
    roots would produce a number attributed to a pipeline it never ran under.
    """
    roots = _anchored_roots(tmp_path)
    extra = os.path.join(str(tmp_path), "results_legacy", "mystery_encoder", STAMP)
    os.makedirs(extra)
    _write_index(os.path.join(extra, runs.PERFORMANCE_INDEX), _cells(0.4))

    with pytest.raises(SystemExit) as caught:
        attribute_effects.collect(roots)
    assert "mystery_encoder" in str(caught.value)


def test_an_incomplete_run_is_refused_rather_than_averaged(tmp_path):
    """A truncated tree parses fine; it must not contribute a real-looking number."""
    roots = _anchored_roots(tmp_path)
    truncated = os.path.join(str(tmp_path), "results_corrected", "bluebert", STAMP)
    with open(
        os.path.join(truncated, runs.PERFORMANCE_INDEX), "w", encoding="utf-8"
    ) as handle:
        handle.write(" FOLD 0: LEN train: 116, LEN test: 13 \n")
        handle.write("PERFORMANCE INDEX of MAX SIMILARITY by MAX\n")
        handle.write(PERFORMANCE_INDEX_HEADER)
        handle.write("0.9\t1\t3\t0.25\t0.25\t0.25\t1.0\n")

    with pytest.raises(SystemExit) as caught:
        attribute_effects.collect(roots)

    message = str(caught.value)
    assert "not a\n        complete 10-fold run" in message
    # The arm and the run directory, not just the root: four roots x four arms is
    # sixteen candidate files, and "results_corrected" alone names four of them.
    assert "arm  bluebert" in message
    assert truncated in message


def test_an_unreadable_index_is_refused_with_the_same_message(tmp_path):
    """``read`` opens the file, so OSError/UnicodeDecodeError land here too.

    Catching only ``PerformanceIndexError`` let a mis-encoded or unreadable file
    out as a bare traceback naming the path but not which of four roots, or which
    arm, asked for it -- on a script whose entire job is attribution.
    """
    roots = _anchored_roots(tmp_path)
    corrupt = os.path.join(str(tmp_path), "results_legacy", "biomedbert", STAMP)
    with open(os.path.join(corrupt, runs.PERFORMANCE_INDEX), "wb") as handle:
        handle.write(b" FOLD 0: LEN train: 116, LEN test: 13 \n\xff\xfe not utf-8\n")

    with pytest.raises(SystemExit) as caught:
        attribute_effects.collect(roots)

    message = str(caught.value)
    assert "unreadable" in message
    assert "arm  biomedbert" in message
    assert corrupt in message


def test_wrong_number_of_roots_is_refused():
    with pytest.raises(SystemExit) as caught:
        attribute_effects.resolve_roots(["one", "two"])
    message = str(caught.value)
    assert "exactly 4 paths" in message
    assert "legacy folds-only preprocess-only corrected" in message


def test_the_same_tree_in_two_columns_is_refused(tmp_path):
    """A repeated root fabricates a column of exact zeros: "this fix did nothing".

    That cell is indistinguishable from the real finding the script exists to
    produce, which is why it refuses rather than reporting it. The comparison is
    on the RESOLVED path, so the three spellings below all collide.
    """
    roots = _anchored_roots(tmp_path)
    roots[3] = roots[0]

    with pytest.raises(SystemExit) as caught:
        attribute_effects.resolve_roots(roots)

    message = str(caught.value)
    assert "names one tree in more than one column" in message
    assert "legacy, corrected" in message
    assert "tree minus itself" in message


def test_duplicate_roots_are_compared_by_resolved_path(tmp_path):
    """'results', './results' and 'results/' are one tree, and are caught as one.

    An exact string comparison would wave every one of these through, and the
    zero column they fabricate looks exactly like a measurement.
    """
    with pytest.raises(SystemExit) as caught:
        attribute_effects.resolve_roots(
            ["results", "results_folds_only", "results_preprocess_only", "./results/"]
        )
    assert "resolve to the same tree" in str(caught.value)


def test_four_distinct_roots_are_accepted():
    """The guard must not fire on the ordinary case."""
    resolved = attribute_effects.resolve_roots(["a", "b", "c", "d"])
    assert resolved == {
        "legacy": "a", "folds-only": "b", "preprocess-only": "c", "corrected": "d",
    }


def test_default_roots_are_the_four_documented_trees():
    resolved = attribute_effects.resolve_roots(None)
    assert resolved == {
        "legacy": "results",
        "folds-only": "results_folds_only",
        "preprocess-only": "results_preprocess_only",
        "corrected": "results_corrected",
    }
    # Order is load-bearing: --roots is positional against ROOT_SPECS and
    # attribute() subtracts the middle two from the first.
    assert [spec.pipeline for spec in attribute_effects.ROOT_SPECS] == [
        "legacy", "folds-only", "preprocess-only", "corrected"
    ]


# ---------------------------------------------------------------------------
# 3. Refusal: the parser anchors
# ---------------------------------------------------------------------------

def test_anchors_pass_on_the_pinned_values(tmp_path, capsys):
    collected = attribute_effects.collect(_anchored_roots(tmp_path))
    capsys.readouterr()
    assert attribute_effects.check_anchors(collected.values) == len(
        attribute_effects.ANCHORS
    )


def test_a_wrong_anchor_value_refuses_and_says_the_parse_is_untrusted(tmp_path, capsys):
    """One digit off in the legacy baseline cell, and no table is printed.

    The anchors are the only thing verifying that the parse lines columns up with
    headers, so a mismatch has to stop the run rather than annotate it -- an
    unverified table looks exactly like a verified one.
    """
    roots = _anchored_roots(tmp_path)
    wrong = os.path.join(str(tmp_path), "results_legacy", "baseline", STAMP)
    _write_index(
        os.path.join(wrong, runs.PERFORMANCE_INDEX),
        _cells(0.4824430641821000, attribute_effects.ANCHORS[("legacy", "baseline", 1.0)]),
    )

    collected = attribute_effects.collect(roots)
    capsys.readouterr()
    with pytest.raises(SystemExit) as caught:
        attribute_effects.check_anchors(collected.values)

    message = str(caught.value)
    # Both sides, labelled. The parsed value is a prefix of the expected one, so
    # the labels are what make the two assertions independent.
    assert "expected 0.4824430641821946" in message
    assert "parsed   0.4824430641821\n" in message
    assert "do not edit ANCHORS to make this pass" in message
    assert "no table was printed" in message
    # The failing cell names its threshold: four anchors, two per pipeline.
    assert "legacy / baseline  TOP-10 @ 0.6" in message


def test_a_wrong_anchor_at_threshold_one_is_caught_too(tmp_path, capsys):
    """The 1.0 cells are anchored in their own right, not inferred from 0.6.

    Threshold 1.0 is the block a reader is told to quote, so an unverified parse
    there is the more dangerous of the two. Its anchors were minted the same way
    -- read back at full precision off the local trees -- and they fail the same
    way.
    """
    roots = _anchored_roots(tmp_path)
    wrong = os.path.join(str(tmp_path), "results_corrected", "baseline", STAMP)
    _write_index(
        os.path.join(wrong, runs.PERFORMANCE_INDEX),
        _cells(attribute_effects.ANCHORS[("corrected", "baseline", 0.6)], 0.2162775515),
    )

    collected = attribute_effects.collect(roots)
    capsys.readouterr()
    with pytest.raises(SystemExit) as caught:
        attribute_effects.check_anchors(collected.values)

    message = str(caught.value)
    assert "corrected / baseline  TOP-10 @ 1.0" in message
    assert "expected 0.2162775515864761" in message


def test_the_anchor_comparison_is_exact_not_rounded(tmp_path, capsys):
    """A value that rounds to the same four decimals still fails.

    ``compare_models`` compares to 4 dp because its expectations were transcribed
    by hand; these were read back at full precision, so the stricter comparison
    is available and is the one worth having -- P == R == FS in most of these
    rows, and a rounded check would pass while the parser read a neighbour.
    """
    roots = _anchored_roots(tmp_path)
    near = os.path.join(str(tmp_path), "results_corrected", "baseline", STAMP)
    _write_index(
        os.path.join(near, runs.PERFORMANCE_INDEX),
        _cells(0.39221680683, attribute_effects.ANCHORS[("corrected", "baseline", 1.0)]),
    )

    collected = attribute_effects.collect(roots)
    capsys.readouterr()
    assert round(collected.values[0.6][attribute_effects.CORRECTED]["baseline"], 4) == 0.3922
    with pytest.raises(SystemExit):
        attribute_effects.check_anchors(collected.values)


def test_zero_anchors_refuses_rather_than_reporting_success():
    """"0 anchors passed" and "verified" print identically, so refuse."""
    with pytest.raises(SystemExit) as caught:
        attribute_effects.check_anchors({0.6: {"legacy": {"bluebert": 0.5}}})
    assert "zero parser anchors ran" in str(caught.value)


def test_anchors_are_the_baseline_cells_compare_models_also_carries():
    """Pinned literals, cross-checked against ``compare_models``' own table.

    Both derive from the same local trees by different routes: that table was
    transcribed by hand to four decimals off the run output, these were read back
    at full precision. Agreement is what makes either one credible, and this
    assertion fails loudly if one of them is edited alone.

    All four anchors are covered -- both pipelines at both thresholds -- because
    ``compare_models`` already carried hand-transcribed ``TOP-10 @ 1.0`` rows for
    every arm, which is exactly the independent source the 1.0 anchors needed.
    """
    import scripts.compare_models as compare_models

    checked = 0
    for (pipeline, arm, threshold), anchor in attribute_effects.ANCHORS.items():
        table = compare_models.SANITY_CHECKS_BY_PIPELINE[pipeline]
        rows = [
            row for row in table
            if row[0] == arm
            and row[1] == attribute_effects.STRATEGY
            and row[2] == threshold
        ]
        assert len(rows) == 1, (pipeline, arm, threshold)
        assert round(anchor, 4) == rows[0][3]["FS"], (pipeline, arm, threshold)
        checked += 1
    assert checked == 4


def test_compare_models_carries_the_unsaturated_cell_for_every_arm():
    """The 1.0 block is a four-arm table, and the sibling script agrees it is.

    Only the baseline is anchored here (the BERT arms print 1.000000 at 0.6, which
    anchors nothing), but the numbers this script publishes at 1.0 cover all four
    arms. ``compare_models``' hand-transcribed rows are the independent record of
    the other three, so their presence is worth pinning: losing them would leave
    the three most interesting rows of the 1.0 attribution with no second source.
    """
    import scripts.compare_models as compare_models

    for pipeline in (attribute_effects.LEGACY, attribute_effects.CORRECTED):
        arms = {
            row[0] for row in compare_models.SANITY_CHECKS_BY_PIPELINE[pipeline]
            if row[1] == attribute_effects.STRATEGY and row[2] == 1.0
        }
        assert arms == set(runs.MODEL_KEYS.values()), pipeline


# ---------------------------------------------------------------------------
# 4. Refusal: provenance
# ---------------------------------------------------------------------------

def test_a_tree_holding_the_wrong_pipeline_is_refused(tmp_path):
    """The failure this script is most exposed to: four trees, named by hand.

    A mis-named root does not look wrong -- it fills the table, swaps two columns
    and lets the residual absorb the difference. ``run_metadata.json`` is what
    makes it loud, for every run written since P14.
    """
    roots = []
    for index, spec in enumerate(attribute_effects.ROOT_SPECS):
        by_arm = {
            arm: _arm_cells(spec.pipeline, arm) for arm in runs.MODEL_KEYS.values()
        }
        # The preprocess-only column is filled with a tree that says it is
        # folds-only: the exact harvest slip that produces a plausible table.
        recorded = "folds-only" if index == 2 else spec.pipeline
        roots.append(_make_root(
            tmp_path, "results_%s" % spec.pipeline.replace("-", "_"),
            by_arm, pipeline=recorded,
        ))

    with pytest.raises(SystemExit) as caught:
        attribute_effects.collect(roots)

    message = str(caught.value)
    assert "says pipeline 'folds-only'" in message
    assert "'preprocess-only'\n    column" in message


def test_missing_metadata_warns_but_does_not_refuse(tmp_path, capsys):
    """Pre-P14 trees carry no metadata, so absence cannot be fatal.

    It is still said out loud, because a column taken on a directory name's word
    is a weaker claim than one taken on the run's own record.
    """
    attribute_effects.collect(_anchored_roots(tmp_path))
    out = capsys.readouterr().out
    assert out.count("[WARN]") == len(attribute_effects.ROOT_SPECS)
    assert "no run_metadata.json" in out
    assert "pre-P14 runs" in out


def test_matching_metadata_is_silent(tmp_path, capsys):
    attribute_effects.collect(_anchored_roots(tmp_path, pipelines=True))
    assert "[WARN]" not in capsys.readouterr().out


def test_root_specs_name_real_pipelines():
    """A typo here would silently downgrade the provenance check to a no-op.

    ``ROOT_SPECS`` is validated at import against the config registry, so this
    asserts the guard's premise rather than re-deriving it: a name that is not
    selectable could never match a ``run_metadata.json`` and every tree would
    read as unrecorded.
    """
    from aicds.config import PIPELINE_NAMES

    for spec in attribute_effects.ROOT_SPECS:
        assert spec.pipeline in PIPELINE_NAMES


# ---------------------------------------------------------------------------
# 5. End to end
# ---------------------------------------------------------------------------

def test_main_runs_end_to_end_and_returns_zero(tmp_path, capsys):
    assert attribute_effects.main(["--roots"] + _realistic_roots(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "[SUCCESS] 4 parser anchors passed" in out
    assert out.count("F-SCORE GRID") == 2
    # The long form: the banner also says "ATTRIBUTION INPUTS", twice.
    assert out.count("ATTRIBUTION   (every part is legacy - fixed") == 2
    assert "# THRESHOLD 0.6 --" in out and "# THRESHOLD 1.0 --" in out
    assert "PREPROCESS-ONLY NUMBERS ARE ATTRIBUTION INPUTS" in out

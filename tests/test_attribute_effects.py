"""``scripts/attribute_effects.py``: the four-root attribution table.

The script is the only thing in the repository that reads more than one
``results*/`` root, and it exists because ``CORRECTED`` bundles two independent
fixes. Everything worth testing follows from that:

1. **The arithmetic**, over plain dicts. :func:`attribute` is pure, so the grid
   tests build values by hand and never touch the filesystem.
2. **The identity** ``leakage + preprocessing + residual == drop``. It holds by
   construction, which is exactly why it is worth pinning: the residual is
   *defined* as the leftover, so an "improvement" that computed it independently
   would break the identity silently and leave a table that no longer adds up.
3. **The refusals.** Two of the four roots do not exist until P29's eight runs
   are harvested, so the missing-root path is the first thing most users will
   see; and the anchors are the only check that the parser read the right column.

Fixtures are synthetic trees under ``tmp_path``, built by driving the real
``PerformanceIndex.txt`` shape -- the same minimal-file approach as
``tests/test_performance_index.py``, extended to a *complete* run because the
script calls ``performance_index.validate`` and a truncated tree is supposed to
be refused. HADM_IDs never appear in these files at all (only aggregate blocks
are written), and the one synthetic ID used for the per-case shape follows the
``999xxx`` convention of the sibling test modules.
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
# aicds.analysis.performance_index.
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


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------

def _write_index(path, f_score):
    """A complete, valid ``PerformanceIndex.txt`` whose TOP-10 @ 0.6 is ``f_score``.

    Complete rather than minimal on purpose: the script calls
    ``performance_index.validate``, which requires all 6 strategies x 5
    thresholds x 10 folds, so a fixture that only carried the one cell under
    test would be refused for the right reason and prove nothing about the
    reading. Every other cell is a filler constant -- if the script ever reads
    one of them, the arithmetic below stops matching.
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
                # Only TOP-10 at 0.6 carries the value under test. The bare `1`
                # for threshold 1.0 is what the writers emit from the set
                # literal, and float() is what makes it the same key as `1.0`.
                cell = (
                    f_score
                    if strategy == attribute_effects.STRATEGY
                    and float(threshold) == attribute_effects.THRESHOLD
                    else filler
                )
                handle.write(
                    "%s\t1\t3\t%s\t%s\t%s\t1.0\n" % (threshold, cell, cell, cell)
                )
    return path


def _make_root(tmp_path, name, by_arm, pipeline=None, stamp=STAMP):
    """One ``ROOT/<arm>/<stamp>/`` tree. ``by_arm`` is ``{arm_key: f_score}``.

    ``pipeline`` writes a ``run_metadata.json`` carrying that name, which is what
    the provenance check reads. Omitting it reproduces every committed tree --
    pre-P14 runs, which warn rather than fail.
    """
    root = os.path.join(str(tmp_path), name)
    for arm, f_score in by_arm.items():
        run_dir = os.path.join(root, arm, stamp)
        os.makedirs(run_dir, exist_ok=True)
        _write_index(os.path.join(run_dir, runs.PERFORMANCE_INDEX), f_score)
        if pipeline is not None:
            with open(
                os.path.join(run_dir, runs.RUN_METADATA), "w", encoding="utf-8"
            ) as handle:
                json.dump({"pipeline": {"name": pipeline}}, handle)
    return root


def _anchored_roots(tmp_path, pipelines=False):
    """All four roots, in ``--roots`` order, satisfying both parser anchors.

    Every arm of a root carries the same f-score (the exact binary fraction for
    that pipeline) except ``baseline``, which carries the real pinned anchor
    where one exists. That split is deliberate: ``check_anchors`` refuses any
    tree that does not reproduce the two committed cells, so an end-to-end
    fixture has to carry them, while the arithmetic under test stays on values
    whose differences are exact.
    """
    roots = []
    for spec in attribute_effects.ROOT_SPECS:
        by_arm = {arm: EXACT[spec.pipeline] for arm in runs.MODEL_KEYS.values()}
        anchor = attribute_effects.ANCHORS.get((spec.pipeline, "baseline"))
        if anchor is not None:
            by_arm["baseline"] = anchor
        roots.append(_make_root(
            tmp_path,
            "results_%s" % spec.pipeline.replace("-", "_"),
            by_arm,
            pipeline=spec.pipeline if pipelines else None,
        ))
    return roots


# ---------------------------------------------------------------------------
# 1. The grid and the arithmetic
# ---------------------------------------------------------------------------

def test_collect_reads_the_top10_cell_at_threshold_06(tmp_path, capsys):
    collected = attribute_effects.collect(_anchored_roots(tmp_path))
    capsys.readouterr()

    assert collected.arms == list(runs.MODEL_KEYS.values())
    assert collected.values[attribute_effects.LEGACY]["baseline"] == (
        attribute_effects.ANCHORS[(attribute_effects.LEGACY, "baseline")]
    )
    assert collected.values[attribute_effects.CORRECTED]["baseline"] == (
        attribute_effects.ANCHORS[(attribute_effects.CORRECTED, "baseline")]
    )
    for spec in attribute_effects.ROOT_SPECS:
        assert collected.values[spec.pipeline]["bluebert"] == EXACT[spec.pipeline]


def test_the_threshold_is_a_key_not_a_row_position(tmp_path):
    """A file whose 0.6 row is not first must still yield the 0.6 value.

    ``WRITE_ORDER`` puts 0.9 in row 0 and 0.6 in row 2, which is what the real
    writers emit, so a parser or caller indexing by position would return the
    filler constant here rather than the value under test. Asserting the filler
    is *not* what comes back is what makes this a real check.
    """
    roots = _anchored_roots(tmp_path)
    collected = attribute_effects.collect(roots)
    assert WRITE_ORDER.index(attribute_effects.THRESHOLD) == 2
    assert collected.values[attribute_effects.FOLDS_ONLY]["biomedbert"] == 0.375
    assert collected.values[attribute_effects.FOLDS_ONLY]["biomedbert"] != 0.111111


def test_attribute_splits_the_drop_into_named_parts():
    values = {pipeline: {"baseline": value} for pipeline, value in EXACT.items()}
    parts = attribute_effects.attribute(values, "baseline")

    assert parts["legacy"] == 0.5
    assert parts["leakage"] == 0.125          # 0.5    - 0.375
    assert parts["preprocessing"] == 0.0625   # 0.5    - 0.4375
    assert parts["drop"] == 0.25              # 0.5    - 0.25
    assert parts["residual"] == 0.0625        # 0.25 - 0.125 - 0.0625


def test_the_identity_holds_exactly_on_binary_fractions():
    """leakage + preprocessing + residual == drop, by construction.

    Pinned because the residual is *defined* as the leftover: anything that
    computed it independently -- a second subtraction order, a rounded print fed
    back in -- would leave a published table whose columns no longer sum, and
    nothing else in the script would notice.
    """
    values = {pipeline: {"baseline": value} for pipeline, value in EXACT.items()}
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
        attribute_effects.LEGACY: {"baseline": 0.4824430641821946},
        attribute_effects.FOLDS_ONLY: {"baseline": 0.4351298701298701},
        attribute_effects.PREPROCESS_ONLY: {"baseline": 0.4402597402597403},
        attribute_effects.CORRECTED: {"baseline": 0.3922168068392324},
    }
    parts = attribute_effects.attribute(values, "baseline")
    total = parts["leakage"] + parts["preprocessing"] + parts["residual"]
    assert total == pytest.approx(parts["drop"], abs=1e-15)
    # The bundled drop is the published 0.090 figure this script exists to split.
    assert round(parts["drop"], 4) == 0.0902


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
    which one of them left the ceiling -- and, worse, would stop printing it for a
    fifth arm that joined.
    """
    values = {
        spec.pipeline: {"baseline": 0.4, "bluebert": 1.0, "biomedbert": 0.9}
        for spec in attribute_effects.ROOT_SPECS
    }
    assert attribute_effects.saturated_arms(
        values, ["baseline", "bluebert", "biomedbert"]
    ) == ["bluebert"]


def test_report_prints_the_banner_the_grid_and_the_attribution(tmp_path, capsys):
    collected = attribute_effects.collect(_anchored_roots(tmp_path))
    capsys.readouterr()
    attribute_effects.report(collected)
    out = capsys.readouterr().out

    # The banner is mandatory on every output, top and bottom: a reader who
    # copies only the tail of this report must still be told what
    # preprocess-only is for.
    assert out.count("PREPROCESS-ONLY NUMBERS ARE ATTRIBUTION INPUTS, NEVER RESULTS.") == 2
    assert "src/aicds/config.py:143-145" in out
    # The residual is labelled as reportable, so nobody reads it as a failed check.
    assert "THE RESIDUAL IS A REPORTABLE NUMBER, NOT AN ERROR" in out
    # Display names from aicds.runs.KEY_LABELS, not directory keys.
    assert "BioSentVec" in out and "Bio_ClinicalBERT" in out
    assert "0.482443" in out and "0.392217" in out
    assert "+0.090226" in out


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


def test_an_existing_root_with_no_runs_is_refused(tmp_path):
    """An empty directory is not "zero effect" -- it is an unusable column."""
    roots = _anchored_roots(tmp_path)
    empty = os.path.join(str(tmp_path), "results_empty")
    os.makedirs(empty)
    roots[2] = empty

    with pytest.raises(SystemExit) as caught:
        attribute_effects.collect(roots)
    assert "holds no runs" in str(caught.value)


def test_an_arm_missing_from_one_root_is_refused(tmp_path):
    """A difference across four roots cannot be taken over three of them."""
    roots = _anchored_roots(tmp_path)
    partial = _make_root(
        tmp_path, "results_partial",
        {"baseline": 0.4, "bluebert": 0.4},
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
    _write_index(os.path.join(extra, runs.PERFORMANCE_INDEX), 0.4)

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
    assert "not a complete 10-fold run" in str(caught.value)


def test_wrong_number_of_roots_is_refused(tmp_path):
    with pytest.raises(SystemExit) as caught:
        attribute_effects.resolve_roots(["one", "two"])
    message = str(caught.value)
    assert "exactly 4 paths" in message
    assert "legacy folds-only preprocess-only corrected" in message


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
    assert attribute_effects.check_anchors(collected.values) == 2


def test_a_wrong_anchor_value_refuses_and_says_the_parse_is_untrusted(tmp_path, capsys):
    """One digit off in the legacy baseline cell, and no table is printed.

    The anchors are the only thing verifying that the parse lines columns up with
    headers, so a mismatch has to stop the run rather than annotate it -- an
    unverified table looks exactly like a verified one.
    """
    roots = _anchored_roots(tmp_path)
    wrong = os.path.join(str(tmp_path), "results_legacy", "baseline", STAMP)
    _write_index(os.path.join(wrong, runs.PERFORMANCE_INDEX), 0.4824430641821000)

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


def test_the_anchor_comparison_is_exact_not_rounded(tmp_path, capsys):
    """A value that rounds to the same four decimals still fails.

    ``compare_models`` compares to 4 dp because its expectations were transcribed
    by hand; these were read back at full precision, so the stricter comparison
    is available and is the one worth having -- P == R == FS in most of these
    rows, and a rounded check would pass while the parser read a neighbour.
    """
    roots = _anchored_roots(tmp_path)
    near = os.path.join(str(tmp_path), "results_corrected", "baseline", STAMP)
    _write_index(os.path.join(near, runs.PERFORMANCE_INDEX), 0.39221680683)

    collected = attribute_effects.collect(roots)
    capsys.readouterr()
    assert round(collected.values[attribute_effects.CORRECTED]["baseline"], 4) == 0.3922
    with pytest.raises(SystemExit):
        attribute_effects.check_anchors(collected.values)


def test_zero_anchors_refuses_rather_than_reporting_success():
    """"0 anchors passed" and "verified" print identically, so refuse."""
    with pytest.raises(SystemExit) as caught:
        attribute_effects.check_anchors({"legacy": {"bluebert": 0.5}})
    assert "zero parser anchors ran" in str(caught.value)


def test_anchors_are_the_two_committed_baseline_cells():
    """Pinned literals, cross-checked against ``compare_models``' own table.

    Both derive from the same two committed trees by different routes: that table
    was transcribed by hand to four decimals off the run output, these were read
    back at full precision. Agreement is what makes either one credible, and this
    assertion fails loudly if one of them is edited alone.
    """
    import scripts.compare_models as compare_models

    expected = {
        "legacy": 0.4824,
        "corrected": 0.3922,
    }
    checked = 0
    for pipeline, want in expected.items():
        table = compare_models.SANITY_CHECKS_BY_PIPELINE[pipeline]
        rows = [
            row for row in table
            if row[0] == "baseline"
            and row[1] == attribute_effects.STRATEGY
            and row[2] == attribute_effects.THRESHOLD
        ]
        assert len(rows) == 1, pipeline
        assert rows[0][3]["FS"] == want
        anchor = attribute_effects.ANCHORS[(pipeline, "baseline")]
        assert round(anchor, 4) == want
        checked += 1
    assert checked == 2


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
        by_arm = {arm: EXACT[spec.pipeline] for arm in runs.MODEL_KEYS.values()}
        anchor = attribute_effects.ANCHORS.get((spec.pipeline, "baseline"))
        if anchor is not None:
            by_arm["baseline"] = anchor
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
    """Every committed tree predates P14, so absence cannot be fatal.

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
    assert attribute_effects.main(["--roots"] + _anchored_roots(tmp_path, pipelines=True)) == 0
    out = capsys.readouterr().out
    assert "[SUCCESS] 2 parser anchors passed" in out
    assert "F-SCORE GRID" in out
    assert "ATTRIBUTION" in out
    assert "PREPROCESS-ONLY NUMBERS ARE ATTRIBUTION INPUTS" in out

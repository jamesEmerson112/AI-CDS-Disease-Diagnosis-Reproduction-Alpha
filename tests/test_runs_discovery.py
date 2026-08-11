"""``aicds.runs.discover``: the one rule that replaced seven discovery rules.

Before this, six-to-seven private rules disagreed about the base directory
(``docs/`` / the repo root / the cwd / a recursive ``**`` / a pytest tmp dir), the
glob (``Prediction_Output_*`` -- which matched *no baseline output at all*, since
the baseline wrote ``Prediction Output_`` with a space), and the marker file. See
``docs/findings/10-output-path-fragmentation.md``.

The replacement has one test for what a run is -- *it contains
``PerformanceIndex.txt``* -- and two accepted layouts. The tests here are organised
around the three ways that can go wrong:

1. **Classification.** Flat, nested, and both in one tree.
2. **Refusal.** A ``PerformanceIndex.txt`` in a directory fitting neither layout
   raises and names the path, instead of being skipped. Skipping is how a run
   vanishes from a comparison table with nobody noticing it was ever there, which
   is the failure mode the fragmentation caused in the first place.
3. **Selection.** One run per arm by *mtime*, announced when there was a choice.
   ``DDMMYYYY`` does not sort chronologically, so the lexical answer and the
   chronological answer genuinely differ on this project's own data --
   :func:`test_mtime_beats_lexical_order_when_they_disagree` builds exactly that
   disagreement.
"""

from __future__ import annotations

import os

import pytest

from aicds import runs


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(REPO_ROOT, "docs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(root, *parts):
    """Create a run directory (with a parsable-enough marker) and return its path.

    The marker's *content* is irrelevant to discovery -- ``discover`` never opens
    it, which is why a broken run is still discoverable and reported by whoever
    tries to read it.
    """
    path = os.path.join(str(root), *parts)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "PerformanceIndex.txt"), "w", encoding="utf-8") as f:
        f.write("marker\n")
    return path


def _touch(path, name, text="x\n"):
    with open(os.path.join(path, name), "w", encoding="utf-8") as f:
        f.write(text)


def _set_mtime(path, epoch):
    os.utime(path, (epoch, epoch))


# ---------------------------------------------------------------------------
# 1. Classification
# ---------------------------------------------------------------------------

def test_flat_classification_on_the_committed_docs_runs():
    """The three committed BERT runs, discovered from the real ``docs/`` tree.

    A real fixture rather than a synthetic one on purpose: these directory names
    are the ones the flat regex has to survive, and ``Bio_ClinicalBERT`` is the
    reason the label group is greedy -- the model name contains an underscore, so a
    lazy match would split it as label ``Bio`` and leave the rest unparsed.
    """
    discovered = runs.discover(DOCS)

    assert [run.key for run in discovered] == [
        "bio_clinical_bert", "biomedbert", "bluebert"
    ]
    assert [run.label for run in discovered] == [
        "Bio_ClinicalBERT", "BiomedBERT", "BlueBERT"
    ]
    assert [run.stamp for run in discovered] == [
        "15022026_11-33-48", "15022026_12-03-36", "15022026_12-24-38"
    ]
    assert [run.date for run in discovered] == ["2026-02-15"] * 3
    for run in discovered:
        assert os.path.isfile(run.performance_index)
        assert run.performance_index == os.path.join(run.path, "PerformanceIndex.txt")


def test_nested_classification_on_a_tmp_tree(tmp_path):
    """``ROOT/<key>/<stamp>/`` -- the layout ``--out`` writes and results*/ uses."""
    _make_run(tmp_path, "baseline", "05082026_18-55-32")
    _make_run(tmp_path, "bluebert", "05082026_19-52-30")

    discovered = runs.discover(str(tmp_path))

    assert [(run.key, run.label, run.stamp) for run in discovered] == [
        ("baseline", "BioSentVec", "05082026_18-55-32"),
        ("bluebert", "BlueBERT", "05082026_19-52-30"),
    ]
    assert [run.date for run in discovered] == ["2026-08-05", "2026-08-05"]


def test_flat_and_nested_in_one_tree(tmp_path):
    """Both layouts coexist, and nothing has to be told which is which.

    This is the shape a half-migrated tree has: an arm harvested into
    ``<key>/<stamp>/`` next to one still sitting where it was written.
    """
    _make_run(tmp_path, "Prediction_Output_BiomedBERT_06082026_02-41-45")
    _make_run(tmp_path, "bluebert", "06082026_02-53-18")

    discovered = runs.discover(str(tmp_path))

    assert [(run.key, run.stamp) for run in discovered] == [
        ("biomedbert", "06082026_02-41-45"),
        ("bluebert", "06082026_02-53-18"),
    ]


def test_a_runs_own_children_are_never_examined(tmp_path):
    """The walk stops at a run; ``Fold*/`` is not searched.

    Not a micro-optimisation. A run writes 10 fold directories and (per P40) up to
    258 per-case files, so an unbounded walk would descend into every one of them
    for every run it already found -- and a stray ``PerformanceIndex.txt`` under
    ``Fold0/`` would then be classified as a run in its own right and raise.
    """
    run_path = _make_run(tmp_path, "baseline", "05082026_18-55-32")
    fold = os.path.join(run_path, "Fold0")
    os.makedirs(fold)
    _touch(fold, "PerformanceIndex.txt")

    discovered = runs.discover(str(tmp_path))
    assert [run.path for run in discovered] == [run_path]


def test_a_directory_without_the_marker_is_not_a_run(tmp_path):
    """One rule, and no ``marker=`` parameter to soften it.

    A directory carrying ``RankMetrics.txt`` but no ``PerformanceIndex.txt`` is not
    a run. That is deliberate: making the marker configurable would let "this run
    predates P5" and "this directory is not a run" produce the same silence, and
    the second is a broken pipeline. Callers ask a *discovered* run for its rank
    metrics via :meth:`Run.file` -- see the accessor test below.
    """
    orphan = os.path.join(str(tmp_path), "bluebert", "06082026_15-53-08")
    os.makedirs(orphan)
    _touch(orphan, "RankMetrics.txt")

    assert runs.discover(str(tmp_path)) == []


def test_empty_and_marker_free_roots_return_no_runs(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "notes", "drafts"))
    _touch(str(tmp_path), "README.md")
    assert runs.discover(str(tmp_path)) == []


def test_missing_root_raises(tmp_path):
    with pytest.raises(runs.RunLayoutError) as caught:
        runs.discover(os.path.join(str(tmp_path), "nope"))
    assert "not a directory" in str(caught.value)


# ---------------------------------------------------------------------------
# 2. Refusal
# ---------------------------------------------------------------------------

def test_unclassifiable_run_raises_and_names_the_path(tmp_path):
    """A marker in a directory fitting neither layout is surfaced, not skipped."""
    path = _make_run(tmp_path, "scratch-rerun")

    with pytest.raises(runs.RunLayoutError) as caught:
        runs.discover(str(tmp_path))

    message = str(caught.value)
    assert path in message
    assert "fits neither run layout" in message
    # Both accepted shapes are quoted, so the message is actionable on its own.
    assert "Prediction_Output_" in message and "<model_key>" in message


def test_flat_run_one_level_too_deep_says_so(tmp_path):
    """Right shape, wrong depth -- the mistake ``discover(repo_root)`` makes.

    The three committed runs live in ``docs/``, so a caller who passes the
    repository root sees them at depth 2, where only the nested shape is legal.
    Rather than a bare refusal the message names the parent to pass instead, which
    turns the most likely wrong call into instructions.
    """
    _make_run(tmp_path, "docs", "Prediction_Output_BlueBERT_06082026_02-53-18")

    with pytest.raises(runs.RunLayoutError) as caught:
        runs.discover(str(tmp_path))

    message = str(caught.value)
    assert "matches the flat shape but sits TWO levels below the root" in message
    assert os.path.join(str(tmp_path), "docs") in message


def test_nested_run_one_level_too_shallow_says_so(tmp_path):
    """The mirror image: ``discover('results/baseline')`` instead of ``results``."""
    _make_run(tmp_path, "05082026_18-55-32")

    with pytest.raises(runs.RunLayoutError) as caught:
        runs.discover(str(tmp_path))

    message = str(caught.value)
    assert "matches the nested shape but sits ONE level below the root" in message


def test_flat_run_with_an_unregistered_model_name_raises(tmp_path):
    """A flat name carries a *display name*, and MODEL_KEYS is the authority.

    Guessing ``label.lower()`` is the exact mistake ``model_key`` refuses: it mints
    ``bio_clinicalbert`` where every reader spells ``bio_clinical_bert``, splitting
    one arm's history across two directories without failing.
    """
    _make_run(tmp_path, "Prediction_Output_MysteryBERT_06082026_02-53-18")

    with pytest.raises(runs.RunLayoutError) as caught:
        runs.discover(str(tmp_path))

    message = str(caught.value)
    assert "MysteryBERT" in message
    assert "MODEL_KEYS" in message


def test_depth_1_finds_a_flat_run_beside_a_subtree_that_would_otherwise_raise(tmp_path):
    """The repository root, in miniature -- and the regression this bounds.

    ``docs/`` holds committed flat runs at depth 2, where only the nested shape is
    legal, so the default walk raises on the whole root however healthy the rest of
    it is: the exception propagates out of the walk instead of the offending
    subtree being skipped, and the valid depth-1 run is never returned. That is
    correct for a caller who *named* this root, and useless for
    ``analyze_performance``'s no-argument auto-detect, which runs from the
    repository root by instruction.
    """
    fresh = _make_run(tmp_path, "Prediction_Output_BlueBERT_09082026_10-00-00")
    _make_run(tmp_path, "docs", "Prediction_Output_BiomedBERT_15022026_12-03-36")

    with pytest.raises(runs.RunLayoutError):
        runs.discover(str(tmp_path))

    assert [run.path for run in runs.discover(str(tmp_path), depth=1)] == [fresh]


def test_depth_1_does_not_see_nested_runs(tmp_path):
    """The cost of the bound, stated: a ``results*/`` tree needs the full walk.

    Not a silent loss -- ``analyze_performance`` prints that auto-detect only sees
    runs written straight into the cwd, and such a tree is passed by name.
    """
    _make_run(tmp_path, "baseline", "05082026_18-55-32")

    assert runs.discover(str(tmp_path), depth=1) == []
    assert [run.key for run in runs.discover(str(tmp_path))] == ["baseline"]


def test_depth_1_still_refuses_an_unclassifiable_run_in_the_root(tmp_path):
    """The bound narrows *which* directories are looked at, not how strictly.

    A marker sitting directly in the root is in scope at either depth, so the
    refusal survives; the bound must not become a way to make bad layouts quiet.
    """
    _make_run(tmp_path, "scratch-rerun")

    with pytest.raises(runs.RunLayoutError):
        runs.discover(str(tmp_path), depth=1)


def test_depth_outside_1_or_2_raises(tmp_path):
    """No deeper layout exists, so an unbounded 3 would only reach ``Fold*/``."""
    with pytest.raises(ValueError) as caught:
        runs.discover(str(tmp_path), depth=3)
    assert "depth must be 1" in str(caught.value)


def test_nested_run_with_an_unregistered_key_is_accepted(tmp_path):
    """Asymmetric with the flat case above, and deliberately so.

    A nested directory name *is* the key -- the tree asserts it, so there is
    nothing to guess. Accepting it preserves what ``compare_models`` did: a new
    arm shows up in the report styled from ``FALLBACK_STYLES``, rather than being
    dropped from a four-model comparison without a word. Its label falls back to
    the key, because no display name is knowable.
    """
    _make_run(tmp_path, "mystery_encoder", "06082026_02-53-18")

    discovered = runs.discover(str(tmp_path))
    assert [(run.key, run.label) for run in discovered] == [
        ("mystery_encoder", "mystery_encoder")
    ]


# ---------------------------------------------------------------------------
# 3. Selection and ordering
# ---------------------------------------------------------------------------

def test_mtime_beats_lexical_order_when_they_disagree(tmp_path, capsys):
    """The tiebreak, on a disagreement taken from this project's real data.

    ``results/bio_clinical_bert/`` holds ``15022026_11-33-48`` (February) and
    ``05082026_19-24-00`` (August). Day-first stamps sort the February run *last*
    lexically while it is five months older, so a name sort silently reports the
    stale run. mtime is the only ordering that is actually chronological here.
    """
    older = _make_run(tmp_path, "bio_clinical_bert", "15022026_11-33-48")
    newer = _make_run(tmp_path, "bio_clinical_bert", "05082026_19-24-00")
    assert sorted([os.path.basename(older), os.path.basename(newer)])[-1] == (
        "15022026_11-33-48"
    ), "the lexical answer must be the WRONG one, or this test proves nothing"

    _set_mtime(older, 1_600_000_000)
    _set_mtime(newer, 1_700_000_000)

    discovered = runs.discover(str(tmp_path))
    assert [run.stamp for run in discovered] == ["05082026_19-24-00"]

    # Printed, not silent: choosing between runs is exactly where a report starts
    # describing last week's numbers.
    out = capsys.readouterr().out
    assert "bio_clinical_bert has 2 runs; using the most recent (05082026_19-24-00)" in out


def test_single_run_per_key_prints_nothing(tmp_path, capsys):
    _make_run(tmp_path, "bluebert", "06082026_02-53-18")
    runs.discover(str(tmp_path))
    assert capsys.readouterr().out == ""


def test_all_runs_returns_every_run_mtime_ascending(tmp_path):
    older = _make_run(tmp_path, "bluebert", "15022026_12-24-38")
    newer = _make_run(tmp_path, "bluebert", "05082026_19-52-30")
    _set_mtime(older, 1_600_000_000)
    _set_mtime(newer, 1_700_000_000)

    discovered = runs.discover(str(tmp_path), all_runs=True)
    assert [run.stamp for run in discovered] == [
        "15022026_12-24-38", "05082026_19-52-30"
    ]


def test_ordering_is_model_keys_then_unregistered_alphabetically(tmp_path):
    """Registry order first, strangers after -- lifted from ``compare_models``.

    The four arms are created here in reverse and in a directory order
    (``baseline`` sorts first alphabetically by luck, ``bluebert`` last) that does
    not distinguish the two rules, so two unregistered keys are added: they must
    land after all four registered ones and alphabetically between themselves.
    """
    for key in ("zeta_encoder", "bluebert", "alpha_encoder", "biomedbert",
                "bio_clinical_bert", "baseline"):
        _make_run(tmp_path, key, "06082026_02-53-18")

    discovered = runs.discover(str(tmp_path))
    assert [run.key for run in discovered] == [
        "baseline", "bio_clinical_bert", "biomedbert", "bluebert",
        "alpha_encoder", "zeta_encoder",
    ]
    assert list(runs.MODEL_KEYS.values()) == [
        "baseline", "bio_clinical_bert", "biomedbert", "bluebert"
    ]


# ---------------------------------------------------------------------------
# 4. The Run value object
# ---------------------------------------------------------------------------

def test_file_accessor_returns_a_path_or_none(tmp_path):
    """``file()`` is how a caller asks for a sibling artifact without a new glob."""
    path = _make_run(tmp_path, "bluebert", "06082026_15-53-01")
    _touch(path, "RankMetrics.txt")

    run = runs.discover(str(tmp_path))[0]

    assert run.file("RankMetrics.txt") == os.path.join(path, "RankMetrics.txt")
    assert os.path.isfile(run.file("RankMetrics.txt"))
    # None, not a non-existent path and not an exception: a pre-P5 run simply did
    # not write this, and the caller's job is to say so and carry on.
    assert run.file("RankMetrics.txt".replace("Rank", "Nope")) is None
    # A directory is not a file, so it does not answer either.
    os.makedirs(os.path.join(path, "symptom_details"))
    assert run.file("symptom_details") is None


def test_run_is_frozen(tmp_path):
    _make_run(tmp_path, "bluebert", "06082026_02-53-18")
    run = runs.discover(str(tmp_path))[0]
    with pytest.raises(Exception):
        run.key = "baseline"


@pytest.mark.parametrize(
    "stamp, expected",
    [
        ("05082026_18-55-32", "2026-08-05"),
        ("15022026_11-33-48", "2026-02-15"),
        # Unparseable input shows itself rather than a wrong date.
        ("not-a-stamp", "not-a-stamp"),
        ("2026-08-05", "2026-08-05"),
    ],
)
def test_format_run_date(stamp, expected):
    assert runs.format_run_date(stamp) == expected


def test_key_labels_is_the_exact_inverse_of_model_keys():
    """Derived, not written twice, so the two directions cannot drift."""
    assert runs.KEY_LABELS == {v: k for k, v in runs.MODEL_KEYS.items()}
    assert len(runs.KEY_LABELS) == len(runs.MODEL_KEYS)

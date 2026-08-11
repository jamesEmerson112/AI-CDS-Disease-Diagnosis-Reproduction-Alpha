"""``analyze_performance``'s no-argument auto-detect, run from where the docs say.

The explicit-dir form (``python scripts/analyze_performance.py <run>``) was the only
one anything exercised, which is how the default form came to hard-exit unnoticed:
``discover`` refuses -- correctly -- a ``PerformanceIndex.txt`` in neither layout,
and the repository root permanently contains three of them, because the committed
``docs/Prediction_Output_*`` oracle runs are *flat* runs sitting at depth 2. Every
no-argument invocation from the root therefore printed a layout error naming
``docs/`` and exited 1, whatever else was present.

``docs/`` is committed, read-only and never going anywhere, and CLAUDE.md says to
run from the repository root, so this is not a state the user can clear. These tests
pin the caller's side of the fix; the depth bound itself is tested in
``test_runs_discovery.py``.
"""

from __future__ import annotations

import os

import pytest

import scripts.analyze_performance as analyze_performance


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_run(root, *parts):
    path = os.path.join(str(root), *parts)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "PerformanceIndex.txt"), "w", encoding="utf-8") as f:
        f.write("marker\n")
    return path


def test_auto_detect_from_the_repository_root_does_not_exit(monkeypatch):
    """The regression, on the real tree rather than a mock of it.

    ``None`` -- "nothing written here yet" -- is the honest answer from a clean
    root, and ``resolve_run_directory`` turns it into an instruction. A path is
    equally fine and means someone left a run lying around; what must never happen
    again is ``SystemExit``, which is what ``docs/`` used to force.
    """
    monkeypatch.chdir(REPO_ROOT)

    directory = analyze_performance.find_latest_output_directory()

    if directory is not None:
        assert os.path.isfile(os.path.join(directory, "PerformanceIndex.txt"))


def test_auto_detect_finds_a_fresh_run_beside_a_docs_subtree(monkeypatch, tmp_path):
    """The root's shape reproduced: a fresh flat run *and* a docs/ tree of runs.

    Before the depth bound this raised, so the fresh run was unreachable even
    though it was the only thing the user asked about.
    """
    fresh = _make_run(tmp_path, "Prediction_Output_BlueBERT_09082026_10-00-00")
    _make_run(tmp_path, "docs", "Prediction_Output_BiomedBERT_15022026_12-03-36")
    monkeypatch.chdir(tmp_path)

    assert analyze_performance.find_latest_output_directory() == fresh


def test_auto_detect_picks_the_newest_flat_run_by_mtime(monkeypatch, tmp_path):
    """Across arms, "most recent" is mtime -- ``DDMMYYYY`` does not sort by date.

    February sorts last lexically while being five months older, so a name sort
    would report the stale run here.
    """
    older = _make_run(tmp_path, "Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48")
    newer = _make_run(tmp_path, "Prediction_Output_BlueBERT_05082026_19-52-30")
    os.utime(older, (1_600_000_000, 1_600_000_000))
    os.utime(newer, (1_700_000_000, 1_700_000_000))
    monkeypatch.chdir(tmp_path)

    assert analyze_performance.find_latest_output_directory() == newer


def test_empty_cwd_reports_no_run_rather_than_a_layout_error(monkeypatch, tmp_path, capsys):
    """The message a user with nothing to plot should get, and its one caveat.

    It names the auto-detect's limit explicitly -- a run harvested into
    ``results*/<model>/<timestamp>/`` is nested, so it has to be passed by name.
    Saying so is what keeps the depth bound from reading as a run silently
    vanishing.
    """
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as caught:
        analyze_performance.resolve_run_directory(None)

    assert caught.value.code == 1
    out = capsys.readouterr().out
    assert "No Prediction_Output_* run directory found in the cwd" in out
    assert "results*/<model>/<timestamp>/" in out


def test_an_unclassifiable_run_in_the_cwd_still_exits(monkeypatch, tmp_path, capsys):
    """The bound must not have turned a broken layout into silence.

    A marker directly in the cwd is in scope at depth 1, so it is still refused by
    name; only *subtrees* the caller did not ask about stopped being opened.
    """
    _make_run(tmp_path, "scratch-rerun")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as caught:
        analyze_performance.find_latest_output_directory()

    assert caught.value.code == 1
    assert "fits neither run layout" in capsys.readouterr().out

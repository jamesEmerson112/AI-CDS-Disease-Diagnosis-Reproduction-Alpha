"""The ``.gitignore`` guard on root-level run output.

A run leaves per-case files NAMED by ``HADM_ID`` under ``Fold*/``.  Those files
are DUA-covered clinical data in a PUBLIC repo -- and because every one of them
is currently EMPTY (``cython_utils`` opens the handles and closes them with
nothing written in between), ``.githooks/pre-commit``'s content rule, which
greps for 20+ distinct HADM_IDs *inside* a file, cannot see them at all.  The
path rule in ``.gitignore`` is the only thing that stops ``git add -A``.

The anchoring is equally load-bearing in the other direction:
``docs/Prediction_Output_*`` is the project's committed regression oracle and
must stay tracked.  An unanchored pattern would match it at any depth -- the
same failure that once had bare ``entity/`` swallowing ``src/entity/``.

These tests create nothing.  ``git check-ignore`` answers for paths that do not
exist, so the assertions are pure queries against the committed rules.
"""

import os
import subprocess

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_available():
    try:
        subprocess.run(
            ["git", "--version"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


pytestmark = pytest.mark.skipif(not _git_available(), reason="git is not available")


def is_ignored(path):
    """True if git would ignore ``path``.  Does not touch the filesystem."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", path],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(
            "git check-ignore failed for %r: %s" % (path, result.stderr.decode(errors="replace"))
        )
    return result.returncode == 0


class TestRootRunOutputIsIgnored:
    def test_a_per_case_file_named_by_hadm_id_is_ignored(self):
        """The exact shape a baseline run leaves behind, empty and unhookable."""
        assert is_ignored("Prediction_Output_BioSentVec_01012026_00-00-00/Fold0/999001.txt")

    def test_the_space_spelling_is_ignored_too(self):
        """``current_time()`` scrubs only '/' and ':', so the space survives.

        Pre-rename SHAs still emit this spelling; checking one out and running
        it must not leave unignored clinical data.
        """
        assert is_ignored("Prediction Output_01012026 00-00-00/x.txt")

    def test_both_symptom_details_spellings_are_ignored(self):
        assert is_ignored("Prediction_Symptom_Details_BiomedBERT_01012026_00-00-00/a.txt")
        assert is_ignored("Prediction Symptom Details_01012026 00-00-00/a.txt")

    def test_any_results_tree_is_ignored(self):
        """The rule is a glob (``results*/``), not a hand-kept list.

        It became a glob after a third pipeline's output directory was about to
        be downloaded into a tree nothing ignored.
        """
        assert is_ignored("results_x/foo.txt")
        assert is_ignored("results/baseline/01012026_00-00-00/PerformanceIndex.txt")


class TestTheCommittedOracleStaysTracked:
    def test_docs_prediction_output_is_not_ignored(self):
        """Anchoring is what makes the guard safe to add at all."""
        assert not is_ignored(
            "docs/Prediction_Output_BiomedBERT_15022026_12-03-36/PerformanceIndex.txt"
        )

    def test_the_oracle_is_really_there(self):
        """Belt and braces: the path above is not hypothetical."""
        assert os.path.isfile(
            os.path.join(
                PROJECT_ROOT,
                "docs",
                "Prediction_Output_BiomedBERT_15022026_12-03-36",
                "PerformanceIndex.txt",
            )
        )

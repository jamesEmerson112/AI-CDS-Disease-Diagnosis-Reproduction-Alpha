"""The run-directory contract, pinned.

``aicds.runs`` decides where every run writes. Two of its properties are load
bearing far away from it:

1. The **default** (``out=None``) layout is frozen by ``tests/test_golden.py``,
   which chdirs into a temporary directory, globs ``Prediction_Output_*``
   non-recursively, asserts **exactly one** match, and cross-checks it against
   ``run_analysis``'s return value. A run that emitted two matching directories
   -- say by naming the symptom-details sibling ``Prediction_Output_Details_`` --
   would fail the golden with a message about directory counts, nowhere near the
   line that caused it. So the flat names are asserted here as exact strings, and
   the one-match property is asserted directly.
2. The **nested** (``out=ROOT``) layout must reproduce the hand-assembled
   ``results*/`` trees that ``scripts/compare_models.py --results-dir`` and
   ``scripts/analyze_rank_metrics.py`` already read. Its directory keys are the
   ones committed on disk, which is why ``MODEL_KEYS`` is a table and not
   ``name.lower()``.

The timestamp regex is **duplicated** from ``test_golden.py`` rather than
imported. Importing it would let a future edit relax both pins at once; two
independent copies mean the golden's own regex still has to be changed
deliberately.
"""

import fnmatch
import glob
import os
import re
import subprocess
import time

import pytest

from aicds import runs


# Duplicated from tests/test_golden.py:115 ON PURPOSE. Do not import it.
TIMESTAMP_RE = re.compile(r"^\d{8}_\d{2}-\d{2}-\d{2}$")

# A stamp that is a legal member of the format but obviously synthetic, so a
# real run's output can never be confused with a test's.
FIXED_STAMP = "01022003_04-05-06"

ALL_NAMES = ["BioSentVec", "Bio_ClinicalBERT", "BiomedBERT", "BlueBERT"]


class TestModelKeys:
    def test_table_is_the_four_arms(self):
        assert runs.MODEL_KEYS == {
            "BioSentVec": "baseline",
            "Bio_ClinicalBERT": "bio_clinical_bert",
            "BiomedBERT": "biomedbert",
            "BlueBERT": "bluebert",
        }

    def test_unknown_name_raises(self):
        """A new encoder is a new arm and needs a deliberate entry."""
        with pytest.raises(KeyError) as excinfo:
            runs.model_key("PubMedBERT")
        message = str(excinfo.value)
        assert "PubMedBERT" in message
        assert "MODEL_KEYS" in message

    def test_lowercasing_would_have_split_an_arm(self):
        """The reason the table exists, stated as an assertion.

        ``"Bio_ClinicalBERT".lower()`` is ``"bio_clinicalbert"``; the committed
        results tree and compare_models both spell it ``bio_clinical_bert``.
        """
        assert runs.model_key("Bio_ClinicalBERT") != "Bio_ClinicalBERT".lower()
        assert runs.model_key("Bio_ClinicalBERT") == "bio_clinical_bert"


class TestTimestamp:
    def test_generated_stamp_matches_the_golden_pattern(self):
        assert TIMESTAMP_RE.match(runs.timestamp())

    def test_explicit_time_is_formatted_day_first(self):
        # 2003-02-01 04:05:06 -> day, month, year.
        moment = time.struct_time((2003, 2, 1, 4, 5, 6, 5, 32, -1))
        assert runs.timestamp(moment) == FIXED_STAMP

    def test_run_dirs_generates_a_matching_stamp_when_none_given(self, tmp_path):
        result = runs.run_dirs("BiomedBERT", out=str(tmp_path))
        assert TIMESTAMP_RE.match(result.stamp)
        assert os.path.basename(result.root.rstrip(os.sep)) == result.stamp


class TestDefaultLayout:
    """out=None -- the shape the golden pins. Exact strings only."""

    @pytest.mark.parametrize("model_name", ALL_NAMES)
    def test_root_and_details_names(self, model_name, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runs.run_dirs(model_name, stamp=FIXED_STAMP)

        # os.getcwd() + '/' is exactly how both arms built this before runs.py.
        cwd = os.getcwd()
        assert result.root == (
            cwd + "/Prediction_Output_" + model_name + "_" + FIXED_STAMP + "/"
        )
        assert result.details_root == (
            cwd
            + "/Prediction_Symptom_Details_"
            + model_name
            + "_"
            + FIXED_STAMP
            + "/"
        )
        assert result.stamp == FIXED_STAMP
        assert result.key == runs.MODEL_KEYS[model_name]

    @pytest.mark.parametrize("model_name", ALL_NAMES)
    def test_details_is_a_sibling_not_a_child(self, model_name, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runs.run_dirs(model_name, stamp=FIXED_STAMP)
        assert not result.details_root.startswith(result.root)
        assert os.path.dirname(result.root.rstrip("/")) == os.path.dirname(
            result.details_root.rstrip("/")
        )

    @pytest.mark.parametrize("model_name", ALL_NAMES)
    def test_glob_pattern_semantics(self, model_name, tmp_path, monkeypatch):
        """``glob.glob("Prediction_Output_*")`` matches the root and nothing else.

        This is the property test_golden.py depends on: exactly one match in the
        run's cwd. The symptom-details sibling must NOT match it.
        """
        monkeypatch.chdir(tmp_path)
        result = runs.run_dirs(model_name, stamp=FIXED_STAMP)

        root_name = os.path.basename(result.root.rstrip("/"))
        details_name = os.path.basename(result.details_root.rstrip("/"))
        assert fnmatch.fnmatch(root_name, "Prediction_Output_*")
        assert not fnmatch.fnmatch(details_name, "Prediction_Output_*")

        # And on a real filesystem, non-recursively, as the golden does it.
        os.makedirs(result.root)
        os.makedirs(result.details_root)
        produced = glob.glob(os.path.join(os.getcwd(), "Prediction_Output_*"))
        assert len(produced) == 1
        assert os.path.basename(produced[0]) == root_name

    def test_committed_docs_directories_are_reproducible(self):
        """The three committed oracle directories come out of run_dirs unchanged.

        ``docs/Prediction_Output_*`` is the project's regression oracle. If the
        default naming ever drifted, fresh runs would stop being comparable to it
        by name -- so derive the names from (model, stamp) and check them against
        what is actually on disk.
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        docs = os.path.join(repo_root, "docs")
        on_disk = sorted(
            name
            for name in os.listdir(docs)
            if name.startswith("Prediction_Output_")
            and os.path.isdir(os.path.join(docs, name))
        )
        assert on_disk == [
            "Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48",
            "Prediction_Output_BiomedBERT_15022026_12-03-36",
            "Prediction_Output_BlueBERT_15022026_12-24-38",
        ]

        # Each name is (model_name, stamp) run back through the writer.
        rebuilt = sorted(
            os.path.basename(runs.run_dirs(name, stamp=stamp).root.rstrip("/"))
            for name, stamp in [
                ("Bio_ClinicalBERT", "15022026_11-33-48"),
                ("BiomedBERT", "15022026_12-03-36"),
                ("BlueBERT", "15022026_12-24-38"),
            ]
        )
        assert rebuilt == on_disk


class TestNestedLayout:
    """out=ROOT -- the results*/ shape the comparison scripts read."""

    @pytest.mark.parametrize("model_name", ALL_NAMES)
    def test_root_is_ROOT_key_stamp(self, model_name, tmp_path):
        out = str(tmp_path / "results")
        result = runs.run_dirs(model_name, out=out, stamp=FIXED_STAMP)
        key = runs.MODEL_KEYS[model_name]

        assert result.key == key
        assert result.root == os.path.join(out, key, FIXED_STAMP) + os.sep
        assert result.root.endswith(os.sep)

    @pytest.mark.parametrize("model_name", ALL_NAMES)
    def test_details_is_nested_inside_the_run(self, model_name, tmp_path):
        out = str(tmp_path / "results")
        result = runs.run_dirs(model_name, out=out, stamp=FIXED_STAMP)

        assert result.details_root == os.path.join(
            out, runs.MODEL_KEYS[model_name], FIXED_STAMP, "symptom_details"
        ) + os.sep
        # Nested, not a sibling: under a shared root a sibling would land beside
        # the timestamps and break "every child of the model dir is a run".
        assert result.details_root.startswith(result.root)

    def test_all_four_arms_land_under_distinct_keys(self, tmp_path):
        out = str(tmp_path / "results")
        keys = [
            runs.run_dirs(name, out=out, stamp=FIXED_STAMP).key for name in ALL_NAMES
        ]
        assert sorted(keys) == [
            "baseline",
            "bio_clinical_bert",
            "biomedbert",
            "bluebert",
        ]

    def test_matches_the_committed_results_tree_shape(self, tmp_path):
        """Reproduce a path from the tree that actually exists on disk.

        ``results/baseline/05082026_18-55-32/`` with ``symptom_details/`` inside
        it is the 2026-08-05 baseline run, hand-moved into place. The nested mode
        exists to make that move unnecessary, so it must land in the same place.
        """
        out = str(tmp_path / "results")
        result = runs.run_dirs("BioSentVec", out=out, stamp="05082026_18-55-32")
        relative = os.path.relpath(result.root.rstrip(os.sep), out)
        assert relative.replace(os.sep, "/") == "baseline/05082026_18-55-32"
        relative_details = os.path.relpath(result.details_root.rstrip(os.sep), out)
        assert (
            relative_details.replace(os.sep, "/")
            == "baseline/05082026_18-55-32/symptom_details"
        )


class TestRunDirsIsFrozen:
    def test_cannot_be_mutated_mid_run(self):
        result = runs.run_dirs("BlueBERT", out="ROOT", stamp=FIXED_STAMP)
        with pytest.raises(Exception):
            result.root = "somewhere_else"


def _git_available():
    try:
        subprocess.run(
            ["git", "--version"],
            cwd=runs.REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


class TestOutRootDataGuard:
    """``--out`` must not aim HADM_ID-named files at an unignored repo path.

    ``.gitignore``'s four anchored root patterns cover the *default* layout, and
    they exist because those per-case files are EMPTY -- so the pre-commit hook's
    content rule (20+ HADM_IDs inside a file) scores them 0 and never sees them.
    ``--out ROOT`` can put the identical filenames one directory over, where no
    rule matches. These tests create nothing: ``git check-ignore`` answers for
    paths that do not exist.
    """

    def test_repo_root_is_the_repo_root(self):
        """Three parents, not four. An off-by-one silently disables the guard."""
        assert os.path.isdir(os.path.join(runs.REPO_ROOT, "src", "aicds"))
        assert os.path.isfile(os.path.join(runs.REPO_ROOT, ".gitignore"))

    def test_default_layout_is_never_checked(self):
        """The golden's path. ``out=None`` returns before git is consulted."""
        assert runs.check_out_root(None) is None

    def test_a_root_outside_the_repo_is_allowed(self, tmp_path):
        out = str(tmp_path / "runs")
        assert runs.check_out_root(out) == os.path.abspath(out)

    def test_no_working_tree_means_nothing_to_stage(self, tmp_path):
        """An unignored path in a directory that is not a git repo is fine."""
        fake_repo = tmp_path / "not-a-repo"
        (fake_repo / "scratch").mkdir(parents=True)
        assert runs.check_out_root(
            str(fake_repo / "scratch"), repo_root=str(fake_repo)
        ) == os.path.abspath(str(fake_repo / "scratch"))

    @pytest.mark.skipif(not _git_available(), reason="git is not available")
    @pytest.mark.parametrize("name", ["scratch", "myruns", "out", "tmp_runs"])
    def test_unignored_repo_internal_roots_are_refused(self, name):
        """``out`` is the sharp one: .gitignore's entry is ``output/``."""
        with pytest.raises(runs.UnignoredOutRoot) as excinfo:
            runs.check_out_root(os.path.join(runs.REPO_ROOT, name))
        message = str(excinfo.value)
        assert ".gitignore" in message
        assert "HADM_ID" in message

    @pytest.mark.skipif(not _git_available(), reason="git is not available")
    @pytest.mark.parametrize("name", ["results", "results_x"])
    def test_the_results_trees_are_allowed(self, name):
        """``--out results`` is the documented happy path (.gitignore results*/)."""
        out = os.path.join(runs.REPO_ROOT, name)
        assert runs.check_out_root(out) == os.path.abspath(out)

    def test_unanswerable_warns_and_continues(self, monkeypatch, capsys):
        """"Don't know" is not "no".

        A user with no working ``git`` cannot stage anything either, so refusing
        would block a legitimate run over a tooling gap -- but the assumption
        expires the moment git appears on PATH, hence the warning.
        """
        monkeypatch.setattr(runs, "_git_ignores", lambda path, repo_root: (False, False))
        out = os.path.join(runs.REPO_ROOT, "scratch")
        assert runs.check_out_root(out) == os.path.abspath(out)
        assert "WARNING" in capsys.readouterr().err

    def test_run_dirs_itself_stays_pure(self, tmp_path):
        """The guard lives at the CLI boundary, not in the path builder.

        ``run_dirs`` is called by library code and by these tests with relative
        roots; making it refuse would couple path construction to a subprocess.
        """
        result = runs.run_dirs("BlueBERT", out="ROOT", stamp=FIXED_STAMP)
        assert result.root == os.path.join("ROOT", "bluebert", FIXED_STAMP) + os.sep

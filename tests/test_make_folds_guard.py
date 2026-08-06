"""The write guard on ``scripts/make_folds.py``.

``data/folds/`` holds the committed legacy folds, which are *inputs* to
``tests/golden/stub768/``.  If a ``--out`` typo overwrites them the byte-exact
golden can never be re-verified again, and the golden is the only thing in this
repo that catches the numbers moving while every other test stays green.  So
the guard is not a nicety -- it protects the safety net itself.

These tests never write to a real fold directory.  Everything that generates
files goes to ``tmp_path``; the only real path touched is ``data/folds``, and
only through ``check_out_dir``, which is pure.
"""

import importlib.util
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "make_folds.py")


def _load():
    spec = importlib.util.spec_from_file_location("make_folds_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_folds = _load()


def _fold_dir(root, index=0):
    path = os.path.join(str(root), "Fold%d" % index)
    os.makedirs(path)
    return path


class TestProtectsTheLegacyFolds:
    def test_refuses_to_write_to_data_folds(self):
        problem = make_folds.check_out_dir("data/folds")
        assert problem is not None
        # The message has to say *why*, or the next person just adds the flag.
        assert "golden" in problem
        assert "--force-legacy" in problem

    def test_refuses_even_when_the_path_is_spelled_indirectly(self):
        """A guard that only string-compares is no guard at all."""
        sneaky = os.path.join("data", "folds_grouped", "..", "folds")
        assert make_folds.check_out_dir(sneaky) is not None

    def test_force_legacy_alone_is_still_not_enough(self):
        """Defence in depth, and it is deliberate.

        ``data/folds`` is both protected *and* populated, so clearing the
        legacy check still leaves the already-has-folds check.  Destroying the
        golden's inputs takes two explicit flags, not one.
        """
        problem = make_folds.check_out_dir("data/folds", force_legacy=True)
        assert problem is not None
        assert "--overwrite" in problem

    def test_both_flags_together_are_the_escape_hatch(self):
        assert make_folds.check_out_dir("data/folds", force_legacy=True, overwrite=True) is None

    def test_the_default_target_is_not_the_protected_one(self):
        """If these ever collide the guard fires on every ordinary run."""
        assert not make_folds._same_path(make_folds.DEFAULT_OUT, make_folds.PROTECTED_OUT)

    def test_main_exits_nonzero_rather_than_writing(self):
        assert make_folds.main(["--out", "data/folds"]) == 2

    def test_the_protected_folds_are_still_there(self):
        """Belt and braces: nothing above may have touched the real thing."""
        assert os.path.isfile(os.path.join(PROJECT_ROOT, "data", "folds", "Fold0", "TestSet.txt"))


class TestProtectsAnyPopulatedDirectory:
    def test_a_fresh_directory_is_fine(self, tmp_path):
        assert make_folds.check_out_dir(str(tmp_path)) is None

    def test_a_directory_that_does_not_exist_yet_is_fine(self, tmp_path):
        assert make_folds.check_out_dir(str(tmp_path / "nope")) is None

    def test_refuses_a_directory_that_already_holds_folds(self, tmp_path):
        _fold_dir(tmp_path)
        problem = make_folds.check_out_dir(str(tmp_path))
        assert problem is not None
        assert "--overwrite" in problem

    def test_overwrite_allows_it(self, tmp_path):
        _fold_dir(tmp_path)
        assert make_folds.check_out_dir(str(tmp_path), overwrite=True) is None

    def test_unrelated_contents_do_not_trip_it(self, tmp_path):
        """Only Fold*/ directories count -- a stray README must not block a run."""
        (tmp_path / "README.md").write_text("notes")
        os.makedirs(str(tmp_path / "scratch"))
        assert make_folds.check_out_dir(str(tmp_path)) is None

    def test_the_message_names_the_stale_folds(self, tmp_path):
        for index in range(3):
            _fold_dir(tmp_path, index)
        problem = make_folds.check_out_dir(str(tmp_path))
        assert "3 fold directories" in problem
        assert "Fold0" in problem


class TestGenerationIsStillDeterministic:
    """The gitignore on data/folds_grouped/ is only defensible if this holds.

    The folds are not committed, on the reasoning that they are a pure function
    of committed data plus a committed script.  That reasoning is load-bearing:
    if generation ever varied, the corrected-pipeline results would not be
    reproducible from the repo alone.
    """

    @pytest.mark.slow
    def test_two_runs_produce_identical_bytes(self, tmp_path):
        import hashlib

        def run(target):
            assert make_folds.main(["--out", str(target), "--overwrite"]) == 0
            digest = hashlib.sha256()
            for root, _dirs, files in sorted(os.walk(str(target))):
                for name in sorted(files):
                    with open(os.path.join(root, name), "rb") as handle:
                        digest.update(handle.read())
            return digest.hexdigest()

        assert run(tmp_path / "a") == run(tmp_path / "b")

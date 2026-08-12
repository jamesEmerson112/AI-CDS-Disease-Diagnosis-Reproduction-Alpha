"""Byte-exact golden regression for the full 10-fold BioSentVec baseline pipeline.

Why a SECOND golden
-------------------
``tests/test_golden.py`` covers the BERT arm only. ``cython_utils.py`` is
*shared* between the arms, so a change there could move the baseline numbers
with the entire suite still green -- which is precisely the failure mode a
golden exists to prevent, and it was unguarded for one of the two arms. This is
P13.

The overlap between the two goldens is smaller than it looks. Running the BERT
arm never executes:

* ``cython_utils.predictS2V`` -- the baseline's whole prediction routine,
  including its mean-of-max symptom aggregation (``:322-357``), its
  ``PRUNING_SIMILARITY`` gate and MAX selection (``:362-385``), and its
  ``np.argsort`` TOP-K path (``:390-415``). The BERT arm reimplements all of
  that inside ``bert_models``;
* ``cython_utils.compute_performance_index`` on the per-case path -- the
  baseline calls it 774 times per run (``:426``/``:431``);
* ``cython_utils.embending_symptoms``/``embending_diagnosis``, the baseline's
  embedding-dict builders, which key on preprocessed text and wrap each vector
  in a one-element row.

So this file is not a duplicate of the BERT golden at a different width. It is
the first coverage those functions have ever had against a byte-exact
reference.

Deliberately standalone
-----------------------
Nothing here imports from ``tests/test_golden.py``, and that is a choice rather
than an oversight. Sharing the canonicalisation helpers would make one file's
refactor able to move the other file's verdict, and two goldens that can fail
together for a reason unrelated to the pipeline are worth less than two that
cannot. The rules below are therefore *copied* from ``test_golden.py`` and must
stay semantically identical to it; they are simple enough that duplication is
cheaper than the coupling. ``test_golden.py`` is also not to be edited.

700 dimensions, not 768
-----------------------
BioSentVec is 700-D and the three transformers are 768-D, so the two goldens are
minted at the width of the arm each one protects. ``StubEncoder`` grew no new
default for this -- ``dim`` was already a constructor argument, and the default
stays 768 so the BERT golden is untouched. What it *did* grow is
``embed_sentence``, because the baseline arm reaches its encoder through
sent2vec's contract (one string in, a ``(1, dim)`` matrix out) and never touches
``.encode``. That addition is purely additive: ``bert_models`` calls no method
that did not already exist.

Byte-for-byte, not float-for-float
----------------------------------
Same reasoning as the BERT golden: the formatting carries information about how
the code works, and a comparator that parsed floats would be blind to it. The
row *order* inside every block is set-iteration order over the literal
``{1, 0.9, 0.8, 0.7, 0.6}``, the columns are tab-separated, and floats print at
``repr`` length (``0.07692307692307693``).

One quirk is specific to this arm and is pinned below, because it is the
clearest evidence the two arms reach their row order by different routes.
**Every** threshold-1 row in this file starts with a bare int ``1`` -- 840 of
them, aggregate and per-case alike -- and **no** row starts with ``1.0``. The
BERT golden carries both spellings (66 int rows and 774 float rows) because
``bert_models:511`` transcribes the set into a hard-coded ordered list of
floats. The baseline never transcribes anything; it iterates the set literal
directly at all seven sites. If a future refactor "unifies" the two arms onto
one threshold list, this arm's 840 rows become ``1.0`` rows and this assertion
is what says so.

NOT A RESULT. Read no science out of this file
----------------------------------------------
Every aggregate row in the reference carries ``PR 1.0`` and ``P == R == FS``,
which is exactly the shape the project's degeneracy finding says the *BERT*
arms have and the baseline does not. That is an artifact of the stub, not a
discovery: ``StubEncoder``'s cosine distribution puts ~94% of pairs above the
0.5 ``PRUNING_SIMILARITY`` cut (measured at dim=768 -- see ``tests/stubs.py``),
so nothing is ever pruned to empty and this arm never abstains. The real
BioSentVec run does abstain, on roughly a quarter of cases: its ``PR`` column
reads 0.7679 throughout and its ``TP+FP`` sums to 9.9, against the 12.9 here.

So the golden pins the *arithmetic and the formatting* of the baseline pipeline,
which is all a regression net needs to do. It does not model the encoder, and
quoting a number out of it as a baseline result would be wrong.

What is excluded, and why
-------------------------
``run_analysis`` appends a wall-clock timing report to the tail of
``PerformanceIndex.txt``. It is nondeterministic by construction, so everything
from its opening banner onward is cut -- see ``TRAILER_DELIMITER``. The run of
80 ``=`` characters appears nowhere in the deterministic body (verified: the
body contains no ``=`` at all outside the ``HADM_ID=`` labels), and the body's
own separators are runs of 108 ``*``.

Provenance of the golden
------------------------
Minted 2026-08-12 from the working tree at commit ``34d6d43`` plus the
``embed_sentence`` addition to ``tests/stubs.py`` in this same change. Settings:
``baseline_sent2vec.run_analysis(encoder=StubEncoder(dim=700), config=LEGACY,
out=None)``, run with the cwd pointed at a scratch directory outside the
repository. Windows 11, Python 3.9, numpy 2.0.2 (the repo venv).

Determinism was proved, not assumed, before the file was committed: two
independent processes, launched in parallel from separate scratch directories,
produced byte-identical output over all 353,589 canonicalised bytes. The two
raw files differ, as they must -- only in the wall-clock trailer this module
cuts.

That check is stronger than the BERT golden's, by accident rather than design.
``PYTHONHASHSEED`` is unset here, so CPython randomised the two processes'
seeds independently, and the seed is not inert on this path:
``preprocess_diagnosis`` routes through ``list(set(...))`` twice, so it really
does change dict insertion order in this arm. The numbers did not move. What
makes that true is that order reaches only *which* equal-valued candidate is
recorded first, never a value -- ``get_diagnosis_similarity_by_description_max``
takes a MAX over the Cartesian product, and retrieval is ordered by the symptom
similarity matrix, which the diagnosis dicts do not touch.

Running it
----------
The cost is the pipeline, not the encoder: ``cython_utils.cosine_similarity`` is
a pure-Python element-wise loop, so wall time scales with ``dim``. Measured
2026-08-12 on the owner's Windows box (12 cores, repo venv): the two mints took
**1867.7s (31m 08s)** and **1833.7s (30m 34s)** running concurrently, and this
test on its own -- the number a developer actually waits for -- took
**1689.92s (28m 09s)**.

Do NOT fold those numbers into the 34-53 minute range the docs quote for
``pytest -m golden`` -- that range was measured on the BERT arm at dim=768 and
describes a different pipeline at a different width. ``pytest -m golden`` now
runs both, so the two costs add.

The test carries both the ``golden`` and ``slow`` markers, so ``addopts``
(``-m 'not network and not slow'``) deselects it from a bare ``pytest`` and
``pytest -m golden`` selects it alongside the BERT one. The cheap checks in
``TestBaselineGoldenFile`` do not run the pipeline and stay in the fast run.
"""

import difflib
import glob
import os
import re
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Golden reference
# ---------------------------------------------------------------------------

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(TESTS_DIR, "golden", "stub700-baseline")
GOLDEN_PATH = os.path.join(GOLDEN_DIR, "PerformanceIndex.txt")

# The golden was minted with these settings. Changing either invalidates it.
STUB_DIM = 700  # BioSentVec's width. The BERT golden's stub is 768.
CONFIG_NAME = "legacy"  # aicds.config.LEGACY -- the published pipeline.

# ---------------------------------------------------------------------------
# Canonicalisation  (semantically identical to test_golden.py's; see docstring)
# ---------------------------------------------------------------------------

# First line of the wall-clock trailer that run_analysis appends to
# PerformanceIndex.txt (`"=" * 80`, immediately above
# "TIMING REPORT - Disease Diagnosis System"). Everything from here on is
# timings, so it is cut.
TRAILER_DELIMITER = "=" * 80

# Output directory names carry a "%d%m%Y_%H-%M-%S" stamp. It does not currently
# leak into the file body, but normalise it anyway so that a future change which
# starts echoing the path does not turn every run into a spurious failure.
_TIMESTAMP_RE = re.compile(r"\d{8}_\d{2}-\d{2}-\d{2}")


def strip_trailer(text):
    """Return everything above the wall-clock trailer, newline-terminated."""
    cut = text.find(TRAILER_DELIMITER)
    if cut == -1:
        raise AssertionError(
            "No timing trailer found in PerformanceIndex.txt. Expected a line "
            "of {n} '=' characters introducing the TIMING REPORT block. Either "
            "the trailer format changed (update TRAILER_DELIMITER) or the run "
            "did not finish.".format(n=len(TRAILER_DELIMITER))
        )
    # run_analysis writes exactly one blank-line separator between the body and
    # the trailer ('\n\n'), so strip exactly one -- CRLF on Windows (the handle
    # is text mode), LF elsewhere. Stripping ANY number of trailing newlines
    # would let 10 appended blank lines compare equal, which is a hole in the
    # byte-exact claim at the tail.
    body = text[:cut]
    for sep in ("\r\n\r\n", "\n\n"):
        if body.endswith(sep):
            body = body[: -len(sep)]
            break
    return body


def normalise_paths(text, run_dir=None):
    """Replace the run-specific output path and timestamp with placeholders."""
    if run_dir:
        run_dir = run_dir.rstrip(os.sep)
        parent = os.path.dirname(run_dir)
        text = text.replace(run_dir, "<RUN_DIR>")
        if parent:
            text = text.replace(parent, "<WORK_DIR>")
    return _TIMESTAMP_RE.sub("<TIMESTAMP>", text)


def canonicalise(text, run_dir=None):
    """Full comparison form: trailer removed, run-specific paths normalised."""
    return normalise_paths(strip_trailer(text), run_dir)


# ---------------------------------------------------------------------------
# Diff reporting
# ---------------------------------------------------------------------------


def first_difference(expected, actual):
    """Index of the first differing line, or None if the texts are identical."""
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    for i in range(min(len(expected_lines), len(actual_lines))):
        if expected_lines[i] != actual_lines[i]:
            return i
    if len(expected_lines) != len(actual_lines):
        return min(len(expected_lines), len(actual_lines))
    return None


def format_first_difference(expected, actual, context=10):
    """Human-readable report of the first differing region.

    Carries three things, because this is the message a developer debugs a
    failed refactor with: where the divergence is, a unified diff of the
    surrounding region, and ``repr()`` of the two offending lines. The reprs
    matter -- an int/float formatting change or a trailing-whitespace change is
    invisible in a rendered diff and obvious in a repr.
    """
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    index = first_difference(expected, actual)

    if index is None:
        return "texts are identical"

    report = [
        "PerformanceIndex.txt diverged from the baseline golden reference.",
        "",
        "  golden: {path}".format(path=GOLDEN_PATH),
        "  golden lines: {n}".format(n=len(expected_lines)),
        "  actual lines: {n}".format(n=len(actual_lines)),
        "  first difference at line {n} (1-indexed)".format(n=index + 1),
        "",
    ]

    if index < len(expected_lines):
        report.append("  golden: {r}".format(r=repr(expected_lines[index])))
    else:
        report.append("  golden: <end of file - golden is shorter>")
    if index < len(actual_lines):
        report.append("  actual: {r}".format(r=repr(actual_lines[index])))
    else:
        report.append("  actual: <end of file - actual is shorter>")

    low = max(0, index - context)
    high = index + context + 1
    diff = difflib.unified_diff(
        expected_lines[low:high],
        actual_lines[low:high],
        fromfile="golden",
        tofile="actual",
        lineterm="",
        n=context,
    )
    report.append("")
    report.append(
        "  unified diff of lines {a}-{b} (hunk headers are window-relative):".format(
            a=low + 1, b=high
        )
    )
    report.append("")
    report.extend("  " + line for line in diff)
    return "\n".join(report)


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


def run_pipeline(dim=STUB_DIM):
    """Run the full 10-fold baseline pipeline under a stub encoder.

    ``run_analysis(out=None)`` builds its output directory from ``os.getcwd()``
    plus a timestamp, so the run is done with the cwd pointed at a temporary
    directory and the result located by glob. Returns ``(raw_text, run_dir)``.

    Note the baseline also drops a sibling ``Prediction_Symptom_Details_*``
    directory into the same cwd. ``Prediction_Output_*`` does not match it, and
    the count assertion below is what would catch that changing.
    """
    from aicds import config as config_module
    from aicds.config import LEGACY
    from aicds.models import baseline_sent2vec
    from tests.stubs import StubEncoder

    # The golden is a statement about ONE config. Assert the object still is
    # the one it was minted under before spending half an hour proving it is
    # not: PipelineConfig is frozen, but the registry entry LEGACY points at is
    # not, and a repointed LEGACY would fail this at line 1 instead of as 5,959
    # unexplained diffs.
    assert config_module.name_of(LEGACY) == CONFIG_NAME, (
        "expected config LEGACY to still be named {want!r}, got {got!r}".format(
            want=CONFIG_NAME, got=config_module.name_of(LEGACY)
        )
    )

    previous_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="golden-baseline-run-") as work_dir:
        # macOS hands out /var/... symlinked to /private/var/...; resolve it so
        # the path we normalise matches the one run_analysis derives from getcwd.
        work_dir = os.path.realpath(work_dir)
        os.chdir(work_dir)
        try:
            returned = baseline_sent2vec.run_analysis(
                encoder=StubEncoder(dim=dim), config=LEGACY, out=None
            )
            produced = sorted(glob.glob(os.path.join(work_dir, "Prediction_Output_*")))
            assert len(produced) == 1, (
                "expected exactly one Prediction_Output_* directory in the "
                "temporary cwd, found {n}: {p}".format(n=len(produced), p=produced)
            )
            run_dir = produced[0] + os.sep
            assert os.path.realpath(returned) == os.path.realpath(run_dir), (
                "run_analysis returned {r} but the glob found {g}".format(
                    r=returned, g=run_dir
                )
            )
            # newline="" disables universal-newline translation. With the
            # default, a whole-file LF -> CRLF change round-trips invisibly and
            # compares equal, which contradicts the byte-exact contract.
            with open(
                os.path.join(produced[0], "PerformanceIndex.txt"), newline=""
            ) as handle:
                raw = handle.read()
        finally:
            os.chdir(previous_cwd)

    return raw, run_dir


def read_golden():
    assert os.path.exists(GOLDEN_PATH), (
        "baseline golden reference missing at {path}. Re-mint it by running "
        "baseline_sent2vec.run_analysis(encoder=StubEncoder(dim={dim}), "
        "config=LEGACY, out=None) and saving the canonicalised "
        "PerformanceIndex.txt.".format(path=GOLDEN_PATH, dim=STUB_DIM)
    )
    with open(GOLDEN_PATH, newline="") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaselineGoldenFile:
    """Cheap checks on the committed reference itself. No pipeline run."""

    def test_golden_exists_and_is_canonical(self):
        golden = read_golden()
        assert golden, "baseline golden reference is empty"
        # Already trailer-free: re-canonicalising must be a no-op, which is what
        # makes the comparison in the slow test a straight equality check.
        assert TRAILER_DELIMITER not in golden
        assert "TIMING REPORT" not in golden
        assert canonicalise(golden + "\n\n" + TRAILER_DELIMITER + "\nx\n") == golden
        assert (
            canonicalise(golden + "\r\n\r\n" + TRAILER_DELIMITER + "\r\nx\r\n") == golden
        )

    def test_golden_has_no_machine_specific_content(self):
        golden = read_golden()
        assert "/Users/" not in golden
        assert "C:\\" not in golden
        assert "/private/" not in golden
        assert "Prediction_Output_" not in golden
        assert _TIMESTAMP_RE.search(golden) is None

    def test_golden_has_the_expected_block_structure(self):
        golden = read_golden()
        # One header per fold, plus the aggregate block.
        for fold in range(10):
            assert " FOLD {n}: LEN train:".format(n=fold) in golden
        assert "10-FOLD PERFORMANCE INDEX of MAX SIMILARITY by MAX" in golden
        for topk in (10, 20, 30, 40, 50):
            assert (
                "10-FOLD PERFORMANCE INDEX of TOP-{k} SIMILARITY by MAX".format(k=topk)
                in golden
            )
        assert golden.rstrip("\r\n").endswith("*" * 108)

    def test_golden_has_a_per_case_block_for_every_test_case(self):
        """129 admissions x 6 blocks (MAX + five TOP-K) = 774 per-case headers.

        This arm writes its per-case blocks from ``predictS2V`` itself
        (``cython_utils.py:425``/``:430``), so the count is a direct statement
        that all 129 cases were scored across the ten folds -- a fold that
        silently lost a case would land here rather than in a shifted number
        somewhere in the middle of the file.
        """
        golden = read_golden()
        lines = golden.splitlines()
        assert sum("HADM_ID=" in line for line in lines) == 774

    def test_golden_preserves_the_baseline_int_threshold_quirk(self):
        """The quirk that distinguishes this arm from the BERT one.

        Every threshold-1 row here is a bare int ``1`` -- both the aggregate
        rows and the per-case rows -- because the baseline iterates the set
        literal ``{1, 0.9, 0.8, 0.7, 0.6}`` directly at all seven sites and
        never transcribes it. The BERT golden carries 774 ``1.0`` rows for
        exactly the opposite reason: ``bert_models:511`` holds a hand-copied
        ordered list of floats. Unifying the two arms onto one threshold list
        would flip these 840 rows to ``1.0`` and this is what says so.
        """
        golden = read_golden()
        lines = golden.splitlines()
        assert sum(line.startswith("1\t") for line in lines) == 840, (
            "expected 840 threshold-1 rows starting with the int '1' -- the "
            "{1, 0.9, ...} set literal formatting quirk has been lost"
        )
        assert sum(line.startswith("1.0") for line in lines) == 0, (
            "the baseline arm has no hard-coded float threshold list; a '1.0' "
            "row means one was introduced"
        )

    def test_golden_preserves_the_bare_int_fscore(self):
        """``cython_utils`` emits int ``0``, not ``0.0``, when R + P == 0.

        Pinned for the same reason the BERT golden pins it: a tolerance-based
        comparator sees nothing, and the count is high enough that the stub is
        demonstrably producing genuine misses rather than saturating.
        """
        golden = read_golden()
        rows = [line.split("\t") for line in golden.splitlines()]
        bare = sum(1 for row in rows if len(row) == 7 and row[5] == "0")
        assert bare == 640, (
            "expected 640 rows with a bare int '0' F-score, found "
            "{n}".format(n=bare)
        )


@pytest.mark.golden
@pytest.mark.slow
class TestBaselinePipelineMatchesGolden:
    """The real thing: run the baseline pipeline, diff the output byte-for-byte."""

    def test_pipeline_output_matches_golden(self):
        golden = read_golden()
        raw, run_dir = run_pipeline()
        actual = canonicalise(raw, run_dir)
        assert actual == golden, format_first_difference(golden, actual)

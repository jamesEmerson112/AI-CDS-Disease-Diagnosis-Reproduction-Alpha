"""Byte-exact golden regression for the full 10-fold BERT pipeline.

Why this exists
---------------
The pipeline is being refactored across several phases. Nothing in it is
trained, so every number it produces is a pure function of the input data, the
embeddings and the arithmetic in ``src/utils/cython_utils.py`` -- which means
any change in any of those is a *behaviour* change, and the whole point of the
refactor is that there should not be one. This test runs the real pipeline
against a deterministic ``StubEncoder`` (no model download, no torch load, no
network) and compares the resulting ``PerformanceIndex.txt`` against a
committed reference minted from the pre-refactor tree.

Byte-for-byte, not float-for-float
----------------------------------
The comparison is a plain string diff, deliberately. A comparator that parsed
floats and compared with a tolerance would be blind to exactly the formatting
quirks that carry information about how the code works:

* aggregate rows print the top threshold as ``1`` (an int, because it comes out
  of the set literal ``{1, 0.9, 0.8, 0.7, 0.6}``) while per-case rows print
  ``1.0`` (a float, from ``bert_models``' own hard-coded list) -- 66 and 774
  rows respectively in the current golden;
* the row *order* within every block is set-iteration order, not sorted order;
* separator widths, the tab-vs-space column layout, and ``repr``-length floats
  such as ``0.15384615384615385`` are all reproduced exactly.

These are load-bearing accidents of the current implementation. If a refactor
changes one, that is a regression in the output contract even though every
number is numerically identical, and this test is the thing that catches it.

That int-vs-float distinction extends to the F-score column: ``cython_utils``
emits an int ``0`` rather than ``0.0`` when ``recall + precision == 0``. This
*is* pinned -- 1597 rows in the current reference carry a bare ``0`` there,
because ``StubEncoder`` produces genuine misses (``TP 0 / FP 1``) on plenty of
test cases. An earlier revision of this docstring claimed the opposite; it was
wrong, and the golden is correspondingly stronger than it advertised.

What is excluded, and why
-------------------------
``bert_models`` appends a wall-clock timing report to the tail of
``PerformanceIndex.txt`` (as well as writing it to ``timing_report.txt``).
That trailer is nondeterministic by construction, so everything from its
opening banner onward is cut before comparison -- see ``TRAILER_DELIMITER``.
The fold-loop body above it is data-driven and stable.

Provenance of the golden
------------------------
Minted from the working tree at commit ``fc5d72b`` plus the (then uncommitted)
``encoder=`` injection patch to ``src/models/bert_models.py`` -- a nine-line
change that only bypasses ``SentenceTransformer`` construction, so the numbers
below are the numbers HEAD produces. Settings: ``model_id='1'``,
``StubEncoder(dim=768)``.

Determinism was proved, not assumed, before the file was committed: three
independent processes -- two at ``PYTHONHASHSEED=0`` and one at
``PYTHONHASHSEED=12345`` -- produced byte-identical output. The hash seed
matters because ``compute_bert_symptom_embeddings`` builds its encode batch
from ``list(set(...))``, so the seed changes batch composition and ordering;
``StubEncoder`` derives every vector from ``sha256(text)`` alone and is
therefore immune.

Running it
----------
The pipeline itself is the cost: ``cython_utils.cosine_similarity`` is a
pure-Python element-wise loop run hundreds of thousands of times per fold, so
wall time scales with the embedding dimension. At the stub's default 768
dimensions a full run takes about 20 minutes on an M-series Mac (measured:
1222.6s). The slow
test carries both the ``golden`` and ``slow`` markers so it can be selected
with ``pytest -m golden`` or excluded with ``pytest -m 'not slow'``; note that
the current ``addopts`` in ``pyproject.toml`` excludes only ``network``, so a
bare ``pytest`` will run it.

The four cheap tests in ``TestGoldenFile`` do not run the pipeline. They guard
the committed reference itself and stay in the fast default run.
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
GOLDEN_DIR = os.path.join(TESTS_DIR, "golden", "stub768")
GOLDEN_PATH = os.path.join(GOLDEN_DIR, "PerformanceIndex.txt")

# The golden was minted with these settings. Changing either invalidates it.
STUB_DIM = 768
MODEL_ID = "1"  # Bio_ClinicalBERT -- only names the output dir under a stub.

# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------

# First line of the wall-clock trailer that bert_models appends to
# PerformanceIndex.txt (`"=" * 80`, immediately above
# "TIMING REPORT - Disease Diagnosis System (...)"). Everything from here on is
# timings, so it is cut. This exact run of 80 '=' characters appears nowhere in
# the deterministic body, which is written entirely by
# cython_utils.compute_aggregated_performance_index /
# print_performance_index plus bert_models' own '*' * 108 separators.
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
    # bert_models writes exactly one blank-line separator between the body and
    # the trailer, so strip exactly one -- CRLF on Windows (open(..., "w") is
    # text mode), LF elsewhere. An earlier version used .rstrip("\n"), which
    # swallowed ANY number of trailing blank lines -- appending 10 of them to
    # the body still compared equal, so the "byte-for-byte" claim had a hole at
    # the tail. Removing exactly one separator keeps that hole closed on both
    # platforms. The two candidates are mutually exclusive (nothing can end with
    # both), so the order is not load-bearing, but keep it explicit.
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

    This is the message a developer will debug a failed refactor with, so it
    carries three things: where the divergence is, a unified diff of the
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
        "PerformanceIndex.txt diverged from the golden reference.",
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


def run_pipeline(dim=STUB_DIM, model_id=MODEL_ID):
    """Run the full 10-fold pipeline under a stub encoder in a throwaway cwd.

    ``bert_models`` builds its output directory from ``os.getcwd()`` plus a
    timestamp, so the run is done with the cwd pointed at a temporary
    directory and the result located by glob. Returns
    ``(raw_text, run_dir)``.
    """
    from src.models import bert_models
    from tests.stubs import StubEncoder

    previous_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="golden-run-") as work_dir:
        # macOS hands out /var/... symlinked to /private/var/...; resolve it so
        # the path we normalise matches the one bert_models derives from getcwd.
        work_dir = os.path.realpath(work_dir)
        os.chdir(work_dir)
        try:
            returned = bert_models.run_analysis(
                model_id=model_id, encoder=StubEncoder(dim=dim)
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
        "golden reference missing at {path}. Re-mint it by running the "
        "pipeline with StubEncoder(dim={dim}) and saving the canonicalised "
        "PerformanceIndex.txt.".format(path=GOLDEN_PATH, dim=STUB_DIM)
    )
    with open(GOLDEN_PATH, newline="") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGoldenFile:
    """Cheap checks on the committed reference itself. No pipeline run."""

    def test_golden_exists_and_is_canonical(self):
        golden = read_golden()
        assert golden, "golden reference is empty"
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

    def test_golden_preserves_the_int_float_formatting_quirks(self):
        """The formatting accidents a tolerance-based comparator would miss."""
        golden = read_golden()
        lines = golden.splitlines()
        # Aggregate rows emit the top threshold as int 1 (set literal ordering).
        assert sum(line.startswith("1\t") for line in lines) == 66, (
            "expected 66 aggregate rows starting with int '1' -- the "
            "{1, 0.9, ...} set literal formatting quirk has been lost"
        )
        # Per-case rows emit it as float 1.0 (bert_models' own THRESHOLDS list).
        assert sum(line.startswith("1.0 ") for line in lines) == 774, (
            "expected 774 per-case rows starting with float '1.0' -- "
            "bert_models' hard-coded threshold list formatting has changed"
        )


@pytest.mark.golden
@pytest.mark.slow
class TestPipelineMatchesGolden:
    """The real thing: run the pipeline, diff the output byte-for-byte."""

    def test_pipeline_output_matches_golden(self):
        golden = read_golden()
        raw, run_dir = run_pipeline()
        actual = canonicalise(raw, run_dir)
        assert actual == golden, format_first_difference(golden, actual)

#!/usr/bin/env python3
"""Split the corrected-pipeline drop into a leakage part and a preprocessing part.

    python scripts/attribute_effects.py
    python scripts/attribute_effects.py --roots results results_folds_only \
                                                results_preprocess_only results_corrected

WHY THIS SCRIPT EXISTS. ``CORRECTED`` changes two independent things at once --
it moves to the ``GroupKFold``-on-``SUBJECT_ID`` split *and* to the fixed
preprocessing (``src/aicds/config.py``) -- so the measured drop between
``results`` and ``results_corrected`` (0.0902 on the baseline arm) attributes to
neither. ``FOLDS_ONLY`` and ``PREPROCESS_ONLY`` each change exactly one of the
two; reading all four together is what separates the effects. That is P29, and
those two configs were built for this table and nothing else.

WHY IT IS A SEPARATE SCRIPT RATHER THAN A FLAG. The comparison is CROSS-ROOT,
and every other consumer here takes exactly one root on purpose:
``compare_models.py --results-dir`` keys its parser expectations by pipeline and
``recorded_pipeline`` *exits* on a tree whose runs disagree about which pipeline
produced them; ``analyze_rank_metrics.py`` takes one directory. Those refusals
are right -- a mixed root is how half a comparison gets attributed to the wrong
pipeline -- and they leave nothing in the repository able to compare four roots.
This script is that tool, and it keeps the refusal: it reads four roots that each
stay internally single-pipeline, and checks each one against the pipeline it is
supposed to hold.

WHY TWO CELLS, AND WHY NEITHER OF THEM IS SELECTABLE. Every number here is a
10-FOLD TOP-10 F-score, reported at exactly two thresholds:

  0.6  THE PUBLISHED CELL. The 0.0902 baseline drop this script exists to split
       is quoted from it -- and the three BERT arms sit on the saturation ceiling
       here (``docs/findings/03-metric-saturation.md``), so their deltas are
       differences between 1.000 and 1.000 and carry no information at all.
  1.0  THE ONLY COSINE SETTING WHERE NO ARM SATURATES, and therefore the only one
       at which the attribution is populated for all four arms.

Reporting 0.6 alone attributes one arm's drop and leaves three rows of zeros to
be misread as "the fixes did not affect the BERT arms". Reporting 1.0 alone
abandons the figure the paper and every earlier finding quote. Both are printed,
each with its own grid, its own attribution table, and its own saturation note
DERIVED FROM THAT THRESHOLD'S DATA rather than asserted.

There is deliberately no ``--threshold`` flag. The two cells are exactly the four
cells :data:`ANCHORS` pins, so a flag would silently disable the only check that
the columns were read correctly, on the invocation most likely to need it.

SIGN CONVENTION, stated once: every part is ``legacy - fixed``, so a POSITIVE
number means the fix REMOVED score, i.e. the legacy figure was inflated by that
much. The residual is ``drop - leakage - preprocessing`` and is a reportable
interaction term, not an error -- the two one-change configs are different
pipelines and their effects have no obligation to add up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, NamedTuple

from aicds import runs
from aicds.analysis.performance_index import PerformanceIndexError, read, validate
from aicds.config import PIPELINE_NAMES


# --------------------------------------------------------------------------
# What is being compared
# --------------------------------------------------------------------------

#: The strategy every cell below comes from.
STRATEGY = "TOP-10"

#: The two threshold cells reported, in print order. Each is a dictionary KEY,
#: never a row position: ``PerformanceIndex.txt`` emits its five threshold rows
#: in *set-iteration* order (0.9, 1, 0.6, 0.8, 0.7), documented as trap 4 in
#: ``aicds.analysis.performance_index``, so row 0 is 0.9 and indexing by position
#: reads the wrong threshold without failing. Not to be confused with
#: ``performance_index.THRESHOLDS``, which is all five the file carries; these
#: are the two this script reports, and the module docstring says why these two.
THRESHOLDS = (0.6, 1.0)

#: Why each threshold is in the report at all, printed as its block header so the
#: reader of a pasted table knows which cell they are holding.
THRESHOLD_NOTES = {
    0.6: "the published cell -- and where the BERT arms saturate",
    1.0: "the only cosine setting at which no arm sits on the ceiling",
}

# Pipeline registry names (aicds.config._BY_NAME), used as this script's own
# column names so a column and the --pipeline that produced it are the same word.
LEGACY = "legacy"
FOLDS_ONLY = "folds-only"
PREPROCESS_ONLY = "preprocess-only"
CORRECTED = "corrected"


class RootSpec(NamedTuple):
    """One column: which pipeline, which default tree, and what it varies."""

    pipeline: str
    dirname: str
    changes: str


#: The four roots, in the order the tables print them. ORDER IS LOAD-BEARING:
#: ``--roots`` is positional against this list, and :func:`attribute` subtracts
#: the middle two from the first.
ROOT_SPECS = (
    RootSpec(LEGACY, "results", "the published pipeline; both defects present"),
    RootSpec(FOLDS_ONLY, "results_folds_only", "grouped folds, legacy preprocessing"),
    RootSpec(PREPROCESS_ONLY, "results_preprocess_only", "legacy folds, fixed preprocessing"),
    RootSpec(CORRECTED, "results_corrected", "grouped folds AND fixed preprocessing"),
)

# A pipeline name with no registry entry would never match any run_metadata.json
# and would silently downgrade the provenance check below to "always absent".
# Checked at import so the failure is at the top of the run, not four roots in.
_UNKNOWN = [spec.pipeline for spec in ROOT_SPECS if spec.pipeline not in PIPELINE_NAMES]
if _UNKNOWN:
    raise ValueError(
        "ROOT_SPECS names pipelines that aicds.config does not define: %s. "
        "Selectable names: %s." % (", ".join(_UNKNOWN), ", ".join(PIPELINE_NAMES))
    )

#: Known-correct values, keyed ``(pipeline, arm, threshold)``, used as a guard on
#: the PARSE -- the same job ``compare_models.SANITY_CHECKS_BY_PIPELINE`` does
#: there. Full precision and compared with ``==`` rather than a tolerance, because
#: both sides are ``float()`` of the same decimal token: the writer emits
#: ``repr``-round-trippable text, so an exact match is achievable and any
#: difference is a real one. A rounded expectation would pass while the parser
#: read a neighbouring column.
#:
#: All four are the BASELINE arm, one per (pipeline, threshold) pair this script
#: reports, and that is not an oversight: at 0.6 the three BERT arms print exactly
#: 1.000000, which anchors nothing -- a parser reading P, R or FS out of a
#: saturated row gets the same number from all three columns.
#:
#: WHERE THEY COME FROM, precisely, because it matters for what a failure means:
#: they were read back off the LOCAL ``results/`` and ``results_corrected/``
#: trees, which are **gitignored working data, not committed artifacts** -- every
#: ``results*/`` path is covered by ``.gitignore:160`` because a run leaves
#: per-case files named by HADM_ID. The committed regression oracle is
#: ``docs/Prediction_Output_*``. So a local tree CAN be overwritten or re-harvested
#: in a way the committed one cannot, and that is one of the things a mismatch
#: here means.
#:
#: DO NOT EDIT THESE TO MAKE A FAILURE GO AWAY. A mismatch means the parse, the
#: roots, or a local tree moved, and which of those it is belongs in the commit
#: message.
ANCHORS = {
    (LEGACY, "baseline", 0.6): 0.4824430641821946,
    (LEGACY, "baseline", 1.0): 0.2801210239036326,
    (CORRECTED, "baseline", 0.6): 0.3922168068392324,
    (CORRECTED, "baseline", 1.0): 0.2162775515864761,
}

#: Printed above and below every table, and inside the refusal that hands a reader
#: the command to CREATE this data. ``PREPROCESS_ONLY`` keeps ``data/folds``, so
#: it carries the patient leakage in full by construction.
BANNER = (
    "PREPROCESS-ONLY NUMBERS ARE ATTRIBUTION INPUTS, NEVER RESULTS.\n"
    "  That pipeline keeps data/folds, so it STILL LEAKS PATIENTS -- 41 of 129\n"
    "  test cases retrieve another admission of their own SUBJECT_ID\n"
    "  (src/aicds/config.py:143-145, docs/findings/05-patient-leakage.md).\n"
    "  It exists to isolate one effect. A number quoted out of that column, or\n"
    "  out of any difference built on it, is a leaked number.\n"
    "  Report 'corrected' or 'drg'. Never this."
)


# --------------------------------------------------------------------------
# Collection -- refuses rather than skips, and reports every gap at once
# --------------------------------------------------------------------------

class Collected(NamedTuple):
    """Everything the report needs, with the arithmetic separated from the I/O.

    ``values`` is nested THRESHOLD FIRST -- ``{threshold: {pipeline: {arm: FS}}}``
    -- and that order is the point: ``values[threshold]`` is exactly the plain
    ``{pipeline: {arm: f_score}}`` that :func:`attribute` and :func:`saturated_arms`
    take, so both stay pure functions of one threshold's slice and a caller can
    still build one by hand. Only :func:`check_anchors` sees the whole structure,
    because an anchor is pinned to a (pipeline, arm, threshold) cell.

    ``runs`` keeps the discovered :class:`aicds.runs.Run` objects for the
    provenance block.

    Note the field is called ``runs`` and so is the module this file imports.
    Reach it as ``collected.runs``; never bind a local named ``runs``, or
    ``runs.discover`` and ``runs.MODEL_KEYS`` stop resolving inside that scope.
    """

    roots: Dict[str, str]
    runs: Dict[str, Dict[str, object]]
    values: Dict[float, Dict[str, Dict[str, float]]]
    arms: List[str]


def resolve_roots(roots=None):
    """``{pipeline: path}`` for the four columns. ``None`` takes the defaults."""
    if roots is None:
        return {spec.pipeline: spec.dirname for spec in ROOT_SPECS}
    if len(roots) != len(ROOT_SPECS):
        raise SystemExit(
            "[ERROR] --roots takes exactly %d paths, in this order: %s. Got %d."
            % (
                len(ROOT_SPECS),
                " ".join(spec.pipeline for spec in ROOT_SPECS),
                len(roots),
            )
        )

    # Two columns pointed at one tree is not a degenerate case worth tolerating:
    # it FABRICATES a column of exact zeros -- a difference between a tree and
    # itself -- which prints as "this fix had no measured effect". That is the
    # single most misreadable cell in the table, and it is indistinguishable from
    # the real finding this script exists to produce. Compared by resolved path,
    # so 'results', './results' and 'results/' collide the way the filesystem says
    # they do; the message quotes what was typed.
    groups = {}
    for spec, root in zip(ROOT_SPECS, roots):
        groups.setdefault(os.path.normcase(os.path.abspath(root)), []).append(
            (spec.pipeline, root)
        )
    duplicates = [group for group in groups.values() if len(group) > 1]
    if duplicates:
        raise SystemExit(
            "[ERROR] --roots names one tree in more than one column, so at least\n"
            "        one 'effect' below would be a tree minus itself: exactly\n"
            "        0.000000, printed as 'this fix did nothing'.\n"
            + "\n".join(
                "  columns %s resolve to the same tree (%s)"
                % (", ".join(pipeline for pipeline, _ in group),
                   " == ".join(root for _, root in group))
                for group in duplicates
            )
            + "\n        The four columns are four different pipelines. Pass four\n"
              "        different trees, in the order %s."
            % " ".join(spec.pipeline for spec in ROOT_SPECS)
        )

    return {spec.pipeline: root for spec, root in zip(ROOT_SPECS, roots)}


#: Why a root cannot enter the table. Named because they are written in
#: :func:`collect` and read in :data:`_ROOT_STATUS`, and a typo in either copy
#: would silently downgrade a status word to the catch-all.
_MISSING = "no such directory"
_UNREADABLE = "unreadable run layout"
_EMPTY = "directory exists but holds no runs"

#: The one-word status printed beside each reason. The column used to say MISSING
#: for all three, which is only true of the first: the other two roots ARE there,
#: one at the wrong depth and one with nothing under it, and neither is fixed by
#: the "produce it with" command block that MISSING points the reader at.
_ROOT_STATUS = {
    _MISSING: "MISSING",
    _UNREADABLE: "UNREADABLE",
    _EMPTY: "EMPTY",
}


def _unusable_roots_error(roots, states, details):
    """The error a first-time user sees, because two roots do not exist yet.

    Written as instructions rather than a diagnosis: the ordinary reason for it
    is that P29's eight runs (4 arms x 2 configs) have not been harvested, which
    is a thing to *do*, not a thing to debug. EVERY unusable root is reported in
    one pass -- fixing them one exit at a time is two pod sessions instead of one
    -- and that includes the malformed-layout ones in ``details``, which used to
    exit on the first.

    ``states`` is ``{pipeline: short reason or None}``; ``details`` carries the
    full :class:`aicds.runs.RunLayoutError` text for the roots that HAVE runs but
    in an unreadable shape. A root in ``details`` is deliberately left out of the
    "produce it with" command block below: re-running the pipeline does not fix a
    tree that is already there at the wrong depth.

    The :data:`BANNER` is repeated at the end because this is the one path that
    hands the reader a command to CREATE preprocess-only data. Telling someone how
    to produce a number without telling them it may never be quoted is how the
    leaked column escapes.
    """
    lines = [
        "[ERROR] %d of the %d attribution roots are unusable, so no effect can be"
        % (sum(1 for state in states.values() if state), len(ROOT_SPECS)),
        "        attributed to either fix.",
        "",
    ]
    for spec in ROOT_SPECS:
        state = states.get(spec.pipeline)
        lines.append(
            "  %-11s %-16s %-26s %s"
            % (_ROOT_STATUS.get(state, "UNUSABLE") if state else "ok",
               spec.pipeline, roots[spec.pipeline], state or spec.changes)
        )

    malformed = [spec for spec in ROOT_SPECS if spec.pipeline in details]
    if malformed:
        lines += [
            "",
            "  These roots exist but hold a PerformanceIndex.txt in neither run",
            "  layout. All of them are reported here rather than one per exit:",
        ]
        for spec in malformed:
            lines.append("")
            lines.append("    %s (%s):" % (spec.pipeline, roots[spec.pipeline]))
            lines += ["      " + line for line in details[spec.pipeline].splitlines()]

    producible = [
        spec for spec in ROOT_SPECS
        if states.get(spec.pipeline) and spec.pipeline not in details
    ]
    if producible:
        lines += [
            "",
            "  The two one-change-at-a-time trees are P29-runs: EIGHT runs, 4 arms x 2",
            "  configs, and they are the whole point of this script. Produce a missing",
            "  root with (from the repository root):",
            "",
        ]
        for spec in producible:
            lines += [
                "    python scripts/run_baseline.py      --pipeline %s --out %s"
                % (spec.pipeline, roots[spec.pipeline]),
                "    python scripts/run_bert_analysis.py --model all --pipeline %s --out %s"
                % (spec.pipeline, roots[spec.pipeline]),
                "",
            ]
        lines += [
            "  The baseline arm is Linux-only -- sent2vec will not build under MSVC --",
            "  so all four arms of a root have to come off the same Linux box.",
            "",
        ]

    lines += [
        "  If your trees are named differently, pass all four in this order:",
        "    --roots %s" % " ".join(roots[spec.pipeline] for spec in ROOT_SPECS),
        "",
        "-" * 78,
        BANNER,
        "-" * 78,
    ]
    return "\n".join(lines)


def _discover(root):
    """``aicds.runs.discover``, returning ``(runs, failure_text)`` rather than raising.

    The failure text is RETURNED rather than raised so :func:`collect` reports
    every unusable root in one pass, matching :func:`_unusable_roots_error`'s
    design. Four roots are normally harvested by one person in one session and go
    wrong the same way; exiting on the first malformed tree costs a second session
    to discover the second. ``RunLayoutError`` names the directory but not which
    of the four columns asked for it, so the caller -- which knows -- pairs the
    text back up with a pipeline name.
    """
    try:
        return runs.discover(root), None
    except runs.RunLayoutError as error:
        return None, str(error)


def _recorded_pipeline(run):
    """``pipeline.name`` from this run's ``run_metadata.json``, or ``None``.

    ``None`` covers "pre-P14 run" and "unreadable metadata" alike, and both are
    treated as absent rather than as an error: it is a provenance file, not a
    result, and it must not be able to stop a table the numbers themselves fully
    support. Every pre-P14 tree predates the writer.
    """
    path = run.file(runs.RUN_METADATA)
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)["pipeline"]["name"]
    except (ValueError, KeyError, TypeError, OSError):
        return None


def _check_provenance(collected):
    """Refuse a root whose runs say they came from a different pipeline.

    The failure this catches is the one this script is most exposed to: four
    roots harvested in one pod session, into directories named by hand. A tree
    filed under the wrong name does not look wrong -- it produces a full table
    with two columns swapped, and the residual absorbs the difference silently.

    Absence is only a warning. It is the normal state of every pre-P14 tree, and
    refusing on it would make this script unusable on exactly the two roots whose
    numbers are pinned.
    """
    mismatches = []
    for spec in ROOT_SPECS:
        unrecorded = 0
        for arm, run in sorted(collected.runs[spec.pipeline].items()):
            recorded = _recorded_pipeline(run)
            if recorded is None:
                unrecorded += 1
            elif recorded != spec.pipeline:
                mismatches.append(
                    "  %s/%s says pipeline '%s', but it is being read as the '%s'\n"
                    "    column (%s)"
                    % (collected.roots[spec.pipeline], arm, recorded, spec.pipeline,
                       run.path)
                )
        if unrecorded:
            print(
                "[WARN] %s: %d of %d runs carry no run_metadata.json, so the '%s'\n"
                "       column is taken on the directory's word (pre-P14 runs)"
                % (collected.roots[spec.pipeline], unrecorded,
                   len(collected.runs[spec.pipeline]), spec.pipeline)
            )
    if mismatches:
        raise SystemExit(
            "[ERROR] a results tree does not hold the pipeline this script is\n"
            "        reading it as. Attribution built on it would be a difference\n"
            "        between two pipelines nobody chose.\n"
            + "\n".join(mismatches)
            + "\n        Point --roots at the right trees, in the order %s."
            % " ".join(spec.pipeline for spec in ROOT_SPECS)
        )


def _f_scores(run, pipeline, root):
    """``{threshold: f_score}`` over :data:`THRESHOLDS` for one run's 10-FOLD block.

    Read and validated ONCE for both cells. The file is ~6000 lines and the two
    thresholds are two keys in one already-parsed dict, so a second read would buy
    nothing except the chance for the two cells to come from different bytes.

    ``validate`` runs first, and is what makes the lookups below safe: it asserts
    the full 6 strategies x 5 thresholds x 10 folds grid, so ``[STRATEGY]`` and
    each threshold cannot be missing by the time they are indexed. It also keeps
    a truncated or in-progress run out of the table entirely -- a half-written
    tree harvested off a pod parses fine and would contribute a real-looking
    number from however many folds finished.

    ``OSError`` and ``UnicodeDecodeError`` are caught alongside
    :class:`PerformanceIndexError` because ``read`` opens the file itself: a
    permission failure, a vanished file, or a mis-encoded one is the same event to
    this script as an incomplete one -- a column that cannot enter the table -- and
    a bare traceback would name the file without naming which of four roots, or
    which arm, asked for it.
    """
    try:
        index = validate(read(run.performance_index))
    except (PerformanceIndexError, OSError, UnicodeDecodeError) as error:
        raise SystemExit(
            "[ERROR] %s / %s: PerformanceIndex.txt is unreadable, or is not a\n"
            "        complete 10-fold run, so it cannot enter an attribution table.\n"
            "        arm  %s\n"
            "        run  %s\n"
            "        %s" % (root, pipeline, run.key, run.path, error)
        )
    return {
        threshold: index.aggregate[STRATEGY][threshold].f_score
        for threshold in THRESHOLDS
    }


def collect(roots=None):
    """Read the four roots. Raises ``SystemExit`` naming exactly what is missing.

    Two passes on purpose. Roots are resolved and discovered first, so a user
    with no ``results_folds_only`` yet gets the instructions immediately instead
    of after four full parses; only then is anything read.
    """
    resolved = resolve_roots(roots)

    states = {}
    details = {}
    discovered = {}
    for spec in ROOT_SPECS:
        root = resolved[spec.pipeline]
        if not os.path.isdir(root):
            states[spec.pipeline] = _MISSING
            continue
        found, failure = _discover(root)
        if failure is not None:
            states[spec.pipeline] = _UNREADABLE
            details[spec.pipeline] = failure
            continue
        if not found:
            states[spec.pipeline] = _EMPTY
            continue
        states[spec.pipeline] = None
        discovered[spec.pipeline] = {run.key: run for run in found}

    if any(states.values()):
        raise SystemExit(_unusable_roots_error(resolved, states, details))

    # Registered arms are required outright; an unregistered key found in any
    # root is required in all of them. A stranger arm is not dropped from the
    # table -- being dropped without a word is the failure aicds.runs.discover
    # refuses upstream, and it would be a worse one here, where the columns are
    # differences.
    required = list(runs.MODEL_KEYS.values())
    strangers = set()
    for by_arm in discovered.values():
        strangers |= set(by_arm) - set(required)
    arms = required + sorted(strangers)

    absent = [
        (spec.pipeline, resolved[spec.pipeline], arm)
        for spec in ROOT_SPECS
        for arm in arms
        if arm not in discovered[spec.pipeline]
    ]
    if absent:
        raise SystemExit(
            "[ERROR] every arm must be present in every root -- a difference\n"
            "        between four roots cannot be taken over three of them.\n"
            + "\n".join(
                "  missing  %-16s %-26s arm '%s'" % (pipeline, root, arm)
                for pipeline, root, arm in absent
            )
            + "\n        Arms expected: %s." % ", ".join(arms)
        )

    # One parse per run, then pivoted threshold-first. Reading the file once per
    # threshold would let two cells of the same row come from different bytes.
    by_run = {
        spec.pipeline: {
            arm: _f_scores(run, spec.pipeline, resolved[spec.pipeline])
            for arm, run in discovered[spec.pipeline].items()
        }
        for spec in ROOT_SPECS
    }

    collected = Collected(
        roots=resolved,
        runs=discovered,
        values={
            threshold: {
                spec.pipeline: {
                    arm: cells[threshold]
                    for arm, cells in by_run[spec.pipeline].items()
                }
                for spec in ROOT_SPECS
            }
            for threshold in THRESHOLDS
        },
        arms=arms,
    )
    _check_provenance(collected)
    return collected


# --------------------------------------------------------------------------
# The arithmetic -- pure, over plain dicts
# --------------------------------------------------------------------------

def check_anchors(values):
    """Verify the parse against the pinned cells. Returns the count checked.

    ``values`` is the whole threshold-first structure, because an anchor names a
    (pipeline, arm, threshold) cell and half the point of adding threshold 1.0 was
    that it comes with two more anchors of its own.

    Refuses on zero checks for the reason ``compare_models`` does: "0 anchors
    passed" and "the parse is verified" print identically, and this table is
    quoted straight into a finding.
    """
    checked = 0
    failures = []
    for (pipeline, arm, threshold), expected in sorted(ANCHORS.items()):
        got = values.get(threshold, {}).get(pipeline, {}).get(arm)
        if got is None:
            continue
        checked += 1
        if got != expected:
            failures.append(
                "  %s / %s  %s @ %.1f  F-score\n"
                "      expected %r\n"
                "      parsed   %r" % (pipeline, arm, STRATEGY, threshold, expected, got)
            )

    if failures:
        raise SystemExit(
            "[ERROR] a parser anchor failed, so nothing below can be trusted and\n"
            "        no table was printed.\n"
            + "\n".join(failures)
            + "\n"
            "        The anchors are exact values read back off the LOCAL results/\n"
            "        and results_corrected/ trees -- gitignored working data, not\n"
            "        committed artifacts (the committed oracle is\n"
            "        docs/Prediction_Output_*). A mismatch means one of: the parser\n"
            "        is reading a different column, --roots points at a tree from\n"
            "        another pipeline, or a local tree was overwritten or\n"
            "        re-harvested. It does NOT mean the expectation is stale:\n"
            "        do not edit ANCHORS to make this pass."
        )
    if not checked:
        raise SystemExit(
            "[ERROR] zero parser anchors ran: none of %s is present, so the parse\n"
            "        is unverified. An unverified table looks exactly like a\n"
            "        verified one, which is why this refuses instead."
            % ", ".join("%s/%s @ %.1f" % key for key in sorted(ANCHORS))
        )
    return checked


def attribute(values, arm):
    """Split one arm's legacy-to-corrected drop into its two named parts.

    ``values`` is ONE threshold's slice -- ``{pipeline: {arm: f_score}}``, i.e.
    ``collected.values[threshold]``. This function has no idea which threshold it
    is on, and that is deliberate: the arithmetic is identical at every cell, and
    a threshold argument here would be a second place to get the cell wrong.

    Every part is ``legacy - fixed``: positive means the fix removed score.

    ``residual`` is the interaction, ``drop - leakage - preprocessing``, and it
    is REPORTABLE rather than an error. The two one-change configs are separate
    pipelines; there is no reason their effects should sum to the bundled one. A
    large residual says the fixes are not independent -- that removing the leaked
    patients changed what fixing the text was worth, or the reverse -- which is
    itself the finding this table exists to produce.
    """
    legacy = values[LEGACY][arm]
    parts = {
        "legacy": legacy,
        "leakage": legacy - values[FOLDS_ONLY][arm],
        "preprocessing": legacy - values[PREPROCESS_ONLY][arm],
        "drop": legacy - values[CORRECTED][arm],
    }
    parts["residual"] = parts["drop"] - parts["leakage"] - parts["preprocessing"]
    return parts


def saturated_arms(values, arms, ceiling=1.0):
    """Arms sitting on the metric ceiling in two or more roots, at one threshold.

    ``values`` is one threshold's slice, as :func:`attribute` takes.

    Derived, not listed by name: which arms saturate is a property of the
    threshold and the encoder, and hard-coding "the three BERT arms" would keep
    printing the note after a run where one of them dropped off the ceiling -- and,
    worse, would print it at threshold 1.0, where nothing saturates at all.
    """
    return [
        arm
        for arm in arms
        if sum(1 for spec in ROOT_SPECS if values[spec.pipeline][arm] >= ceiling) >= 2
    ]


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

_ARM = 18
_COL = 17


def _label(arm):
    return runs.KEY_LABELS.get(arm, arm)


def _row(first, cells, width=_COL):
    # rstrip so no row carries trailing spaces: this output gets pasted into
    # findings and diffs, where invisible padding is noise.
    return (("%-*s " % (_ARM, first)) + " ".join("%-*s" % (width, c) for c in cells)).rstrip()


def _rule(cells, width=_COL):
    """The full padded width of a row, which a stripped header no longer gives."""
    return "-" * (_ARM + 1 + len(cells) * (width + 1) - 1)


def _roots_block(collected):
    print("ROOTS")
    for spec in ROOT_SPECS:
        dates = sorted({run.date for run in collected.runs[spec.pipeline].values()})
        print(
            "  %-16s %-26s %d arms   %s"
            % (spec.pipeline, collected.roots[spec.pipeline],
               len(collected.runs[spec.pipeline]), ", ".join(dates))
        )
    print()


def _threshold_block(collected, threshold):
    """One threshold's grid, attribution table and saturation note.

    Every threshold gets the FULL apparatus, including its own saturation note,
    rather than one grid with a shared footnote: the two cells disagree about
    which rows carry information, and a reader who pastes half this report has to
    be holding the half that says so.
    """
    values, arms = collected.values[threshold], collected.arms

    print("#" * 86)
    print("# THRESHOLD %.1f -- %s"
          % (threshold, THRESHOLD_NOTES.get(threshold, "")))
    print("#" * 86)
    print()

    print("F-SCORE GRID   (10-FOLD %s aggregate at threshold %.1f)" % (STRATEGY, threshold))
    names = [spec.pipeline for spec in ROOT_SPECS]
    print(_row("arm", names))
    print(_rule(names))
    for arm in arms:
        print(_row(
            _label(arm),
            ["%.6f" % values[spec.pipeline][arm] for spec in ROOT_SPECS],
        ))
    print(_rule(names))
    print()

    print("ATTRIBUTION   (every part is legacy - fixed; POSITIVE = the fix removed score)")
    columns = ["legacy FS", "leakage", "preprocessing", "bundled drop", "residual"]
    print(_row("arm", columns, width=14))
    print(_rule(columns, width=14))
    for arm in arms:
        parts = attribute(values, arm)
        print(_row(
            _label(arm),
            [
                "%.6f" % parts["legacy"],
                "%+.6f" % parts["leakage"],
                "%+.6f" % parts["preprocessing"],
                "%+.6f" % parts["drop"],
                "%+.6f" % parts["residual"],
            ],
            width=14,
        ))
    print(_rule(columns, width=14))
    print("  leakage        legacy - folds-only        (grouped folds alone)")
    print("  preprocessing  legacy - preprocess-only   (fixed text handling alone)")
    print("  bundled drop   legacy - corrected         (both, as CORRECTED ships them)")
    print("  residual       drop - leakage - preprocessing")
    print("                 THE RESIDUAL IS A REPORTABLE NUMBER, NOT AN ERROR. The two")
    print("                 one-change configs are separate pipelines and their effects")
    print("                 need not add up; a large residual says the fixes interact.")
    print()

    saturated = saturated_arms(values, arms)
    if saturated:
        print("SATURATION -- read these rows as empty, not as zero effect:")
        print("  %s" % ", ".join(_label(arm) for arm in saturated))
        print("  scored the ceiling 1.000000 in two or more roots AT THIS THRESHOLD, so")
        print("  their deltas are differences between two ceilings and measure the")
        print("  threshold, not the fixes. Threshold %.1f is where the biomedical" % threshold)
        print("  embeddings saturate; see docs/findings/03-metric-saturation.md.")
        clean = [
            other for other in THRESHOLDS
            if other != threshold
            and not saturated_arms(collected.values[other], arms)
        ]
        if clean:
            print("  Threshold %s in this same report has no arm on the ceiling; the"
                  % ", ".join("%.1f" % other for other in clean))
            print("  four-arm attribution there is the one to read.")
    else:
        print("NO SATURATION at this threshold: every arm is below %.6f in every root," % 1.0)
        print("  so all %d rows above carry information and the attribution is" % len(arms))
        print("  populated for every arm. This is the block to quote.")
    print()


def report(collected):
    """Print the banner, the provenance block, and one block per threshold."""
    print("=" * 86)
    print(BANNER)
    print("=" * 86)
    print()

    _roots_block(collected)

    for threshold in THRESHOLDS:
        _threshold_block(collected, threshold)

    print("=" * 86)
    print(BANNER)
    print("=" * 86)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Reads four results roots. Never writes to any of them.",
    )
    parser.add_argument(
        "--roots", nargs=len(ROOT_SPECS), default=None,
        metavar=tuple(spec.pipeline.upper().replace("-", "_") for spec in ROOT_SPECS),
        help="override the four default roots (%s), in that exact order"
        % ", ".join(spec.dirname for spec in ROOT_SPECS),
    )
    args = parser.parse_args(argv)

    collected = collect(args.roots)
    checked = check_anchors(collected.values)
    print("[SUCCESS] %d parser anchors passed" % checked)
    report(collected)
    return 0


if __name__ == "__main__":
    sys.exit(main())

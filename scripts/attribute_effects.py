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

WHY ONE CELL, WITH NO FLAG TO MOVE IT. Every number here is the 10-FOLD TOP-10
F-score at threshold 0.6 -- the cell the published baseline figure and the 0.0902
drop are both quoted from. It is fixed rather than selectable because the two
parser anchors below are pinned to exactly that cell: a ``--threshold`` flag
would silently disable the only check that the columns were read correctly, on
the invocation most likely to need it.

WHAT THE THRESHOLD COSTS, AND WHY THE OUTPUT SAYS SO. At 0.6 the three BERT arms
sit on the saturation ceiling (``docs/findings/03-metric-saturation.md``), so
their deltas are differences between 1.000 and 1.000 and carry no information.
That is a property of the threshold, not of the fixes, and :func:`report` prints
it as a note derived from the data rather than leaving four rows of zeros to be
read as "the fixes did not affect the BERT arms".

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
from typing import Dict, List, NamedTuple, Optional

from aicds import runs
from aicds.analysis.performance_index import PerformanceIndexError, read, validate
from aicds.config import PIPELINE_NAMES


# --------------------------------------------------------------------------
# What is being compared
# --------------------------------------------------------------------------

#: The one cell every number in this script comes from. ``THRESHOLD`` is a
#: dictionary KEY, never a row position: ``PerformanceIndex.txt`` emits its five
#: threshold rows in *set-iteration* order (0.9, 1, 0.6, 0.8, 0.7), documented as
#: trap 4 in ``aicds.analysis.performance_index``, so row 0 is 0.9 and indexing by
#: position reads the wrong threshold without failing.
STRATEGY = "TOP-10"
THRESHOLD = 0.6

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

#: Known-correct values, read back off the two committed trees, used as a guard on
#: the PARSE -- the same job ``compare_models.SANITY_CHECKS_BY_PIPELINE`` does
#: there. Full precision and compared with ``==`` rather than a tolerance, because
#: both sides are ``float()`` of the same decimal token: the writer emits
#: ``repr``-round-trippable text, so an exact match is achievable and any
#: difference is a real one. A rounded expectation would pass while the parser
#: read a neighbouring column.
#:
#: DO NOT EDIT THESE TO MAKE A FAILURE GO AWAY. Both trees are frozen; a mismatch
#: means the parse, the roots, or the trees moved, and which of those it is
#: belongs in the commit message.
ANCHORS = {
    (LEGACY, "baseline"): 0.4824430641821946,
    (CORRECTED, "baseline"): 0.3922168068392324,
}

#: Printed above and below every table. ``PREPROCESS_ONLY`` keeps ``data/folds``,
#: so it carries the patient leakage in full by construction.
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

    ``values`` is a plain ``{pipeline: {arm_key: f_score}}`` so :func:`attribute`
    and :func:`check_anchors` are pure functions over data a caller can build by
    hand; ``runs`` keeps the discovered :class:`aicds.runs.Run` objects for the
    provenance block.

    Note the field is called ``runs`` and so is the module this file imports.
    Reach it as ``collected.runs``; never bind a local named ``runs``, or
    ``runs.discover`` and ``runs.MODEL_KEYS`` stop resolving inside that scope.
    """

    roots: Dict[str, str]
    runs: Dict[str, Dict[str, object]]
    values: Dict[str, Dict[str, float]]
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
    return {spec.pipeline: root for spec, root in zip(ROOT_SPECS, roots)}


def _missing_roots_error(roots, states):
    """The error a first-time user sees, because two roots do not exist yet.

    Written as instructions rather than a diagnosis: the ordinary reason for it
    is that P29's eight runs (4 arms x 2 configs) have not been harvested, which
    is a thing to *do*, not a thing to debug. Every root is reported in one pass
    -- fixing them one exit at a time is two pod sessions instead of one.
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
            "  %-9s %-16s %-26s %s"
            % ("MISSING" if state else "ok", spec.pipeline, roots[spec.pipeline],
               state or spec.changes)
        )
    lines += [
        "",
        "  The two one-change-at-a-time trees are P29-runs: EIGHT runs, 4 arms x 2",
        "  configs, and they are the whole point of this script. Produce a missing",
        "  root with (from the repository root):",
        "",
    ]
    for spec in ROOT_SPECS:
        if not states.get(spec.pipeline):
            continue
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
        "  If your trees are named differently, pass all four in this order:",
        "    --roots %s" % " ".join(roots[spec.pipeline] for spec in ROOT_SPECS),
    ]
    return "\n".join(lines)


def _discover(root, pipeline):
    """``aicds.runs.discover``, with the failing column named.

    ``RunLayoutError`` already says which directory and why; what it cannot know
    is which of four roots the caller passed, and that is the first thing a
    reader of this script's output needs.
    """
    try:
        return runs.discover(root)
    except runs.RunLayoutError as error:
        raise SystemExit(
            "[ERROR] the %s root (%s) does not hold a readable run layout:\n"
            "        %s" % (pipeline, root, error)
        )


def _recorded_pipeline(run):
    """``pipeline.name`` from this run's ``run_metadata.json``, or ``None``.

    ``None`` covers "pre-P14 run" and "unreadable metadata" alike, and both are
    treated as absent rather than as an error: it is a provenance file, not a
    result, and it must not be able to stop a table the numbers themselves fully
    support. Every committed ``results*/`` tree predates the writer.
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

    Absence is only a warning. It is the normal state of every committed tree,
    and refusing on it would make this script unusable on exactly the two roots
    whose numbers are pinned.
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


def _f_score(run, pipeline, root):
    """The 10-FOLD TOP-10 F-score at threshold 0.6 for one run.

    ``validate`` runs first, and is what makes the lookup below safe: it asserts
    the full 6 strategies x 5 thresholds x 10 folds grid, so ``[STRATEGY]`` and
    ``[THRESHOLD]`` cannot be missing by the time they are indexed. It also keeps
    a truncated or in-progress run out of the table entirely -- a half-written
    tree harvested off a pod parses fine and would contribute a real-looking
    number from however many folds finished.
    """
    try:
        index = validate(read(run.performance_index))
    except PerformanceIndexError as error:
        raise SystemExit(
            "[ERROR] %s / %s: PerformanceIndex.txt is not a complete 10-fold run,\n"
            "        so it cannot enter an attribution table.\n"
            "        %s" % (root, pipeline, error)
        )
    return index.aggregate[STRATEGY][THRESHOLD].f_score


def collect(roots=None):
    """Read the four roots. Raises ``SystemExit`` naming exactly what is missing.

    Two passes on purpose. Roots are resolved and discovered first, so a user
    with no ``results_folds_only`` yet gets the instructions immediately instead
    of after four full parses; only then is anything read.
    """
    resolved = resolve_roots(roots)

    states = {}
    discovered = {}
    for spec in ROOT_SPECS:
        root = resolved[spec.pipeline]
        if not os.path.isdir(root):
            states[spec.pipeline] = "no such directory"
            continue
        found = _discover(root, spec.pipeline)
        if not found:
            states[spec.pipeline] = "directory exists but holds no runs"
            continue
        states[spec.pipeline] = None
        discovered[spec.pipeline] = {run.key: run for run in found}

    if any(states.values()):
        raise SystemExit(_missing_roots_error(resolved, states))

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

    collected = Collected(
        roots=resolved,
        runs=discovered,
        values={
            spec.pipeline: {
                arm: _f_score(run, spec.pipeline, resolved[spec.pipeline])
                for arm, run in discovered[spec.pipeline].items()
            }
            for spec in ROOT_SPECS
        },
        arms=arms,
    )
    _check_provenance(collected)
    return collected


# --------------------------------------------------------------------------
# The arithmetic -- pure, over plain dicts
# --------------------------------------------------------------------------

def check_anchors(values):
    """Verify the parse against the two pinned cells. Returns the count checked.

    Refuses on zero checks for the reason ``compare_models`` does: "0 anchors
    passed" and "the parse is verified" print identically, and this table is
    quoted straight into a finding.
    """
    checked = 0
    failures = []
    for (pipeline, arm), expected in sorted(ANCHORS.items()):
        got = values.get(pipeline, {}).get(arm)
        if got is None:
            continue
        checked += 1
        if got != expected:
            failures.append(
                "  %s / %s  %s @ %.1f  F-score\n"
                "      expected %r\n"
                "      parsed   %r" % (pipeline, arm, STRATEGY, THRESHOLD, expected, got)
            )

    if failures:
        raise SystemExit(
            "[ERROR] a parser anchor failed, so nothing below can be trusted and\n"
            "        no table was printed.\n"
            + "\n".join(failures)
            + "\n"
            "        Both anchors are exact values read off frozen committed trees.\n"
            "        A mismatch means one of: the parser is reading a different\n"
            "        column, --roots points at a tree from another pipeline, or a\n"
            "        committed tree was overwritten. It does NOT mean the\n"
            "        expectation is stale -- do not edit ANCHORS to make this pass."
        )
    if not checked:
        raise SystemExit(
            "[ERROR] zero parser anchors ran: neither %s is present, so the parse\n"
            "        is unverified. An unverified table looks exactly like a\n"
            "        verified one, which is why this refuses instead."
            % " nor ".join("%s/%s" % key for key in sorted(ANCHORS))
        )
    return checked


def attribute(values, arm):
    """Split one arm's legacy-to-corrected drop into its two named parts.

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
    """Arms sitting on the metric ceiling in two or more roots.

    Derived, not listed by name: which arms saturate is a property of the
    threshold and the encoder, and hard-coding "the three BERT arms" would keep
    printing the note after a run where one of them dropped off the ceiling.
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


def report(collected):
    """Print the banner, the provenance block, the grid and the attribution."""
    values, arms = collected.values, collected.arms

    print("=" * 86)
    print(BANNER)
    print("=" * 86)
    print()

    print("ROOTS")
    for spec in ROOT_SPECS:
        dates = sorted({run.date for run in collected.runs[spec.pipeline].values()})
        print(
            "  %-16s %-26s %d arms   %s"
            % (spec.pipeline, collected.roots[spec.pipeline],
               len(collected.runs[spec.pipeline]), ", ".join(dates))
        )
    print()

    print("F-SCORE GRID   (10-FOLD %s aggregate at threshold %.1f)" % (STRATEGY, THRESHOLD))
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
        print("  scored the ceiling 1.000000 in two or more roots, so their deltas are")
        print("  differences between two ceilings and measure the threshold, not the")
        print("  fixes. Threshold %.1f is where the biomedical embeddings saturate;" % THRESHOLD)
        print("  see docs/findings/03-metric-saturation.md. Threshold 1.0 is the only")
        print("  cosine setting where no arm sits on the ceiling -- and it is not this")
        print("  cell, which is pinned to the published figure and to the two anchors.")
        print()

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

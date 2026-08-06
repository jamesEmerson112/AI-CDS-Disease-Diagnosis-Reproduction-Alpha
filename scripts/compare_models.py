"""
Cross-model comparison report for the AI-CDS reproduction.

Discovers every run under ``results/<model>/<timestamp>/PerformanceIndex.txt``,
parses the six ``10-FOLD PERFORMANCE INDEX`` aggregate blocks out of each, and
writes a multi-page PDF to ``results/model_comparison.pdf``.

This script is read-only with respect to every input. It never touches ``docs/``,
which is the project's regression oracle.

Usage (from the repository root):

    python scripts/compare_models.py
    python scripts/compare_models.py --results-dir results --out results/model_comparison.pdf

Format notes the parser has to survive (verified against the real files):

* Six aggregate sections per file: ``MAX`` plus ``TOP-10/20/30/40/50``, each
  headed ``10-FOLD PERFORMANCE INDEX of ... SIMILARITY by MAX``.
* Columns are ``threshold TP FP P R FS PR``.
* The five threshold rows are emitted in *set-iteration* order
  ``0.9, 1, 0.6, 0.8, 0.7`` -- not sorted. We sort them ourselves.
* The threshold token is a bare ``1`` in aggregate rows (it comes from a set
  literal), not ``1.0``. ``float()`` normalises both.
* The BioSentVec file is tab-separated throughout; the BERT files use runs of
  spaces in their per-case blocks. A bare ``str.split()`` handles both.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; this script only writes files
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

# --------------------------------------------------------------------------
# Presentation constants
# --------------------------------------------------------------------------

STRATEGIES = ["MAX", "TOP-10", "TOP-20", "TOP-30", "TOP-40", "TOP-50"]
THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 1.0]

# Categorical palette, assigned by *entity* and never cycled or re-ranked.
# Validated all-pairs on a light surface (worst CVD dE 9.2, worst normal-vision
# dE 16.3). Aqua sits at 2.74:1 against the page, below the 3:1 contrast gate --
# the relief for that is the direct labels and the value tables on pages 2 and 5,
# so identity is never carried by colour alone.
#
# Dash patterns are a deliberate second channel: the three BERT curves lie exactly
# on top of one another wherever they saturate at 1.000, and interleaved dashes are
# the only way to see that all three are present.
MODEL_REGISTRY = OrderedDict(
    [
        (
            "baseline",
            {
                "label": "BioSentVec (baseline)",
                "arm": "baseline",
                "color": "#2a78d6",
                "marker": "o",
                "dash": (0, ()),  # solid -- the reference arm
            },
        ),
        (
            "bio_clinical_bert",
            {
                "label": "Bio_ClinicalBERT",
                "arm": "bert",
                "color": "#eb6834",
                "marker": "s",
                "dash": (0, (6, 3)),
            },
        ),
        (
            "biomedbert",
            {
                "label": "BiomedBERT",
                "arm": "bert",
                "color": "#1baf7a",
                "marker": "^",
                "dash": (0, (2, 2)),
            },
        ),
        (
            "bluebert",
            {
                "label": "BlueBERT",
                "arm": "bert",
                "color": "#4a3aa7",
                "marker": "D",
                "dash": (0, (7, 2, 1, 2)),
            },
        ),
    ]
)

FALLBACK_STYLES = [
    {"color": "#e87ba4", "marker": "v", "dash": (0, (4, 2))},
    {"color": "#e34948", "marker": "P", "dash": (0, (1, 2))},
]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#7a786f"
GRID = "#d9d8d2"
WARN_FILL = "#fdf0e8"
WARN_EDGE = "#eb6834"
TABLE_HEADER = "#e8e7e2"

# Known-correct values, used as a build-time guard on the parser. If the parse
# ever disagrees with these the parser is wrong -- do not edit the expectations.
#
# These are PIPELINE-SPECIFIC, and that is the whole point of keying them. The
# checks verify the parser lines up columns with headers; they cannot do that
# against a number from a different pipeline, so a single flat list made
# --results-dir results_corrected fail all 16 checks and exit before writing the
# PDF. The checks were right to fire. Deleting them to unblock the PDF would
# have removed the only thing verifying the parse.
SANITY_CHECKS_BY_PIPELINE = {
    # Committed docs/Prediction_Output_* -- data/folds, legacy preprocessing,
    # cosine self-grading. Frozen forever; the golden pins the same path.
    "legacy": [
        ("baseline", "TOP-10", 0.6, {"P": 0.5624, "R": 0.4256, "FS": 0.4824, "PR": 0.7679}),
        ("baseline", "TOP-10", 1.0, {"P": 0.3254, "R": 0.2474, "FS": 0.2801}),
        ("bio_clinical_bert", "TOP-10", 1.0, {"P": 0.2853, "R": 0.2853, "FS": 0.2853}),
        ("biomedbert", "TOP-10", 1.0, {"P": 0.2545, "R": 0.2545, "FS": 0.2545}),
        ("bluebert", "TOP-10", 1.0, {"P": 0.2391, "R": 0.2391, "FS": 0.2391}),
    ],
    # RunPod Linux, 2026-08-06: folds_grouped + corrected preprocessing, still
    # cosine self-graded. Source of the README's headline table.
    "corrected": [
        ("baseline", "TOP-10", 0.6, {"P": 0.4692, "R": 0.3410, "FS": 0.3922, "PR": 0.7558}),
        ("baseline", "TOP-10", 1.0, {"P": 0.2512, "R": 0.1923, "FS": 0.2163}),
        ("bio_clinical_bert", "TOP-10", 1.0, {"P": 0.2491, "R": 0.2491, "FS": 0.2491}),
        ("biomedbert", "TOP-10", 1.0, {"P": 0.1981, "R": 0.1981, "FS": 0.1981}),
        ("bluebert", "TOP-10", 1.0, {"P": 0.1821, "R": 0.1821, "FS": 0.1821}),
    ],
    # RunPod Linux, 2026-08-06: folds_grouped + corrected preprocessing + the
    # encoder-independent DRG grader. These are BIT-IDENTICAL to the corrected
    # entries above at threshold 1.0 -- all 144 numbers, 4 arms x 6 aggregators x
    # 6 columns. That is not a copy-paste error, it is the finding: cosine reaches
    # exactly 1.0 only when two diagnosis embeddings are parallel, which (the
    # dicts being keyed by text) means identical text -- exactly what an exact DRG
    # match tests. See docs/findings/12-drg-grader.md.
    #
    # The DIFFERENCE is that here all five threshold rows carry these values,
    # not just the 1.0 row: a binary grader returns only 0.0 or 1.0, so the
    # threshold sweep collapses. Keyed at 1.0 anyway, so a regression that
    # un-collapsed the sweep would still be caught by this table.
    "drg": [
        ("baseline", "TOP-10", 1.0, {"P": 0.2512, "R": 0.1923, "FS": 0.2163, "PR": 0.7558}),
        ("bio_clinical_bert", "TOP-10", 1.0, {"P": 0.2491, "R": 0.2491, "FS": 0.2491, "PR": 1.0000}),
        ("biomedbert", "TOP-10", 1.0, {"P": 0.1981, "R": 0.1981, "FS": 0.1981, "PR": 1.0000}),
        ("bluebert", "TOP-10", 1.0, {"P": 0.1821, "R": 0.1821, "FS": 0.1821, "PR": 1.0000}),
    ],
}

# Which pipeline a results tree came from is not recorded in the output (that is
# P14, run_metadata.json). Until it is, infer from the directory name and let
# --pipeline override. Inference is a convenience, never a guess that proceeds
# silently: an unrecognised tree fails loudly below.
_PIPELINE_BY_DIRNAME = {
    "results": "legacy",
    "results_corrected": "corrected",
    "results_drg": "drg",
}

# What each pipeline changed, for the cover page. This exists because the cover
# used to state the preprocessing confound ("119 of 145 descriptions differ")
# unconditionally -- which P3 retired. A corrected-pipeline PDF asserting a
# confound that was fixed is worse than no PDF: it reads as a caveat the author
# checked.
PIPELINE_NOTES = {
    "legacy": {
        "label": "legacy",
        "detail": "data/folds + original preprocessing + cosine self-grading",
        "confound": (
            "One confound remains: the arms preprocess diagnosis text differently. The baseline calls\n"
            "preprocess_sentence on diagnosis strings and the BERT path does not, so 119 of 145 (82.1%) differ.\n"
            "The folds also split on HADM_ID, so 41 of 129 test cases share a patient with their own retrieval pool."
        ),
    },
    "corrected": {
        "label": "corrected",
        "detail": "data/folds_grouped + fixed preprocessing + cosine self-grading",
        "confound": (
            "Two of the four documented confounds are GONE here: patient leakage (41 leaked cases -> 0, GroupKFold on\n"
            "SUBJECT_ID) and divergent cross-arm preprocessing (26/145 matching -> 145/145). Two remain untouched:\n"
            "every arm still grades its own predictions by cosine in its own embedding space (self-grading), and the\n"
            "metric is rank-blind, so a hit at rank 1 and a hit at rank 50 count the same. Still not an encoder ranking."
        ),
    },
    "drg": {
        "label": "drg",
        "detail": "data/folds_grouped + fixed preprocessing + exact DRG-label grading",
        "confound": (
            "Leakage, preprocessing divergence AND self-grading are all removed here: a prediction counts only on an\n"
            "exact DRG label match, so all four arms are scored by one ruler rather than each by its own embedding\n"
            "space. Two caveats stay. The ceiling is 76/129 = 58.9% -- only that many test cases have their correct\n"
            "label anywhere in their training pool -- and the metric is still rank-blind, so TOP-K remains a free win."
        ),
    },
}


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

AGGREGATE_HEADER = re.compile(r"^10-FOLD PERFORMANCE INDEX of (.+?) SIMILARITY by MAX")
TOPK = re.compile(r"TOP-(\d+)")


def _strategy_from_header(descriptor):
    """Map the header descriptor ('MAX' / 'TOP-30') onto a strategy key."""
    match = TOPK.search(descriptor)
    if match:
        return "TOP-%s" % match.group(1)
    if descriptor.strip() == "MAX":
        return "MAX"
    return None


def parse_performance_index(path):
    """Return {strategy: {threshold: {TP,FP,P,R,FS,PR}}} for the aggregate blocks.

    Only the six ``10-FOLD`` sections are read; the ~5900 lines of per-fold and
    per-case blocks above them are ignored.
    """
    results = {}
    current = None

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()

            header = AGGREGATE_HEADER.match(line)
            if header:
                current = _strategy_from_header(header.group(1))
                if current is not None:
                    results.setdefault(current, {})
                continue

            if current is None:
                continue

            if not line or line.startswith("*") or line.startswith("="):
                # Blank line or banner ends the section.
                current = None
                continue

            parts = line.split()
            if len(parts) != 7:
                # The column header row ("TP FP P R FS PR") has 6 tokens.
                continue

            try:
                threshold = float(parts[0])
                values = [float(token) for token in parts[1:]]
            except ValueError:
                continue

            results[current][threshold] = dict(
                zip(("TP", "FP", "P", "R", "FS", "PR"), values)
            )

            if len(results[current]) == len(THRESHOLDS):
                current = None

    return results


def validate_parse(model_key, data):
    """Raise if a parsed file is not the expected 6 strategies x 5 thresholds."""
    missing = [s for s in STRATEGIES if s not in data]
    if missing:
        raise ValueError(
            "%s: missing aggregate sections %s" % (model_key, ", ".join(missing))
        )
    for strategy in STRATEGIES:
        found = sorted(data[strategy])
        if found != THRESHOLDS:
            raise ValueError(
                "%s / %s: expected thresholds %s, parsed %s"
                % (model_key, strategy, THRESHOLDS, found)
            )


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def _format_run_date(timestamp_dir):
    """'05082026_18-55-32' -> '2026-08-05'. Returns the raw name if unparseable."""
    match = re.match(r"^(\d{2})(\d{2})(\d{4})_", timestamp_dir)
    if not match:
        return timestamp_dir
    day, month, year = match.groups()
    return "%s-%s-%s" % (year, month, day)


def discover_runs(results_dir):
    """Find results/<model>/<timestamp>/PerformanceIndex.txt.

    Returns a list of run dicts in MODEL_REGISTRY order, with any unregistered
    model directories appended alphabetically.
    """
    if not os.path.isdir(results_dir):
        raise SystemExit("[ERROR] results directory not found: %s" % results_dir)

    found = {}
    for model_key in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_key)
        if not os.path.isdir(model_dir):
            continue

        candidates = []
        for stamp in sorted(os.listdir(model_dir)):
            run_dir = os.path.join(model_dir, stamp)
            perf = os.path.join(run_dir, "PerformanceIndex.txt")
            if os.path.isdir(run_dir) and os.path.isfile(perf):
                candidates.append((stamp, run_dir, perf))
        if not candidates:
            continue

        # Timestamp names are DDMMYYYY_HH-MM-SS, which does not sort
        # chronologically, so fall back to mtime when a model has several runs.
        candidates.sort(key=lambda item: os.path.getmtime(item[1]))
        stamp, run_dir, perf = candidates[-1]
        if len(candidates) > 1:
            print(
                "[INFO] %s has %d runs; using the most recent (%s)"
                % (model_key, len(candidates), stamp)
            )

        found[model_key] = {
            "key": model_key,
            "timestamp": stamp,
            "run_dir": run_dir,
            "perf_file": perf,
            "date": _format_run_date(stamp),
        }

    ordered_keys = [k for k in MODEL_REGISTRY if k in found]
    ordered_keys += [k for k in sorted(found) if k not in MODEL_REGISTRY]

    # The host is a property of WHEN a run happened, not of which model it used --
    # the same encoder has now been run on both machines. Deriving it from the run
    # directory's DDMMYYYY prefix is what keeps the provenance box honest; the old
    # per-model constant silently mislabelled the 2026-08-05 Linux runs as "Mac"
    # the first time all four arms came off one box.
    #
    # An unmapped date now resolves to "unknown host" rather than falling through
    # to MODEL_REGISTRY's per-model constant. That constant says "M-series Mac"
    # for every arm, so the fallback did not degrade gracefully -- it asserted a
    # specific wrong machine. The 2026-08-06 runs hit exactly that: they came off
    # RunPod and the box would have labelled them Mac. An honest blank beats a
    # confident error in a provenance box.
    hosts_by_date = {
        "15022026": "M-series Mac",
        "05082026": "RunPod Linux",
        "06082026": "RunPod Linux",
    }

    runs = []
    for index, key in enumerate(ordered_keys):
        run = found[key]
        style = MODEL_REGISTRY.get(key)
        if style is None:
            fallback = FALLBACK_STYLES[index % len(FALLBACK_STYLES)]
            style = dict(fallback, label=key, arm="unknown", host="unknown host")
        run.update(
            {
                "label": style["label"],
                "arm": style["arm"],
                "host": hosts_by_date.get(
                    str(run.get("timestamp", ""))[:8], "unknown host"
                ),
                "color": style["color"],
                "marker": style["marker"],
                "dash": style["dash"],
            }
        )
        runs.append(run)

    if not runs:
        raise SystemExit(
            "[ERROR] no runs found. Expected %s/<model>/<timestamp>/PerformanceIndex.txt"
            % results_dir
        )
    return runs


# --------------------------------------------------------------------------
# Derived diagnostics (computed from the data, not hard-coded)
# --------------------------------------------------------------------------

def describe_run(run):
    """Attach degeneracy / saturation / prediction-rate facts to a parsed run."""
    data = run["data"]
    rows = [data[s][t] for s in STRATEGIES for t in THRESHOLDS]

    run["degenerate"] = all(
        abs(r["P"] - r["R"]) < 1e-12 and abs(r["P"] - r["FS"]) < 1e-12 for r in rows
    )
    run["saturated_rows"] = sum(1 for r in rows if r["FS"] >= 0.99999)
    run["total_rows"] = len(rows)

    prediction_rates = sorted({round(r["PR"], 12) for r in rows})
    run["prediction_rate"] = prediction_rates[0]
    run["prediction_rate_constant"] = len(prediction_rates) == 1
    return run


def resolve_pipeline(results_dir, requested):
    """Decide which pipeline's expectations to check the parse against.

    Refuses rather than guesses. A results tree whose pipeline cannot be
    established, or one whose pipeline has no recorded expectations, exits with
    instructions -- because the alternative is a PDF built from an unverified
    parse, which looks exactly like a verified one.
    """
    known = ", ".join(sorted(SANITY_CHECKS_BY_PIPELINE))

    if requested is None:
        basename = os.path.basename(os.path.normpath(results_dir))
        requested = _PIPELINE_BY_DIRNAME.get(basename)
        if requested is None:
            raise SystemExit(
                "[ERROR] cannot tell which pipeline produced '%s'.\n"
                "        Pass --pipeline explicitly. Known: %s." % (results_dir, known)
            )
        print("[INFO] pipeline inferred from directory name: %s" % requested)
    else:
        print("[INFO] pipeline: %s (explicit)" % requested)

    if requested not in SANITY_CHECKS_BY_PIPELINE:
        raise SystemExit(
            "[ERROR] no parser expectations recorded for pipeline '%s'.\n"
            "        Recorded: %s.\n"
            "        Add a SANITY_CHECKS_BY_PIPELINE entry from that run's own\n"
            "        10-FOLD TOP-10 rows. Do NOT add an empty list -- zero checks\n"
            "        reports as success and the parse goes unverified."
            % (requested, known)
        )
    if requested not in PIPELINE_NOTES:
        # Caught here rather than as a KeyError three pages into rendering.
        raise SystemExit(
            "[ERROR] pipeline '%s' has parser expectations but no PIPELINE_NOTES\n"
            "        entry, so the cover page cannot say what it changed." % requested
        )
    return requested


def run_sanity_checks(runs_by_key, pipeline):
    """Compare the parse against known-correct values for one pipeline."""
    failures = []
    checked = 0
    for model_key, strategy, threshold, expected in SANITY_CHECKS_BY_PIPELINE[pipeline]:
        run = runs_by_key.get(model_key)
        if run is None:
            print("[WARN] sanity check skipped, no run for '%s'" % model_key)
            continue
        row = run["data"][strategy][threshold]
        for column, want in expected.items():
            checked += 1
            got = round(row[column], 4)
            if abs(got - want) > 5e-5:
                failures.append(
                    "%s %s @ %.1f  %s: expected %.4f, parsed %.4f"
                    % (model_key, strategy, threshold, column, want, got)
                )
    return checked, failures


# --------------------------------------------------------------------------
# Page helpers
# --------------------------------------------------------------------------

def _new_page(title, subtitle=None):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.945, title, fontsize=17, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.06, 0.912, subtitle, fontsize=9.5, color=INK_SECONDARY)
    return fig


def _style_axes(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=3, width=0.8)


def _annotate_endpoints(ax, x_data, values, x_label, gap=0.042, ymin=0.02):
    """Label the right-hand endpoint of each curve without letting labels collide.

    Values that round to the same 3 decimals get one shared label -- three BERT
    curves at exactly 1.000 must not print "1.000" three times at three different
    heights. Remaining labels are pushed apart to `gap`, the block is re-centred on
    the data it describes, and a thin leader ties each label back to its point.
    """
    unique = []
    seen = set()
    for value in values:
        key = "%.3f" % value
        if key not in seen:
            seen.add(key)
            unique.append((key, value))
    unique.sort(key=lambda item: item[1], reverse=True)

    placed = []
    previous = None
    for key, value in unique:
        y = value if previous is None else min(value, previous - gap)
        placed.append([key, value, y])
        previous = y

    # Re-centre the label block on the values, so leaders splay symmetrically.
    shift = (
        sum(item[1] for item in placed) - sum(item[2] for item in placed)
    ) / float(len(placed))
    for item in placed:
        item[2] = max(ymin, item[2] + shift)

    for key, value, y in placed:
        ax.annotate(
            key,
            xy=(x_data, value), xytext=(x_label, y), textcoords="data",
            fontsize=7.6, color=INK_SECONDARY, va="center", ha="left",
            arrowprops=dict(arrowstyle="-", linewidth=0.7, color=GRID,
                            shrinkA=3, shrinkB=2),
            annotation_clip=False,
        )


def _draw_table(ax, col_labels, cell_text, col_widths, cell_colors=None, fontsize=8.0):
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colWidths=col_widths,
        cellColours=cell_colors,
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.32)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER)
            cell.set_text_props(fontweight="bold", color=INK)
    return table


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

def page_cover(pdf, runs, pipeline):
    notes = PIPELINE_NOTES[pipeline]
    fig = _new_page(
        "Four-model comparison: BioSentVec baseline vs three BERT encoders",
        "Reproduction of Comito et al. (2022) -- 10-fold aggregate results, "
        "pipeline '%s' (%s), generated by scripts/compare_models.py"
        % (notes["label"], notes["detail"]),
    )

    # ---- Provenance warning, first thing on the page ----------------------
    warn = fig.add_axes([0.06, 0.625, 0.88, 0.255])
    warn.axis("off")
    warn.add_patch(
        plt.Rectangle(
            (0, 0), 1, 1, transform=warn.transAxes,
            facecolor=WARN_FILL, edgecolor=WARN_EDGE, linewidth=1.6,
        )
    )
    # One machine or several? The hardware confound is only real when the runs
    # actually came off different boxes, so say which situation this page shows
    # rather than warning unconditionally.
    one_host = len({r["host"] for r in runs}) == 1

    warn.text(
        0.022, 0.87,
        "PROVENANCE -- one machine; preprocessing still differs" if one_host
        else "PROVENANCE WARNING -- cross-arm deltas are confounded",
        fontsize=12, fontweight="bold", color="#a83c14", va="top",
    )

    lines = []
    for run in runs:
        lines.append(
            "    %-24s  %s, %s   (run %s)"
            % (run["label"], run["host"], run["date"], run["timestamp"])
        )
    warn.text(
        0.022, 0.68, "\n".join(lines),
        fontsize=8.6, color=INK, va="top", family="monospace",
    )

    hardware = (
        "All four arms ran on the SAME machine, so hardware is not a confound here -- and re-running the three\n"
        "BERT encoders on a second platform reproduced their numbers bit-for-bit, to all 17 significant figures."
        if one_host else
        "The arms ran on DIFFERENT hardware, so any baseline-vs-BERT gap below mixes encoder with machine. Treat\n"
        "the three BERT models as comparable to each other and the baseline as a separate reference point."
    )
    warn.text(
        0.022, 0.36,
        hardware + "\n" + notes["confound"],
        fontsize=8.0, color=INK, va="top", linespacing=1.45,
    )

    # ---- How to read ------------------------------------------------------
    fig.text(0.06, 0.570, "How to read these numbers", fontsize=12,
             fontweight="bold", color=INK)

    degenerate = [r["label"] for r in runs if r["degenerate"]]
    non_degenerate = [r["label"] for r in runs if not r["degenerate"]]
    saturated = [
        "%s (%d/%d rows)" % (r["label"], r["saturated_rows"], r["total_rows"])
        for r in runs if r["saturated_rows"]
    ]

    bullets = [
        (
            '1.  "F-score" is accuracy for the BERT models.',
            "Across every parsed aggregate row for %s, precision == recall == F-score exactly. Each test case\n"
            "increments exactly one of TP or FP, so tp+fp == n, precision reduces to tp/n -- which is recall -- and the\n"
            "harmonic mean is that same number. Only %s reports a genuine P != R."
            % (", ".join(degenerate) or "none", ", ".join(non_degenerate) or "none"),
        ),
        (
            "2.  The BERT models saturate below threshold 1.0.",
            "Aggregate rows scoring exactly 1.000 -- %s.\n"
            "Biomedical embeddings are compact (mean pairwise cosine 0.72-0.93 even between unrelated diagnoses) and\n"
            "the MAX-over-Cartesian-product aggregator amplifies that, so nearly every patient pair clears 0.6.\n"
            "Threshold 1.0 is the only setting at which the three BERT encoders separate."
            % ("; ".join(saturated) or "none"),
        ),
        (
            "3.  The PR column is the diagnostic, not the F column.",
            "PR is the prediction rate: the share of test cases the system answered at all. The baseline abstains when no\n"
            "training patient clears PRUNING_SIMILARITY = 0.5; the BERT arm never abstains, which is exactly why its\n"
            "precision and recall collapse onto each other. See page 5.",
        ),
        (
            "4.  TOP-K rising with K is an artifact.",
            "One hit inside K predictions suffices and there is no penalty for the other K-1, so the TOP-K curve on page 4\n"
            "can only go up. It is not evidence that larger K retrieves better.",
        ),
    ]

    y = 0.533
    for heading, body in bullets:
        fig.text(0.065, y, heading, fontsize=9.6, fontweight="bold", color=INK, va="top")
        fig.text(0.085, y - 0.026, body, fontsize=8.3, color=INK_SECONDARY,
                 va="top", linespacing=1.5)
        # Advance by the height this bullet actually consumed, so a body that
        # grows a line does not collide with the next heading.
        y -= 0.030 + 0.0215 * (body.count("\n") + 1) + 0.018

    fig.text(
        0.06, 0.038,
        "Both arms additionally share a patient-leakage defect: folds split on HADM_ID, but 129 admissions come from "
        "100 patients, so 41 of 129\ntest cases (31.8%) retrieve another admission of the same SUBJECT_ID. Measured "
        "inflation at threshold 1.0 is +0.11 to +0.26 -- roughly ten\ntimes the encoder differences shown here. "
        "See docs/findings/05-patient-leakage.md.",
        fontsize=7.6, color=INK_MUTED, va="bottom", linespacing=1.5,
    )

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def page_summary_table(pdf, runs):
    fig = _new_page(
        "TOP-10 summary: all models, all thresholds",
        "10-fold aggregate. P = precision, R = recall, F = F-score, PR = prediction rate "
        "(share of test cases answered).",
    )

    col_labels = ["Model", "Provenance", "Thr.", "P", "R", "F", "PR", "P == R == F ?"]
    cell_text = []
    cell_colors = []

    for run in runs:
        tint = "#ffffff"
        for index, threshold in enumerate(THRESHOLDS):
            row = run["data"]["TOP-10"][threshold]
            same = (
                abs(row["P"] - row["R"]) < 1e-12 and abs(row["P"] - row["FS"]) < 1e-12
            )
            cell_text.append(
                [
                    run["label"] if index == 0 else "",
                    "%s, %s" % (run["host"], run["date"]) if index == 0 else "",
                    "%.1f" % threshold,
                    "%.4f" % row["P"],
                    "%.4f" % row["R"],
                    "%.4f" % row["FS"],
                    "%.4f" % row["PR"],
                    "yes -- F is accuracy" if same else "no",
                ]
            )
            cell_colors.append([tint] * len(col_labels))

    ax = fig.add_axes([0.05, 0.10, 0.90, 0.78])
    table = _draw_table(
        ax, col_labels, cell_text,
        col_widths=[0.20, 0.19, 0.07, 0.10, 0.10, 0.10, 0.10, 0.16],
        fontsize=8.2,
    )

    # Colour the model name cell by the model's series colour so the tables and
    # the charts share one identity channel.
    for row_index, run_index in enumerate(range(len(runs))):
        cell = table[(row_index * len(THRESHOLDS) + 1, 0)]
        cell.set_text_props(fontweight="bold", color=runs[run_index]["color"])

    fig.text(
        0.05, 0.295,
        "Read the last column first. For the three BERT models every single row has P == R == F, so their \"F1\" is "
        "accuracy, not F1.\nThe baseline is the only arm whose precision and recall differ -- because it is the only "
        "arm that ever declines to answer (PR < 1).",
        fontsize=8.2, color=INK_SECONDARY, va="top", linespacing=1.55,
    )

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _plot_threshold_panel(ax, runs, strategy, annotate_at=1.0):
    _style_axes(ax)
    for run in runs:
        values = [run["data"][strategy][t]["FS"] for t in THRESHOLDS]
        ax.plot(
            THRESHOLDS, values,
            color=run["color"], linewidth=2.0, linestyle=run["dash"],
            marker=run["marker"], markersize=6.5,
            markeredgecolor="white", markeredgewidth=1.2,
            label=run["label"], clip_on=False, zorder=3,
        )

    _annotate_endpoints(
        ax, annotate_at,
        [run["data"][strategy][annotate_at]["FS"] for run in runs],
        x_label=1.035,
    )

    ax.set_xticks(THRESHOLDS)
    ax.set_xlim(0.575, 1.16)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("Similarity threshold", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("F-score  (= accuracy for the BERT arm)", fontsize=9, color=INK_SECONDARY)
    ax.set_title(
        "%s strategy" % strategy, fontsize=11, fontweight="bold", color=INK, loc="left"
    )
    ax.axhline(1.0, color=WARN_EDGE, linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.8, zorder=1)
    ax.text(
        0.585, 1.015, "saturation ceiling", fontsize=7.2, color=WARN_EDGE, va="bottom"
    )


def page_threshold_curves(pdf, runs):
    fig = _new_page(
        "F-score vs similarity threshold",
        "Direct labels mark threshold 1.0 -- the only setting at which the three BERT "
        "encoders separate. Dash patterns distinguish curves that coincide.",
    )

    ax_left = fig.add_axes([0.075, 0.335, 0.40, 0.525])
    ax_right = fig.add_axes([0.565, 0.335, 0.40, 0.525])
    _plot_threshold_panel(ax_left, runs, "MAX")
    _plot_threshold_panel(ax_right, runs, "TOP-10")

    handles, labels = ax_left.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.215),
        ncol=4, frameon=False, fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(INK)

    fig.text(
        0.075, 0.175,
        "The flat run at 1.000 across thresholds 0.6-0.9 is the saturation finding: the BERT similarity distribution "
        "sits so high that essentially every\npatient pair clears the threshold, so the metric stops discriminating. "
        "BlueBERT is the only BERT model that dips below the ceiling before 1.0.\nThe baseline never approaches it -- "
        "sent2vec's space is far less compact, and the baseline also abstains on 23.2% of cases.",
        fontsize=8.2, color=INK_SECONDARY, va="top", linespacing=1.55,
    )

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _plot_topk_panel(ax, runs, threshold):
    _style_axes(ax)
    positions = list(range(len(STRATEGIES)))
    for run in runs:
        values = [run["data"][s][threshold]["FS"] for s in STRATEGIES]
        ax.plot(
            positions, values,
            color=run["color"], linewidth=2.0, linestyle=run["dash"],
            marker=run["marker"], markersize=6.5,
            markeredgecolor="white", markeredgewidth=1.2,
            label=run["label"], clip_on=False, zorder=3,
        )

    _annotate_endpoints(
        ax, positions[-1],
        [run["data"][STRATEGIES[-1]][threshold]["FS"] for run in runs],
        x_label=positions[-1] + 0.35,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(["MAX", "10", "20", "30", "40", "50"])
    ax.set_xlim(-0.25, len(STRATEGIES) + 0.55)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("Retrieval strategy (MAX, then TOP-K)", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("F-score  (= accuracy for the BERT arm)", fontsize=9, color=INK_SECONDARY)
    ax.set_title(
        "threshold %.1f" % threshold, fontsize=11, fontweight="bold", color=INK, loc="left"
    )


def page_topk_curves(pdf, runs):
    fig = _new_page(
        "F-score vs TOP-K, at the two extreme thresholds",
        "Left: threshold 0.6, where the BERT arm is saturated. Right: threshold 1.0, "
        "the only informative setting.",
    )

    ax_left = fig.add_axes([0.075, 0.335, 0.40, 0.525])
    ax_right = fig.add_axes([0.565, 0.335, 0.40, 0.525])
    _plot_topk_panel(ax_left, runs, 0.6)
    _plot_topk_panel(ax_right, runs, 1.0)

    handles, labels = ax_left.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.215),
        ncol=4, frameon=False, fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(INK)

    fig.text(
        0.075, 0.175,
        "These curves can only rise: a case counts as a hit if the correct diagnosis appears anywhere in the K "
        "predictions, and the other K-1 cost nothing.\nThe rise from MAX to TOP-10 is therefore a measure of how much "
        "slack the metric grants, not of retrieval quality. A genuine set-level P/R/F1\nover the diagnosis sets is "
        "tracked in docs/plans/metric-redesign.md. Note also that only 75 of 129 test cases (58.1%) have their correct\n"
        "DRG present anywhere in their fold's training pool, so a perfect retriever caps at 0.581 under exact matching.",
        fontsize=8.2, color=INK_SECONDARY, va="top", linespacing=1.55,
    )

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def page_prediction_rate(pdf, runs):
    fig = _new_page(
        "Prediction rate (PR): the column that explains the metric collapse",
        "PR = share of the 129 test cases for which the system emitted any prediction at all.",
    )

    ax = fig.add_axes([0.24, 0.50, 0.62, 0.33])
    _style_axes(ax)
    ax.grid(axis="y", visible=False)

    labels = [run["label"] for run in runs]
    values = [run["prediction_rate"] for run in runs]
    positions = list(range(len(runs)))[::-1]

    ax.barh(
        positions, values,
        color=[run["color"] for run in runs],
        height=0.55, zorder=3,
    )
    for pos, value in zip(positions, values):
        ax.text(
            value + 0.012, pos, "%.4f" % value,
            va="center", fontsize=9, fontweight="bold", color=INK,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Prediction rate", fontsize=9, color=INK_SECONDARY)
    ax.axvline(1.0, color=GRID, linewidth=1.0, zorder=1)

    # Value table -- also the accessibility relief for the low-contrast series hue.
    ax_table = fig.add_axes([0.10, 0.315, 0.80, 0.13])
    cell_text = []
    for run in runs:
        pr = run["prediction_rate"]
        cell_text.append(
            [
                run["label"],
                "%.4f" % pr,
                "%.1f%%" % (100.0 * (1.0 - pr)),
                "%.0f of 129" % round(129.0 * (1.0 - pr)),
                "constant" if run["prediction_rate_constant"] else "VARIES",
                "yes" if run["degenerate"] else "no",
            ]
        )
    _draw_table(
        ax_table,
        ["Model", "PR", "Abstention rate", "Cases not answered", "PR across all 30 rows", "P == R == F ?"],
        cell_text,
        col_widths=[0.24, 0.11, 0.16, 0.17, 0.18, 0.14],
        fontsize=8.2,
    )

    fig.text(
        0.06, 0.265,
        "This is the single most informative difference between the two arms, and it is the mechanism behind the "
        "degeneracy on page 2.",
        fontsize=9.4, fontweight="bold", color=INK, va="top",
    )
    fig.text(
        0.06, 0.232,
        "The baseline answers 76.8% of test cases: on the remaining 23.2% no training patient clears "
        "PRUNING_SIMILARITY = 0.5, so it abstains and\nthe case counts against recall without ever entering "
        "precision. That is precisely why the baseline is the only arm with P != R.\n\n"
        "All three BERT models sit at PR = 1.0000. Their embedding space is compact enough that some candidate always "
        "clears 0.5, so they never\nabstain -- every case increments exactly one of TP or FP, tp + fp equals the case "
        "count, and precision, recall and F-score become the same\nnumber. The BERT \"F1\" is accuracy. A PR of 1.0 is "
        "not a strength: it means the abstention mechanism that the published pipeline relies on\nhas been switched "
        "off by the encoder, so the two arms are not being scored by the same rule even before the preprocessing "
        "difference is\nconsidered.",
        fontsize=8.4, color=INK_SECONDARY, va="top", linespacing=1.55,
    )

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def page_full_grid(pdf, runs):
    fig = _new_page(
        "Appendix: F-score for every strategy and threshold",
        "Same values as the charts, in full precision. Cells at exactly 1.000 are shaded -- that is saturation, "
        "not a perfect result.",
    )

    col_labels = ["Model", "Strategy"] + ["thr %.1f" % t for t in THRESHOLDS]
    cell_text = []
    cell_colors = []
    for run in runs:
        for index, strategy in enumerate(STRATEGIES):
            row = [run["label"] if index == 0 else "", strategy]
            colors = ["#ffffff", "#ffffff"]
            for threshold in THRESHOLDS:
                value = run["data"][strategy][threshold]["FS"]
                row.append("%.4f" % value)
                colors.append(WARN_FILL if value >= 0.99999 else "#ffffff")
            cell_text.append(row)
            cell_colors.append(colors)

    ax = fig.add_axes([0.10, 0.06, 0.80, 0.82])
    table = _draw_table(
        ax, col_labels, cell_text,
        col_widths=[0.22, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13],
        cell_colors=cell_colors,
        fontsize=7.6,
    )
    for run_index, run in enumerate(runs):
        table[(run_index * len(STRATEGIES) + 1, 0)].set_text_props(
            fontweight="bold", color=run["color"]
        )

    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_report(runs, out_path, pipeline):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    notes = PIPELINE_NOTES[pipeline]

    with PdfPages(out_path) as pdf:
        page_cover(pdf, runs, pipeline)
        page_summary_table(pdf, runs)
        page_threshold_curves(pdf, runs)
        page_topk_curves(pdf, runs)
        page_prediction_rate(pdf, runs)
        page_full_grid(pdf, runs)

        # The pipeline goes in the metadata as well as on the cover. Two PDFs from
        # different pipelines are otherwise indistinguishable once the filename is
        # lost, and these get emailed around.
        info = pdf.infodict()
        info["Title"] = (
            "AI-CDS four-model comparison, pipeline '%s' (10-fold aggregate)"
            % notes["label"]
        )
        info["Subject"] = (
            "BioSentVec baseline vs Bio_ClinicalBERT / BiomedBERT / BlueBERT. "
            "Pipeline: %s. Not an encoder ranking -- see the cover page." % notes["detail"]
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a four-model comparison PDF from results/<model>/<timestamp>/."
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="directory containing <model>/<timestamp>/PerformanceIndex.txt (default: results)",
    )
    parser.add_argument(
        "--out", default=None,
        help="output PDF path (default: <results-dir>/model_comparison.pdf)",
    )
    parser.add_argument(
        "--pipeline", default=None,
        help="which pipeline's parser expectations to check against: %s. "
        "Default: inferred from the results directory name. The expectations are "
        "pipeline-specific, so a mismatch fails every check rather than a few."
        % ", ".join(sorted(SANITY_CHECKS_BY_PIPELINE)),
    )
    args = parser.parse_args(argv)

    out_path = args.out or os.path.join(args.results_dir, "model_comparison.pdf")
    pipeline = resolve_pipeline(args.results_dir, args.pipeline)

    print("[INFO] discovering runs under %s/" % args.results_dir)
    runs = discover_runs(args.results_dir)

    for run in runs:
        run["data"] = parse_performance_index(run["perf_file"])
        validate_parse(run["key"], run["data"])
        describe_run(run)
        print(
            "[INFO]   %-20s %s  (%s, %s)  PR=%.4f  P==R==F: %s"
            % (
                run["key"], run["timestamp"], run["host"], run["date"],
                run["prediction_rate"], "yes" if run["degenerate"] else "no",
            )
        )

    runs_by_key = {run["key"]: run for run in runs}
    checked, failures = run_sanity_checks(runs_by_key, pipeline)
    if failures:
        print("[ERROR] parser sanity checks FAILED:")
        for failure in failures:
            print("          " + failure)
        print(
            "        If this results tree is NOT pipeline '%s', pass the right\n"
            "        --pipeline. A mismatch fails every check at once, which is\n"
            "        what a wrong pipeline looks like." % pipeline
        )
        return 1
    if not checked:
        # Every arm was missing, so the loop skipped everything and reported
        # zero failures. "0 checks passed" is indistinguishable from a verified
        # parse in the output, so refuse instead.
        print(
            "[ERROR] zero parser sanity checks ran -- none of pipeline '%s'"
            % pipeline
        )
        print("        expected arms were found under %s/." % args.results_dir)
        return 1
    print("[SUCCESS] %d parser sanity checks passed (pipeline: %s)" % (checked, pipeline))

    build_report(runs, out_path, pipeline)
    print("[SUCCESS] wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

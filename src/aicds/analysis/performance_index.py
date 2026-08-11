"""The one parser for ``PerformanceIndex.txt``.

Four hand-rolled parsers for this file exist in ``scripts/`` today
(``compare_models.parse_performance_index``,
``analyze_performance.parse_performance_index``,
``build_dashboard_data.parse_performance_file``,
``build_readme_plots.parse_performance_file``). They disagree about which
sections exist, which columns survive, and what the metric is called -- one emits
``"F1"``, one ``"FS"``, one drops ``PR`` entirely, and one accumulates into
module-level globals so two files parsed in one process silently merge. This
module replaces all four. ``tests/test_performance_index_equivalence.py`` pins it
against every one of them on real fixtures before they are deleted.

The format is written in three places and nowhere specified:
``cython_utils.compute_performance_index`` /
``compute_aggregated_performance_index`` / ``print_performance_index`` emit the
data rows, and ``bert_models`` / ``baseline_sent2vec`` emit the section headers
and banners around them. Every failure mode in it is a *whitespace* failure,
which is why :class:`PerformanceIndexError` reports the 1-based line number and
``repr()`` of the offending line rather than a bare message.

The traps, each of which cost a parser somewhere
------------------------------------------------

1. **Three header shapes, and the per-case one must be tested first.** A loose
   ``.+ SIMILARITY by MAX`` pattern swallows the per-case header, because
   ``"0 - HADM_ID=999001: PERFORMANCE INDEX of MAX SIMILARITY by MAX"`` matches
   it too. Order: per-case, then ``<k>-FOLD``, then fold-aggregate.

2. **The optional leading space on the fold-aggregate header is load-bearing.**
   ``MAX`` is written at column 0 and every ``TOP-K`` with one leading space, in
   *both* arms (``bert_models.py:702`` vs ``:708``,
   ``baseline_sent2vec.py:429`` vs ``:434``). Anchoring on ``^PERFORMANCE`` finds
   only the MAX block; anchoring on ``^ PERFORMANCE`` finds only the TOP-K ones.

3. **The fold delimiter has a leading *and* a trailing space** --
   ``" FOLD 0: LEN train: 116, LEN test: 13 \\n"``. A fold-aggregate block with no
   preceding fold header means the file is malformed (or a section header was
   mis-recognised), so it raises rather than inventing a fold number.

4. **A data row is exactly 7 whitespace-separated tokens; the column header is
   6.** That is the only reliable discriminator, because the separators are not
   consistent: the fold and 10-FOLD rows are TAB-separated with a bare integer
   ``1`` for threshold 1.0 (``str(1)`` over the set literal ``{1, 0.9, ...}``),
   while ``bert_models``' per-case rows are hand-padded with spaces and print
   ``1.0``. ``float(token)`` normalises the two spellings. Rows also arrive in
   *set-iteration* order -- 0.9, 1, 0.6, 0.8, 0.7 -- not sorted order, so nothing
   here may assume position.

5. **Per-case blocks are recognised, counted, and their values discarded.** This
   is deliberate and is the one place this parser refuses to be more useful. The
   baseline's per-case rows are ``compute_performance_index`` called on the
   *cumulative* fold confusion matrix (``cython_utils.py:312-318`` -- the same
   ``confusion_matrix_max`` that keeps accumulating across test cases), so row
   *i* is the running total after *i+1* cases. The BERT arm's per-case rows are
   single-trial: ``tp, fp = 1, 0`` or ``0, 1`` for that one case
   (``bert_models.py:647-663``). The two are not the same quantity, and exposing
   them through one field is an invitation to average across arms. Only
   :attr:`PerformanceIndex.per_case_blocks` comes out.

6. **``k_fold`` is parsed from the ``<k>-FOLD`` header, not assumed to be 10.**
   :func:`validate` compares it against what the caller expects. (Note that
   ``print_performance_index`` divides by the ``K_FOLD`` *constant* regardless of
   how many folds actually ran, so a short run's aggregate is arithmetically
   wrong -- that is a writer defect this parser reports faithfully rather than
   papering over.)

7. **The trailer starts at the first line of exactly 80 ``=``.** Below it are
   ``"name: N seconds"`` timings and ``MODEL: <name>``; above it, no line ever
   matches those. The constant is duplicated here rather than imported from
   ``tests/test_golden.py``: a net must not import what it polices. Note that
   ``"Fold 0:  80.29 seconds (1.34 minutes)"`` matches the seconds pattern too,
   which is how :attr:`PerformanceIndex.fold_times` is derived -- both existing
   clones already behave this way and it is kept on purpose. A file with **no**
   trailer at all is legal (``tests/golden/stub768`` is trailer-stripped by
   design): ``timing`` comes back empty and ``model`` is ``None``.

Parsing and validation are separate on purpose. A truncated or in-progress file
must still parse -- that is exactly when you want to look at it -- so structural
expectations (6 strategies x 5 thresholds x 10 folds) live in :func:`validate`
and are opt-in. The k-FOLD aggregate is written *last* (line 5921 of 6000 in the
committed BiomedBERT file), so :func:`parse` cannot require it; its floor is one
aggregate block of either kind, per-fold or k-FOLD. A file with neither -- pure
garbage, or a prefix carrying only per-case rows -- is rejected at end of parse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


__all__ = [
    "MetricRow",
    "PerformanceIndex",
    "PerformanceIndexError",
    "STRATEGIES",
    "THRESHOLDS",
    "parse",
    "read",
    "validate",
]


#: The six aggregation strategies every complete run writes, in write order.
STRATEGIES: Tuple[str, ...] = ("MAX", "TOP-10", "TOP-20", "TOP-30", "TOP-40", "TOP-50")

#: The five score thresholds, sorted. The *file* emits them in set-iteration
#: order (0.9, 1, 0.6, 0.8, 0.7); this tuple is for validation only.
THRESHOLDS: Tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 1.0)

# Trap 7. Deliberately duplicated from tests/test_golden.py's TRAILER_DELIMITER
# rather than imported: the golden test is the safety net for this repo's
# numbers, and a net must not import what it polices. If the writers ever change
# the banner, both copies should fail independently.
TRAILER_DELIMITER = "=" * 80

# Trap 1: per-case first. This one is anchored on the row index and HADM_ID that
# only per-case headers carry.
_PER_CASE_HEADER = re.compile(
    r"^(?P<case>\d+) - HADM_ID=(?P<hadm>[^:]+): "
    r"PERFORMANCE INDEX of (?P<what>.+?) SIMILARITY by MAX\s*$"
)
_KFOLD_HEADER = re.compile(
    r"^(?P<k>\d+)-FOLD PERFORMANCE INDEX of (?P<what>.+?) SIMILARITY by MAX\s*$"
)
# Trap 2: the leading space is optional, and that is the whole point.
_FOLD_AGG_HEADER = re.compile(
    r"^ ?PERFORMANCE INDEX of (?P<what>.+?) SIMILARITY by MAX\s*$"
)
# Trap 3: leading AND trailing space.
_FOLD_DELIMITER = re.compile(
    r"^ FOLD (?P<fold>\d+): LEN train: (?P<train>\d+), LEN test: (?P<test>\d+) $"
)

_TOPK = re.compile(r"^TOP-(\d+)$")

# Trailer-mode patterns. Kept identical in shape to the two existing clones in
# scripts/ so the equivalence test compares like with like.
_TIMING_LINE = re.compile(r"^([^:]+):\s+([0-9]+(?:\.[0-9]+)?)\s+seconds")
_MODEL_LINE = re.compile(r"^MODEL:\s+([^(]+)")
_FOLD_TIME_KEY = re.compile(r"^Fold (\d+)$")


class PerformanceIndexError(ValueError):
    """A ``PerformanceIndex.txt`` that does not parse.

    Carries ``path``, the 1-based ``lineno``, and ``line`` (the raw text, shown
    via ``repr`` so trailing spaces and tabs are visible) because every failure
    in this format is a whitespace failure and a message without the bytes is
    useless. ``lineno``/``line`` are ``None`` for failures found by
    :func:`validate`, which inspects the parsed structure rather than the text.
    """

    def __init__(self, message, path=None, lineno=None, line=None):
        self.path = path
        self.lineno = lineno
        self.line = line
        where = path if path is not None else "<stream>"
        if lineno is not None:
            detail = "%s:%d: %s" % (where, lineno, message)
        else:
            detail = "%s: %s" % (where, message)
        if line is not None:
            detail = "%s (line was %r)" % (detail, line)
        super(PerformanceIndexError, self).__init__(detail)


@dataclass(frozen=True)
class MetricRow:
    """One ``threshold TP FP P R FS PR`` row.

    All seven fields are floats even where the writer emitted integers: the
    threshold column is a bare ``1`` in aggregate rows and ``1.0`` in per-case
    rows, and ``float()`` is what makes those the same key. ``f_score`` is
    frequently a bare ``0`` in the file (1597 rows of the golden) rather than
    ``0.0`` -- the formatting difference is real and load-bearing for the golden
    comparison, but it is not information this parser carries.
    """

    threshold: float
    tp: float
    fp: float
    precision: float
    recall: float
    f_score: float
    prediction_rate: float

    def as_dict(self) -> Dict[str, float]:
        """The legacy six-key row shape the four old parsers all produced.

        Keys are the file's own column names (``TP FP P R FS PR``), which is what
        ``compare_models`` and ``build_readme_plots`` emit.
        ``build_dashboard_data`` calls the same column ``F1``; the equivalence
        test asserts ``F1 == FS == f_score`` rather than pretending one of the
        two names is correct. The threshold is deliberately absent -- it is the
        dict *key* in every consumer.
        """
        return {
            "TP": self.tp,
            "FP": self.fp,
            "P": self.precision,
            "R": self.recall,
            "FS": self.f_score,
            "PR": self.prediction_rate,
        }


@dataclass(frozen=True)
class PerformanceIndex:
    """A parsed ``PerformanceIndex.txt``.

    ``folds`` is ``{fold_index: {strategy: {threshold: MetricRow}}}`` and
    ``aggregate`` is ``{strategy: {threshold: MetricRow}}`` -- plain dicts, not
    ``defaultdict``s, so a missing strategy raises ``KeyError`` at the call site
    instead of materialising an empty one (which is how
    ``analyze_performance``'s module-level ``defaultdict`` globals hid their
    merging).

    ``per_case_blocks`` is a count and nothing more; see trap 5 in the module
    docstring for why the values are dropped.
    """

    folds: Dict[int, Dict[str, Dict[float, MetricRow]]]
    aggregate: Dict[str, Dict[float, MetricRow]]
    timing: Dict[str, float] = field(default_factory=dict)
    fold_times: List[float] = field(default_factory=list)
    model: Optional[str] = None
    k_fold: int = 0
    per_case_blocks: int = 0
    path: Optional[str] = None


def _strategy(descriptor: str) -> str:
    """``'MAX'`` / ``'TOP-30'`` -> the same string, validated.

    Returned verbatim rather than normalised, because the descriptor in the file
    *is* the strategy key every consumer uses. Anything else is a format
    surprise the caller should hear about.
    """
    text = descriptor.strip()
    if text == "MAX" or _TOPK.match(text):
        return text
    return ""


def read(path: str) -> PerformanceIndex:
    """Parse the ``PerformanceIndex.txt`` at ``path``."""
    with open(path, "r", encoding="utf-8", newline=None) as handle:
        return parse(handle, path=path)


def parse(stream: Iterable[str], path: Optional[str] = None) -> PerformanceIndex:
    """Parse an iterable of lines (a file object, or a list, or a ``StringIO``).

    Line endings may be CRLF or LF; the real files are CRLF on every platform
    because the writers use text mode. ``\\r`` is stripped explicitly so the
    fold delimiter's trailing space (trap 3) is still the last character.
    """
    folds: Dict[int, Dict[str, Dict[float, MetricRow]]] = {}
    aggregate: Dict[str, Dict[float, MetricRow]] = {}
    timing: Dict[str, float] = {}
    model: Optional[str] = None
    k_fold: Optional[int] = None
    per_case_blocks = 0

    current_fold: Optional[int] = None
    # ("per_case", strategy) | ("fold", strategy) | ("kfold", strategy) | None
    section: Optional[Tuple[str, str]] = None
    in_trailer = False

    for lineno, raw in enumerate(stream, start=1):
        line = raw.rstrip("\n").rstrip("\r")

        if in_trailer:
            stripped = line.strip()
            timing_match = _TIMING_LINE.match(stripped)
            if timing_match:
                # Trap 7: "Fold 0: 80.29 seconds (1.34 minutes)" lands here too,
                # keyed "Fold 0". fold_times below is derived from exactly that.
                timing[timing_match.group(1).strip()] = float(timing_match.group(2))
                continue
            model_match = _MODEL_LINE.match(stripped)
            if model_match:
                model = model_match.group(1).strip()
            continue

        if line == TRAILER_DELIMITER:
            in_trailer = True
            section = None
            continue

        # Trap 1: per-case FIRST.
        per_case = _PER_CASE_HEADER.match(line)
        if per_case:
            name = _strategy(per_case.group("what"))
            if not name:
                raise PerformanceIndexError(
                    "unrecognised strategy %r in per-case header"
                    % per_case.group("what"),
                    path,
                    lineno,
                    line,
                )
            per_case_blocks += 1
            section = ("per_case", name)
            continue

        kfold = _KFOLD_HEADER.match(line)
        if kfold:
            name = _strategy(kfold.group("what"))
            if not name:
                raise PerformanceIndexError(
                    "unrecognised strategy %r in k-fold header" % kfold.group("what"),
                    path,
                    lineno,
                    line,
                )
            parsed_k = int(kfold.group("k"))
            if k_fold is not None and k_fold != parsed_k:
                raise PerformanceIndexError(
                    "k-fold header says %d but an earlier one said %d"
                    % (parsed_k, k_fold),
                    path,
                    lineno,
                    line,
                )
            k_fold = parsed_k
            if name in aggregate:
                # Trap 7 / rule 7: a repeat would silently overwrite, which is
                # how a partially-rewritten file would read as valid.
                raise PerformanceIndexError(
                    "repeated %d-FOLD section for strategy %s" % (parsed_k, name),
                    path,
                    lineno,
                    line,
                )
            aggregate[name] = {}
            section = ("kfold", name)
            continue

        # Trap 2: the leading space is optional. Reached only after the two more
        # specific shapes above have been ruled out.
        fold_agg = _FOLD_AGG_HEADER.match(line)
        if fold_agg:
            name = _strategy(fold_agg.group("what"))
            if not name:
                raise PerformanceIndexError(
                    "unrecognised strategy %r in fold-aggregate header"
                    % fold_agg.group("what"),
                    path,
                    lineno,
                    line,
                )
            if current_fold is None:
                # Trap 3: no " FOLD n: ..." delimiter has been seen, so there is
                # no honest place to put these rows.
                raise PerformanceIndexError(
                    "fold-aggregate block with no preceding FOLD header",
                    path,
                    lineno,
                    line,
                )
            per_fold = folds.setdefault(current_fold, {})
            if name in per_fold:
                raise PerformanceIndexError(
                    "repeated %s section inside FOLD %d" % (name, current_fold),
                    path,
                    lineno,
                    line,
                )
            per_fold[name] = {}
            section = ("fold", name)
            continue

        if "PERFORMANCE INDEX" in line:
            # Rule 7: matched none of the three shapes. Almost always a
            # whitespace change in a writer.
            raise PerformanceIndexError(
                "PERFORMANCE INDEX line matches none of the three header shapes",
                path,
                lineno,
                line,
            )

        fold_delim = _FOLD_DELIMITER.match(line)
        if fold_delim:
            current_fold = int(fold_delim.group("fold"))
            section = None
            continue

        stripped = line.strip()
        if not stripped or set(stripped) == {"*"} or set(stripped) == {"="}:
            # Blank line or a banner terminates the current section. The
            # baseline writes a '*' banner *between* 10-FOLD blocks
            # (baseline_sent2vec.py:453) where the BERT arm writes none, so
            # banners cannot be treated as end-of-file.
            section = None
            continue

        if section is None:
            # Outside any section: ignore. Nothing in either writer puts content
            # here today, but a future informational line should not break
            # reading the numbers.
            continue

        tokens = stripped.split()
        if len(tokens) == 6:
            # Trap 4: the column header ("\t TP \t FP \t  P \t R \t FS \t PR").
            continue
        if len(tokens) != 7:
            raise PerformanceIndexError(
                "expected 7 whitespace-separated tokens in a data row, got %d"
                % len(tokens),
                path,
                lineno,
                line,
            )

        try:
            values = [float(token) for token in tokens]
        except ValueError:
            raise PerformanceIndexError(
                "non-numeric token in a 7-column data row",
                path,
                lineno,
                line,
            )

        row = MetricRow(*values)
        kind, name = section
        if kind == "per_case":
            # Trap 5: counted at the header, values dropped here.
            continue
        if kind == "kfold":
            aggregate[name][row.threshold] = row
        else:
            folds[current_fold][name][row.threshold] = row

    if not aggregate and not folds:
        # Nothing recognisable was read: neither a k-FOLD aggregate nor a single
        # fold-aggregate block. A file with folds but no aggregate is a *truncated
        # run*, which must still parse -- the k-FOLD block is written last (line
        # 5921 of 6001 in the committed BiomedBERT file), so requiring it would
        # reject every in-progress run, i.e. exactly the file you want to read.
        # Structural completeness is validate()'s job; ``folds`` only gains an
        # entry when a fold-aggregate header is seen, so a per-case-only prefix
        # still lands here.
        raise PerformanceIndexError(
            "no k-FOLD aggregate and no fold-aggregate section found; file is "
            "not a PerformanceIndex.txt",
            path,
        )

    fold_times = [
        timing[key]
        for key in sorted(
            (k for k in timing if _FOLD_TIME_KEY.match(k)),
            key=lambda k: int(_FOLD_TIME_KEY.match(k).group(1)),
        )
    ]

    return PerformanceIndex(
        folds=folds,
        aggregate=aggregate,
        timing=timing,
        fold_times=fold_times,
        model=model,
        k_fold=k_fold if k_fold is not None else 0,
        per_case_blocks=per_case_blocks,
        path=path,
    )


def validate(
    index: PerformanceIndex,
    strategies: Iterable[str] = STRATEGIES,
    thresholds: Iterable[float] = THRESHOLDS,
    k_fold: int = 10,
) -> PerformanceIndex:
    """Raise unless ``index`` is a *complete* run; return it unchanged if so.

    Separate from :func:`parse` because a truncated file must still parse -- a
    run that died in fold 6 is exactly when you want to read what it wrote.
    Callers that need the full 6 x 5 x 10 grid ask for it here.
    """
    want_strategies = list(strategies)
    want_thresholds = sorted(thresholds)
    path = index.path

    if index.k_fold != k_fold:
        raise PerformanceIndexError(
            "file reports a %d-FOLD aggregate, expected %d"
            % (index.k_fold, k_fold),
            path,
        )

    missing = [s for s in want_strategies if s not in index.aggregate]
    if missing:
        raise PerformanceIndexError(
            "aggregate is missing strategies: %s" % ", ".join(missing), path
        )
    for name in want_strategies:
        found = sorted(index.aggregate[name])
        if found != want_thresholds:
            raise PerformanceIndexError(
                "aggregate / %s: expected thresholds %s, parsed %s"
                % (name, want_thresholds, found),
                path,
            )

    if sorted(index.folds) != list(range(k_fold)):
        raise PerformanceIndexError(
            "expected folds %s, parsed %s"
            % (list(range(k_fold)), sorted(index.folds)),
            path,
        )
    for fold in range(k_fold):
        per_fold = index.folds[fold]
        missing = [s for s in want_strategies if s not in per_fold]
        if missing:
            raise PerformanceIndexError(
                "FOLD %d is missing strategies: %s" % (fold, ", ".join(missing)),
                path,
            )
        for name in want_strategies:
            found = sorted(per_fold[name])
            if found != want_thresholds:
                raise PerformanceIndexError(
                    "FOLD %d / %s: expected thresholds %s, parsed %s"
                    % (fold, name, want_thresholds, found),
                    path,
                )

    return index

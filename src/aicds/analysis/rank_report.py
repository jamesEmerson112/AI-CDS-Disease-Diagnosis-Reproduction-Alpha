"""Accumulate per-case relevance vectors and emit ``RankMetrics.txt``.

Separate from ``rank_metrics`` on purpose: that module is pure (no I/O, no
config) and differentially validated, so nothing here may change what it
computes. All the aggregation, population and formatting decisions live here,
where they can be argued with.

WHY A SIBLING FILE. ``PerformanceIndex.txt`` is compared byte-for-byte by
``pytest -m golden`` and is the project's only regression net. Writing a new file
next to it cannot break that; adding a column to it would. See CLAUDE.md.

--------------------------------------------------------------------------------
THE TWO AGGREGATIONS, AND WHY BOTH ARE PRINTED
--------------------------------------------------------------------------------
The legacy 10-FOLD block is the **unweighted mean of the ten per-fold rates**,
not a pooled rate. ``cython_utils.py:733`` does ``values[R] += recall`` per fold
and ``:745`` divides by ``K_FOLD``, i.e. ``mean(TP_f / n_f)`` -- NOT
``sum(TP_f) / sum(n_f)``. Confirmed against all four committed drg runs: the
printed 10-FOLD R matches mean-of-folds exactly and pooled not at all.

The two differ here because **the grouped folds are uneven** -- 15, 13, 13, 13,
13, 13, 13, 12, 12, 12 -- since GroupKFold cannot split subject 41976's fifteen
admissions. Mean-of-folds therefore over-weights a 12-case fold relative to the
15-case one.

Measured difference on the drg runs: 0.0015 (baseline) to 0.0067
(Bio_ClinicalBERT). **It does not reorder the arms** -- verified with the
population held fixed. An earlier comparison appeared to reorder them, but that
was the N1 population trap: it compared all-cases mean-of-folds against
answered-only pooled, which is two changes at once.

``MEAN_OF_FOLDS`` is primary because it makes the cross-check against
``PerformanceIndex.txt`` exact. ``POOLED`` is printed beside it because the
difference is a real choice and a reader is entitled to see its size rather than
take our word that it is small.

--------------------------------------------------------------------------------
THE THREE POPULATIONS
--------------------------------------------------------------------------------
Every block uses ONE denominator for ALL FOUR metrics and states it. Never one
population per column -- that is N1 in ``rank_metrics``, and it is how the same
retrieval gets two different rankings.

  winnable   the 76 arm-invariant cases whose label is reachable at all.
             THE HEADLINE. Abstention scores 0.0 for every metric, precision
             included.
  all-cases  all 129. Abstention scores 0.0. Reproduces legacy R.
  answered   abstentions excluded entirely. Reproduces legacy P.

Cross-check, exact rather than approximate, and the reason the populations are
worth this much ceremony:

  mean-of-folds  Hit@K over all-cases  ==  printed legacy 10-FOLD R  (0.192308)
  pooled         Hit@K over all-cases  ==  sum(TP) / 129             (0.193798)

Both are "legacy recall". Naming which one is what makes the claim checkable.
"""

from __future__ import annotations

import os

from aicds.analysis.populations import winnable_by_fold
from aicds.analysis.rank_metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)
from aicds.config import LEGACY
from aicds.utils.Constants import K_FOLD

__all__ = ["RankAccumulator", "REPORT_KS", "binarise"]

# 1 and 5 are additions -- the pipeline's own range starts at 10, but the top of
# the list is where a rank-aware metric is most informative and it costs nothing
# to compute, since the vector is already in hand.
REPORT_KS = [1, 5, 10, 20, 30, 40, 50]

POPULATIONS = ["winnable", "all-cases", "answered"]

# The candidate list is capped upstream at TOP_K_UPPER_BOUND - TOP_K_INCR = 50,
# so no metric here can see past rank 50. Naming it MRR@50 rather than MRR is the
# difference between a bounded statement and an overclaim.
MRR_CAP = 50


def binarise(similarities):
    """Relevance vector from a graded similarity vector, in rank order.

    ONE rule for both graders, deliberately. Under ``drg-exact`` the values are
    already exactly 0.0 or 1.0, so this is the identity. Under ``cosine`` it
    reproduces threshold 1.0 -- which the 2026-08-06 four-arm run proved
    bit-identical to ``drg-exact`` across all 144 aggregate numbers, because
    cosine reaches exactly 1.0 only when two diagnosis embeddings are parallel,
    which (the dicts being keyed by text) means identical text.

    So there is no grader special-casing anywhere in this module, and the rank
    metrics are directly comparable to the threshold-1.0 column the README
    already reports.
    """
    return [1.0 if value >= 1.0 else 0.0 for value in similarities]


def _mean(values):
    """Mean, or ``None`` for an empty population.

    ``None`` rather than 0.0: a fold in which an arm answered nothing has no
    measurement, and printing 0.0 would assert it scored zero. Fold 7 has only
    3 winnable cases of 12, so small-n blocks are real here, not hypothetical.
    """
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value):
    """``repr``, never ``%.16f``.

    ``%.16f`` gives sixteen digits after the point, not seventeen significant
    digits, so it is NOT round-trip safe: measured over random ``a/b`` with
    ``b <= 129`` -- exactly this project's value space -- **25.1% of values do not
    survive the round trip**. ``0.24743589743589745`` prints as
    ``0.2474358974358974`` and reads back as a different float.

    That would matter twice over. The per-fold values feed a paired *t*-test, so
    perturbing their last bits perturbs the significance. And this file claims an
    exact identity against ``PerformanceIndex.txt``; a lossy format makes that
    claim unverifiable from the file it is printed in.

    ``repr`` is round-trip safe, and it is also what ``cython_utils`` emits --
    ``str(x) is repr(x)`` for floats on Python 3 -- so the two artifacts are
    textually comparable, not merely numerically close.
    """
    return "n/a" if value is None else repr(float(value))


class RankAccumulator:
    """Collects one relevance vector per test case, then writes the report.

    Usage from either arm::

        acc = RankAccumulator(config)
        ...                                     # per test case, per fold
        acc.add(fold, hadm_id, top_similarities_max)
        ...
        acc.write(output_dir)                   # once, at the end

    ``add`` takes the RAW similarity vector and binarises internally, so a caller
    cannot accidentally pass an already-thresholded list under one convention and
    a graded one under another.
    """

    def __init__(self, config=LEGACY):
        self.config = config
        self.by_fold = {}
        self.expected = {}
        self._winnable = {
            fold: winnable for fold, (winnable, _) in winnable_by_fold(config).items()
        }

    def begin_fold(self, fold, nrow):
        """Declare how many test cases this fold has, before any are added.

        This exists because of a real hazard. Both arms skip a test case with
        ``continue`` when its admission is missing (``bert_models.py:571-573``),
        and the accumulation happens *after* that skip -- so a skipped case would
        never be recorded, while legacy still counts it in ``nrow = len(x_test)``
        for recall. The all-cases denominators would then differ by the number of
        skips.

        Nothing is skipped today, so this is latent. Declaring the expected count
        turns the latent divergence into a loud one: ``fold_stats`` raises rather
        than quietly averaging over a smaller population.
        """
        self.expected[fold] = nrow
        self.by_fold.setdefault(fold, [])

    def add(self, fold, hadm_id, similarities):
        """Record one test case. ``similarities`` may be empty (an abstention)."""
        self.by_fold.setdefault(fold, []).append(
            (str(hadm_id), binarise(similarities))
        )

    def _check_complete(self, fold):
        expected = self.expected.get(fold)
        if expected is None:
            return
        actual = len(self.by_fold.get(fold, []))
        if actual != expected:
            raise ValueError(
                "fold %d: %d test cases recorded but %d expected. A case was "
                "skipped without being recorded as an abstention, so every "
                "all-cases denominator here would disagree with "
                "PerformanceIndex.txt's nrow. Record skipped cases with an "
                "empty similarity list rather than omitting them."
                % (fold, actual, expected)
            )

    # -- populations ------------------------------------------------------

    def _select(self, population, fold, rows):
        """Rows belonging to one population, and whether abstention counts.

        Returns ``(rows, drop_abstentions)``. For ``answered`` the abstaining
        cases are removed from the denominator; for the other two they stay and
        score 0.0 -- including for precision, which is the N1 fix.
        """
        if population == "winnable":
            allowed = self._winnable.get(fold, set())
            return [r for r in rows if r[0] in allowed], False
        if population == "all-cases":
            return list(rows), False
        if population == "answered":
            return [r for r in rows if r[1]], True
        raise ValueError("unknown population %r" % population)

    # -- per-fold ---------------------------------------------------------

    def fold_stats(self, fold, population):
        """One fold, one population: n, coverage, MRR and the per-K metrics."""
        self._check_complete(fold)
        rows, drop = self._select(population, fold, self.by_fold.get(fold, []))
        return self._stats_from_rows(rows, drop)

    def _stats_from_rows(self, rows, drop):
        """Compute the metric block for an already-selected set of rows.

        Shared by ``fold_stats`` and the pooled aggregate so the two cannot
        diverge. An earlier version built a throwaway accumulator via
        ``__new__`` and re-ran the population filter on it -- which skipped
        ``__init__``, so the instance had no ``expected`` map and the completeness
        check crashed; worse, re-filtering an already-filtered set risked widening
        ``winnable`` back to all cases. Computing directly from rows removes both
        hazards.
        """
        n = len(rows)
        if not n:
            return {"n": 0, "answered": 0, "coverage": None, "mrr": None, "per_k": {}}

        answered = sum(1 for _, rel in rows if rel)
        stats = {
            "n": n,
            "answered": answered,
            "coverage": answered / n,
            "mrr": _mean([reciprocal_rank(rel) for _, rel in rows]),
            "per_k": {},
        }
        for k in REPORT_KS:
            hits = [hit_at_k(rel, k) for _, rel in rows]
            ndcgs = [ndcg_at_k(rel, k) for _, rel in rows]
            # precision_at_k returns None on an empty list. In the two
            # abstention-inclusive populations that None is replaced by 0.0, so
            # all four metrics share the denominator; in `answered` no empty list
            # is present, so nothing is replaced.
            precisions = []
            for _, rel in rows:
                p = precision_at_k(rel, k)
                precisions.append(p if p is not None else (None if drop else 0.0))
            stats["per_k"][k] = {
                "hit": _mean(hits),
                "precision": _mean(precisions),
                "ndcg": _mean(ndcgs),
            }
        return stats

    # -- aggregate --------------------------------------------------------

    def aggregate(self, population, pooled=False):
        """Across all folds. ``pooled`` weights by case, else by fold.

        Mean-of-folds is the default because it is what ``PerformenceIndex.txt``
        prints, which is what makes the cross-check exact. Pooled is offered so
        the size of that choice is visible.
        """
        folds = sorted(self.by_fold)
        if pooled:
            # Select per fold -- `winnable` membership is fold-specific, since a
            # label reachable in one fold's training pool need not be in
            # another's -- then compute over the concatenation. Every test case
            # appears in exactly one fold, so there is no double-counting.
            rows, drop = [], False
            for fold in folds:
                self._check_complete(fold)
                selected, drop = self._select(
                    population, fold, self.by_fold.get(fold, [])
                )
                rows.extend(selected)
            stats = self._stats_from_rows(rows, drop)
            stats["empty_folds"] = sum(
                1 for f in folds if not self._select(population, f, self.by_fold.get(f, []))[0]
            )
            return stats

        per_fold = [self.fold_stats(f, population) for f in folds]
        if not any(s["n"] for s in per_fold):
            return {"n": 0, "answered": 0, "coverage": None, "mrr": None,
                    "per_k": {}, "empty_folds": len(per_fold)}

        # DIVIDE BY K_FOLD, ALWAYS -- not by the number of non-empty folds.
        #
        # Legacy does exactly this: cython_utils.py:744-746 divides the summed
        # per-fold values by K_FOLD unconditionally, and a fold with tp+fp == 0
        # contributes precision = 0 (:683) rather than being skipped. Dropping
        # empty folds instead would compute sum/9 where legacy computes sum/10,
        # and the answered-population aggregate would stop reproducing legacy P.
        #
        # This is latent rather than active today -- every fold has at least 3
        # winnable cases and all four arms answer on most -- which is precisely
        # why it is worth fixing now. It would otherwise surface as a
        # fourth-decimal disagreement after some unrelated data edit and read as
        # a metric bug.
        empty = sum(1 for s in per_fold if not s["n"])

        def across(pick):
            """Sum the per-fold values, treating an empty fold as 0.0, over K_FOLD."""
            total = 0.0
            for stats in per_fold:
                if not stats["n"]:
                    continue                      # contributes 0.0
                value = pick(stats)
                total += 0.0 if value is None else value
            return total / K_FOLD

        out = {
            "n": sum(s["n"] for s in per_fold),
            "answered": sum(s["answered"] for s in per_fold),
            "coverage": across(lambda s: s["coverage"]),
            "mrr": across(lambda s: s["mrr"]),
            "per_k": {},
            # Surfaced in the block heading: an aggregate over fewer than K_FOLD
            # populated folds is still divided by K_FOLD, so a reader must know.
            "empty_folds": empty,
        }
        for k in REPORT_KS:
            out["per_k"][k] = {
                key: across(lambda s, key=key: s["per_k"][k][key])
                for key in ("hit", "precision", "ndcg")
            }
        return out

    # -- output -----------------------------------------------------------

    def write(self, output_dir, filename="RankMetrics.txt"):
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as out:
            self._write_header(out)
            for fold in sorted(self.by_fold):
                self._write_block(out, "FOLD %d" % fold,
                                  lambda p, f=fold: self.fold_stats(f, p))
            self._write_block(out, "%d-FOLD  (mean of per-fold rates)" % K_FOLD,
                              lambda p: self.aggregate(p, pooled=False))
            self._write_block(out, "%d-FOLD  (pooled over cases)" % K_FOLD,
                              lambda p: self.aggregate(p, pooled=True))
        return path

    def _write_header(self, out):
        out.write("=" * 78 + "\n")
        out.write("RANK METRICS\n")
        out.write("=" * 78 + "\n")
        out.write("pipeline      %s   folds: %s   grader: %s\n"
                  % (self.config.preprocess_version, self.config.fold_dir,
                     self.config.grader))
        out.write("relevance     similarity >= 1.0, in rank order\n")
        out.write("MRR           reported as MRR@%d -- the candidate list is capped\n"
                  % MRR_CAP)
        out.write("              upstream at %d, so a hit at rank %d+ scores 0\n"
                  % (MRR_CAP, MRR_CAP + 1))
        out.write("\n")
        out.write("READ THIS BEFORE COMPARING ANY TWO CELLS:\n")
        out.write("  Every block below uses ONE population for ALL FOUR metrics, named in\n")
        out.write("  its heading. Cells from different populations are NOT comparable --\n")
        out.write("  the same retrieval scores 0.2551 answered-only and 0.1938 all-cases.\n")
        out.write("  Compare like with like or the ranking is an artifact of the choice.\n")
        out.write("\n")
        out.write("  P@K is a WITHIN-ARM diagnostic only. An arm that truncates its own\n")
        out.write("  candidate list raises reported P@K 3.53x while delivering strictly\n")
        out.write("  less; an abstain-unless-confident gate reaches 37.9x. Coverage is\n")
        out.write("  printed beside it for exactly this reason.\n")
        out.write("\n")
        out.write("  Hit@K and nDCG@K both RISE with K by construction. That is the\n")
        out.write("  artifact, preserved so it stays visible. Only P@K and MRR break it.\n")
        out.write("\n")

    def _write_block(self, out, title, stats_for):
        out.write("=" * 78 + "\n")
        out.write("%s\n" % title)
        out.write("=" * 78 + "\n")
        for population in POPULATIONS:
            stats = stats_for(population)
            out.write("\npopulation=%s  n=%d  answered=%d  coverage=%s\n"
                      % (population, stats["n"], stats["answered"],
                         _fmt(stats["coverage"])))
            out.write("  MRR@%d\t%s\n" % (MRR_CAP, _fmt(stats["mrr"])))
            if not stats["per_k"]:
                out.write("  (no cases in this population)\n")
                continue
            out.write("  K\tHit@K\tP@K\tnDCG@K\n")
            for k in REPORT_KS:
                row = stats["per_k"][k]
                out.write("  %d\t%s\t%s\t%s\n"
                          % (k, _fmt(row["hit"]), _fmt(row["precision"]),
                             _fmt(row["ndcg"])))
        out.write("\n")

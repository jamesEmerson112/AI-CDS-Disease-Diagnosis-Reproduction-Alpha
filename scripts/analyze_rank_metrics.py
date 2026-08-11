#!/usr/bin/env python3
"""Parse RankMetrics.txt across arms and run the paired t-test on per-fold MRR.

    python scripts/analyze_rank_metrics.py results_p5
    python scripts/analyze_rank_metrics.py results_p5 --population answered

WHY PAIRED. All arms are evaluated on the *same* ten folds, so the per-fold
values are matched pairs and an unpaired test would throw away the pairing and
badly understate power. With 9 degrees of freedom, |t| > 2.262 is p < 0.05.

WHICH POPULATION, AND WHY IT DECIDES THE ANSWER. MRR removed the K knob but it
is NOT abstention-neutral: an abstained case scores RR = 0, while an arm that
always offers 50 candidates has some chance of a hit. So on `winnable` and
`all-cases`, MRR still rewards willingness to guess -- the same bias that made
TOP-K meaningless, in a quieter form.

The baseline abstains; all three BERT arms never do. So:

  winnable / all-cases  PENALISE abstention -- declining scores the same as
                        answering wrongly
  answered              EXCLUDE abstention -- but those cases are self-selected,
                        the arm chose which to answer, and n is not matched
                        (baseline 98 vs BERT 129)

**Neither population is neutral, and there is no third one that is.** Abstention
is a property of the arm, not of the metric, so every convention takes a side.
Measured 2026-08-06: the ranking FLIPS between them and every sign inverts --
baseline last on winnable/all-cases, first on answered. All three are therefore
printed, because the disagreement between them IS the finding, and a reader shown
only one of them is being misled. See docs/findings/13-rank-aware-metrics.md.

Do NOT use the ratio MRR_all-cases / coverage to bound the self-selection effect.
It is algebraically identical to MRR_answered (abstentions contribute 0 to the
numerator, and coverage is exactly the denominator ratio), so it returns ~1.0 by
construction and looks like evidence while carrying none. The real test needs the
BERT arms restricted to the baseline's answered cases, which requires per-case
relevance vectors that RankMetrics.txt does not carry.
RankCases.txt (P40) carries them, and THE SELF-SELECTION TEST below is that
legitimate replacement: each arm's MRR recomputed over exactly the baseline's
answered cases, printed beside its MRR over all of its own.
"""

from __future__ import annotations

import argparse
import collections
import math
import re
import sys

from aicds import runs
from aicds.analysis.rank_metrics import reciprocal_rank
from aicds.analysis.rank_report import RANK_CASES_FILENAME

POPULATIONS = ["winnable", "all-cases", "answered"]
ARM_ORDER = ["baseline", "bio_clinical_bert", "biomedbert", "bluebert"]
LABELS = {
    "baseline": "BioSentVec",
    "bio_clinical_bert": "Bio_ClinicalBERT",
    "biomedbert": "BiomedBERT",
    "bluebert": "BlueBERT",
}
# Two-tailed critical values for a paired t-test on 10 folds (9 df).
T_CRIT = {0.05: 2.262, 0.01: 3.250, 0.001: 4.781}

BLOCK = re.compile(r"^(FOLD (\d+)|(\d+)-FOLD\s+\((.+?)\))\s*$")
POP = re.compile(r"^population=(\S+)\s+n=(\d+)\s+answered=(\d+)\s+coverage=(\S+)")
MRR = re.compile(r"^\s+MRR@(\d+)\s+(\S+)")
KROW = re.compile(r"^\s+(\d+)\t(\S+)\t(\S+)\t(\S+)")


def parse_rank_metrics(path):
    """-> {block_name: {population: {n, answered, coverage, mrr, per_k}}}."""
    blocks, block, population = {}, None, None
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.rstrip("\n")
        m = BLOCK.match(line)
        if m:
            # Collapse internal whitespace: the writer emits "10-FOLD  (mean of
            # per-fold rates)" with two spaces, and matching that by eye is how a
            # lookup silently returns nothing and every aggregate row prints
            # "(missing)" while the per-fold parse quietly succeeds.
            block = " ".join(m.group(1).split())
            blocks.setdefault(block, {})
            population = None
            continue
        if block is None:
            continue
        m = POP.match(line)
        if m:
            population = m.group(1)
            blocks[block][population] = {
                "n": int(m.group(2)),
                "answered": int(m.group(3)),
                "coverage": _num(m.group(4)),
                "mrr": None,
                "per_k": {},
            }
            continue
        if population is None:
            continue
        m = MRR.match(line)
        if m:
            blocks[block][population]["mrr"] = _num(m.group(2))
            continue
        m = KROW.match(line)
        if m:
            blocks[block][population]["per_k"][int(m.group(1))] = {
                "hit": _num(m.group(2)),
                "precision": _num(m.group(3)),
                "ndcg": _num(m.group(4)),
            }
    return blocks


def _num(token):
    return None if token == "n/a" else float(token)


# --------------------------------------------------------------------------
# RankCases.txt: the per-case record (P40) and the self-selection test
# --------------------------------------------------------------------------

#: One row of RankCases.txt. ``rank`` is 1-based, or None for "nothing relevant"
#: -- which covers both an abstention and an answer that missed; ``candidates``
#: is what tells those two apart.
Case = collections.namedtuple("Case", "fold hadm status candidates rank")


def parse_rank_cases(path):
    """-> ``[Case, ...]`` in file order.

    Every header line starts with ``#``, so the rule is "keep what does not" and
    split on tabs. A malformed row RAISES rather than being skipped: this file is
    a per-case census whose whole value is that its population is exactly the
    run's, and quietly dropping a row would shrink a denominator by one and
    change an MRR in the fourth decimal with nothing to show for it.
    """
    cases = []
    with open(path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                raise ValueError(
                    "%s:%d: expected 5 tab-separated fields "
                    "(fold, hadm_id, status, candidates, first_relevant_rank), "
                    "got %d: %r" % (path, number, len(fields), line)
                )
            fold, hadm, status, candidates, rank = fields
            cases.append(Case(
                fold=int(fold),
                hadm=hadm,
                status=status,
                candidates=int(candidates),
                rank=None if rank == "none" else int(rank),
            ))
    return cases


def _relevance(case):
    """A relevance vector with the same first-relevant position as the run's.

    ``reciprocal_rank`` depends on NOTHING but the index of the first relevant
    entry -- verified bit-exact over 30540 append/splice checks in
    ``rank_metrics``' own validation -- so (rank-1) zeros followed by a 1.0 gives
    the identical RR to the original candidate vector. That is why this script
    can recompute MRR from a rank column without the vectors themselves, and why
    it does not open-code ``1 / (1 + i)``: the arithmetic stays in the one pure,
    differentially validated module that owns it.
    """
    if case.rank is None:
        return [0.0] * case.candidates
    return [0.0] * (case.rank - 1) + [1.0]


def mrr(cases):
    """Mean reciprocal rank over cases, POOLED per case. ``None`` if empty.

    Pooled, not the mean-of-folds that RankMetrics.txt prints as primary, and the
    difference is deliberate: this section compares arms on a *fixed case set*,
    so weighting a 12-case fold like a 15-case one would let fold composition
    move the very quantity under test. The two aggregations differ by 0.0015 to
    0.0067 on the measured runs and do not reorder the arms (rank_report's module
    docstring), but they are different numbers and the heading says which.
    """
    if not cases:
        return None
    return sum(reciprocal_rank(_relevance(c)) for c in cases) / len(cases)


def paired_t(a, b):
    """Paired t on matched per-fold values. Returns (t, mean_diff, n) or None."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    diffs = [x - y for x, y in pairs]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0:
        # Identical on every fold. A t of 0 is the honest report; inf would be a
        # lie in the other direction.
        return (0.0, mean, n)
    return (mean / math.sqrt(var / n), mean, n)


def significance(t):
    for alpha in (0.001, 0.01, 0.05):
        if abs(t) > T_CRIT[alpha]:
            return "p < %g" % alpha
    return "not significant"


def discover(results_dir):
    """-> ``({arm: RankMetrics.txt}, {arm: RankCases.txt})`` for the newest runs.

    A *run* is a directory with a ``PerformanceIndex.txt``; that is
    ``aicds.runs.discover``'s one rule and this script does not get its own. The
    consequence is deliberate: the arm chosen here is the newest run, not the
    newest run that happens to carry rank metrics. If that run predates P5 it is
    reported as missing and skipped, rather than being silently replaced by an
    older run whose numbers came from different code -- which is exactly the
    substitution the old private glob would have made, without saying so.

    The two files are looked up INDEPENDENTLY. A pre-P40 run still has its
    RankMetrics.txt analysed in full; it only drops out of the self-selection
    section, which is the one thing that genuinely cannot be computed without a
    per-case record. Coupling them would silently retire a run's aggregates over
    a missing sibling.
    """
    found, cases = {}, {}
    for run in runs.discover(results_dir):
        rank_cases = run.file(RANK_CASES_FILENAME)
        if rank_cases is None:
            print("[WARN] %s: no RankCases.txt (pre-P40 run?)" % run.key)
        else:
            cases[run.key] = rank_cases
        rank_metrics = run.file("RankMetrics.txt")
        if rank_metrics is None:
            print("[WARN] %s: no RankMetrics.txt (pre-P5 run?)" % run.key)
            continue
        found[run.key] = rank_metrics
    return found, cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir")
    parser.add_argument("--population", default=None, choices=POPULATIONS,
                        help="default: report all three")
    args = parser.parse_args(argv)

    # Named `found`, not `runs`: `runs` is now the aicds.runs module.
    found, case_files = discover(args.results_dir)
    if not found:
        raise SystemExit("[ERROR] no <arm>/<timestamp>/RankMetrics.txt under %s"
                         % args.results_dir)

    parsed = {arm: parse_rank_metrics(path) for arm, path in found.items()}
    arms = [a for a in ARM_ORDER if a in parsed] + \
           [a for a in sorted(parsed) if a not in ARM_ORDER]
    print("arms found: %s\n" % ", ".join(arms))

    populations = [args.population] if args.population else POPULATIONS
    for population in populations:
        _report(parsed, arms, population)

    cases = {arm: parse_rank_cases(path) for arm, path in case_files.items()}
    _self_selection(cases)


def _report(parsed, arms, population):
    print("=" * 78)
    print("POPULATION: %s" % population)
    if population == "answered":
        print("  Abstentions EXCLUDED. May flatter the abstaining arm: these cases are")
        print("  self-selected, and n is not matched (baseline 98 vs BERT 129).")
    else:
        print("  Abstentions score 0 -> PENALISED, the same as answering wrongly.")
        print("  The baseline abstains; the BERT arms never do.")
    print("  NEITHER population is neutral. Compare against the other two before")
    print("  quoting any ranking from this block.")
    print("=" * 78)

    agg = "10-FOLD (mean of per-fold rates)"
    header = "%-18s %-10s %-20s %-10s" % ("arm", "coverage", "MRR@50", "n")
    print(header)
    print("-" * len(header))
    summary = {}
    for arm in arms:
        block = parsed[arm].get(agg, {}).get(population)
        if not block:
            print("%-18s (missing)" % LABELS.get(arm, arm))
            continue
        summary[arm] = block
        print("%-18s %-10.4f %-20.6f %-10d"
              % (LABELS.get(arm, arm), block["coverage"], block["mrr"], block["n"]))

    per_fold = {}
    for arm in arms:
        values = []
        for fold in range(10):
            block = parsed[arm].get("FOLD %d" % fold, {}).get(population)
            values.append(block["mrr"] if block else None)
        per_fold[arm] = values

    print("\nPAIRED t-TEST on per-fold MRR@50   (9 df: |t| > 2.262 is p < 0.05)")
    print("%-38s %-11s %-9s %s" % ("pair", "mean diff", "t", "verdict"))
    print("-" * 78)
    any_sig = False
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            result = paired_t(per_fold[a], per_fold[b])
            if result is None:
                continue
            t, mean, n = result
            verdict = significance(t)
            any_sig |= verdict != "not significant"
            print("%-38s %+11.6f %-9.3f %s"
                  % ("%s vs %s" % (LABELS.get(a, a), LABELS.get(b, b)),
                     mean, t, verdict))
    print("-" * 78)
    if any_sig:
        print("AT LEAST ONE PAIR SEPARATES on this population. Before reporting it as")
        print("an encoder result, check whether it separates on the OTHER TWO as well.")
        print("The abstention asymmetry alone reorders all four arms, so a result that")
        print("holds on only one population is a statement about the convention.")
    else:
        print("NO PAIR SEPARATES. With the threshold knob gone (drg-exact) and the K")
        print("knob gone (MRR), the encoders remain statistically indistinguishable.")
    print()


def _self_selection(cases, baseline_arm="baseline"):
    """THE SELF-SELECTION TEST (P40): each arm's MRR on the baseline's own cases.

    Finding 13 left exactly one question open. The baseline declines to answer
    31 of 129 cases and the three BERT arms decline none, so `answered` flatters
    the baseline *if* the cases it chose are the easy ones -- and no aggregate
    can tell you whether they are, because the two arms are being scored on
    different case sets. The fix is not a correction factor, it is a matched
    comparison: score every arm on the 98 cases the baseline answered, and read
    the restricted column against the all-cases one.

    Read the delta, not the restricted number alone. A BERT arm scoring HIGHER
    restricted than overall says the baseline's answered set is genuinely easier
    for everybody -- self-selection is real and `answered` overstates the
    baseline. A delta near zero says the set is unremarkable and the abstention
    asymmetry is about willingness, not difficulty. Both outcomes are reportable;
    neither is the outcome this project is hoping for.

    What this still does NOT do: it does not make abstention neutral. The
    baseline's own row is a restriction to its own answered set, i.e. its
    `answered` population by construction -- printed as the anchor the other rows
    are read against, not as a fifth comparison.
    """
    print("=" * 78)
    print("THE SELF-SELECTION TEST   (P40; per-case, from RankCases.txt)")
    print("=" * 78)

    if not cases:
        print("  No RankCases.txt found for any arm -- every run here predates P40.")
        print("  Re-run the arms to get per-case records; the aggregates above are")
        print("  unaffected.\n")
        return

    baseline = cases.get(baseline_arm)
    if baseline is None:
        print("  No RankCases.txt for the %r arm, so there is no answered set to" % baseline_arm)
        print("  restrict the others to. Arms with per-case records: %s"
              % ", ".join(sorted(cases)))
        print("  Skipped -- a restriction needs BOTH sides.\n")
        return

    # (fold, hadm) rather than hadm alone. A HADM_ID appears as a test case in
    # exactly one fold today, so the pair is redundant -- and it stays correct if
    # that ever stops being true, whereas a bare ID would silently union folds.
    answered = {(c.fold, c.hadm) for c in baseline if c.status == "answered"}
    print("  restriction   the %d cases %s ANSWERED, of %d it was scored on"
          % (len(answered), LABELS.get(baseline_arm, baseline_arm), len(baseline)))
    print("  MRR           pooled per case (NOT the mean-of-folds primary in")
    print("                RankMetrics.txt) -- a fixed case set is the point here")
    print()

    ordered = [a for a in ARM_ORDER if a in cases] + \
              [a for a in sorted(cases) if a not in ARM_ORDER]
    header = "%-18s %-8s %-14s %-16s %-11s" % (
        "arm", "n(all)", "MRR(all)", "MRR(restricted)", "delta")
    print(header)
    print("-" * len(header))
    for arm in ordered:
        rows = cases[arm]
        restricted = [c for c in rows if (c.fold, c.hadm) in answered]
        all_mrr = mrr(rows)
        restricted_mrr = mrr(restricted)
        label = LABELS.get(arm, arm)
        if arm == baseline_arm:
            label += " *"
        if all_mrr is None or restricted_mrr is None:
            print("%-18s %-8d (no cases in one of the two sets)" % (label, len(rows)))
            continue
        print("%-18s %-8d %-14.6f %-16.6f %+11.6f"
              % (label, len(rows), all_mrr, restricted_mrr,
                 restricted_mrr - all_mrr))
        if len(restricted) != len(answered):
            # Loud, because it means the arms were not evaluated on the same
            # admissions -- a different fold split, or a run from another
            # pipeline harvested into this tree. Every number in the row above is
            # then a comparison between two different experiments.
            print("%-18s [WARN] only %d of the %d restricted cases are present in "
                  "this arm's record" % ("", len(restricted), len(answered)))
    print("-" * len(header))
    print("  * the baseline's restricted set IS its own answered set, so its two")
    print("    columns are the all-cases and answered populations of the blocks")
    print("    above -- the anchor, not a fifth comparison.")
    print("  Do NOT convert any of this into MRR_all-cases / coverage. That ratio")
    print("  is algebraically identical to MRR_answered and carries no information;")
    print("  the restriction above is what replaces it. See this file's docstring.")
    print()


if __name__ == "__main__":
    sys.exit(main())

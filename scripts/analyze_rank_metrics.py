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
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

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
    runs = {}
    for arm in sorted(os.listdir(results_dir)):
        hits = glob.glob(os.path.join(results_dir, arm, "*", "RankMetrics.txt"))
        if not hits:
            continue
        hits.sort(key=os.path.getmtime)
        runs[arm] = hits[-1]
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir")
    parser.add_argument("--population", default=None, choices=POPULATIONS,
                        help="default: report all three")
    args = parser.parse_args(argv)

    runs = discover(args.results_dir)
    if not runs:
        raise SystemExit("[ERROR] no <arm>/<timestamp>/RankMetrics.txt under %s"
                         % args.results_dir)

    parsed = {arm: parse_rank_metrics(path) for arm, path in runs.items()}
    arms = [a for a in ARM_ORDER if a in parsed] + \
           [a for a in sorted(parsed) if a not in ARM_ORDER]
    print("arms found: %s\n" % ", ".join(arms))

    populations = [args.population] if args.population else POPULATIONS
    for population in populations:
        _report(parsed, arms, population)


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


if __name__ == "__main__":
    sys.exit(main())

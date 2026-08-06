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

  winnable / all-cases  mixes ranking quality with willingness to answer
  answered              isolates ranking quality -- USE THIS for cross-arm claims

Both are printed, because the disagreement between them IS the finding, and a
reader who only sees the flattering one is being misled.
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
            block = m.group(1).strip()
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
        print("  Abstentions excluded -> ranking quality ISOLATED from willingness")
        print("  to answer. THIS is the population for cross-arm claims.")
    else:
        print("  Abstentions score 0 -> this mixes ranking quality with willingness")
        print("  to answer. The baseline abstains; the BERT arms never do.")
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
        print("an encoder result, check the confounds -- on winnable/all-cases the")
        print("most likely explanation is the abstention asymmetry, not the encoder.")
    else:
        print("NO PAIR SEPARATES. With the threshold knob gone (drg-exact) and the K")
        print("knob gone (MRR), the encoders remain statistically indistinguishable.")
    print()


if __name__ == "__main__":
    sys.exit(main())

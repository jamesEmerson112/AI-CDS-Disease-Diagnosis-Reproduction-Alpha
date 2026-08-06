"""P5 rank-aware retrieval metrics: Hit@K, Precision@K, MRR, nDCG@K.

**This file was not hand-written. It is the validated merge of three
independently authored implementations** (impl A's bodies plus impl C's ``k``
coercion), and the bodies below should be treated as pinned. Differentially
tested: bit-identical to all three on the whole guaranteed domain (every binary
relevance vector of length 0..12 x 15 values of k -- exhaustive, so a proof
rather than a sample -- plus 174k sampled cases up to length 60); differs from
impl B off the binary domain, where B is wrong; differs from impl C only in
rejecting inputs C silently accepts. The merge itself was then re-validated at
1,130,358 bit-exact comparisons against all three.

``MRR`` is the reason this module exists. It is the only metric here with no K,
and under the ``drg-exact`` grader there is no threshold either -- so it is the
one number this project can report without an experimenter-chosen knob in it.
Every other function here is context or diagnostic; read N3 and N4 before
putting any of them in a cross-arm table.

Background: docs/findings/12-drg-grader.md (the encoder-independent grader that
makes the relevance signal binary) and docs/plans/metric-redesign.md (which
carries the corrected version of the claims refuted in N4 below).

--------------------------------------------------------------------------------
WHAT THESE FUNCTIONS DO NOT DO -- read this before averaging anything
--------------------------------------------------------------------------------
These are per-case functions. Every question about abstention is a question about
the CALLER's aggregation, and the caller gets it wrong by default. Three facts,
all measured:

(N1) The four metrics do not agree on the population. precision_at_k returns None
     on abstention (dropped from the average); hit_at_k / ndcg_at_k /
     reciprocal_rank return 0.0 (kept, scored as a failure). The baseline arm
     abstains on 31 of 129 cases and the three BERT arms on none, so a table with
     these four columns has an ANSWERED-ONLY precision cell next to ALL-CASES hit
     / nDCG / MRR cells, in the same row, unlabelled. Measured: baseline hit@10 is
     0.2551 answered-only against 0.1938 all-cases, and BiomedBERT is 0.2016 in
     both -- so the two populations put the arms in DIFFERENT orders. Fix: fix the
     population once (the 76 arm-invariant R>0 cases), print coverage separately,
     and label every cell.

(N2) hit_at_k([], k) == 0.0 does NOT reproduce the legacy metric, contrary to what
     it looks like. The legacy call site (cython_utils.py:299-303, and
     bert_models.py:653-661 identically) increments NEITHER TP nor FP on an empty
     candidate list, so the legacy pipeline already EXCLUDES abstentions from the
     confusion matrix. That is why the baseline's rows read TP+FP=9.8 against a
     mean fold size of 12.9 with PR=0.7558, while every BERT row reads TP+FP=12.9
     with PR=1.0. Averaging hit_at_k over all cases reproduces legacy RECALL
     (verified exactly: 25/129 = 0.193798); averaging over answered cases
     reproduces legacy PRECISION (25/98 = 0.255102). Neither alone is "the legacy
     number", and 0.0 has erased which cases abstained, so PR cannot be recovered
     downstream. Report all three.

(N3) The min(k, len) denominator is not a defence of abstention -- it is a
     separate, unbounded lever. On the measured Bio_ClinicalBERT list lengths only
     3 of 129 lists are shorter than 10, so at K=10 min(k,len) moves P@10 by 1.26%
     versus a plain k. Meanwhile truncating a 50-candidate list to 1 -- strictly
     less delivered, hit@10 falling 0.2636 -> 0.0930 -- RAISES reported P@10 from
     0.0264 to 0.0930, a 3.5x inflation, and an oracle abstain-unless-rank-1 gate
     reports P@10 = 1.0000 while delivering a WORSE hit@10 than the guesser. A
     non-oracle confidence gate reaches +237% reported P@10 at -29% true hit@10.
     Keep min(k, len) only because the spec mandates it; treat precision_at_k as a
     per-arm diagnostic, never a cross-arm headline, and always print coverage
     beside it.

(N4) nDCG@k is NOT immune to expanding k, despite the original rationale in
     docs/plans/metric-redesign.md (corrected there 2026-08-06, under "C.
     Rank-aware retrieval metrics"). IDCG@k sums min(R, k) slots, so once k >= R the
     IDCG is constant while DCG@k keeps growing: nDCG@k is then non-decreasing in
     k, exactly like the metric it replaces. R > 10 in 5 of 129 cases and R > 20 in
     none, so score(TOP-50) >= score(TOP-10) still holds by construction for nDCG
     on 124 of 129 cases at K=10 and on all 129 at K>=20. What nDCG fixes is the
     SIZE of the free gain (a rank-50 hit contributes 1/log2(51) = 0.176 instead of
     1.000), not its direction. Only precision_at_k (strictly decreasing in k when
     hits are front-loaded) and reciprocal_rank (no k at all) actually break the
     artifact.

--------------------------------------------------------------------------------
RESOLVED AMBIGUITIES -- every one that produced a real three-way disagreement
--------------------------------------------------------------------------------
R1. IDCG gain for the ideal ordering: the SORTED ACTUAL VALUES,
    sorted(relevance, reverse=True)[:min(n_relevant, k)], not a constant unit
    gain. Identical bit-for-bit on binary input (trailing zeros add exactly 0.0),
    so it cannot move a drg-exact number -- but the unit-gain reading returns
    nDCG = 2.0 for relevance=[2.0] and 0.5 for the already-ideal [0.5,0.5,0.5],
    i.e. it breaks both "nDCG <= 1" and "an ideal ordering scores 1.0".
    sklearn.metrics.ndcg_score agrees with the reading implemented here on every
    graded case tested. THIS IS THE ONE VALUE DISAGREEMENT FOUND; do not
    "simplify" the ideal back to a unit gain.
R2. Non-integer k RAISES TypeError (via operator.index), rather than truncating.
    Truncation is a silent wrong answer in the abstention-favouring direction:
    k=0.5 truncates to 0 and makes precision_at_k return None, i.e. it reports
    "no observation" for a positive k. operator.index (not isinstance(k, int)) is
    load-bearing: numpy.int64 is not an int subclass on Windows. An integral float
    (10.0) is accepted so JSON/numpy round-trips still work.
R3. relevance=None RAISES TypeError; it is not treated as an empty list. Mapping
    None to abstention hands a plumbing bug the abstention exemption (N1/N3), and
    abstention is the privileged outcome under this metric set.
R4. relevance must be a SEQUENCE (len + indexing). One-shot iterables raise. An
    implementation that accepts a generator scores the first metric correctly and
    then reports the SAME case as an abstention on the other three, silently
    splitting the population -- measured.
R5. precision_at_k with k <= 0 returns None (the denominator min(k, len) is <= 0,
    so there is no observation window). All three independent implementations
    chose this despite all three flagging it as their likeliest divergence; it
    needed no arbitration.
R6. "relevant" means rel > 0.0 -- one helper, shared by all four functions, so
    they cannot disagree. Indistinguishable from == 1.0 on the drg-exact domain.
R7. precision_at_k's numerator is a COUNT of relevant entries, not a sum of
    gains ("number of relevant"). Identical on binary input.
R8. precision_at_k's denominator is min(k, len(relevance)), not k -- spec-
    mandated. See N3 for why that is a hazard rather than a protection.
R9. hit_at_k([], k) is 0.0, not None -- it matches the legacy PREDICATE exactly
    (verified: 36828 comparisons against the AST-extracted
    cython_utils.containGreaterOrEqualsValue, zero mismatches, including k <= 0
    and k > len). See N2 for why that is not the same as matching the legacy
    METRIC.
R10. math.log2 with a fixed left-to-right sum. All three independent
    implementations chose the same, so no ULP disagreement was ever observable:
    the three are bit-identical, tolerance 0.0, over 376786 exhaustive
    comparisons. Against sklearn the last bit does move (max 8.9e-16), so an
    external cross-check needs ~1e-12; an internal regression test can and should
    assert bit equality.
"""

from __future__ import annotations

import math
import operator
from typing import Optional, Sequence

__all__ = ["hit_at_k", "precision_at_k", "reciprocal_rank", "ndcg_at_k"]


def _as_k(k) -> int:
    """Coerce k to int, rejecting fractional values (R2).

    operator.index accepts int and numpy integer types; numpy.int64 is not an int
    subclass on Windows, so isinstance(k, int) would spuriously reject it.
    """
    try:
        return operator.index(k)
    except TypeError:
        if isinstance(k, float) and k.is_integer():
            return int(k)
        raise


def _is_relevant(value) -> bool:
    """A candidate is relevant iff its gain is strictly positive (R6)."""
    return float(value) > 0.0


def hit_at_k(relevance: Sequence[float], k) -> float:
    """1.0 if any of the first k entries is relevant, else 0.0.

    The EXISTING metric, reproduced so the artifact stays comparable: a hit at
    rank 1 and a hit at rank 50 both score 1.0, and hit_at_k(rel, 50) >=
    hit_at_k(rel, 10) holds by construction. Empty list -> 0.0; k <= 0 -> 0.0;
    k past the end is truncated. See N2 before treating an average of this as
    "the legacy number".
    """
    kk = _as_k(k)
    if kk <= 0:
        return 0.0
    for i in range(min(kk, len(relevance))):
        if _is_relevant(relevance[i]):
            return 1.0
    return 0.0


def precision_at_k(relevance: Sequence[float], k) -> Optional[float]:
    """(number relevant in the first k) / min(k, len(relevance)), or None.

    None means "no observation" -- NOT a measured zero -- and is returned exactly
    when the denominator would be non-positive: an empty relevance list
    (abstention) or k <= 0. The caller excludes None from averages.

    Contrast [] (abstained -> None) with [0.0, 0.0] (answered, missed -> 0.0):
    that distinction is the entire point, and it is why None is not simply
    "return None when the score is zero".

    Denominator is min(k, len), not k (R8) -- but read N3: that is a hazard, not
    a protection.
    """
    kk = _as_k(k)
    if kk <= 0:
        return None
    denominator = min(kk, len(relevance))
    if denominator <= 0:
        return None
    hits = 0
    for i in range(denominator):
        if _is_relevant(relevance[i]):
            hits += 1
    return hits / denominator


def reciprocal_rank(relevance: Sequence[float]) -> float:
    """1 / (1 + index of the first relevant entry); 0.0 if none is relevant.

    Deliberately NOT k-limited. This is the only one of the four that is
    structurally immune to list length: appending candidates after the first
    relevant one cannot change it (verified bit-exact over 30540 append/splice
    checks). Rank 1 -> 1.0, rank 10 -> 0.1, rank 50 -> 0.02.
    """
    for i in range(len(relevance)):
        if _is_relevant(relevance[i]):
            return 1.0 / (1.0 + i)
    return 0.0


def ndcg_at_k(relevance: Sequence[float], k) -> float:
    """DCG@k / IDCG@k with a log2(i + 2) discount; 0.0 when IDCG == 0.

    DCG@k  = sum of rel_i / log2(i + 2) for i in [0, min(k, len(relevance)))
    IDCG@k = the same sum over the ideal ordering: the relevance values sorted
             descending, truncated to min(n_relevant, k) entries, where
             n_relevant is counted over the WHOLE list, not over the first k.

    Counting n_relevant over the whole list is what makes a relevant candidate
    stranded at rank 50 lower nDCG@10 instead of being invisible. It also means
    nDCG@k is not monotonic in list LENGTH: appending a relevant candidate at
    rank k+1 lowers nDCG@k. That is the standard convention and it is the spec,
    but it is the same shape of bias P5 removes, so do not report nDCG alone.

    The ideal uses the SORTED ACTUAL VALUES (R1). Do not replace it with a unit
    gain: identical on binary input, wrong off it.

    Returns 0.0 when k <= 0, when the list is empty, and when no candidate
    anywhere in the list is relevant. See N4: this is NOT immune to expanding k.
    """
    kk = _as_k(k)
    if kk <= 0:
        return 0.0

    limit = min(kk, len(relevance))
    dcg = 0.0
    for i in range(limit):
        dcg += float(relevance[i]) / math.log2(i + 2)

    n_relevant = 0
    for value in relevance:
        if _is_relevant(value):
            n_relevant += 1
    ideal_len = min(n_relevant, kk)
    if ideal_len <= 0:
        return 0.0

    idcg = 0.0
    ideal = sorted((float(v) for v in relevance), reverse=True)[:ideal_len]
    for i, value in enumerate(ideal):
        idcg += value / math.log2(i + 2)
    if idcg == 0.0:
        return 0.0

    return dcg / idcg

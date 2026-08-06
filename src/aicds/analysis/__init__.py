"""Analysis helpers that read the pipeline's outputs rather than producing them.

Kept out of ``aicds.utils`` on purpose. ``cython_utils`` owns everything both arms
must share in order to stay comparable -- preprocessing, fold loading, grading,
confusion-matrix arithmetic -- and every line in it is load-bearing on the
byte-exact golden. Rank-aware reporting is *additive*: it reads the relevance
vector the pipeline already computes and writes a separate file. Putting it here
keeps that distinction visible.
"""

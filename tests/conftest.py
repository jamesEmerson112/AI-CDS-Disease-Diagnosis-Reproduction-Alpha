"""Shared pytest configuration.

PYTHONHASHSEED matters here beyond the usual reasons. `bert_models` builds its
encode batch from `list(unique_symptoms)` over a set of strings, so hash order
determines the order text reaches the encoder. sentence-transformers then sorts
by length and composes batches, which changes padding and therefore the last
ulp of the embeddings. Pinning the seed keeps runs comparable.

Setting it here only affects hash randomization for code imported afterwards;
for a fully deterministic interpreter, export PYTHONHASHSEED=0 before running.
"""

import os

os.environ.setdefault("PYTHONHASHSEED", "0")
# Silence the tokenizers fork warning that clutters test output.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

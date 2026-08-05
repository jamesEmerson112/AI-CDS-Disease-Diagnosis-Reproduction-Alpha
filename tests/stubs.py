"""
Deterministic test doubles for the BERT pipeline.

StubEncoder stands in for ``sentence_transformers.SentenceTransformer`` so the
full 10-fold pipeline can run with no model download, no torch load and no
network. It implements exactly the three attributes
``src/models/bert_models.py`` touches: ``.device`` and ``.max_seq_length``
(printed only, never used in control flow) and ``.encode(...)``.

Note that the stub removes model-loading and encoding time, not the pipeline's
own cost: ``cython_utils.cosine_similarity`` is a pure-Python element-wise loop
run ~295k times per fold, so wall time scales with ``dim``. A measured full
10-fold ``run_analysis(model_id='1', encoder=StubEncoder())`` takes 1245s at the
default dim=768 and 128s at ``StubEncoder(dim=64)``. The cosine distribution
below is substantially unchanged by that reduction (mean 0.6145 at dim=64
against 0.6320 at dim=768), so dim=64 is the better choice for anything that
has to run often.

Determinism contract
--------------------
Every vector is derived solely from ``sha256(text)``. Therefore:

* the same string always produces the same vector, in-process, across
  processes, and regardless of ``PYTHONHASHSEED``;
* the vector does not depend on batch composition or on the position of the
  string within the list passed to ``encode``;
* distinct strings produce distinct vectors.

Similarity structure
--------------------
Independent Gaussian vectors in 768 dimensions are nearly orthogonal (pairwise
cosine ~0 +/- 0.04), which would make every pipeline score fall below the 0.5
pruning cut and the 0.6-1.0 thresholds -- a golden file of all zeros detects
nothing. Real biomedical embeddings sit at the opposite extreme (mean pairwise
cosine 0.72-0.93), which saturates every threshold instead.

So each vector is built as a hash-weighted mixture over a small fixed set of
"topic" directions plus per-string noise. The topic mixture creates genuine
structure -- strings sharing a dominant topic are similar, strings on different
topics are not -- and the resulting cosine distribution straddles the pipeline's
operating range instead of collapsing to either end of it.

Measured over the 573 unique preprocessed symptoms of the real vocabulary
(163,878 pairs) at the default dim=768::

    mean 0.6320   min 0.3122   max 0.8658   std 0.0763
    p5 0.4927  p25 0.5819  p50 0.6412  p75 0.6896  p95 0.7412
    94.1% of pairs above the 0.5 pruning cut, 68.4% above the 0.6 threshold

The 145 unique diagnosis descriptions behave the same (mean 0.6348, min 0.3692,
max 0.8309). Because no distinct pair reaches 0.9, the threshold-0.9 and
threshold-1.0 rows fire only on exact string matches -- which is also true of
the real encoders, and is what makes those rows discriminating rather than
saturated.
"""

import hashlib

import numpy as np

DEFAULT_DIM = 768

# Number of latent "topic" directions the mixture draws from. Small enough that
# the vocabulary reuses topics (producing genuinely similar pairs), large enough
# that it does not collapse into one blob.
N_TOPICS = 6

# Every vector carries a fixed component along one shared direction. This is a
# deliberate imitation of the anisotropy of real contextual embeddings, and it
# is what places the cosine distribution around the pipeline's 0.5 pruning cut
# and 0.6-1.0 thresholds instead of down near orthogonality.
BASE_SCALE = 0.84

# Relative weight of the per-string topic signal against per-string noise.
# Raising TOPIC_SCALE widens the spread; raising NOISE_SCALE narrows it toward
# the BASE_SCALE-implied floor.
TOPIC_SCALE = 1.0
NOISE_SCALE = 0.55

# Fixed seeds for the shared direction and the topic basis. Not derived from any
# input, so both are constants of this module.
_BASE_DIRECTION_SEED = 20220606
_TOPIC_BASIS_SEED = 20220607


def _seed_from_text(text):
    """Map a string to a stable 64-bit seed via sha256.

    Uses the cryptographic digest rather than ``hash()`` so the value is
    independent of PYTHONHASHSEED and identical across processes.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _topic_basis(dim):
    """Fixed orthonormal-ish basis of N_TOPICS directions in ``dim`` dims."""
    rng = np.random.default_rng(_TOPIC_BASIS_SEED)
    basis = rng.standard_normal((N_TOPICS, dim))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    return basis


def _base_direction(dim):
    """Fixed unit direction shared by every vector."""
    rng = np.random.default_rng(_BASE_DIRECTION_SEED)
    vector = rng.standard_normal(dim)
    return vector / np.linalg.norm(vector)


class StubEncoder:
    """Deterministic hash-based stand-in for a SentenceTransformer."""

    def __init__(self, dim=DEFAULT_DIM, device="cpu-stub", max_seq_length=512):
        self.dim = dim
        self.device = device
        self.max_seq_length = max_seq_length
        self._basis = _topic_basis(dim)
        self._base = _base_direction(dim)
        self._cache = {}

    def embed(self, text):
        """Return the vector for a single string (float32, length ``dim``)."""
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        rng = np.random.default_rng(_seed_from_text(text))

        # Sparse-ish mixture weights: cubing concentrates mass on one or two
        # topics, so strings land in recognisable neighbourhoods rather than
        # all averaging out to the basis centroid.
        weights = rng.random(N_TOPICS) ** 3
        weights /= weights.sum()

        vector = BASE_SCALE * self._base
        vector += TOPIC_SCALE * (weights @ self._basis)
        vector += NOISE_SCALE * rng.standard_normal(self.dim) / np.sqrt(self.dim)

        vector = vector.astype(np.float32)
        self._cache[text] = vector
        return vector

    def encode(
        self,
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
        **kwargs
    ):
        """Encode a list of strings.

        ``batch_size`` and ``show_progress_bar`` are accepted and ignored --
        encoding is per-string by construction, so batching cannot affect the
        result. ``convert_to_numpy`` is likewise ignored; a numpy array is
        always returned, which is what the caller's ``.shape[1]`` and ``zip``
        usage requires. ``normalize_embeddings`` is honoured because it changes
        the values.
        """
        if isinstance(texts, str):
            texts = [texts]

        matrix = np.vstack([self.embed(t) for t in texts]) if texts else np.zeros(
            (0, self.dim), dtype=np.float32
        )

        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms

        return matrix

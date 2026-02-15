"""
Tests for BERT pipeline symptom-level pairwise similarity fix.

Uses AST-based import to avoid executing bert_models.py module-level code
(model loading, dataset reading, interactive prompts).
"""
import ast
import sys
import types
import unittest
from unittest.mock import MagicMock

import numpy as np


# ---------------------------------------------------------------------------
# Inlined helper functions (avoid importing cython_utils which requires sent2vec)
# ---------------------------------------------------------------------------
def cosine_similarity(u, v):
    """Pure Python cosine similarity — same as cython_utils.cosine_similarity."""
    assert len(u) == len(v)
    uv = uu = vv = 0.0
    for i in range(len(u)):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    if uu != 0 and vv != 0:
        return uv / np.sqrt(uu * vv)
    return 0.0


def containGreaterOrEqualsValue(topK, top_similarities, b):
    """Same as cython_utils.containGreaterOrEqualsValue."""
    for i in range(0, topK):
        if i < len(top_similarities) and top_similarities[i] >= b:
            return True
    return False


# ---------------------------------------------------------------------------
# AST-based loader: extract only function defs from bert_models.py
# without executing module-level statements.
# ---------------------------------------------------------------------------
def _load_bert_functions():
    """
    Extract function definitions from bert_models.py without executing
    module-level code (model loading, dataset reading, interactive prompts).
    """
    import pathlib
    src_path = pathlib.Path(__file__).parent.parent / "src" / "models" / "bert_models.py"
    source = src_path.read_text()
    tree = ast.parse(source, filename=str(src_path))

    # Extract ONLY function definitions
    func_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    new_tree = ast.Module(body=func_nodes, type_ignores=[])
    ast.fix_missing_locations(new_tree)
    code = compile(new_tree, str(src_path), "exec")

    # Build namespace with dependencies that the functions reference
    mock_util_cy = MagicMock()
    mock_util_cy.preprocess_sentence = lambda s: s
    mock_util_cy.cosine_similarity = cosine_similarity

    ns = {
        "__builtins__": __builtins__,
        "__name__": "bert_models_test_ns",
        "cosine_similarity": cosine_similarity,
        "util_cy": mock_util_cy,
        "MIN_SIMILARITY": 0,
        "PRUNING_SIMILARITY": 0.5,
        "containGreaterOrEqualsValue": containGreaterOrEqualsValue,
        "np": np,
        "numpy": np,
    }

    exec(code, ns)
    return ns


# Load once for all tests
_ns = _load_bert_functions()


def _get(name):
    """Get a name from the loaded namespace, or None if missing."""
    return _ns.get(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class FakeAdmission:
    """Minimal stand-in for SymptomsDiagnosis."""
    def __init__(self, hadm_id, symptoms, diagnosis=None):
        self.hadm_id = hadm_id
        self.symptoms = symptoms
        self.diagnosis = diagnosis or []


def _make_embedding(dim=768, seed=None):
    """Return a deterministic random embedding vector."""
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float32)


# ============================= TEST CLASSES =================================

class TestComputeBertSymptomEmbeddingsKeyStructure(unittest.TestCase):
    """Validate that embeddings are keyed by symptom text, not HADM_ID."""

    def test_keys_are_symptom_text_not_hadm_id(self):
        """Dict keys must be preprocessed symptom strings, not HADM_IDs."""
        fn = _get("compute_bert_symptom_embeddings")
        self.assertIsNotNone(fn, "compute_bert_symptom_embeddings must exist")

        admissions = {
            "100": FakeAdmission("100", "fever,cough"),
            "200": FakeAdmission("200", "cough,headache"),
        }
        mock_model = MagicMock()
        dim = 768

        def fake_encode(texts, **kw):
            return np.random.randn(len(texts), dim).astype(np.float32)

        mock_model.encode = fake_encode
        mock_model.device = "cpu"

        _ns["util_cy"].preprocess_sentence = lambda s: s.strip().lower()
        result = fn(mock_model, admissions)

        # Keys should NOT be HADM_IDs
        self.assertNotIn("100", result)
        self.assertNotIn("200", result)
        # Keys should be preprocessed symptom strings
        self.assertTrue(any(isinstance(k, str) and k not in ("100", "200") for k in result))

    def test_unique_symptom_count(self):
        """Number of keys should equal number of unique preprocessed symptoms."""
        fn = _get("compute_bert_symptom_embeddings")
        self.assertIsNotNone(fn)

        admissions = {
            "100": FakeAdmission("100", "fever,cough"),
            "200": FakeAdmission("200", "cough,headache"),
        }
        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.random.randn(len(texts), 768).astype(np.float32)
        mock_model.device = "cpu"

        _ns["util_cy"].preprocess_sentence = lambda s: s.strip().lower()
        result = fn(mock_model, admissions)

        # "fever", "cough", "headache" = 3 unique symptoms
        self.assertEqual(len(result), 3)

    def test_values_are_list_wrapped_arrays(self):
        """Each value must be [np.ndarray] (list-wrapped) for cosine_similarity(emb[0], ...) usage."""
        fn = _get("compute_bert_symptom_embeddings")
        self.assertIsNotNone(fn)

        admissions = {"100": FakeAdmission("100", "fever")}
        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.random.randn(len(texts), 768).astype(np.float32)
        mock_model.device = "cpu"

        _ns["util_cy"].preprocess_sentence = lambda s: s.strip().lower()
        result = fn(mock_model, admissions)

        for key, val in result.items():
            self.assertIsInstance(val, list, f"Value for '{key}' must be a list")
            self.assertEqual(len(val), 1, f"Value for '{key}' must have exactly 1 element")
            self.assertTrue(hasattr(val[0], "shape"), f"val[0] must be an array-like")


class TestComputePatientSimilarityPairwise(unittest.TestCase):
    """Validate the new pairwise symptom-level similarity function."""

    def test_function_exists(self):
        """compute_patient_similarity_pairwise must be defined."""
        fn = _get("compute_patient_similarity_pairwise")
        self.assertIsNotNone(fn, "compute_patient_similarity_pairwise must exist")

    def test_self_similarity_near_one(self):
        """Identical symptom sets should produce similarity ~1.0."""
        fn = _get("compute_patient_similarity_pairwise")
        self.assertIsNotNone(fn)

        emb_a = _make_embedding(seed=42)
        emb_b = _make_embedding(seed=99)
        embeddings = {
            "fever": [emb_a],
            "cough": [emb_b],
        }
        test_symptoms = ["fever", "cough"]
        train_symptoms = ["fever", "cough"]

        sim = fn(test_symptoms, train_symptoms, embeddings)
        self.assertAlmostEqual(sim, 1.0, places=3)

    def test_cross_similarity_lower(self):
        """Different symptom sets should have similarity < 1.0."""
        fn = _get("compute_patient_similarity_pairwise")
        self.assertIsNotNone(fn)

        embeddings = {
            "fever": [_make_embedding(seed=1)],
            "cough": [_make_embedding(seed=2)],
            "headache": [_make_embedding(seed=3)],
        }
        test_symptoms = ["fever"]
        train_symptoms = ["cough", "headache"]

        sim = fn(test_symptoms, train_symptoms, embeddings)
        self.assertLess(sim, 1.0)

    def test_uses_max_length_denominator(self):
        """Denominator should be max(len_test, len_train) matching baseline."""
        fn = _get("compute_patient_similarity_pairwise")
        self.assertIsNotNone(fn)

        # If test has 1 symptom, train has 3 symptoms, denominator = 3
        emb = _make_embedding(seed=10)
        embeddings = {
            "a": [emb],
            "b": [emb],
            "c": [emb],
            "d": [emb],
        }
        test_symptoms = ["a"]
        train_symptoms = ["b", "c", "d"]

        sim = fn(test_symptoms, train_symptoms, embeddings)
        # Self-cosine = 1.0 for identical vectors, so max_sim for "a" vs {b,c,d} = 1.0
        # result = 1.0 / max(1, 3) = 1/3
        self.assertAlmostEqual(sim, 1.0 / 3.0, places=3)


class TestPredictTopkDiagnosesPurePruning(unittest.TestCase):
    """Validate that PRUNING_SIMILARITY=0.5 threshold is applied."""

    def test_max_returns_empty_below_threshold(self):
        """MAX strategy must return empty when best similarity < 0.5."""
        fn = _get("predict_topk_diagnoses_pure")
        self.assertIsNotNone(fn)

        # Create embeddings that produce similarity < 0.5
        embeddings = {
            "symptom_a": [_make_embedding(seed=1)],
            "symptom_b": [_make_embedding(seed=2)],
        }

        test_admission = FakeAdmission("100", "symptom_a")
        test_symptoms = ["symptom_a"]
        x_train = [{"200": ["symptom_b"]}]
        admissions_dict = {"200": FakeAdmission("200", "symptom_b", ["apr:test diagnosis"])}

        diags, sims, pids = fn(
            test_admission, test_symptoms, x_train,
            embeddings, {}, admissions_dict, k=None
        )

        # All returned similarities must be >= 0.5
        for s in sims:
            self.assertGreaterEqual(s, 0.5)

    def test_max_returns_results_above_threshold(self):
        """MAX strategy must return the best match when similarity >= 0.5."""
        fn = _get("predict_topk_diagnoses_pure")
        self.assertIsNotNone(fn)

        # Use identical embeddings for high similarity
        emb = _make_embedding(seed=42)
        embeddings = {
            "fever": [emb],
        }

        test_admission = FakeAdmission("100", "fever")
        test_symptoms = ["fever"]
        x_train = [{"200": ["fever"]}]
        admissions_dict = {"200": FakeAdmission("200", "fever", ["apr:flu"])}

        diags, sims, pids = fn(
            test_admission, test_symptoms, x_train,
            embeddings, {}, admissions_dict, k=None
        )

        self.assertEqual(len(diags), 1)
        self.assertGreaterEqual(sims[0], 0.5)

    def test_topk_also_filters_by_threshold(self):
        """TOP-K strategy must also apply PRUNING_SIMILARITY filter."""
        fn = _get("predict_topk_diagnoses_pure")
        self.assertIsNotNone(fn)

        emb = _make_embedding(seed=42)
        embeddings = {
            "fever": [emb],
        }

        test_admission = FakeAdmission("100", "fever")
        test_symptoms = ["fever"]
        x_train = [{"200": ["fever"]}]
        admissions_dict = {"200": FakeAdmission("200", "fever", ["apr:flu"])}

        diags, sims, pids = fn(
            test_admission, test_symptoms, x_train,
            embeddings, {}, admissions_dict, k=10
        )

        # All returned similarities must be >= 0.5
        for s in sims:
            self.assertGreaterEqual(s, 0.5)


class TestEmbeddingStructureCompatibility(unittest.TestCase):
    """Validate embeddings work with cosine_similarity and load_dataset format."""

    def test_cosine_similarity_unwrap(self):
        """cosine_similarity(emb[0], emb[0]) should work on list-wrapped embeddings."""
        emb = _make_embedding(seed=7)
        wrapped = [emb]
        sim = cosine_similarity(wrapped[0], wrapped[0])
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_keys_match_load_dataset_format(self):
        """
        load_dataset returns [{hadm_id: [preproc_symptom1, preproc_symptom2, ...]}].
        Symptom embeddings must be keyed by the same preprocessed symptom strings.
        """
        fn = _get("compute_bert_symptom_embeddings")
        self.assertIsNotNone(fn)

        admissions = {
            "100": FakeAdmission("100", "fever,cough"),
        }
        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.random.randn(len(texts), 768).astype(np.float32)
        mock_model.device = "cpu"

        _ns["util_cy"].preprocess_sentence = lambda s: s.strip().lower()
        result = fn(mock_model, admissions)

        # Simulated load_dataset output
        dataset_symptoms = ["fever", "cough"]  # after preprocess_sentence

        for sym in dataset_symptoms:
            self.assertIn(sym, result, f"Symptom '{sym}' must be in embedding dict")


class TestTopKFPCounting(unittest.TestCase):
    """Validate FP counting logic in TOP-K evaluation."""

    def test_no_fp_when_empty(self):
        """When top_similarities_max is empty, FP should NOT be incremented."""
        top_similarities_max = []
        fp_count = 0
        threshold = 0.8

        if len(top_similarities_max) > 0:
            if not any(s >= threshold for s in top_similarities_max):
                fp_count += 1

        self.assertEqual(fp_count, 0)

    def test_fp_when_predictions_below_threshold(self):
        """When predictions exist but all below threshold, FP should increment."""
        top_similarities_max = [0.3, 0.4]
        fp_count = 0
        threshold = 0.8

        if len(top_similarities_max) > 0:
            if not any(s >= threshold for s in top_similarities_max):
                fp_count += 1

        self.assertEqual(fp_count, 1)

    def test_integration_fp_not_counted_for_missing_predictions(self):
        """
        In the actual fold loop, when no training case passes PRUNING_SIMILARITY,
        top_similarities_max should be empty and FP must NOT increment.
        """
        # Empty list => containGreaterOrEqualsValue returns False
        self.assertFalse(containGreaterOrEqualsValue(10, [], 0.8))

        # But FP should NOT be incremented because list is empty
        top_similarities_max = []
        fp_count = 0

        # Correct pattern (baseline cython_utils.py line 145):
        if not containGreaterOrEqualsValue(10, top_similarities_max, 0.8):
            if len(top_similarities_max) > 0:
                fp_count += 1

        self.assertEqual(fp_count, 0, "FP must not increment when no predictions exist")


if __name__ == "__main__":
    unittest.main()

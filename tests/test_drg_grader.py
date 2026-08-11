"""Tests for the encoder-independent DRG grader (TODO P4).

Why this file exists
--------------------
Before P4 every arm graded its own predictions with its own embedding space, so a
more compressed space marked its own work more leniently. These tests pin the
replacement: a grader that consults no embedding at all, so all four arms are
measured with one ruler.

Two of these tests exist because of specific traps rather than general diligence:

* ``test_prefix_variants_all_match`` -- ``preprocess_diagnosis`` rebuilds the DRG
  system prefix with a *substring* test and an unordered ``set``, so the same
  description legitimately appears as ``"apr:x"``, ``"apr,hcfa:x"`` and
  ``"hcfa,apr:x"`` in one run. A grader comparing whole labels would score those
  as three different diagnoses.
* ``test_determinism_across_hash_seeds`` -- that prefix instability is real and
  process-dependent, so determinism has to be demonstrated in *subprocesses*
  under different ``PYTHONHASHSEED`` values. Setting the variable inside this
  process would prove nothing: CPython reads it before any Python code runs.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from aicds.config import (
    CORRECTED,
    GRADER_DRG,
    LEGACY,
    PIPELINE_NAMES,
    SUPPORTED_GRADERS,
    PipelineConfig,
    from_name,
    require_supported_grader,
)
from aicds.utils.cython_utils import (
    drg_descriptions,
    get_diagnosis_relevance,
    get_diagnosis_similarity_baseline,
    get_diagnosis_similarity_by_drgcode,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# drg_descriptions -- prefix stripping and normalisation
# ---------------------------------------------------------------------------

class TestDrgDescriptions:
    def test_strips_single_prefix(self):
        assert drg_descriptions(["hcfa:septicemia age >17"]) == {"septicemia age >17"}

    def test_strips_combined_prefix(self):
        assert drg_descriptions(["apr,hcfa:septicemia age >17"]) == {"septicemia age >17"}

    def test_lowercases_and_trims(self):
        assert drg_descriptions(["MS:  Other Pneumonia  "]) == {"other pneumonia"}

    def test_label_without_prefix_is_used_whole(self):
        assert drg_descriptions(["bare description"]) == {"bare description"}

    def test_empty_description_is_dropped_not_matched(self):
        # A prefix with nothing after the colon must not become an empty-string
        # key that matches another empty one and manufactures a true positive.
        assert drg_descriptions(["apr:", "ms:   "]) == set()

    def test_splits_on_first_colon_only(self):
        # Verified against the real data: 0 of the 145 descriptions contain a
        # colon, so this only matters if that ever changes.
        assert drg_descriptions(["apr:a:b"]) == {"a:b"}


# ---------------------------------------------------------------------------
# drg-exact
# ---------------------------------------------------------------------------

class TestDrgExact:
    def test_exact_match_scores_one(self):
        got = get_diagnosis_similarity_by_drgcode(["hcfa:other pneumonia"], ["hcfa:other pneumonia"])
        assert got == 1.0

    def test_no_overlap_scores_zero(self):
        got = get_diagnosis_similarity_by_drgcode(["hcfa:other pneumonia"], ["ms:acute liver failure"])
        assert got == 0.0

    def test_prefix_variants_all_match(self):
        """The same DRG under every prefix spelling preprocess_diagnosis can emit."""
        truth = ["apr:intracranial hemorrhage"]
        for spelling in ("apr:", "hcfa:", "apr,hcfa:", "hcfa,apr:", "apr,hcfa,ms:"):
            predicted = [spelling + "intracranial hemorrhage"]
            assert get_diagnosis_similarity_by_drgcode(truth, predicted) == 1.0, spelling

    def test_case_differences_still_match(self):
        got = get_diagnosis_similarity_by_drgcode(["HCFA:Other Pneumonia"], ["ms:other pneumonia"])
        assert got == 1.0

    def test_any_of_several_true_labels_suffices(self):
        truth = ["apr:other pneumonia", "ms:respiratory failure"]
        got = get_diagnosis_similarity_by_drgcode(truth, ["ms:respiratory failure"])
        assert got == 1.0

    def test_empty_truth_scores_zero_and_does_not_raise(self):
        assert get_diagnosis_similarity_by_drgcode([], ["ms:x"]) == 0.0

    def test_empty_prediction_scores_zero(self):
        assert get_diagnosis_similarity_by_drgcode(["ms:x"], []) == 0.0

    def test_returns_float_not_bool(self):
        # The published grader seeds its running max with int 0, and that int-vs-float
        # distinction reaches the printed output elsewhere in this pipeline. Keep the
        # new grader unambiguously float so it cannot introduce a formatting change.
        got = get_diagnosis_similarity_by_drgcode(["a:x"], ["a:x"])
        assert isinstance(got, float) and not isinstance(got, bool)


# ---------------------------------------------------------------------------
# The graded (fraction-of-truth) variant, repaired alongside
# ---------------------------------------------------------------------------

class TestGradedBaseline:
    def test_all_matched_is_one(self):
        truth = ["apr:a", "ms:b"]
        assert get_diagnosis_similarity_baseline(truth, ["hcfa:a", "hcfa:b"]) == 1.0

    def test_half_matched_is_half(self):
        truth = ["apr:a", "ms:b"]
        assert get_diagnosis_similarity_baseline(truth, ["hcfa:a"]) == 0.5

    def test_none_matched_is_zero(self):
        assert get_diagnosis_similarity_baseline(["apr:a"], ["ms:z"]) == 0.0

    def test_empty_truth_does_not_divide_by_zero(self):
        """The original divided by len(gt_diagnosis) with no guard."""
        assert get_diagnosis_similarity_baseline([], ["ms:z"]) == 0.0

    def test_duplicate_truth_labels_do_not_inflate_the_denominator(self):
        # preprocess_diagnosis dedups, but the same description can still arrive
        # twice under different prefixes. Set semantics keep that from making a
        # perfect prediction score 0.5.
        truth = ["apr:a", "hcfa:a"]
        assert get_diagnosis_similarity_baseline(truth, ["ms:a"]) == 1.0


# ---------------------------------------------------------------------------
# Dispatch and config wiring
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_legacy_config_selects_cosine(self):
        """The golden path must reach the published grader, embeddings and all."""
        embeddings = {"x": [[1.0, 0.0]], "y": [[1.0, 0.0]]}
        got = get_diagnosis_relevance(embeddings, ["a:x"], ["a:y"], LEGACY)
        assert got == pytest.approx(1.0)

    def test_corrected_config_still_selects_cosine(self):
        # CORRECTED fixes the pipeline INPUTS. Bundling the grader into it would
        # have made its delta uninterpretable and retroactively invalidated the
        # corrected results already published.
        assert CORRECTED.grader == "cosine"

    def test_drg_config_ignores_embeddings_entirely(self):
        # Passing None proves no embedding is consulted: a cosine grader would
        # raise on subscripting it.
        got = get_diagnosis_relevance(None, ["a:x"], ["b:x"], GRADER_DRG)
        assert got == 1.0

    def test_unknown_grader_raises_rather_than_falling_back(self):
        rogue = PipelineConfig(grader="nonesuch")
        with pytest.raises(ValueError, match="no grader implements"):
            get_diagnosis_relevance(None, ["a:x"], ["a:x"], rogue)

    def test_require_supported_grader_rejects_unimplemented(self):
        with pytest.raises(ValueError, match="has no consumer"):
            require_supported_grader(PipelineConfig(grader="drg-graded"))

    def test_every_selectable_pipeline_has_a_live_grader(self):
        """No config may be selectable-but-unhonoured. This is the footgun guard.

        Iterates PIPELINE_NAMES rather than a hand-written list: the list form
        silently stopped covering the registry the moment `corrected2` was
        added (five names enumerated, six registered). No live bug then --
        CORRECTED2.grader is 'cosine' -- but a guard that does not follow the
        registry is not a guard.
        """
        assert len(PIPELINE_NAMES) >= 6, PIPELINE_NAMES
        for name in PIPELINE_NAMES:
            assert from_name(name).grader in SUPPORTED_GRADERS, name

    def test_drg_graded_is_not_selectable(self):
        with pytest.raises(ValueError):
            from_name("drg-graded")


# ---------------------------------------------------------------------------
# Determinism -- must be measured in subprocesses
# ---------------------------------------------------------------------------

SEED_PROBE = textwrap.dedent(
    """
    import hashlib, os, sys
    sys.path.insert(0, os.path.join(r"{root}", "src"))
    from aicds.utils.cython_utils import preprocess_diagnosis, get_diagnosis_similarity_by_drgcode

    raw = open(os.path.join(r"{root}", "data", "raw", "Symptoms-Diagnosis.txt"),
               encoding="utf-8").read().splitlines()
    labels = [preprocess_diagnosis(line.split(";")[-1]) for line in raw if line.strip()]

    # Every ordered pair of admissions, graded. Any hash-order sensitivity in
    # the grader would move this digest.
    digest = hashlib.sha256()
    for a in labels:
        for b in labels:
            digest.update(b"%d" % int(get_diagnosis_similarity_by_drgcode(a, b)))
    print(digest.hexdigest())
    """
)


class TestDeterminism:
    def test_determinism_across_hash_seeds(self):
        """Same grades under different hash seeds, in separate processes.

        preprocess_diagnosis is genuinely nondeterministic -- it builds the DRG
        prefix from an unordered set, so the prefix *spelling* varies run to run.
        The grader is immune only because it strips the prefix and compares
        description sets, whose content is stable. This test is what keeps that
        immunity from being an assumption.
        """
        probe = SEED_PROBE.format(root=REPO_ROOT)
        digests = {}
        for seed in ("0", "1", "12345", "99991"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            out = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True, text=True, env=env, cwd=REPO_ROOT,
            )
            assert out.returncode == 0, out.stderr
            digests[seed] = out.stdout.strip()

        assert len(set(digests.values())) == 1, digests


# ---------------------------------------------------------------------------
# The ceiling -- the denominator every drg-exact number must be read against
# ---------------------------------------------------------------------------

def _admission_labels():
    path = os.path.join(REPO_ROOT, "data", "raw", "Symptoms-Diagnosis.txt")
    labels = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split(";")
            labels[fields[0]] = drg_descriptions(fields[-1].split("--"))
    return labels


def _fold_ids(fold_dir, fold, name):
    path = os.path.join(REPO_ROOT, "data", fold_dir, "Fold%d" % fold, name)
    with open(path) as handle:
        return [line[: line.index("_")] for line in handle if "_" in line]


def _exact_match_ceiling(fold_dir):
    """Test cases whose own DRG label is present anywhere in their training pool."""
    labels = _admission_labels()
    reachable = total = 0
    for fold in range(10):
        pool = set()
        for hadm in _fold_ids(fold_dir, fold, "TrainingSet.txt"):
            pool |= labels.get(hadm, set())
        for hadm in _fold_ids(fold_dir, fold, "TestSet.txt"):
            mine = labels.get(hadm)
            if not mine:
                continue
            total += 1
            if mine & pool:
                reachable += 1
    return reachable, total


class TestExactMatchCeiling:
    """A perfect retriever cannot score 1.0 under drg-exact. Pin the real bound.

    Reported numbers are meaningless without this denominator: most of the
    unreachable cases are diagnoses that occur once in the entire dataset, so
    no amount of encoder quality can retrieve them.
    """

    def test_legacy_folds_ceiling(self):
        reachable, total = _exact_match_ceiling("folds")
        assert (reachable, total) == (75, 129)

    @pytest.mark.skipif(
        not os.path.isdir(os.path.join(REPO_ROOT, "data", "folds_grouped")),
        reason="data/folds_grouped is generated, not committed -- run scripts/make_folds.py",
    )
    def test_grouped_folds_ceiling(self):
        reachable, total = _exact_match_ceiling("folds_grouped")
        assert (reachable, total) == (76, 129)

    def test_fixing_leakage_barely_moved_the_ceiling(self):
        """Worth pinning: the leakage fix changed retrievability by one case.

        It is easy to assume regrouping the folds also improved what is findable.
        It did not -- 75 to 76 of 129. The two defects are independent.
        """
        if not os.path.isdir(os.path.join(REPO_ROOT, "data", "folds_grouped")):
            pytest.skip("data/folds_grouped not generated")
        legacy, _ = _exact_match_ceiling("folds")
        grouped, _ = _exact_match_ceiling("folds_grouped")
        assert abs(grouped - legacy) <= 1

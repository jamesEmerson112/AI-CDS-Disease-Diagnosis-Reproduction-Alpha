"""Pipeline configuration: the seam that lets corrections be *selectable*.

Why this module exists
----------------------
Nothing in this repository is trained. Every number that reaches
``PerformanceIndex.txt`` is a pure function of the input files and the
arithmetic in ``aicds.utils.cython_utils``, which means **any behaviour change
is a numerical change**. The project's only defence against silent numerical
drift is ``tests/golden/stub768/PerformanceIndex.txt`` -- a full 10-fold run
compared **byte for byte**, not with a float tolerance.

That safety net only works while the path it exercises stays frozen. So the
known correctness defects (patient leakage across folds, ``w/o`` collapsing to
``w`` in ``preprocess_sentence``, comma-split symptom fragments -- see
``docs/findings/`` and ``docs/plans/correctness-fixes.md``) cannot simply
*replace* the current behaviour: doing that would move every number and force a
re-mint of the golden, which would destroy the one artifact capable of proving
the move was intentional.

The resolution is this module. Each fix lands as a new *selectable* branch:

* ``LEGACY``    -- reproduces the published pipeline bit-for-bit, forever.
                   This is the default of every function that takes a config,
                   so any caller that does not opt in is numerically untouched.
* ``CORRECTED`` -- the fixed pipeline. Free to move the numbers, because it is
                   a different configuration rather than a redefinition of the
                   old one, and gets its own reference artifacts.

Rules of the road
-----------------
* ``PipelineConfig`` is frozen (hashable, no accidental mutation mid-run: a
  config that changed between fold 3 and fold 7 would produce a result that
  belongs to neither arm).
* Adding a field means adding a **default that preserves legacy behaviour**.
  If ``PipelineConfig()`` ever stops meaning "exactly what the golden ran",
  this seam has failed at its one job.
* Fields name a *variant*, not a boolean flag, so a third option (say a second
  preprocessing revision) does not require a new parameter everywhere.

Fields
------
fold_dir
    Directory under ``data/`` holding the ``Fold{n}/`` train/test splits.
    ``"folds"`` is the committed split, which divides on ``HADM_ID`` and
    therefore leaks patients across folds (129 admissions come from only 100
    subjects). ``"folds_grouped"`` is the ``GroupKFold``-on-``SUBJECT_ID``
    replacement. Consumed by ``cython_utils.load_dataset``.
preprocess_version
    Which ``preprocess_sentence``/``preprocess_diagnosis`` behaviour to use.
    ``"legacy"`` keeps the ``w/o`` -> ``w`` negation loss and the naive
    comma split. ``"corrected"`` protects ``w/o`` before slash-padding and
    rejoins comma fragments to their predecessor. Note the rejoin is a
    *heuristic* and still misses nine occurrences where the intra-label comma
    carries no trailing space -- see ``split_symptoms`` and TODO P27.
grader
    How a prediction is scored against the truth, consumed by
    ``cython_utils.get_diagnosis_relevance``.

    ``"cosine"`` is the published behaviour and the defect that motivated this
    field: it scores predictions by cosine similarity **in the same embedding
    space that produced the retrieval**, so every arm marks its own work with
    its own ruler. A more compressed space grades itself more leniently, which
    is why the most compact of the three BERT models scores highest at the
    strictest thresholds. Any cross-encoder claim built on it is confounded.

    ``"drg-exact"`` compares DRG label text instead: 1.0 if any true label
    equals any predicted label, else 0.0. No embeddings are consulted, so all
    four arms are graded identically and threshold 1.0 finally means something
    checkable -- *the predicted DRG label is exactly the true one*.

    ``"drg-graded"`` is ``"drg-exact"`` plus partial credit for near-miss
    labels, since the three coding systems describe the same condition in
    different words ("SEPTICEMIA AGE >17" vs "Septicemia & Disseminated
    Infections"). Exact matches still return 1.0.

    Note the DRG graders are deterministic even though ``preprocess_diagnosis``
    is not: its hash-order instability lives in the reconstructed *prefix*,
    which the grader strips before comparing, and the description *set* it
    produces is content-stable.

Why there are six configs and not two
-------------------------------------
``CORRECTED`` changes **two independent things at once**: it moves to the
grouped folds *and* to the fixed preprocessing. A ``LEGACY`` vs ``CORRECTED``
delta therefore cannot say how much of the movement came from removing patient
leakage and how much from fixing the text handling. For a project whose entire
point is attributing an effect to a cause, that is not good enough, so both
one-change-at-a-time configs exist as well. They cost nothing to define; run
them only when the attribution question is actually being asked.

The last two, ``GRADER_DRG`` and ``GRADER_DRG_GRADED``, are a different kind of
correction and are kept separate for that reason. ``CORRECTED`` fixes the
*inputs* -- which patients land in which fold, and what text gets encoded. The
grader configs fix the *measurement*: they stop each arm marking its own work
with its own embedding space. Bundling them into ``CORRECTED`` would have made
its delta uninterpretable in exactly the way described above, so the grader
change gets its own axis and its own runs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Which variant of each pipeline stage to run.

    The default constructor MUST always describe the legacy pipeline, because
    every config-aware function defaults its parameter to ``LEGACY``.
    """

    fold_dir: str = "folds"
    preprocess_version: str = "legacy"
    grader: str = "cosine"


#: The published pipeline. Bit-identical to what minted the golden reference.
#: Do not change these values -- add new fields with legacy-preserving defaults.
LEGACY = PipelineConfig()

#: The correctness-fixed pipeline. Expected to produce different numbers than
#: ``LEGACY``; that difference is the finding, not a regression.
CORRECTED = PipelineConfig(fold_dir="folds_grouped", preprocess_version="corrected")

#: Grouped folds, legacy text handling. Isolates the PATIENT LEAKAGE effect.
FOLDS_ONLY = PipelineConfig(fold_dir="folds_grouped", preprocess_version="legacy")

#: Legacy folds, fixed text handling. Isolates the PREPROCESSING effect.
#: Still leaks patients, so it is an attribution instrument only -- never
#: report a number from this config as a result.
PREPROCESS_ONLY = PipelineConfig(fold_dir="folds", preprocess_version="corrected")

#: CORRECTED plus an encoder-independent grader: a predicted diagnosis counts
#: only if its DRG label matches the true one exactly. This is the first config
#: under which the four arms are graded by the SAME ruler, which is what makes a
#: cross-encoder comparison mean anything (TODO P4).
#:
#: Its ceiling is not 1.0 and cannot be. Only 76 of 129 test cases have their
#: correct label present anywhere in their own fold's training pool under
#: ``folds_grouped``, so a perfect retriever scores 58.9% at threshold 1.0.
#: Always report that denominator alongside the number.
GRADER_DRG = PipelineConfig(
    fold_dir="folds_grouped", preprocess_version="corrected", grader="drg-exact"
)

# A partial-credit grader ("drg-graded") was designed and rejected; see
# docs/findings/12-drg-grader.md. Three candidate similarity functions were built
# and measured against the 72 label pairs that co-occur on a single admission --
# pairs known to describe one episode, so a usable proxy for "should score high".
# The best of them cleared 54 of 72 at threshold 0.6, but needed ~156 hand-written
# lexicon strings mined from the same 146 diagnosis titles it then scores, with no
# held-out set to show it had not simply memorised them. Its two lowest rungs ran
# at 0.13 and 0.08 precision against non-co-occurring pairs, i.e. roughly seven of
# every eight admits were false credit.
#
# The whole point of P4 is to remove an arbitrary ruler. Replacing cosine with a
# more elaborate arbitrary ruler would concede the argument, so there is no
# ``drg-graded`` config and ``from_name`` cannot return one. ``drg-exact`` has
# zero free parameters, which is precisely its value.


# Selecting a pipeline from outside the process.
#
# HISTORICAL: this env var used to be the ONLY hook for the baseline arm, because
# baseline_sent2vec.py executed its whole fold loop at import time and there was
# no moment after import at which a caller could rebind its config -- by then the
# run was over. That is no longer true: the module now exposes
# ``run_analysis(encoder=None, config=LEGACY)``, so a caller passes the config
# directly, exactly as the BERT arm does.
#
# The env var stays as the script-level fallback shared by BOTH arms: each entry
# point resolves ``from_name(args.pipeline) if args.pipeline else from_env()``, so
# ``--pipeline`` wins and the variable covers the case where no flag is given.
# LEGACY remains the default of ``from_env``, so a process that sets nothing is
# bit-identical and the golden is unaffected.
_ENV_VAR = "AICDS_PIPELINE"

_BY_NAME = {
    "legacy": LEGACY,
    "corrected": CORRECTED,
    "folds-only": FOLDS_ONLY,
    "preprocess-only": PREPROCESS_ONLY,
    "drg": GRADER_DRG,
}


def name_of(config):
    """The registry name of ``config``, or ``None`` if it is not a registry entry.

    The inverse of :func:`from_name`, and the reason it can be written as a
    value lookup rather than an identity check is that ``PipelineConfig`` is a
    frozen dataclass: two configs with the same three fields ARE equal, so a
    caller that rebuilt ``PipelineConfig(fold_dir="folds_grouped",
    preprocess_version="corrected", grader="drg-exact")`` by hand still gets
    back ``"drg"``. That is the honest answer -- the name describes the values,
    not the object.

    ``None`` is a first-class result, not a failure. A run may legitimately be
    driven by a config assembled in a test or in an ad-hoc experiment, and
    ``run_metadata.json`` records the name *alongside* all three fields for
    exactly this case: the null says "no registry entry describes this", and the
    fields still identify the run completely. Never make this raise -- a caller
    recording provenance must not be the thing that kills a finished run.
    """
    for name, known in _BY_NAME.items():
        if known == config:
            return name
    return None

#: Grader values that a consumer actually honours. This exists because a config
#: field with no reader is worse than no field at all: for a short window during
#: development, ``--pipeline drg`` was selectable, printed a plausible banner,
#: ran a full 10-fold pass and produced *cosine* numbers labelled as DRG ones.
#: Nothing failed. Any new grader name must land in the same commit as its
#: consumer, and this set is what makes forgetting that a loud error.
SUPPORTED_GRADERS = frozenset({"cosine", "drg-exact"})


#: Selectable pipeline names, for argparse ``choices=``. Derived from the
#: registry so a new config cannot exist-but-be-unselectable, which is the bug
#: this seam shipped with initially.
PIPELINE_NAMES = tuple(sorted(_BY_NAME))

#: argparse ``help=`` for ``--pipeline``, shared by every entry point. It lives
#: here rather than in one script because BOTH arms take the flag and the text
#: describes the registry above: two copies would drift, and the copy that
#: drifted would be the one describing a config the reader did not run.
PIPELINE_HELP = (
    "legacy (default) reproduces the committed numbers byte-for-byte; "
    "corrected uses the GroupKFold folds AND the fixed preprocessing. "
    "folds-only and preprocess-only change one thing each, so the two "
    "effects can be told apart -- note preprocess-only still leaks patients "
    "and is an attribution instrument, not a result. "
    "drg adds the encoder-independent grader on top of corrected: a "
    "prediction counts only on an exact DRG label match, so all four arms "
    "are scored by one ruler instead of each by its own embedding space. "
    # %% -- argparse runs this string through % expansion, so a bare
    # percent sign here breaks --help with an unrelated ValueError.
    "Its ceiling is 58.9%%, not 100%%. "
    "Falls back to $AICDS_PIPELINE, then legacy; an explicit --pipeline wins."
)


def require_supported_grader(config):
    """Raise unless ``config.grader`` has a consumer. Call this EARLY.

    A grader nobody reads does not crash -- it silently produces numbers from
    whichever grader *is* wired, under the label of the one requested. That is
    the worst failure mode this repository has, because the output looks
    entirely normal. Call this before a run starts rather than letting the
    dispatch discover it 12 minutes in, after the model load and the encode.
    """
    if config.grader not in SUPPORTED_GRADERS:
        raise ValueError(
            "pipeline grader %r has no consumer -- supported: %s. A grader must "
            "land in the same commit as the code that honours it."
            % (config.grader, ", ".join(sorted(SUPPORTED_GRADERS)))
        )
    return config


def from_name(name):
    """Resolve 'legacy' or 'corrected' to a PipelineConfig. Raises on anything else."""
    try:
        config = _BY_NAME[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            "unknown pipeline %r -- expected one of %s"
            % (name, ", ".join(sorted(_BY_NAME)))
        )
    return require_supported_grader(config)


def from_env(default=LEGACY):
    """Read AICDS_PIPELINE. Unset means LEGACY, so the default path never moves."""
    import os

    raw = os.environ.get(_ENV_VAR)
    if raw is None or not raw.strip():
        return default
    return from_name(raw)

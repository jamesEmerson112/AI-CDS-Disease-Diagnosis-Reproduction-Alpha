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

Why there are four configs and not two
--------------------------------------
``CORRECTED`` changes **two independent things at once**: it moves to the
grouped folds *and* to the fixed preprocessing. A ``LEGACY`` vs ``CORRECTED``
delta therefore cannot say how much of the movement came from removing patient
leakage and how much from fixing the text handling. For a project whose entire
point is attributing an effect to a cause, that is not good enough, so both
one-change-at-a-time configs exist as well. They cost nothing to define; run
them only when the attribution question is actually being asked.
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


# Selecting a pipeline from outside the process.
#
# baseline_sent2vec.py executes its whole fold loop at import time, so there is
# no moment after import at which a caller could rebind its config -- by then
# the run is over. An environment variable is read at import, which is the one
# hook that arrives early enough for both arms. LEGACY stays the default, so a
# process that sets nothing is bit-identical and the golden is unaffected.
_ENV_VAR = "AICDS_PIPELINE"

_BY_NAME = {
    "legacy": LEGACY,
    "corrected": CORRECTED,
    "folds-only": FOLDS_ONLY,
    "preprocess-only": PREPROCESS_ONLY,
}


#: Selectable pipeline names, for argparse ``choices=``. Derived from the
#: registry so a new config cannot exist-but-be-unselectable, which is the bug
#: this seam shipped with initially.
PIPELINE_NAMES = tuple(sorted(_BY_NAME))


def from_name(name):
    """Resolve 'legacy' or 'corrected' to a PipelineConfig. Raises on anything else."""
    try:
        return _BY_NAME[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            "unknown pipeline %r -- expected one of %s"
            % (name, ", ".join(sorted(_BY_NAME)))
        )


def from_env(default=LEGACY):
    """Read AICDS_PIPELINE. Unset means LEGACY, so the default path never moves."""
    import os

    raw = os.environ.get(_ENV_VAR)
    if raw is None or not raw.strip():
        return default
    return from_name(raw)

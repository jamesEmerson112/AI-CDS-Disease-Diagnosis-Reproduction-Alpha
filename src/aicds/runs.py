"""The run-directory contract: where an arm writes, and how a reader finds it.

Both arms used to build their own output paths inline, from ``os.getcwd()`` plus a
timestamp, with no shared vocabulary between the writers and the five discovery
sites that look for their results. That produced the fragmentation documented in
``docs/findings/10-output-path-fragmentation.md``: the BERT arm wrote
``Prediction_Output_{Model}_{stamp}/`` while the baseline wrote
``Prediction Output_{stamp}/`` -- *with a space*, and no model name -- so **zero
globs in the repository ever matched baseline output**. `build_dashboard_data`
"recovered" a model name from it by stripping a prefix that was not there and got
back the whole directory name, silently, with no exception.

This module is the one place that shape is decided. It is the *writer* side of the
contract today; a later commit adds the reader side beside it, which is why it sits
at the top of the package rather than under ``aicds.analysis`` -- that package's
docstring scopes it to code that *reads* the pipeline's outputs, so a model
importing from it would invert the layering.

Why an explicit key table
-------------------------
``MODEL_KEYS`` maps display name to directory key by hand. The obvious shortcut,
``name.lower()``, is wrong in a way that does not raise: it turns
``"Bio_ClinicalBERT"`` into ``"bio_clinicalbert"`` while every committed
``results*/`` tree, every key in ``compare_models.MODEL_REGISTRY``
(``scripts/compare_models.py:59``; the reader-side copies are
``analyze_rank_metrics.ARM_ORDER`` and ``LABELS``), and every
sanity check keyed by arm spells it ``bio_clinical_bert``. The run would succeed
and simply file itself under a second directory, splitting one arm's history in
two. ``model_key`` therefore raises on an unknown name: a new encoder is a new
arm, and it deserves a deliberate entry here rather than an invented key.

Why two directory shapes
------------------------
``out=None`` -- the flat, cwd-relative default -- is **frozen by the golden**.
``tests/test_golden.py`` chdirs into a temporary directory, globs
``Prediction_Output_*`` non-recursively, asserts exactly one match, and
cross-checks it against ``run_analysis``'s return value. Its timestamp regex
(``\\d{8}_\\d{2}-\\d{2}-\\d{2}``) also fixes the stamp format. Nothing about the
default may move. Note it uses the display **name**, not the key, and that the
symptom-details directory is a *sibling* of the run root.

``out=ROOT`` -- ``ROOT/{key}/{stamp}/`` with ``symptom_details/`` nested inside --
reproduces the layout the ``results*/`` trees were hand-assembled into after each
run, and which ``scripts/compare_models.py --results-dir`` and
``scripts/analyze_rank_metrics.py`` already read. Nesting the details directory is
the deliberate difference from the default: under a shared root a sibling would
land as ``ROOT/{key}/symptom_details_{stamp}`` next to the timestamps and break
"every child of the model directory is a run".

The baseline's space spelling is retired here. Both arms now emit
``Prediction_Output_{Name}_{stamp}``; the underscore wins because
``test_golden.py`` pins it and because nothing ever read the other one
(finding 10, "Fix direction is forced").

Why ``--out`` needs a guard
---------------------------
The default layout is covered: ``.gitignore``'s four anchored root patterns
exist precisely because a run leaves per-case files **named by HADM_ID** under
``Fold*/``, and because those files are *empty* the pre-commit hook's content
rule (20+ distinct HADM_IDs *inside* a file) scores them 0 and cannot see them
at all. ``--out ROOT`` reopens that hole one directory over: ``--out scratch``,
``--out myruns`` and ``--out out`` are all **unignored** (``.gitignore``'s entry
is ``output/``, not ``out/``), so the same DUA-covered filenames land somewhere
``git add -A`` will happily stage. ``check_out_root`` closes it by asking git
itself, and is called from both CLI entry points before the model load.
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass

# runs.py lives at <repo>/src/aicds/runs.py -- three parents to the root. (Note
# Constants.CH_DIR walks four; it sits one package deeper.)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Display name (as the arm reports it) -> directory key (as readers spell it).
# Hand-maintained on purpose; see the module docstring. Keys must match the
# directory names already present under results*/ and the keys in
# scripts/compare_models.py MODEL_REGISTRY (and analyze_rank_metrics.ARM_ORDER).
MODEL_KEYS = {
    "BioSentVec": "baseline",
    "Bio_ClinicalBERT": "bio_clinical_bert",
    "BiomedBERT": "biomedbert",
    "BlueBERT": "bluebert",
}

# Pinned by tests/test_golden.py's timestamp regex. Day-first, and ':' avoided
# because it is illegal in a Windows path component.
TIMESTAMP_FORMAT = "%d%m%Y_%H-%M-%S"


def model_key(model_name):
    """Directory key for a display model name.

    Raises ``KeyError`` on anything not in ``MODEL_KEYS``. That is the point:
    guessing a key (``name.lower()``) would file a new arm's runs under a name no
    reader knows, without failing.
    """
    try:
        return MODEL_KEYS[model_name]
    except KeyError:
        raise KeyError(
            "unknown model name %r -- add it to aicds.runs.MODEL_KEYS with a "
            "deliberate directory key. Known names: %s. Do not fall back to "
            "name.lower(): that mints keys like 'bio_clinicalbert' and splits "
            "one arm's run history across two directories."
            % (model_name, ", ".join(sorted(MODEL_KEYS)))
        )


def timestamp(now=None):
    """Run stamp in ``TIMESTAMP_FORMAT``. ``now`` is a struct_time, for tests."""
    if now is None:
        return time.strftime(TIMESTAMP_FORMAT)
    return time.strftime(TIMESTAMP_FORMAT, now)


@dataclass(frozen=True)
class RunDirs:
    """The two directories one run writes, plus the identifiers behind them.

    Frozen for the same reason ``PipelineConfig`` is: a run whose output root
    changed halfway through would leave results belonging to neither location.

    ``root`` and ``details_root`` both end in a separator, because every caller
    in both arms builds sub-paths by string concatenation (``root + 'Fold0/'``)
    and those concatenations are load-bearing on the golden -- including the
    ``root + '/PerformanceIndex.txt'`` double slash, which reaches the console.
    """

    root: str
    details_root: str
    key: str
    stamp: str


def run_dirs(model_name, out=None, stamp=None):
    """Resolve the output directories for one run. Creates nothing.

    ``out=None`` gives the legacy flat layout in the current working directory --
    ``Prediction_Output_{Name}_{stamp}/`` with a sibling
    ``Prediction_Symptom_Details_{Name}_{stamp}/``. This is what the golden pins;
    do not change it.

    ``out=ROOT`` gives ``ROOT/{key}/{stamp}/`` with ``symptom_details/`` nested
    inside it -- the ``results*/`` layout the comparison scripts already read.
    """
    key = model_key(model_name)
    if stamp is None:
        stamp = timestamp()

    if out is None:
        # os.getcwd() + '/' rather than os.path.join: the exact string the two
        # arms produced before this module existed, forward slash and all.
        base = os.getcwd()
        root = base + "/Prediction_Output_" + model_name + "_" + stamp + "/"
        details_root = (
            base + "/Prediction_Symptom_Details_" + model_name + "_" + stamp + "/"
        )
    else:
        root = os.path.join(out, key, stamp) + os.sep
        details_root = os.path.join(out, key, stamp, "symptom_details") + os.sep

    return RunDirs(root=root, details_root=details_root, key=key, stamp=stamp)


class UnignoredOutRoot(ValueError):
    """``--out`` resolves inside the repo and no ignore rule covers it."""


def _is_inside(path, root):
    """True if ``path`` is at or below ``root``.

    ``os.path.relpath`` rather than ``commonpath`` because the latter raises on
    two different Windows drives, which is the ordinary case for ``--out D:/runs``
    -- and that case is *safe*, so it must answer False rather than blow up.
    ``normcase`` because ``C:/Users`` and ``c:/users`` are one directory.
    """
    path = os.path.normcase(os.path.abspath(path))
    root = os.path.normcase(os.path.abspath(root))
    try:
        relative = os.path.relpath(path, root)
    except ValueError:
        return False
    return not (relative == os.pardir or relative.startswith(os.pardir + os.sep))


def _git_ignores(path, repo_root):
    """``(answered, ignored)`` from ``git check-ignore``. Touches no files.

    ``answered`` is False when git could not be asked (not installed, or an exit
    status outside the documented 0/1). Callers must not read ``ignored`` then:
    "no" and "don't know" are different, and conflating them either blocks a
    legitimate run or waves clinical data through.
    """
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", path],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False, False
    if result.returncode not in (0, 1):
        return False, False
    return True, result.returncode == 0


def check_out_root(out, repo_root=None):
    """Refuse an ``--out`` root that would leave clinical data unignored.

    Returns the resolved absolute root, or ``None`` for ``out=None``.

    ``out=None`` returns immediately, which is what keeps the golden contract
    untouched: ``test_golden.py`` never passes ``out``, so this function is not
    even reached on the path it pins.

    The check asks git rather than re-implementing ``.gitignore`` semantics, and
    it probes a *representative per-case file* (``ROOT/{key}/{stamp}/Fold0/…``)
    rather than the root, because that deep path is the thing actually at risk
    and ``check-ignore`` reports an ancestor's rule for it anyway.

    Three outcomes, deliberately not two:

    * **outside the repo, or no working tree at ``repo_root``** -- returns. There
      is nothing here for ``git add`` to stage.
    * **inside and ignored** -- returns. ``--out results`` is the happy path.
    * **inside and not ignored** -- raises ``UnignoredOutRoot``.

    A fourth case, *git could not be asked*, warns on stderr and continues. It
    is not a refusal because the state is unknown rather than bad, and a user
    with no working ``git`` cannot stage anything either; it is not silence
    because that assumption stops holding the moment git appears on PATH.
    """
    if out is None:
        return None

    root = REPO_ROOT if repo_root is None else repo_root
    resolved = os.path.abspath(out)

    if not _is_inside(resolved, root):
        return resolved
    # .git is a file, not a directory, inside a linked worktree.
    if not os.path.exists(os.path.join(root, ".git")):
        return resolved

    probe = os.path.join(resolved, "baseline", "01012026_00-00-00", "Fold0", "999001.txt")
    answered, ignored = _git_ignores(probe, root)

    if not answered:
        sys.stderr.write(
            "WARNING: could not ask git whether %s is ignored, so it was not "
            "checked. Run output includes per-case files named by HADM_ID; if "
            "this root is inside the repository, add a pattern to .gitignore "
            "before committing anything. See docs/guides/data-use.md.\n" % resolved
        )
        return resolved

    if not ignored:
        raise UnignoredOutRoot(
            "--out %s resolves inside the repository (%s) and no .gitignore rule "
            "covers it.\n"
            "A run writes per-case files NAMED by HADM_ID under {key}/{stamp}/Fold*/ "
            "-- DUA-covered MIMIC-III data in a PUBLIC repo. They are also EMPTY, so "
            ".githooks/pre-commit cannot catch them: its rule counts 20+ HADM_IDs "
            "*inside* a file and an empty file scores 0. The ignore rule is the only "
            "defence.\n"
            "Use --out results/ (or any results*/ path), pick a root outside the "
            "repository, or add an anchored pattern to .gitignore first. See "
            "docs/guides/data-use.md." % (out, root)
        )

    return resolved

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

This module is the one place that shape is decided. Both halves live here: the
*writer* side (:func:`run_dirs`) and the *reader* side (:func:`discover`). It sits
at the top of the package rather than under ``aicds.analysis`` because that
package's docstring scopes it to code that *reads* the pipeline's outputs, so a
model importing from it would invert the layering -- and the writer is the half a
model needs.

Why one rule for "is this a run"
-------------------------------
:func:`discover` replaces six-to-seven private discovery rules that disagreed
about the base directory, the glob, and the marker file. It has exactly **one**
test: *a directory is a run iff it contains ``PerformanceIndex.txt``*. There is
deliberately no ``marker=`` parameter. A caller that wants ``RankMetrics.txt``
asks a discovered run for it via :meth:`Run.file` and gets ``None`` if it is
absent -- which is the honest answer for a pre-P5 run. Making the marker
configurable would turn "this run predates rank metrics" and "this directory is
not a run" into the same silence, and the second of those is a broken pipeline
worth surfacing.

The corollary is that a directory holding a ``PerformanceIndex.txt`` in a shape
neither layout allows raises :class:`RunLayoutError` rather than being skipped.
Skipping is how a run vanishes from a comparison table without anybody noticing
it was ever there; that is exactly the failure finding 10 documents, and a
half-migrated tree is the likeliest way to reproduce it.

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

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
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

# Directory key -> display name. The inverse of MODEL_KEYS, derived rather than
# written out, so the two cannot drift. The reader needs this direction: a nested
# run's key is its parent directory's name, and the label is what a legend prints.
KEY_LABELS = {key: name for name, key in MODEL_KEYS.items()}

# Pinned by tests/test_golden.py's timestamp regex. Day-first, and ':' avoided
# because it is illegal in a Windows path component.
TIMESTAMP_FORMAT = "%d%m%Y_%H-%M-%S"

#: The one marker that makes a directory a run. See the module docstring.
PERFORMANCE_INDEX = "PerformanceIndex.txt"

#: The provenance file every run writes LAST. Deliberately NOT the run marker:
#: ``PERFORMANCE_INDEX`` keeps that job, so every pre-P14 run stays discoverable
#: and asking a ``Run`` for this file returns ``None`` rather than erasing it
#: from a comparison. See :func:`write_run_metadata`.
RUN_METADATA = "run_metadata.json"

#: Schema version of ``run_metadata.json``. Bump it when a key's MEANING
#: changes; a reader that finds a version it does not know should say so rather
#: than guess. Adding a key is not a bump -- readers take what they need.
METADATA_VERSION = 1

# The same shape TIMESTAMP_FORMAT writes, and the same one tests/test_golden.py
# asserts. Spelled as a regex here because the reader has to recognise it in a
# directory name, and time.strptime would accept forms strftime never emits.
_STAMP = r"\d{8}_\d{2}-\d{2}-\d{2}"

# The flat, cwd-relative layout. `.+` is greedy but the stamp is anchored to the
# end of the name, so "Bio_ClinicalBERT_15022026_11-33-48" splits at the last
# underscore-delimited stamp and the label keeps its own underscores.
_FLAT_RUN = re.compile(r"^Prediction_Output_(?P<label>.+)_(?P<stamp>" + _STAMP + r")$")

# The nested `ROOT/{key}/{stamp}/` layout: the run directory's own name is just
# the stamp, and its parent carries the key.
_STAMP_ONLY = re.compile(r"^" + _STAMP + r"$")


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


# --------------------------------------------------------------------------
# Reader side: finding the runs a writer left behind
# --------------------------------------------------------------------------

class RunLayoutError(ValueError):
    """A directory holds a ``PerformanceIndex.txt`` in neither run shape.

    Raised rather than skipped on purpose; see "Why one rule" in the module
    docstring. The message always names the offending path, because the useful
    next action is to look at it.
    """


def format_run_date(stamp):
    """``'05082026_18-55-32'`` -> ``'2026-08-05'``.

    Returns ``stamp`` unchanged when it does not start with a day-first date, so
    an unrecognised name degrades to showing itself rather than to a wrong date.
    Lifted verbatim from ``scripts/compare_models._format_run_date``, which was
    the only implementation and had no business living in a reporting script.
    """
    match = re.match(r"^(\d{2})(\d{2})(\d{4})_", stamp)
    if not match:
        return stamp
    day, month, year = match.groups()
    return "%s-%s-%s" % (year, month, day)


@dataclass(frozen=True)
class Run:
    """One discovered run directory.

    Frozen, like :class:`RunDirs`: a reader that mutated a run's ``key`` halfway
    through a report would attribute half its rows to the wrong arm.

    ``key`` is the directory key (``bio_clinical_bert``), ``label`` the display
    name (``Bio_ClinicalBERT``); a caller that prints one and groups by the other
    is the mistake ``MODEL_KEYS`` exists to prevent. ``performance_index`` is
    always a real file -- its existence is what made this directory a run.
    """

    key: str
    label: str
    stamp: str
    date: str
    path: str
    performance_index: str

    def file(self, name):
        """Absolute-or-as-given path to a sibling artifact, or ``None``.

        ``None`` means "this run did not write that file", which is a fact about
        the run (``RankMetrics.txt`` did not exist before P5) rather than an
        error. Callers are expected to say so and carry on; see
        ``scripts/analyze_rank_metrics.py``.
        """
        candidate = os.path.join(self.path, name)
        return candidate if os.path.isfile(candidate) else None


def _unclassifiable(root, path, name, depth):
    """The error text for a run directory in neither shape.

    Both accepted shapes are quoted, and the two *near misses* -- right shape,
    wrong depth -- get a specific instruction, because that is what a
    half-migrated tree looks like and "pass a different root" is the fix.
    """
    hint = ""
    if depth == 2 and _FLAT_RUN.match(name):
        hint = (
            "\nIt matches the flat shape but sits TWO levels below the root. Pass "
            "its parent directory (%s) as the root instead."
            % os.path.dirname(path)
        )
    elif depth == 1 and _STAMP_ONLY.match(name):
        hint = (
            "\nIt matches the nested shape but sits ONE level below the root, so "
            "there is no model-key directory above it. Pass its grandparent (%s) "
            "as the root instead." % os.path.dirname(os.path.dirname(path))
        )
    return (
        "%s contains %s but fits neither run layout, so there is no honest key to "
        "file it under.\n"
        "Accepted: <root>/Prediction_Output_<Model>_<DDMMYYYY_HH-MM-SS>/ (flat) or "
        "<root>/<model_key>/<DDMMYYYY_HH-MM-SS>/ (nested).\n"
        "Discovered from root %s." % (path, PERFORMANCE_INDEX, root) + hint
    )


def _flat_run(root, path, name):
    match = _FLAT_RUN.match(name)
    if not match:
        raise RunLayoutError(_unclassifiable(root, path, name, depth=1))
    label = match.group("label")
    if label not in MODEL_KEYS:
        # Deliberately asymmetric with the nested case below. A flat directory
        # name carries a *display name*, and MODEL_KEYS is the authority on those
        # -- guessing label.lower() is the exact mistake model_key() refuses. A
        # nested directory name carries a *key*, which the tree itself asserts,
        # so there is nothing to guess there.
        raise RunLayoutError(
            "%s is a flat run for model name %r, which is not in "
            "aicds.runs.MODEL_KEYS. Known: %s. Add it with a deliberate directory "
            "key rather than letting this run be filed under an invented one."
            % (path, label, ", ".join(sorted(MODEL_KEYS)))
        )
    stamp = match.group("stamp")
    return Run(
        key=MODEL_KEYS[label],
        label=label,
        stamp=stamp,
        date=format_run_date(stamp),
        path=path,
        performance_index=os.path.join(path, PERFORMANCE_INDEX),
    )


def _nested_run(root, path, parent, name):
    if not _STAMP_ONLY.match(name):
        raise RunLayoutError(_unclassifiable(root, path, name, depth=2))
    # An unregistered parent name is accepted and becomes its own label. That
    # preserves what compare_models did: a new arm's directory shows up in the
    # report under its directory name, styled from FALLBACK_STYLES, rather than
    # being dropped from a four-model comparison without a word.
    return Run(
        key=parent,
        label=KEY_LABELS.get(parent, parent),
        stamp=name,
        date=format_run_date(name),
        path=path,
        performance_index=os.path.join(path, PERFORMANCE_INDEX),
    )


def discover(root, all_runs=False, depth=2):
    """Every run under ``root``, ordered by arm. Reads no ``PerformanceIndex.txt``.

    A directory is a run iff it contains ``PerformanceIndex.txt`` -- the single
    rule. The walk is bounded at depth 2 because both layouts put a run there or
    shallower, and an unbounded walk would descend into the ``Fold*/`` trees of
    every run it already found. A run directory's own children are never
    examined, for the same reason.

    ``depth=1`` narrows that to the *flat* layout alone: only ``root``'s immediate
    children are considered, and a child that is not itself a run is not opened.
    It exists because "refuse, don't skip" (above) makes the depth-2 walk unusable
    on a root that merely *contains* a tree of runs it was not pointed at. The
    repository root is exactly that root: ``docs/`` holds three committed flat runs
    at depth 2, where only the nested shape is legal, so ``discover(repo_root)``
    raises no matter what else is present -- including a legitimate fresh run
    sitting at depth 1. A caller auto-detecting "whatever an arm just wrote into
    the cwd" wants only the layout an arm writes with no ``--out``, which is flat
    and always at depth 1 (``run_dirs``, frozen by the golden). A caller that wants
    a ``results*/`` tree names it, and gets the full depth-2 walk.

    Ordering is ``MODEL_KEYS`` order first, then unregistered keys
    alphabetically -- lifted from ``compare_models.discover_runs`` so its report
    pages keep the arm order they were laid out for.

    By default one run per key survives: the one whose directory has the newest
    mtime, **not** the lexically last stamp. ``DDMMYYYY`` does not sort
    chronologically, so ``15022026`` sorts after ``05082026`` while being five
    months older. When a key has more than one run the choice is *printed*, which
    is what makes a stale run visible; silently picking is how a report ends up
    describing last week's numbers. ``all_runs=True`` returns every run, still
    grouped by key and mtime-ascending within a key.

    Raises :class:`RunLayoutError` if ``root`` is not a directory, or if any
    directory holding a ``PerformanceIndex.txt`` fits neither layout. Note that
    ``depth=1`` narrows *which* directories are looked at, not how strictly they
    are judged: an unclassifiable run sitting directly in ``root`` still raises.
    """
    if depth not in (1, 2):
        raise ValueError(
            "depth must be 1 (flat runs directly under root) or 2 (both layouts), "
            "got %r. There is no deeper layout: a run's own children are Fold*/ "
            "and symptom_details/." % (depth,)
        )
    if not os.path.isdir(root):
        raise RunLayoutError("run root is not a directory: %s" % root)

    found = {}
    for name in sorted(os.listdir(root)):
        depth1 = os.path.join(root, name)
        if not os.path.isdir(depth1):
            continue

        if os.path.isfile(os.path.join(depth1, PERFORMANCE_INDEX)):
            run = _flat_run(root, depth1, name)
            found.setdefault(run.key, []).append(run)
            # Do not descend: this run's children are Fold*/ and symptom_details/.
            continue

        if depth < 2:
            # Not a flat run, and the caller asked for flat runs only. Crucially
            # this does not *open* the directory, so a subtree of runs the caller
            # was not pointed at (docs/, results*/) can neither be found nor
            # refused.
            continue

        for child in sorted(os.listdir(depth1)):
            depth2 = os.path.join(depth1, child)
            if not os.path.isdir(depth2):
                continue
            if not os.path.isfile(os.path.join(depth2, PERFORMANCE_INDEX)):
                continue
            run = _nested_run(root, depth2, name, child)
            found.setdefault(run.key, []).append(run)

    registered = list(MODEL_KEYS.values())
    ordered_keys = [key for key in registered if key in found]
    ordered_keys += [key for key in sorted(found) if key not in registered]

    runs = []
    for key in ordered_keys:
        candidates = sorted(found[key], key=lambda run: os.path.getmtime(run.path))
        if all_runs:
            runs.extend(candidates)
            continue
        if len(candidates) > 1:
            print(
                "[INFO] %s has %d runs; using the most recent (%s)"
                % (key, len(candidates), candidates[-1].stamp)
            )
        runs.append(candidates[-1])
    return runs


# --------------------------------------------------------------------------
# Provenance: what produced this run (P14)
# --------------------------------------------------------------------------
#
# Before this, NOTHING recorded which pipeline produced a run. Attribution rested
# entirely on the results-directory name -- scripts/compare_models.py carried a
# comment saying exactly that -- which is why the P29 attribution experiment had
# to harvest into a config-named root after *each* batch: both batches emit
# identically-named Prediction_Output_<Model>_<stamp> directories into one cwd,
# so sorting them afterwards would have meant inferring the config from timestamp
# order alone, in the one experiment whose entire purpose is attribution.
#
# Two decisions here are load-bearing.
#
# The pipeline is recorded as its NAME *and* all three of its fields. A name with
# no reader was the P4 lesson (`--pipeline drg` was selectable, printed a
# plausible banner, and produced cosine numbers filed as DRG ones); a name with
# no fields is its sibling. If the registry is ever edited, the fields still say
# what ran, and `name` degrades to null rather than to a lie.
#
# The fold split is recorded by CONTENT HASH, not by directory name.
# ``data/folds_grouped/`` is gitignored and regenerated by scripts/make_folds.py,
# so "fold_dir: folds_grouped" alone does not identify a split -- the git SHA
# cannot cover a file git never saw. The hash is what makes a corrected or DRG
# run reproducible at all.


def _git_state(repo_root):
    """``(sha, dirty)`` for the working tree at ``repo_root``.

    ``(None, None)`` when git could not be asked -- not installed, not a working
    tree, or a non-zero exit. ``dirty`` is deliberately null rather than False in
    that case, the same distinction :func:`_git_ignores` draws with its
    ``answered`` flag: "clean" and "don't know" are different claims, and a
    provenance record that asserts the first when it means the second is worse
    than one that admits the gap.

    Never raises. A finished 13-minute baseline run must not die here.
    """
    def _git(*args):
        result = subprocess.run(
            ["git", "-C", repo_root] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return None
        return result.stdout.decode("utf-8", "replace")

    try:
        sha = _git("rev-parse", "HEAD")
        if sha is None:
            return None, None
        status = _git("status", "--porcelain")
        if status is None:
            return sha.strip(), None
        return sha.strip(), bool(status.strip())
    except Exception:
        return None, None


def fold_dir_digest(fold_dir, repo_root=None):
    """SHA-256 over the fold split ``fold_dir`` names, or ``None`` if absent.

    ``fold_dir`` is the ``PipelineConfig`` field -- ``"folds"``,
    ``"folds_grouped"`` -- resolved against ``<repo_root>/data/`` exactly as
    ``cython_utils.load_dataset`` resolves it.

    Only the files the pipeline actually reads contribute -- the two names
    ``cython_utils.load_dataset`` opens, ``Constants.TRAIN`` and
    ``Constants.TEST``. They are taken from ``Constants`` rather than spelled
    again here so the allowlist cannot drift away from the reader. Each
    contributes in sorted-relative-path order, as ``path bytes`` then ``file
    bytes``. The path is hashed alongside the content so that renaming
    ``Fold3`` to ``Fold7`` changes the digest: which admission sits in which
    fold IS the split, and a content-only hash would call two different
    experiments identical.

    **The allowlist is what makes the digest portable, and it was not always
    here.** Hashing *every* file under the directory pulled in whatever the
    host filesystem had left lying about: every ``data/folds/Fold*/`` on the
    owner's Windows checkout carries a hidden 105-byte ``desktop.ini`` that git
    has never tracked, so the byte-identical legacy split hashed to
    ``b7554079...`` there and ``93f0943a...`` on the Linux pod -- and this
    project's whole cross-machine story is Windows dev to RunPod Linux. It
    failed the other way too: Explorer rewrites ``desktop.ini`` when a folder
    view changes, so an untouched split would start reporting itself as a
    different one. Silently, in the one field whose entire job is to say
    whether two runs saw the same data. Relative paths, forward slashes, and
    binary reads do the rest of the cross-platform work; they were never the
    part that was broken.

    ``None`` means the directory is not there -- ordinary for ``folds_grouped``,
    which is gitignored and only exists once ``scripts/make_folds.py`` has run.
    """
    from aicds.utils.Constants import TEST, TRAIN

    root = REPO_ROOT if repo_root is None else repo_root
    resolved = os.path.join(root, "data", fold_dir)
    if not os.path.isdir(resolved):
        return None

    hashed_names = (TRAIN, TEST)
    relative_paths = []
    for directory, _subdirs, files in os.walk(resolved):
        for name in files:
            if name not in hashed_names:
                continue
            full = os.path.join(directory, name)
            relative_paths.append(os.path.relpath(full, resolved).replace(os.sep, "/"))

    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        digest.update(relative.encode("utf-8"))
        with open(os.path.join(resolved, relative.replace("/", os.sep)), "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def build_run_metadata(config, model_name, stamp, repo_root=None):
    """The ``run_metadata.json`` payload, as a plain dict. Writes nothing.

    Split out from :func:`write_run_metadata` so it can be asserted on directly,
    and so the never-raises wrapper has exactly one place to wrap.
    """
    from aicds.config import name_of
    from aicds.utils.Constants import K_FOLD

    root = REPO_ROOT if repo_root is None else repo_root
    notes = []

    sha, dirty = _git_state(root)
    if sha is None:
        notes.append("git_unavailable")

    digest = fold_dir_digest(config.fold_dir, repo_root=root)
    if digest is None:
        # Not an error: folds_grouped/ is regenerated output. Recorded so a
        # reader can tell "the split was not hashed" from "the split hashed to
        # nothing", which is the same distinction the git branch above draws.
        notes.append("fold_dir_missing")

    return {
        "metadata_version": METADATA_VERSION,
        "git": {"sha": sha, "dirty": dirty},
        "pipeline": {
            # name may be null: a config built by hand is still fully described
            # by the three fields below it.
            "name": name_of(config),
            "fold_dir": config.fold_dir,
            "preprocess_version": config.preprocess_version,
            "grader": config.grader,
        },
        "fold_dir_sha256": digest,
        "model": {"key": model_key(model_name), "label": model_name},
        "stamp": stamp,
        "k_fold": K_FOLD,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": _numpy_version(),
        },
        "notes": notes,
    }


def _numpy_version():
    """numpy's version string, or ``None``. Imported here, not at module scope.

    ``aicds.runs`` is imported by the two arms *and* by every reporting script;
    it has no numeric work of its own and should not drag a numeric stack in to
    answer a provenance question.
    """
    try:
        import numpy

        return numpy.__version__
    except Exception:
        return None


def write_run_metadata(run_dir, config, model_name, stamp, repo_root=None):
    """Write ``<run_dir>/run_metadata.json``. Returns its path, or ``None``.

    Call this LAST, after the handle that owns ``PerformanceIndex.txt`` has
    closed -- the same after-the-close discipline as ``RankAccumulator.write``,
    and for the same reason. ``test_golden.strip_trailer`` raises "No timing
    trailer found" if that file lacks its trailer, so anything that could throw
    before the trailer is written surfaces as a GOLDEN FAILURE THAT LOOKS LIKE
    THE PIPELINE CHANGED. Writing it last has a second payoff: the file's
    presence means the run reached the end, so a reader can tell a completed run
    from an interrupted one without parsing anything.

    **This function never raises.** Every failure -- git missing, an unwritable
    directory, an unregistered model name, a fold directory that vanished
    mid-run -- prints ``[WARN] run_metadata.json not written: <err>`` and
    returns ``None``. A run whose numbers are already on disk must not be lost
    to a subprocess call about provenance. The warning is loud enough to notice
    and cheap enough to ignore, which is the correct trade for a file that
    describes the run rather than being it.
    """
    try:
        payload = build_run_metadata(config, model_name, stamp, repo_root=repo_root)
        path = os.path.join(run_dir, RUN_METADATA)
        # newline="\n" so the bytes are identical on Windows and Linux: this file
        # is a provenance record that may well be compared across the two.
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path
    except Exception as error:
        print("[WARN] run_metadata.json not written: %s" % (error,))
        return None

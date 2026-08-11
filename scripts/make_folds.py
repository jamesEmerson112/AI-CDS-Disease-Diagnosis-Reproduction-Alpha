#!/usr/bin/env python3
"""Generate leakage-free cross-validation folds by grouping on SUBJECT_ID.

Why this exists
---------------
The committed folds under ``data/folds/`` split on HADM_ID, but the 129 admissions
come from only 100 patients (SUBJECT_ID 41976 alone holds 15 of them).  41 of the
129 test cases (31.8%) therefore have *another admission from the same patient*
sitting in their own retrieval pool.  Measured inflation from that leakage is
+0.11 to +0.26, against encoder differences of 0.015-0.046 -- roughly 10x the
effect the study is trying to measure.  See ``docs/findings/05-patient-leakage.md``.

``sklearn.model_selection.GroupKFold`` keeps every admission of a patient inside a
single fold, so no test case can ever retrieve itself-by-proxy.

Output
------
``data/folds_grouped/Fold{0..9}/{TrainingSet,TestSet}.txt``, in the exact on-disk
format ``load_dataset`` (``src/aicds/utils/cython_utils.py``) expects:

    {HADM_ID}_{raw SYMPTOMS field verbatim}.\\n

Three format details are load-bearing and are asserted after every write:

* **The appended period.**  Verified across all 1290 lines of the legacy folds:
  ``line.partition("_")[2] == raw_SYMPTOMS + "."`` with zero exceptions, and no
  raw SYMPTOMS field already ends in ``.``.
* **LF, never CRLF.**  All 20 legacy fold files are pure LF with zero CR bytes
  despite this being a Windows checkout under ``.gitattributes`` ``* text=auto``.
  (The *raw* dataset, by contrast, is CRLF -- so line endings are normalised on
  read, which never touches field 4 because the CR sits at the end of field 5.)
* **A trailing newline on every file.**  ``cython_utils.py`` slices
  ``line[line.index("_") + 1 : len(line) - 1]``, dropping the final character
  unconditionally.  A file without a final newline silently eats the last
  symptom's last character and raises nothing.

The generated folds are a deterministic pure function of committed data plus this
committed script, so they are *not* committed -- ``data/folds_grouped/`` is
gitignored.  Committing them would add ~1290 more lines of DUA-covered MIMIC-III
text to a public repository.

Note that ``.githooks/pre-commit`` will **not** save you here, despite catching
the equivalent mistake under ``data/folds/``.  Its path rule is
``^data/(raw|folds)/``, which requires a literal ``/`` straight after ``folds``
and so never matches ``folds_grouped/``; and its 20-distinct-HADM_ID fallback
greps for a literal ``HADM_ID[=: ]*\\d{6}`` token, which fold lines
(``123456_symptom,...``) do not contain -- measured 0 matches in a 114-line
file.  The .gitignore entry is the only thing standing between ``git add -A``
and a DUA violation.

Usage
-----
Run from the repository root::

    python scripts/make_folds.py
    python scripts/make_folds.py --verify
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fallback for a plain checkout where the package has not been pip-installed.
src_root = os.path.join(project_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from aicds.entity.SymptomsDiagnosis import SymptomsDiagnosis  # noqa: E402
from aicds.utils.Constants import K_FOLD  # noqa: E402

DEFAULT_DATA = "data/raw/Symptoms-Diagnosis.txt"
DEFAULT_OUT = "data/folds_grouped"

# The split of record (finding 14): digest of the folds_grouped every
# committed results tree was produced on. numpy 2.0.x reproduces it;
# numpy 1.x produces a different, equally valid split. Keep in sync with
# tests/test_populations.py::CANONICAL_FOLDS_GROUPED_DIGEST.
CANONICAL_FOLDS_GROUPED_DIGEST = (
    "b36f721638987fe0280393a28983a90c65127a04ffe21bcd2341024a6ec5084f"
)

# data/folds/ holds the committed LEGACY folds.  They are inputs to
# tests/golden/stub768/, so overwriting them makes the byte-exact golden
# impossible to re-verify -- and the golden is the only regression protection
# this repo has.  Guarded by --force-legacy rather than left to care.
PROTECTED_OUT = "data/folds"

FIELD_SEPARATOR = ";"
N_FIELDS = 6
ID_SEPARATOR = "_"
LINE_SUFFIX = "."


class Record(object):
    """One admission, carrying only what fold generation needs."""

    __slots__ = ("hadm_id", "subject_id", "symptoms")

    def __init__(self, hadm_id, subject_id, symptoms):
        self.hadm_id = hadm_id
        self.subject_id = subject_id
        self.symptoms = symptoms

    def to_line(self):
        """Render the fold-file representation, sans newline."""
        return self.hadm_id + ID_SEPARATOR + self.symptoms + LINE_SUFFIX


def read_records(path):
    """Parse the raw dataset, preserving file order.

    Reads in universal-newline text mode: the raw file is CRLF-terminated, and
    the CR lands at the end of field 5 (DIAGNOSIS), never inside field 4
    (SYMPTOMS), which is copied verbatim.
    """
    with open(path, "r", encoding="utf-8", newline=None) as handle:
        text = handle.read()

    records = []
    seen = set()
    for lineno, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue  # the file has no trailing newline, but tolerate one

        fields = line.split(FIELD_SEPARATOR)
        if len(fields) != N_FIELDS:
            raise ValueError(
                "%s line %d: expected %d %r-separated fields, found %d"
                % (path, lineno, N_FIELDS, FIELD_SEPARATOR, len(fields))
            )

        hadm_id = fields[SymptomsDiagnosis.CONST_HADM_ID]
        subject_id = fields[SymptomsDiagnosis.CONST_SUBJECT_ID]
        symptoms = fields[SymptomsDiagnosis.CONST_SYMPTOMS]

        # An "_" anywhere in either field would break the reader's ID parse,
        # which splits on the FIRST "_" and keeps everything after it.
        if ID_SEPARATOR in hadm_id:
            raise ValueError(
                "%s line %d: HADM_ID %r contains %r; the fold-file ID parse "
                "would truncate it" % (path, lineno, hadm_id, ID_SEPARATOR)
            )
        if ID_SEPARATOR in symptoms:
            raise ValueError(
                "%s line %d: SYMPTOMS for HADM_ID %s contains %r; the fold-file "
                "ID parse would mis-split the line"
                % (path, lineno, hadm_id, ID_SEPARATOR)
            )
        if hadm_id in seen:
            raise ValueError("%s line %d: duplicate HADM_ID %s" % (path, lineno, hadm_id))
        seen.add(hadm_id)

        records.append(Record(hadm_id, subject_id, symptoms))

    if not records:
        raise ValueError("%s: no records parsed" % path)
    return records


def make_splits(records, n_splits):
    """Return ``[(train_idx, test_idx), ...]``, each sorted into raw-file order.

    ``GroupKFold`` is used in its plain no-arg form on purpose: ``shuffle=`` and
    ``random_state=`` need sklearn >= 1.6, while the dependency manifests floor
    at >= 0.23.0 -- and an unshuffled split is deterministic, which is what a
    reproduction study wants anyway.
    """
    from sklearn.model_selection import GroupKFold

    groups = [record.subject_id for record in records]
    n_groups = len(set(groups))
    if n_groups < n_splits:
        raise ValueError(
            "cannot build %d folds from %d distinct SUBJECT_IDs" % (n_splits, n_groups)
        )

    indices = list(range(len(records)))
    splitter = GroupKFold(n_splits=n_splits)
    return [
        (sorted(train_idx.tolist()), sorted(test_idx.tolist()))
        for train_idx, test_idx in splitter.split(indices, groups=groups)
    ]


def write_fold_file(path, records):
    """Write one fold file as LF text with a guaranteed trailing newline."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.to_line() + "\n")

    # Re-read the bytes rather than trusting the write: both of these
    # invariants fail silently downstream instead of raising.
    with open(path, "rb") as handle:
        blob = handle.read()
    if b"\r" in blob:
        raise AssertionError("%s: contains CR bytes; fold files must be pure LF" % path)
    if not blob.endswith(b"\n"):
        raise AssertionError(
            "%s: missing trailing newline; load_dataset would drop the last "
            "symptom's final character" % path
        )


def _same_path(a, b):
    """Compare two paths tolerantly enough for Windows and for '..' segments."""
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def existing_fold_dirs(out_dir):
    """Fold*/ directories already present under out_dir, so we never clobber blind."""
    if not os.path.isdir(out_dir):
        return []
    return sorted(
        name
        for name in os.listdir(out_dir)
        if name.startswith("Fold") and os.path.isdir(os.path.join(out_dir, name))
    )


def check_out_dir(out_dir, force_legacy=False, overwrite=False):
    """Return an error string if writing to out_dir would destroy something.

    Two separate hazards, deliberately gated by two separate flags:

    * out_dir IS the committed legacy fold set.  Nothing about a GroupKFold run
      should ever land there; that set is a fixed historical artifact and an
      input to the golden.
    * out_dir already holds folds from an earlier run.  Regenerating with a
      different --n-splits leaves the surplus Fold*/ directories behind, so the
      result is a silent mixture of two runs rather than a clean overwrite.
    """
    if _same_path(out_dir, PROTECTED_OUT) and not force_legacy:
        return (
            "refusing to write to %s -- those are the committed LEGACY folds and "
            "inputs to tests/golden/stub768/. Overwriting them means the "
            "byte-exact golden can never be re-verified.\n"
            "Write somewhere else (the default is %s), or pass --force-legacy if "
            "you genuinely mean to replace the legacy fold set." % (PROTECTED_OUT, DEFAULT_OUT)
        )

    present = existing_fold_dirs(out_dir)
    if present and not overwrite:
        return (
            "refusing to write to %s -- it already contains %d fold director%s "
            "(%s%s).\n"
            "Re-run with --overwrite to replace them. Note that fold directories "
            "beyond the new --n-splits are NOT removed, so a smaller run leaves "
            "stale folds behind." % (
                out_dir,
                len(present),
                "y" if len(present) == 1 else "ies",
                ", ".join(present[:4]),
                ", ..." if len(present) > 4 else "",
            )
        )
    return None


def write_folds(records, splits, out_dir):
    """Materialise every fold directory; returns the list of directories."""
    written = []
    for fold_index, (train_idx, test_idx) in enumerate(splits):
        fold_dir = os.path.join(out_dir, "Fold%d" % fold_index)
        os.makedirs(fold_dir, exist_ok=True)
        write_fold_file(
            os.path.join(fold_dir, "TrainingSet.txt"), [records[i] for i in train_idx]
        )
        write_fold_file(
            os.path.join(fold_dir, "TestSet.txt"), [records[i] for i in test_idx]
        )
        written.append(fold_dir)
    return written


def _read_fold_ids(path):
    """Read back the HADM_IDs a generated fold file contains, in file order."""
    with open(path, "r", encoding="utf-8", newline="\n") as handle:
        text = handle.read()
    return [line.partition(ID_SEPARATOR)[0] for line in text.split("\n") if line]


def verify(records, out_dir, n_splits):
    """Re-read the generated files and report sizes plus residual leakage.

    Deliberately parses the artefact on disk rather than the in-memory split, so
    a format or write bug shows up here instead of surviving to a training run.
    Returns True when every check passes.
    """
    subject_of = {record.hadm_id: record.subject_id for record in records}
    ok = True

    print("fold   n_train   n_test   leaked   test SUBJECT_IDs")
    print("----   -------   ------   ------   ----------------")

    all_test_ids = []
    total_leaked = 0
    for fold_index in range(n_splits):
        fold_dir = os.path.join(out_dir, "Fold%d" % fold_index)
        train_ids = _read_fold_ids(os.path.join(fold_dir, "TrainingSet.txt"))
        test_ids = _read_fold_ids(os.path.join(fold_dir, "TestSet.txt"))
        all_test_ids.extend(test_ids)

        train_subjects = set(subject_of[h] for h in train_ids)
        test_subjects = set(subject_of[h] for h in test_ids)
        leaked = sum(1 for h in test_ids if subject_of[h] in train_subjects)
        total_leaked += leaked

        if leaked:
            ok = False
        if set(train_ids) & set(test_ids):
            print("  FAIL Fold%d: HADM_ID present in both train and test" % fold_index)
            ok = False
        if len(train_ids) + len(test_ids) != len(records):
            print(
                "  FAIL Fold%d: train+test = %d, expected %d"
                % (fold_index, len(train_ids) + len(test_ids), len(records))
            )
            ok = False

        print(
            "%4d   %7d   %6d   %6d   %16d"
            % (fold_index, len(train_ids), len(test_ids), leaked, len(test_subjects))
        )

    print("")
    print("total test cases across folds : %d (expected %d)" % (len(all_test_ids), len(records)))
    if len(all_test_ids) != len(records):
        print("  FAIL: test cases do not sum to the dataset size")
        ok = False

    duplicates = len(all_test_ids) - len(set(all_test_ids))
    print("duplicate test HADM_IDs       : %d (expected 0)" % duplicates)
    if duplicates:
        ok = False

    missing = set(subject_of) - set(all_test_ids)
    print("admissions never tested       : %d (expected 0)" % len(missing))
    if missing:
        ok = False

    print("leaked test cases (all folds) : %d (expected 0)" % total_leaked)
    print(
        "  (the legacy HADM_ID-split folds in data/folds/ score 41 here -- that "
        "is the bug this script fixes)"
    )

    # Finding 14 (P42): GroupKFold's tie-break among the ~85 single-admission
    # subjects routes through np.argsort, whose unstable-sort behaviour changed
    # between numpy 1.x and 2.x -- identical inputs produce a DIFFERENT (but
    # equally valid, still leak-free) assignment per numpy major. Every
    # committed results tree was produced on the numpy-2.0 split, so that one
    # is canonical, recorded here as a hash (the fold files themselves are
    # DUA-covered and gitignored; their digest is not). A mismatch is a
    # warning, not a FAIL: the split this environment built is internally
    # sound, it just is not the split of record, and no number computed on it
    # is comparable with the committed trees. tests/test_populations.py pins
    # the same constant.
    from aicds.runs import fold_dir_digest

    digest = fold_dir_digest(os.path.basename(out_dir.rstrip("/\\")))
    print("")
    print("fold content digest           : %s" % digest)
    if digest != CANONICAL_FOLDS_GROUPED_DIGEST:
        print(
            "  WARNING: NOT the canonical split every committed result used\n"
            "  (canonical %s,\n"
            "   produced by numpy 2.0.x; numpy 1.x ties argsort differently --\n"
            "   see docs/findings/14-fold-split-environment-dependence.md)"
            % CANONICAL_FOLDS_GROUPED_DIGEST
        )
    print("")
    print("VERIFY: %s" % ("PASS" if ok else "FAIL"))
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate GroupKFold-on-SUBJECT_ID cross-validation folds."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="raw ';'-delimited dataset (default: %s)" % DEFAULT_DATA,
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="output directory for Fold{0..N-1}/ (default: %s)" % DEFAULT_OUT,
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=K_FOLD,
        help="number of folds (default: %d, from aicds.utils.Constants.K_FOLD)" % K_FOLD,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="after writing, re-read the files and report per-fold sizes and leakage",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow writing into a directory that already contains Fold*/ dirs",
    )
    parser.add_argument(
        "--force-legacy",
        action="store_true",
        help="allow writing into %s, the committed legacy folds. Almost certainly "
        "not what you want: they are inputs to the byte-exact golden." % PROTECTED_OUT,
    )
    args = parser.parse_args(argv)

    problem = check_out_dir(args.out, args.force_legacy, args.overwrite)
    if problem is not None:
        sys.stderr.write("ERROR: %s\n" % problem)
        return 2

    records = read_records(args.data)
    n_subjects = len(set(record.subject_id for record in records))
    print("read %d admissions from %d patients (%s)" % (len(records), n_subjects, args.data))

    splits = make_splits(records, args.n_splits)
    write_folds(records, splits, args.out)
    print("wrote %d folds to %s/" % (len(splits), args.out.rstrip("/")))
    print(
        "fold sizes are UNEVEN by design: whole patients stay together, and one "
        "patient holds 15 admissions."
    )

    if args.verify:
        print("")
        if not verify(records, args.out, args.n_splits):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

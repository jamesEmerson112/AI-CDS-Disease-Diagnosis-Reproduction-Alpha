"""
Characterization tests pinning invariants of the raw dataset and fold-loading logic.

These tests exist to catch accidental changes to:
  - data/raw/Symptoms-Diagnosis.txt (129 admission records, 6 semicolon-delimited fields)
  - data/folds/Fold{0..9}/{TrainingSet,TestSet}.txt (the committed 10-fold partition)
  - src.utils.cython_utils.load_dataset's trailing-newline-strip behaviour

They pin what IS today, not what SHOULD be -- known quirks (no trailing newline on
the raw file, the unconditional last-character strip in load_dataset) are pinned
as-is so a future change (fold regeneration, dataset edits, a "fix" to load_dataset)
is caught rather than silently altering downstream results. All values below were
derived by running the actual code against the actual committed files, then
hard-coded.
"""

import os

from nltk.corpus import stopwords

from src.entity.SymptomsDiagnosis import SymptomsDiagnosis
from src.utils import cython_utils
from src.utils.Constants import CH_DIR

RAW_DATASET_PATH = os.path.join(CH_DIR, "data", "raw", "Symptoms-Diagnosis.txt")
FOLDS_DIR = os.path.join(CH_DIR, "data", "folds")
N_FOLDS = 10


def _read_raw_lines():
    """Read Symptoms-Diagnosis.txt the way any correct loader must: split on
    '\\n' and drop empty trailing entries. The file has NO trailing newline,
    so naive `wc -l` (128) undercounts the true record count (129) by one.
    """
    with open(RAW_DATASET_PATH, "rb") as f:
        raw = f.read()
    lines = raw.decode("utf-8").split("\n")
    return [line for line in lines if line != ""]


class TestRawDatasetShape:
    def test_file_has_no_trailing_newline(self):
        with open(RAW_DATASET_PATH, "rb") as f:
            raw = f.read()
        assert not raw.endswith(b"\n"), (
            "Symptoms-Diagnosis.txt now ends with a newline -- this changes "
            "what naive line-counting tools (e.g. `wc -l`) report and is "
            "worth knowing about, even though split('\\n')-based loaders are "
            "unaffected."
        )

    def test_record_count_is_129(self):
        lines = _read_raw_lines()
        assert len(lines) == 129

    def test_every_line_has_six_semicolon_delimited_fields(self):
        lines = _read_raw_lines()
        for line in lines:
            fields = line.split(";")
            assert len(fields) == 6

    def test_field_order_matches_symptomsdiagnosis_entity(self):
        # Spot-check the first record against the field order documented in
        # src/entity/SymptomsDiagnosis.py: HADM_ID, SUBJECT_ID, ADMITTIME,
        # DISCHTIME, SYMPTOMS, DIAGNOSIS.
        lines = _read_raw_lines()
        fields = lines[0].split(";")
        assert fields[SymptomsDiagnosis.CONST_HADM_ID] == "142345"
        assert fields[SymptomsDiagnosis.CONST_SUBJECT_ID] == "10006"
        assert fields[SymptomsDiagnosis.CONST_ADMITTIME] == "2164-10-23 21:09:00"
        assert fields[SymptomsDiagnosis.CONST_DISCHTIME] == "2164-11-01 17:15:00"
        assert fields[SymptomsDiagnosis.CONST_SYMPTOMS].startswith("Sepsis,")
        assert fields[SymptomsDiagnosis.CONST_DIAGNOSIS].startswith("HCFA:SEPTICEMIA")


class TestSubjectIds:
    def _subject_ids(self):
        return [
            line.split(";")[SymptomsDiagnosis.CONST_SUBJECT_ID]
            for line in _read_raw_lines()
        ]

    def test_100_distinct_subject_ids_across_129_admissions(self):
        subject_ids = self._subject_ids()
        assert len(subject_ids) == 129
        assert len(set(subject_ids)) == 100

    def test_subject_41976_holds_15_admissions(self):
        subject_ids = self._subject_ids()
        counts = {}
        for sid in subject_ids:
            counts[sid] = counts.get(sid, 0) + 1
        assert counts["41976"] == 15
        assert max(counts.values()) == 15


class TestDiagnosisPreprocessing:
    def test_145_unique_diagnosis_descriptions(self):
        descriptions = set()
        for line in _read_raw_lines():
            raw_diagnosis_field = line.split(";")[SymptomsDiagnosis.CONST_DIAGNOSIS]
            diagnosis_list = cython_utils.preprocess_diagnosis(raw_diagnosis_field)
            for d in diagnosis_list:
                description = d[d.index(":") + 1:]
                descriptions.add(description)
        assert len(descriptions) == 145


class TestFoldPartition:
    def _line_count(self, path):
        with open(path, "rb") as f:
            return f.read().count(b"\n")

    def test_fold_sizes_116_13_except_fold9_117_12(self):
        for i in range(N_FOLDS):
            train_path = os.path.join(FOLDS_DIR, f"Fold{i}", "TrainingSet.txt")
            test_path = os.path.join(FOLDS_DIR, f"Fold{i}", "TestSet.txt")
            n_train = self._line_count(train_path)
            n_test = self._line_count(test_path)

            if i == 9:
                assert n_train == 117
                assert n_test == 12
            else:
                assert n_train == 116
                assert n_test == 13
            assert n_train + n_test == 129

    def test_every_hadm_id_appears_exactly_once_across_all_test_sets(self):
        all_dataset_hadm_ids = {
            line.split(";")[SymptomsDiagnosis.CONST_HADM_ID]
            for line in _read_raw_lines()
        }

        seen = []
        for i in range(N_FOLDS):
            test_path = os.path.join(FOLDS_DIR, f"Fold{i}", "TestSet.txt")
            with open(test_path, "rb") as f:
                raw = f.read()
            lines = [l for l in raw.decode("utf-8").split("\n") if l != ""]
            for line in lines:
                seen.append(line[: line.index("_")])

        assert len(seen) == 129, "TestSet.txt files across all 10 folds should total 129 HADM_IDs"
        assert len(set(seen)) == 129, "duplicate HADM_ID across TestSet.txt files"
        assert set(seen) == all_dataset_hadm_ids, (
            "TestSet.txt partition does not exactly cover every admission in "
            "the raw dataset"
        )

    def test_fold_line_format_is_hadmid_underscore_symptoms_period_no_header(self):
        # "<HADM_ID>_<comma-separated symptoms><trailing period>", no header row.
        test_path = os.path.join(FOLDS_DIR, "Fold0", "TestSet.txt")
        with open(test_path, "rb") as f:
            raw = f.read()
        lines = [l for l in raw.decode("utf-8").split("\n") if l != ""]

        assert len(lines) == 13
        for line in lines:
            assert "_" in line
            hadm_id, _, rest = line.partition("_")
            assert hadm_id.isdigit(), f"HADM_ID prefix is not numeric: {hadm_id!r}"
            assert rest.endswith("."), f"line does not end in a trailing period: {rest[-15:]!r}"
            assert hadm_id != "HADM_ID", "fold files must not have a header row"


class TestTrailingNewlineInvariant:
    """The most load-bearing invariant in this file.

    load_dataset (src/utils/cython_utils.py ~line 172) does:

        symptoms = line[line.index("_") + 1: len(line) - 1]

    which unconditionally drops the LAST CHARACTER of every line, on the
    assumption that the line ends with '\\n'. Today that assumption holds --
    every committed fold file ends with a trailing newline byte, so the
    dropped character is always the (already-insignificant) newline.

    If a future fold-file generator ever omits the trailing newline, this
    same code silently eats the final character of real symptom text
    instead, with no error raised. The two tests below (a) pin that every
    committed fold file currently satisfies the assumption, and (b)
    demonstrate on a throwaway tmp_path file what happens the moment that
    assumption breaks.
    """

    def test_all_committed_fold_files_end_with_a_newline_byte(self):
        for i in range(N_FOLDS):
            for name in ("TrainingSet.txt", "TestSet.txt"):
                path = os.path.join(FOLDS_DIR, f"Fold{i}", name)
                with open(path, "rb") as f:
                    raw = f.read()
                assert raw.endswith(b"\n"), f"Fold{i}/{name} is missing its trailing newline"

    def test_load_dataset_strips_exactly_the_newline_when_present(self, tmp_path, monkeypatch):
        cython_utils.stop_words = set(stopwords.words("english"))

        fold_dir = tmp_path / "data" / "folds" / "Fold0"
        fold_dir.mkdir(parents=True)
        (fold_dir / "TestSet.txt").write_bytes(b"999_symptomone,symptomtwo.\n")

        monkeypatch.setattr(cython_utils, "CH_DIR", str(tmp_path))
        dataset = cython_utils.load_dataset(0, "TestSet.txt")

        # With a trailing newline, the dropped character IS the newline, so
        # both symptoms survive intact (their trailing periods are stripped
        # by preprocess_sentence's punctuation filter, not by the slice).
        assert dataset == [{"999": ["symptomone", "symptomtwo"]}]

    def test_missing_trailing_newline_silently_loses_the_final_character(self, tmp_path, monkeypatch):
        """This is the regression this whole class exists to catch: a fold
        file generator that forgets the trailing newline does not raise --
        it silently corrupts the last symptom of the last line.
        """
        cython_utils.stop_words = set(stopwords.words("english"))

        fold_dir = tmp_path / "data" / "folds" / "Fold0"
        fold_dir.mkdir(parents=True)
        # No trailing '\n' here -- the real character 'z' is what gets eaten.
        (fold_dir / "TestSet.txt").write_bytes(b"999_symptomone,symptomz")

        monkeypatch.setattr(cython_utils, "CH_DIR", str(tmp_path))
        dataset = cython_utils.load_dataset(0, "TestSet.txt")

        # 'symptomz' loses its trailing 'z' and becomes 'symptom'.
        assert dataset == [{"999": ["symptomone", "symptom"]}]
        assert dataset != [{"999": ["symptomone", "symptomz"]}]

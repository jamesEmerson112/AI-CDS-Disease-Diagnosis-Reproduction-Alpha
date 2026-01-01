"""
Verification tests to ensure project reorganization is correct.
Run after reorganization: python -m pytest tests/test_reorganization.py -v
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestImports:
    """Verify all modules can be imported correctly."""

    def test_import_entity_modules(self):
        from src.entity.SymptomsDiagnosis import SymptomsDiagnosis
        from src.entity.Admission import Admission
        from src.entity.Symptom import Symptom
        from src.entity.Drgcodes import Drgcodes

    def test_import_utils_modules(self):
        from src.utils.Constants import CH_DIR, K_FOLD, TOP_K_LOWER_BOUND
        from src.utils import cython_utils

    def test_import_constants_values(self):
        from src.utils.Constants import K_FOLD, PRUNING_SIMILARITY
        assert K_FOLD == 10
        assert PRUNING_SIMILARITY == 0.5


class TestDataPaths:
    """Verify data files are accessible at new locations."""

    def test_folds_directory_exists(self):
        folds_path = os.path.join(project_root, "data", "folds")
        assert os.path.isdir(folds_path), "data/folds/ directory missing"

    def test_all_folds_present(self):
        for i in range(10):
            fold_path = os.path.join(project_root, f"data/folds/Fold{i}")
            assert os.path.isdir(fold_path), f"Fold{i} missing"

    def test_fold_files_present(self):
        for i in range(10):
            train = os.path.join(project_root, f"data/folds/Fold{i}/TrainingSet.txt")
            test = os.path.join(project_root, f"data/folds/Fold{i}/TestSet.txt")
            assert os.path.isfile(train), f"TrainingSet.txt missing in Fold{i}"
            assert os.path.isfile(test), f"TestSet.txt missing in Fold{i}"

    def test_raw_data_exists(self):
        raw_path = os.path.join(project_root, "data/raw/Symptoms-Diagnosis.txt")
        assert os.path.isfile(raw_path), "Symptoms-Diagnosis.txt missing"


class TestDirectoryStructure:
    """Verify all expected directories exist."""

    def test_src_structure(self):
        dirs = ["src", "src/models", "src/entity", "src/utils", "src/evaluation"]
        for d in dirs:
            path = os.path.join(project_root, d)
            assert os.path.isdir(path), f"{d}/ missing"

    def test_other_directories(self):
        dirs = ["scripts", "tests", "data", "output", "docs", "config", "archive"]
        for d in dirs:
            path = os.path.join(project_root, d)
            assert os.path.isdir(path), f"{d}/ missing"

    def test_output_subdirectories(self):
        dirs = ["output/baseline", "output/bio_clinical_bert", "output/biomedbert", "output/bluebert"]
        for d in dirs:
            path = os.path.join(project_root, d)
            assert os.path.isdir(path), f"{d}/ missing"


class TestConfigFiles:
    """Verify config files are in place."""

    def test_requirements_files(self):
        assert os.path.isfile(os.path.join(project_root, "config/requirements.txt"))
        assert os.path.isfile(os.path.join(project_root, "config/requirements_bert.txt"))

    def test_environment_yml(self):
        assert os.path.isfile(os.path.join(project_root, "config/environment.yml"))


class TestConstantsPathResolution:
    """Verify Constants.py correctly resolves paths after move."""

    def test_ch_dir_valid(self):
        from src.utils.Constants import CH_DIR
        assert os.path.isdir(CH_DIR), f"CH_DIR={CH_DIR} is not valid"

    def test_ch_dir_is_project_root(self):
        from src.utils.Constants import CH_DIR
        # CH_DIR should point to project root
        assert os.path.isfile(os.path.join(CH_DIR, "pyproject.toml")), "CH_DIR should be project root"

    def test_can_find_folds_from_constants(self):
        from src.utils.Constants import CH_DIR
        folds_path = os.path.join(CH_DIR, "data", "folds")
        assert os.path.isdir(folds_path), "Cannot find folds from CH_DIR"


class TestEntityClasses:
    """Verify entity classes work correctly."""

    def test_symptoms_diagnosis_instantiation(self):
        from src.entity.SymptomsDiagnosis import SymptomsDiagnosis
        sd = SymptomsDiagnosis()
        assert hasattr(sd, 'symptoms')
        assert hasattr(sd, 'diagnosis')

    def test_admission_instantiation(self):
        from src.entity.Admission import Admission
        adm = Admission()
        assert hasattr(adm, 'hadm_id')

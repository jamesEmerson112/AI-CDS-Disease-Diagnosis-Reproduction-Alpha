"""Tests for cython_utils.split_symptoms -- FIX 3b, the comma-split fragments.

Background (docs/findings/06-preprocessing-defects.md defect 2): the SYMPTOMS
field is ','-delimited, but ICD-9 short titles contain commas of their own, so
'Pneumonia, organism NOS' reaches the pipeline as 'Pneumonia' plus an orphan
' organism NOS'. 80 of the committed file's 1,805 symptom tokens (4.4%) are
such orphans. They do two kinds of damage: severity distinctions collapse
('Pressure ulcer, stage I' and '... stage IV' both become 'Pressure ulcer'),
and the fragments are embedded as if they were symptoms -- ' organism NOS'
appears in 26 admissions, and since patient similarity is mean-of-max, any two
of those 26 receive a spurious 1.0 contribution from a token carrying no
clinical content.

The recovery rule is exact rather than heuristic: the field separator is a bare
',' while every intra-label comma is followed by a space, so a part beginning
with a space can only be the tail of the part before it.

Two halves to this file, and the split matters:

  * LEGACY tests pin `text.split(',')` exactly. They exist because the golden
    reference (tests/golden/stub768/PerformanceIndex.txt) was minted through
    this path and must stay bit-identical forever.
  * CORRECTED tests assert the repair, including the headline acceptance
    number measured against the committed dataset: 573 unique symptom strings
    -> 564.

Expected values were obtained by running the code and reading its output,
never by calling the function again inside an assertion.
"""

import os

import pytest
from nltk.corpus import stopwords

from aicds.config import CORRECTED, LEGACY, PipelineConfig
from aicds.entity.SymptomsDiagnosis import SymptomsDiagnosis
from aicds.utils import cython_utils
from aicds.utils.Constants import CH_DIR

RAW_DATASET_PATH = os.path.join(CH_DIR, "data", "raw", "Symptoms-Diagnosis.txt")

# Same incantation the other characterization tests use; preprocess_sentence
# reads this module-level global.
cython_utils.stop_words = set(stopwords.words("english"))


def _symptom_fields():
    """The SYMPTOMS field of all 129 committed admissions, unmodified."""
    with open(RAW_DATASET_PATH, "r") as f:
        lines = f.readlines()
    return [line.split(";")[SymptomsDiagnosis.CONST_SYMPTOMS] for line in lines]


def _unique_symptom_strings(config):
    """Reproduce what the pipeline actually embeds: every SYMPTOMS field split
    and preprocessed under `config`, deduplicated by string.

    This is the same sequence as bert_models.compute_bert_symptom_embeddings
    and cython_utils.embending_symptoms, so the count it returns is literally
    the number of vectors either arm computes.
    """
    unique = set()
    total = 0
    for field in _symptom_fields():
        for symptom in cython_utils.split_symptoms(field, config):
            total += 1
            unique.add(cython_utils.preprocess_sentence(symptom, config))
    return unique, total


class TestSplitSymptomsLegacy:
    """LEGACY is a plain comma split -- pinned, because the golden ran it."""

    def test_default_argument_is_legacy(self):
        text = "Pneumonia, organism NOS,Fever"
        assert cython_utils.split_symptoms(text) == cython_utils.split_symptoms(text, LEGACY)

    def test_legacy_is_exactly_str_split(self):
        for text in [
            "Pneumonia, organism NOS,Pressure ulcer, stage IV,Fever",
            "fever,cough",
            "no commas here",
            "",
        ]:
            assert cython_utils.split_symptoms(text, LEGACY) == text.split(","), text

    def test_legacy_shreds_labels_containing_commas(self):
        # The defect itself, pinned. ' organism NOS' is the orphan.
        assert cython_utils.split_symptoms("Pneumonia, organism NOS,Fever", LEGACY) == [
            "Pneumonia",
            " organism NOS",
            "Fever",
        ]

    def test_legacy_collapses_severity(self):
        # Both pressure ulcers reduce to the identical token 'Pressure ulcer';
        # the stage is stranded in a fragment.
        assert cython_utils.split_symptoms(
            "Pressure ulcer, stage I,Pressure ulcer, stage IV", LEGACY
        ) == ["Pressure ulcer", " stage I", "Pressure ulcer", " stage IV"]


class TestSplitSymptomsCorrected:
    """CORRECTED rejoins any part beginning with a space onto its predecessor."""

    def test_fragment_is_rejoined(self):
        assert cython_utils.split_symptoms("Pneumonia, organism NOS,Fever", CORRECTED) == [
            "Pneumonia, organism NOS",
            "Fever",
        ]

    def test_severity_is_preserved(self):
        # The two ulcers are now distinct strings, so stage I and stage IV no
        # longer embed to the same vector.
        result = cython_utils.split_symptoms(
            "Pressure ulcer, stage I,Pressure ulcer, stage IV", CORRECTED
        )
        assert result == ["Pressure ulcer, stage I", "Pressure ulcer, stage IV"]
        assert result[0] != result[1]

    def test_multiple_labels_each_recover(self):
        assert cython_utils.split_symptoms(
            "Ac kidny fail, tubr necr,Dysphagia, oropharyngeal", CORRECTED
        ) == ["Ac kidny fail, tubr necr", "Dysphagia, oropharyngeal"]

    def test_consecutive_fragments_chain_onto_one_label(self):
        # A label with two internal commas produces two orphans in a row;
        # both must land on the same predecessor, not on each other.
        assert cython_utils.split_symptoms(
            "Ac kidny fail, tubr necr, tubr necr,Fever", CORRECTED
        ) == ["Ac kidny fail, tubr necr, tubr necr", "Fever"]

    def test_separator_without_a_following_space_still_splits(self):
        # The rule keys on the space, not the comma -- ordinary ','-separated
        # symptoms are unaffected.
        assert cython_utils.split_symptoms("fever,cough", CORRECTED) == ["fever", "cough"]

    def test_no_commas_is_a_single_symptom(self):
        assert cython_utils.split_symptoms("no commas here", CORRECTED) == ["no commas here"]

    def test_leading_space_with_no_predecessor_is_kept(self):
        # A first part cannot be a continuation of anything, so it survives as
        # its own symptom rather than being silently dropped. No SYMPTOMS field
        # in the committed file starts with a space (verified: 0 of 129), so
        # this pins defensive behaviour rather than an observed case.
        assert cython_utils.split_symptoms(" leading fragment,real", CORRECTED) == [
            " leading fragment",
            "real",
        ]

    def test_correction_is_a_no_op_when_there_are_no_fragments(self):
        for text in ["fever,cough", "no commas here", "a,b,c", ""]:
            assert cython_utils.split_symptoms(text, CORRECTED) == cython_utils.split_symptoms(
                text, LEGACY
            ), text


class TestSplitSymptomsVersionValidation:
    def test_unknown_version_raises(self):
        bogus = PipelineConfig(preprocess_version="corrrected")
        with pytest.raises(ValueError) as excinfo:
            cython_utils.split_symptoms("Pneumonia, organism NOS", bogus)
        assert "corrrected" in str(excinfo.value)


class TestSplitSymptomsOnTheCommittedDataset:
    """The acceptance measurement, run against data/raw/Symptoms-Diagnosis.txt.

    docs/findings/06-preprocessing-defects.md predicted 573 -> 564 unique
    symptom strings and 80 fragments; these assert it end to end through the
    real API rather than a one-off script.
    """

    def test_legacy_yields_573_unique_symptom_strings(self):
        unique, total = _unique_symptom_strings(LEGACY)
        assert len(unique) == 573
        assert total == 1805

    def test_corrected_yields_564_unique_symptom_strings(self):
        unique, total = _unique_symptom_strings(CORRECTED)
        assert len(unique) == 564
        # 80 orphan fragments stop being separate tokens.
        assert total == 1725
        assert 1805 - total == 80

    def test_every_fragment_is_recovered(self):
        legacy_fragments = [
            part
            for field in _symptom_fields()
            for part in cython_utils.split_symptoms(field, LEGACY)
            if part.startswith(" ")
        ]
        corrected_fragments = [
            part
            for field in _symptom_fields()
            for part in cython_utils.split_symptoms(field, CORRECTED)
            if part.startswith(" ")
        ]
        assert len(legacy_fragments) == 80
        assert len(set(legacy_fragments)) == 18
        assert corrected_fragments == []

    def test_the_organism_nos_spurious_match_disappears(self):
        # ' organism NOS' -> 'organism nos' is the worst offender: 26
        # admissions share it, and mean-of-max hands every one of those pairs a
        # free 1.0 on a token with no clinical content.
        legacy_unique, _ = _unique_symptom_strings(LEGACY)
        corrected_unique, _ = _unique_symptom_strings(CORRECTED)
        assert "organism nos" in legacy_unique
        assert "organism nos" not in corrected_unique
        assert "pneumonia organism nos" in corrected_unique
        assert "pneumonia organism nos" not in legacy_unique


class TestLoadDatasetThreadsTheConfig:
    """load_dataset must forward `config` to BOTH split_symptoms and
    preprocess_sentence.

    Forgetting either one is silent: the fold loader and the embedding builder
    would key on different strings, every lookup would miss, and every patient
    pair would score zero rather than raising. A synthetic fold directory keeps
    this off the committed folds (and off data/folds_grouped, which is
    gitignored and may not exist on a given machine).
    """

    def _write_fold(self, tmp_path, line):
        fold_dir = tmp_path / "data" / "folds" / "Fold0"
        fold_dir.mkdir(parents=True)
        (fold_dir / "TestSet.txt").write_bytes(line)
        return fold_dir

    def test_legacy_shreds_and_destroys_negation(self, tmp_path, monkeypatch):
        cython_utils.stop_words = set(stopwords.words("english"))
        self._write_fold(
            tmp_path,
            b"999_Pneumonia, organism NOS,Tracheostomy w/o Extensive Procedure\n",
        )
        monkeypatch.setattr(cython_utils, "CH_DIR", str(tmp_path))

        assert cython_utils.load_dataset(0, "TestSet.txt", LEGACY) == [
            {"999": ["pneumonia", "organism nos", "tracheostomy w extensive procedure"]}
        ]

    def test_corrected_rejoins_and_keeps_negation(self, tmp_path, monkeypatch):
        cython_utils.stop_words = set(stopwords.words("english"))
        self._write_fold(
            tmp_path,
            b"999_Pneumonia, organism NOS,Tracheostomy w/o Extensive Procedure\n",
        )
        monkeypatch.setattr(cython_utils, "CH_DIR", str(tmp_path))

        # fold_dir stays 'folds' so this exercises preprocess_version alone --
        # CORRECTED's fold_dir='folds_grouped' is a separate fix with its own
        # tests.
        config = PipelineConfig(fold_dir="folds", preprocess_version="corrected")
        assert cython_utils.load_dataset(0, "TestSet.txt", config) == [
            {
                "999": [
                    "pneumonia organism nos",
                    "tracheostomy without extensive procedure",
                ]
            }
        ]

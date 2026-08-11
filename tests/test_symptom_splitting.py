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

Both recovery rules are HEURISTICS, and this docstring used to deny it: it
claimed the rule was "exact rather than heuristic" because the field separator
is a bare ',' while every intra-label comma is followed by a space. That is
false -- nine occurrences in the committed file use a bare comma with no
trailing space, which is why rule 1 leaves a residual at all.
cython_utils.split_symptoms:116-120 records the same correction and says the
claim must never be restored; the tests below pin the residual precisely so a
future reader cannot re-derive it.

  * RULE 1 (`corrected`): a part beginning with a space is the tail of the part
    before it. Recovers 80 of 89 fragments.
  * RULE 2 (`corrected2` only, P27): a lower-case part rejoins a capitalised
    predecessor when the join fits the 24-character CMS short-title cap.
    Recovers the remaining 9. Bounded, not exact -- see
    TestSplitSymptomsCorrected2's cost pins.

Three halves to this file, and the split matters:

  * LEGACY tests pin `text.split(',')` exactly. They exist because the golden
    reference (tests/golden/stub768/PerformanceIndex.txt) was minted through
    this path and must stay bit-identical forever.
  * CORRECTED tests assert rule 1, including the headline acceptance number
    measured against the committed dataset: 573 unique symptom strings -> 564.
    `corrected` is FROZEN -- results_corrected/ and results_drg/ were minted
    through it -- so every expectation here stays as it is. In particular
    `test_separator_without_a_following_space_still_splits` is CORRECT, not
    stale: rule 2 lives in a different variant precisely so this one keeps
    passing untouched.
  * CORRECTED2 tests assert rule 2 on top: 1,716 tokens, 561 unique, zero
    residual, firing exactly 9 times across exactly 4 labels. Those five pin
    the DATASET and are measurably blind to a widened predicate (a cap of 25,
    30 or infinity leaves all five unmoved); the PREDICATE is pinned separately
    by `test_the_cap_constant_is_pinned_at_24`, the 25-character boundary
    fixture, and the two KNOWN_COST tests.

Expected values were obtained by running the code and reading its output,
never by calling the function again inside an assertion.
"""

import os
from collections import Counter

import pytest

from aicds.config import CORRECTED, CORRECTED2, LEGACY, PIPELINE_NAMES, PipelineConfig, from_name
from aicds.entity.SymptomsDiagnosis import SymptomsDiagnosis
from aicds.utils import cython_utils
from aicds.utils.Constants import CH_DIR

RAW_DATASET_PATH = os.path.join(CH_DIR, "data", "raw", "Symptoms-Diagnosis.txt")


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


def _residual_fragments(config):
    """Count parts that are still shredded label tails under `config`.

    A fragment is a part that cannot start a label: one beginning with a space
    (rule 1's target) or with a lower-case letter (rule 2's). Every symptom in
    the committed file is a capitalised ICD-9 short title, so anything else is
    a tail that lost its head.
    """
    return sum(
        1
        for field in _symptom_fields()
        for part in cython_utils.split_symptoms(field, config)
        if part.startswith(" ") or part[:1].islower()
    )


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


class TestSplitSymptomsCorrected2:
    """CORRECTED2 = rule 1 PLUS rule 2, the bounded bare-comma rejoin (P27).

    Rule 2 fires when all three hold: the part starts lower-case, the
    accumulated predecessor starts upper-case (the ICD-9 short-title tell), and
    the join is <= 24 characters (the CMS short-title field width).
    """

    def test_rule_1_still_fires_under_corrected2(self):
        # corrected2 is a SUPERSET: it adds a rule, it removes none.
        assert cython_utils.split_symptoms("Pneumonia, organism NOS,Fever", CORRECTED2) == [
            "Pneumonia, organism NOS",
            "Fever",
        ]

    def test_the_four_residual_labels_each_round_trip(self):
        # One fixture per label in the measured residual. These four -- and only
        # these four -- are what rule 2 exists to recover.
        assert cython_utils.split_symptoms("Pressure ulcer,stage NOS", CORRECTED2) == [
            "Pressure ulcer,stage NOS"
        ]
        assert cython_utils.split_symptoms("Pressure ulcer,stage III", CORRECTED2) == [
            "Pressure ulcer,stage III"
        ]
        assert cython_utils.split_symptoms("Vascular dementia,uncomp", CORRECTED2) == [
            "Vascular dementia,uncomp"
        ]
        assert cython_utils.split_symptoms("Dysphagia,pharyngoesoph", CORRECTED2) == [
            "Dysphagia,pharyngoesoph"
        ]

    def test_corrected_leaves_all_four_shredded(self):
        # The contrast that makes the pins above mean something, and the proof
        # that `corrected` is untouched by this commit.
        for label in [
            "Pressure ulcer,stage NOS",
            "Pressure ulcer,stage III",
            "Vascular dementia,uncomp",
            "Dysphagia,pharyngoesoph",
        ]:
            assert len(cython_utils.split_symptoms(label, CORRECTED)) == 2, label

    def test_a_recovered_label_still_separates_from_the_next_symptom(self):
        assert cython_utils.split_symptoms("Pressure ulcer,stage NOS,Fever", CORRECTED2) == [
            "Pressure ulcer,stage NOS",
            "Fever",
        ]

    def test_lower_case_predecessor_still_splits(self):
        # Condition 2 (capitalised predecessor) is what keeps ordinary
        # lower-case symptom lists splitting under corrected2. Same expectation
        # as the CORRECTED test of the same name; it must not regress.
        assert cython_utils.split_symptoms("fever,cough", CORRECTED2) == ["fever", "cough"]

    def test_KNOWN_COST_capitalised_short_pair_is_over_joined(self):
        # THE ACCEPTED COST, PINNED AS ACTUAL BEHAVIOUR, NOT ASPIRATION.
        #
        # 'Fever,cough' is two genuine symptoms, and corrected2 joins them: the
        # predecessor is capitalised, the tail is lower-case, and the join is 11
        # characters -- all three conditions hold. The 24-char cap does NOT save
        # this case (an earlier write-up claimed it did; it is 11 chars, well
        # inside the cap), and neither does the upper-case condition.
        #
        # Accepted rather than fixed because no such field exists in the
        # committed data: the real-data sentinels below measure exactly 9
        # firings across exactly 4 labels, and those are the guard. A dataset
        # change that introduced a 'Fever,cough' would move those counts and
        # fail loudly.
        assert cython_utils.split_symptoms("Fever,cough", CORRECTED2) == ["Fever,cough"]
        # ... while CORRECTED, being frozen, still splits it.
        assert cython_utils.split_symptoms("Fever,cough", CORRECTED) == ["Fever", "cough"]

    def test_KNOWN_COST_a_long_bare_comma_label_is_not_recovered(self):
        # The cap cuts the other way too: a genuine bare-comma label wider than
        # the 24-char short title stays shredded. That is the price of bounding
        # the false join above, and it is why rule 2 is described as bounded
        # rather than complete.
        assert len("Acute respiratry failure,uncomp") == 31
        assert cython_utils.split_symptoms("Acute respiratry failure,uncomp", CORRECTED2) == [
            "Acute respiratry failure",
            "uncomp",
        ]

    def test_the_cap_constant_is_pinned_at_24(self):
        # THE PREDICATE PIN. The real-data measurements below do NOT constrain
        # the cap upwards -- MEASURED by mutating it: 25, 27, 30, 31, 40 and no
        # cap at all all leave 1716 tokens / 561 unique / 0 residual / the same
        # 9-firing multiset, because no part in the committed file exceeds 24
        # characters, so a wider cap has nothing extra to absorb. Only a
        # NARROWING (23 -> 1724/563/8 residual/1 firing) moves them. So the
        # constant is asserted directly rather than inferred from the data.
        assert cython_utils._SHORT_TITLE_CAP == 24

    def test_a_25_char_join_is_over_the_cap(self):
        # The boundary, chosen to sit in the blind spot the two KNOWN_COST tests
        # leave: 'Acute respiratry failure,uncomp' is 31 characters, so it only
        # catches caps >= 31, and nothing at all caught caps 25-30 before this
        # fixture (verified: the whole file passed at cap 25, 27 and 30).
        #
        # 'Pressure ulcerx' is the real 'Pressure ulcer' plus one character, so
        # the join is 25 -- one over -- and rule 2 declines it.
        assert len("Pressure ulcerx,stage NOS") == 25
        assert cython_utils.split_symptoms("Pressure ulcerx,stage NOS", CORRECTED2) == [
            "Pressure ulcerx",
            "stage NOS",
        ]
        # ... while the genuine 24-character label one character shorter joins.
        assert len("Pressure ulcer,stage NOS") == 24
        assert cython_utils.split_symptoms("Pressure ulcer,stage NOS", CORRECTED2) == [
            "Pressure ulcer,stage NOS"
        ]

    def test_a_non_alphabetic_tail_does_not_rejoin(self):
        # ''.islower() and '3'.islower() are both False, so digits and empty
        # parts fall through to the plain split rather than being absorbed.
        assert cython_utils.split_symptoms("Dysphagia,3rd degree", CORRECTED2) == [
            "Dysphagia",
            "3rd degree",
        ]
        assert cython_utils.split_symptoms("Dysphagia,", CORRECTED2) == ["Dysphagia", ""]

    def test_lower_case_first_part_has_no_predecessor_and_survives(self):
        assert cython_utils.split_symptoms("uncomp,Fever", CORRECTED2) == ["uncomp", "Fever"]

    def test_no_commas_and_empty_are_unchanged(self):
        assert cython_utils.split_symptoms("no commas here", CORRECTED2) == ["no commas here"]
        assert cython_utils.split_symptoms("", CORRECTED2) == [""]


class TestCorrected2Predicates:
    """`corrected2` is a superset for rule 1 and narrow for rule 2."""

    def test_use_corrected_preprocessing_is_true_for_both_corrected_names(self):
        assert cython_utils.use_corrected_preprocessing(CORRECTED) is True
        assert cython_utils.use_corrected_preprocessing(CORRECTED2) is True
        assert cython_utils.use_corrected_preprocessing(LEGACY) is False

    def test_use_corrected2_rejoin_is_true_for_corrected2_alone(self):
        # Every registry config is checked, so a future config that quietly
        # picked up preprocess_version='corrected2' would show up here.
        for name in PIPELINE_NAMES:
            config = from_name(name)
            expected = name == "corrected2"
            assert cython_utils.use_corrected2_rejoin(config) is expected, name

    def test_the_w_slash_o_fix_fires_under_corrected2(self):
        # The other site gated on use_corrected_preprocessing. corrected2 must
        # inherit it, or it would be corrected-minus-a-fix rather than
        # corrected-plus-a-rule.
        assert (
            cython_utils.preprocess_sentence("Tracheostomy w/o Extensive Procedure", CORRECTED2)
            == "tracheostomy without extensive procedure"
        )


class TestSplitSymptomsVersionValidation:
    def test_unknown_version_raises(self):
        bogus = PipelineConfig(preprocess_version="corrrected")
        with pytest.raises(ValueError) as excinfo:
            cython_utils.split_symptoms("Pneumonia, organism NOS", bogus)
        assert "corrrected" in str(excinfo.value)

    def test_a_near_miss_of_corrected2_also_raises(self):
        # 'corrected3' does not exist. Resolving it to either corrected arm
        # would file a run under the wrong preprocessing revision.
        bogus = PipelineConfig(preprocess_version="corrected3")
        with pytest.raises(ValueError) as excinfo:
            cython_utils.split_symptoms("Pressure ulcer,stage NOS", bogus)
        assert "corrected3" in str(excinfo.value)

    def test_both_predicates_reject_the_same_unknown_value(self):
        # They must agree about which names exist; a version accepted by one and
        # rejected by the other would be a half-corrected run.
        bogus = PipelineConfig(preprocess_version="corrected_v2")
        with pytest.raises(ValueError):
            cython_utils.use_corrected_preprocessing(bogus)
        with pytest.raises(ValueError):
            cython_utils.use_corrected2_rejoin(bogus)


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

    def test_corrected2_yields_561_unique_symptom_strings(self):
        # MEASURED on data/raw/Symptoms-Diagnosis.txt through the shipped API,
        # not predicted: 1,725 -> 1,716 tokens, 564 -> 561 unique. The nine
        # tokens that disappear are the nine bare-comma fragments.
        unique, total = _unique_symptom_strings(CORRECTED2)
        assert total == 1716
        assert len(unique) == 561
        assert 1725 - total == 9

    def test_corrected2_leaves_zero_residual_fragments(self):
        assert _residual_fragments(LEGACY) == 89
        assert _residual_fragments(CORRECTED) == 9
        assert _residual_fragments(CORRECTED2) == 0

    def test_the_lower_case_fragment_sentinel_is_exactly_nine(self):
        # THE REGRESSION SENTINEL named in split_symptoms' docstring. Under
        # CORRECTED the bare-comma residual is 9 occurrences across 4 labels;
        # if the dataset ever changes it must not silently grow, and if rule 2
        # ever widens it must not silently absorb more.
        lower_case_parts = [
            part
            for field in _symptom_fields()
            for part in cython_utils.split_symptoms(field, CORRECTED)
            if part[:1].islower()
        ]
        assert sorted(Counter(lower_case_parts).items()) == [
            ("pharyngoesoph", 1),
            ("stage III", 3),
            ("stage NOS", 4),
            ("uncomp", 1),
        ]
        assert len(lower_case_parts) == 9

    def test_rule_2_fires_on_exactly_four_labels_nine_times(self):
        # Written as an equality against the full multiset, so a rule that
        # widened -- even by one extra join anywhere in the 129 admissions --
        # FAILS rather than passing with a bigger number.
        fired = [
            part
            for field in _symptom_fields()
            for part in cython_utils.split_symptoms(field, CORRECTED2)
            if part not in cython_utils.split_symptoms(field, CORRECTED)
        ]
        assert sorted(Counter(fired).items()) == [
            ("Dysphagia,pharyngoesoph", 1),
            ("Pressure ulcer,stage III", 3),
            ("Pressure ulcer,stage NOS", 4),
            ("Vascular dementia,uncomp", 1),
        ]
        assert len(set(fired)) == 4
        assert len(fired) == 9

    def test_corrected2_changes_nothing_else_about_the_split(self):
        # Every admission whose parts rule 2 does not touch must come out of
        # corrected2 byte-identical to corrected. MEASURED: 129 fields, 9
        # differing -- one per firing, so no admission carries two joins and
        # 120 of 129 admissions are numerically unmoved by P27.
        differing = [
            field
            for field in _symptom_fields()
            if cython_utils.split_symptoms(field, CORRECTED2)
            != cython_utils.split_symptoms(field, CORRECTED)
        ]
        assert len(_symptom_fields()) == 129
        assert len(differing) == 9

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
        self._write_fold(
            tmp_path,
            b"999_Pneumonia, organism NOS,Tracheostomy w/o Extensive Procedure\n",
        )
        monkeypatch.setattr(cython_utils, "CH_DIR", str(tmp_path))

        assert cython_utils.load_dataset(0, "TestSet.txt", LEGACY) == [
            {"999": ["pneumonia", "organism nos", "tracheostomy w extensive procedure"]}
        ]

    def test_corrected_rejoins_and_keeps_negation(self, tmp_path, monkeypatch):
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

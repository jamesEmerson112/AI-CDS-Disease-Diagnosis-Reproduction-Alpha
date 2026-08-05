"""
Characterization tests for src/utils/cython_utils.py::preprocess_sentence and
::preprocess_diagnosis.

These pin CURRENT behaviour, bugs included. They must not change a single
character of what the functions under test produce; if a listed assertion
ever needs to change, that is a deliberate behaviour change to preprocessing
and should be called out as such (and the goldens under docs/ regenerated
deliberately), not an incidental side effect of refactoring.

Every expected value below was obtained by actually running the function and
reading its output -- never by calling the function again inside the
assertion. Follows tests/test_reorganization.py style: plain classes, bare
assert, no fixtures.

Non-determinism note (see TestPreprocessDiagnosisNonDeterminism at the
bottom): PYTHONHASHSEED pinning in tests/conftest.py happens via
os.environ.setdefault() *after* the interpreter has already started, so it
has no effect on this process's str-hash seed (CPython reads
PYTHONHASHSEED only at start-up, before any Python code runs). Every
preprocess_diagnosis path that goes through `list(set(...))` over more than
one element is therefore genuinely order-random across separate `pytest`
invocations run without PYTHONHASHSEED exported in the shell beforehand.
Multi-element assertions below are written to be tolerant of that (compare
by content/sets, not list position); only genuinely single-element results
are asserted as an exact literal.
"""

from nltk.corpus import stopwords

from aicds.utils import cython_utils

# preprocess_sentence reads the module-level `stop_words` global, which the
# module itself does not define -- every real call site monkey-patches it in
# (baseline_sent2vec.py:66, bert_models.py:240, bert_eval.py:63,
# analyze_score_distributions.py:48), all with this identical expression.
cython_utils.stop_words = set(stopwords.words("english"))


def _diagnosis_by_description(result):
    """Parse preprocess_diagnosis() output rows ('p1,p2,...:description')
    into {description: frozenset(prefixes)}, order-independent."""
    parsed = {}
    for row in result:
        prefix_part, _, description = row.partition(":")
        parsed[description] = frozenset(prefix_part.split(","))
    return parsed


class TestPreprocessSentenceNegationDestruction:
    """Bug (a): '/' is padded with spaces before tokenising, so 'w/o'
    tokenises to ['w', '/', 'o']; '/' is dropped as punctuation and 'o' is
    an NLTK English stopword, so it is dropped too. 'w/o' -> 'w'.
    See docs/findings/06-preprocessing-defects.md. Pinned as-is."""

    def test_w_o_collapses_to_w(self):
        assert (
            cython_utils.preprocess_sentence("Tracheostomy w/o Extensive Procedure")
            == "tracheostomy w extensive procedure"
        )

    def test_w_o_and_plain_w_collide(self):
        # The defect: two clinically opposite phrases become byte-identical.
        with_negation = cython_utils.preprocess_sentence(
            "Tracheostomy w/o Extensive Procedure"
        )
        without_negation = cython_utils.preprocess_sentence(
            "Tracheostomy w Extensive Procedure"
        )
        assert with_negation == "tracheostomy w extensive procedure"
        assert without_negation == "tracheostomy w extensive procedure"
        assert with_negation == without_negation

    def test_w_o_collapses_to_w_second_example(self):
        assert (
            cython_utils.preprocess_sentence("Dvrtcli colon w/o hmrhg")
            == "dvrtcli colon w hmrhg"
        )


class TestPreprocessSentenceStopwordLoss:
    """Bug (b): ordinary stopwords (including single letters and short
    words like 'or', 'of', 'no', 'some') are stripped along with the '/'
    negation defect, silently deleting logical connectives.
    See docs/findings/06-preprocessing-defects.md. Pinned as-is."""

    def test_or_and_w_o_both_lost(self):
        assert (
            cython_utils.preprocess_sentence(
                "SEPTICEMIA OR SEVERE SEPSIS W/O MV 96+ HOURS W MCC"
            )
            == "septicemia severe sepsis w mv 96+ hours w mcc"
        )

    def test_of_and_no_lost(self):
        assert (
            cython_utils.preprocess_sentence("plain text no punctuation")
            == "plain text punctuation"
        )


class TestPreprocessSentenceLowercasing:
    def test_all_caps_lowered(self):
        assert (
            cython_utils.preprocess_sentence("MiXeD CaSe Text") == "mixed case text"
        )

    def test_already_lower_unchanged_aside_from_stopwords(self):
        assert (
            cython_utils.preprocess_sentence("ALREADY LOWER already")
            == "already lower already"
        )


class TestPreprocessSentencePadding:
    """Pin the four explicit .replace() padding rules and their interaction
    order: '/' , '.-' , '.' , then apostrophe -- applied in that literal
    order, so a '.-' substring is padded by the '.-' rule first and then the
    lone '.' inside the resulting ' .- ' token is padded *again* by the
    plain '.' rule, splitting it into separate '.' and '-' tokens."""

    def test_slash_padding_isolates_and_drops_slash(self):
        assert (
            cython_utils.preprocess_sentence("Some/Value.-Test") == "value test"
        )
        # 'some' is dropped as an NLTK stopword (not part of the slash bug);
        # '/' is dropped as punctuation; '.-' is split into '.' and '-' and
        # both dropped as punctuation.

    def test_dot_dash_padding(self):
        assert (
            cython_utils.preprocess_sentence("end.-of sentence") == "end sentence"
        )
        # 'of' is an NLTK stopword; '.-' -> '.' + '-', both punctuation.

    def test_dot_padding(self):
        assert (
            cython_utils.preprocess_sentence("Diagnosis: A.B. Test")
            == "diagnosis b test"
        )
        # 'a' is an NLTK stopword (single-letter), ':' and '.' are
        # punctuation.

    def test_apostrophe_padding(self):
        assert (
            cython_utils.preprocess_sentence("Patient's condition")
            == "patient condition"
        )
        # "'s" -> "'" (punctuation, dropped) + "s" (NLTK stopword, dropped).


class TestPreprocessSentencePunctuationRemoval:
    def test_comma_and_exclamation_removed(self):
        assert cython_utils.preprocess_sentence("Hello, World!") == "hello world"


class TestPreprocessDiagnosisSingleEntryNoDoubleDash:
    """A diagnosis field with one entry and no '--' delimiter: split('--')
    yields a single-element list, so every set() dedup step is a no-op and
    the prefix-reconstruction loop has exactly one candidate -- no
    hash-order dependence is possible here."""

    def test_single_entry_passthrough(self):
        assert cython_utils.preprocess_diagnosis("ms:sepsis") == ["ms:sepsis"]

    def test_single_entry_lowercased_and_rstripped(self):
        assert cython_utils.preprocess_diagnosis("ms:Diabetes Mellitus  ") == [
            "ms:diabetes mellitus"
        ]


class TestPreprocessDiagnosisDedupRounds:
    """Pin both dedup rounds:
      round 1 -- `diagnosis_list = list(set(diagnosis.split('--')))`
                 dedups identical *raw* (prefix-and-all) entries.
      round 2 -- `diagnosis_no_drgtype = list(set(diagnosis_no_drgtype))`
                 dedups descriptions that are only identical *after* the
                 apr:/hcfa:/ms: prefix has been stripped.
    Each case below collapses to a single final row, so the result content
    is exact and order-independent regardless of hash seed."""

    def test_round_one_collapses_identical_raw_duplicates(self):
        # 'apr:Foo' appears twice, byte-for-byte identical after lowering --
        # round 1 alone collapses it to one element, round 2 is then a
        # no-op on the already-unique 'foo'.
        assert cython_utils.preprocess_diagnosis("apr:Foo--apr:Foo") == ["apr:foo"]

    def test_round_two_collapses_same_description_different_prefix(self):
        # 'apr:Foo' and 'hcfa:Foo' are distinct raw strings -- round 1 does
        # NOT collapse them (2 elements survive). Only after the apr:/hcfa:
        # prefixes are stripped do both become 'foo', which round 2 then
        # collapses to a single element. Both original prefixes are folded
        # back onto that one surviving row.
        result = cython_utils.preprocess_diagnosis("apr:Foo--hcfa:Foo")
        assert len(result) == 1
        parsed = _diagnosis_by_description(result)
        assert parsed == {"foo": frozenset({"apr", "hcfa"})}


class TestPreprocessDiagnosisMultipleDistinctEntries:
    """Two descriptions that share no substring: prefix reconstruction is
    unambiguous per-row, but which row comes first in the returned list is
    hash-order dependent, so compare as an order-independent mapping."""

    def test_two_unrelated_descriptions_each_single_prefix(self):
        result = cython_utils.preprocess_diagnosis("apr:Sepsis--hcfa:Pneumonia")
        assert len(result) == 2
        parsed = _diagnosis_by_description(result)
        assert parsed == {
            "sepsis": frozenset({"apr"}),
            "pneumonia": frozenset({"hcfa"}),
        }

    def test_three_prefixes_same_description(self):
        # All three raw entries share the description 'sepsis' after prefix
        # stripping -> round 2 collapses to a single row carrying all three
        # prefixes.
        result = cython_utils.preprocess_diagnosis(
            "apr:Sepsis--hcfa:Sepsis--ms:Sepsis"
        )
        assert len(result) == 1
        # Content (description + prefix set) is pinned exactly; only the
        # comma-joined prefix ORDER within the single surviving string is
        # hash-order dependent (see TestPreprocessDiagnosisNonDeterminism
        # below), so compare via the order-independent parse rather than
        # the literal string.
        parsed = _diagnosis_by_description(result)
        assert parsed == {"sepsis": frozenset({"apr", "hcfa", "ms"})}


class TestPreprocessDiagnosisPrefixReconstructionSubstringBug:
    """UNDOCUMENTED bug found while characterizing this function (not in
    docs/findings/06-preprocessing-defects.md, which only covers
    preprocess_sentence and the symptom/diagnosis-arm divergence): the
    prefix-reconstruction loop uses substring containment --
    `if x in d` -- against the ORIGINAL prefixed strings, not an exact
    match of the stripped description. When one stripped description is a
    substring of another entry's full 'prefix:description' text, it
    spuriously acquires that entry's prefix too.

    Concretely, for 'apr:Sepsis--hcfa:Severe Sepsis--ms:Sepsis': the
    description 'sepsis' is a substring of 'hcfa:severe sepsis' (because
    "severe sepsis" ends in "sepsis"), so the 'sepsis' row incorrectly
    picks up the 'hcfa' prefix as well as its own 'apr' and 'ms'. Pinned
    as-is; a fix would need exact-match prefix reconstruction instead of
    substring containment."""

    def test_severe_sepsis_substring_pollutes_sepsis_prefix(self):
        result = cython_utils.preprocess_diagnosis(
            "apr:Sepsis--hcfa:Severe Sepsis--ms:Sepsis"
        )
        assert len(result) == 2
        parsed = _diagnosis_by_description(result)
        assert parsed == {
            "severe sepsis": frozenset({"hcfa"}),
            # Bug: 'hcfa' leaks in here via substring match on
            # 'hcfa:severe sepsis', even though this row's own raw entries
            # were only 'apr:sepsis' and 'ms:sepsis'.
            "sepsis": frozenset({"apr", "hcfa", "ms"}),
        }

    def test_single_letter_description_pollutes_from_unrelated_prefix_label(self):
        # 'a' is a substring of the LITERAL WORD "hcfa" itself, so the 'a'
        # row spuriously picks up 'hcfa' as a prefix even though 'a' and
        # 'b' were never associated with the same drg-type prefix in the
        # input.
        result = cython_utils.preprocess_diagnosis("apr:A--hcfa:B")
        assert len(result) == 2
        parsed = _diagnosis_by_description(result)
        assert parsed == {
            "a": frozenset({"apr", "hcfa"}),
            "b": frozenset({"hcfa"}),
        }


class TestPreprocessDiagnosisNonDeterminism:
    """Documents, rather than asserts a single behaviour: preprocess_diagnosis
    routes every multi-element intermediate through Python's `set()`, whose
    string iteration order depends on the process's str-hash seed. Because
    tests/conftest.py sets PYTHONHASHSEED via os.environ.setdefault() at
    *import* time -- after the interpreter has already fixed its hash seed
    for this process -- that pinning has no effect here unless
    PYTHONHASHSEED=0 is exported in the shell BEFORE `pytest` starts. This
    test only asserts the invariant that survives regardless of seed
    (the returned prefix set / description content), not any particular
    string ordering. This is a repo characteristic, not something this test
    file changes."""

    def test_prefix_join_order_is_content_stable_across_calls(self):
        # Calling twice in the same process must at least agree with itself
        # (same seed within one process), even though a fresh process could
        # differ.
        first = cython_utils.preprocess_diagnosis("apr:Sepsis--hcfa:Sepsis")
        second = cython_utils.preprocess_diagnosis("apr:Sepsis--hcfa:Sepsis")
        assert first == second
        assert _diagnosis_by_description(first) == {
            "sepsis": frozenset({"apr", "hcfa"})
        }

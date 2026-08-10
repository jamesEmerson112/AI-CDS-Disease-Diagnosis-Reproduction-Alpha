# Preprocessing defects: destroyed negation, shredded labels, divergent arms

> **In plain words.** Before any AI model sees the text, a cleanup step lowercases it, splits
> it into words, and throws away "unimportant" words. Three bugs lived in that cleanup. The
> worst: `w/o` (medical shorthand for *without*) got chopped into `w`, `/`, `o` — the slash was
> discarded as punctuation, the `o` was discarded because it's on a list of throwaway words —
> leaving just `w`, which means *with*. So "tracheostomy WITHOUT extensive procedure" and
> "tracheostomy WITH extensive procedure" — opposite medical events — became the same sentence.
> Second bug: symptom lists are comma-separated, but some symptom names *contain* commas
> ("Pneumonia, organism NOS"), so splitting on commas shredded them into meaningless fragments
> that then matched each other perfectly across patients. Third: the two competing systems
> didn't even run the same cleanup, so any comparison between them was partly measuring the
> cleanup difference. **All three are now fixed under the `corrected` pipeline** — with nine
> stray fragments left, tracked as P27.

> **Status: FIXED 2026-08-05 (`c2115ba`), selectable via `--pipeline corrected`.** Under
> `corrected`: `w/o` survives as `without`, the leading-space rejoin rule recovers 80 of the 89
> orphan fragments (the last nine are **P27**, still open), and both arms preprocess diagnosis
> text identically — 145/145 descriptions match, up from 26/145. `legacy` keeps all three
> defects on purpose, so the golden regression stays byte-exact. Path note: written before the
> `src/aicds` package move; line numbers have drifted.

**Bottom line: three defects in the text-preparation path, two of them clinically severe.** The
worst inverts the meaning of clinical phrases — `w/o` ("without") becomes `w` ("with"),
collapsing opposite DRG groups into identical strings. All three sit upstream of embedding, so
they move every number in the project.

Unlike the [metric](04-metric-degeneracy.md) and [fold](05-patient-leakage.md) problems, these
are plain bugs. Nobody would defend them.

## 1. `w/o` collapses to `w` — negation is destroyed

`preprocess_sentence` (`cython_utils.py`) pads `/` with spaces before tokenising:

```python
def preprocess_sentence(text):
    text = text.replace('/', ' / ')
    ...
    tokens = [token for token in word_tokenize(text)
              if token not in punctuation and token not in stop_words]
```

So `w/o` tokenises to `["w", "/", "o"]`. The `/` is discarded as punctuation, and **`o` is a
member of NLTK's English stopword list**, so it is discarded too. What survives is `w`.

Verified directly:

```
"Tracheostomy w/o Extensive Procedure"  ->  'tracheostomy w extensive procedure'
"Tracheostomy w   Extensive Procedure"  ->  'tracheostomy w extensive procedure'
```

**Two clinically opposite DRG groups become the same string.** Further examples:

| Source | After preprocessing | Reads as |
|---|---|---|
| `Dvrtcli colon w/o hmrhg` | `dvrtcli colon w hmrhg` | *with* haemorrhage |
| `SEPTICEMIA OR SEVERE SEPSIS W/O MV 96+ HOURS W MCC` | `septicemia severe sepsis w mv 96+ hours w mcc` | *with* mechanical ventilation |

Note the second example also loses `OR` — another stopword — so the disjunction disappears as
well.

**Scope.** `preprocess_sentence` is applied to symptom text in **both arms**, and to diagnosis
text in the **baseline arm only** under `legacy` (see defect 3). Negation appears throughout
MIMIC-III's ICD-9 short titles and the DRG names, so this is not a rare edge case.

**Fix — landed.** Under `corrected`, `w/o` is protected before the slash-padding step and
survives as `without`. As predicted, the change moved every number in the project, which is
exactly why it lives behind the config seam rather than replacing the legacy behaviour: the
golden still covers `legacy` byte-for-byte, and `corrected` gets its own runs
([11](11-corrected-pipeline-first-results.md)). One measured consequence worth knowing: the
collision this bug created is the *only* thing that would have made the DRG grader and
threshold-1.0 cosine disagree — so fixing it here is what made those two graders provably
equivalent ([12](12-drg-grader.md)).

## 2. Comma-delimited symptoms shred labels that contain commas

The SYMPTOMS field is comma-separated, and `load_dataset` splits on it naively (and again in
`embending_symptoms`):

```python
symptoms_list = symptoms.split(',')
```

But ICD-9 short titles contain commas. **80 of 1,805 symptom tokens (4.4%) are orphan
fragments:**

| Intact source label | What the pipeline sees |
|---|---|
| `Pneumonia, organism NOS` | `Pneumonia` + `" organism NOS"` |
| `Pressure ulcer, stage I` | `Pressure ulcer` + `" stage I"` |
| `Pressure ulcer, stage IV` | `Pressure ulcer` + `" stage IV"` |
| `Pressure ulcer, low back` | `Pressure ulcer` + `" low back"` |

Most frequent fragments: `" organism NOS"` (26), `" low back"` (15), `" initial"` (9),
`" tubr necr"` (6), `" stage II"` (5).

Two consequences, and the second is worse:

1. **Severity information is destroyed.** A stage I and a stage IV pressure ulcer both collapse
   to the identical token `Pressure ulcer`.
2. **The fragments are embedded as if they were symptoms, creating spurious perfect matches.**
   `" organism NOS"` appears in 26 admissions. Patient similarity is mean-of-max over symptom
   pairs, so any two of those 26 patients receive a **1.0** contribution from a token carrying
   no clinical content — inflating patient similarity for reasons unrelated to the encoder
   under test.

**Fix — landed, with a nine-fragment remainder.** All the recoverable fragments begin with a
leading space, because the field separator is `,` while intra-label commas are followed by a
space (`, `). The rule *"if a token starts with a space, rejoin it to the previous token"* is
what `corrected` applies:

```
unique symptom strings before fix: 573
unique symptom strings after  fix: 564
recovered: 'Ac kidny fail, tubr necr', 'Dysphagia, oropharyngeal',
           'AMI anterolateral, init', 'Liver laceration, major', ...
```

It recovers 80 of the 89 fragments; the last **nine** do not follow the leading-space pattern
and remain open as **P27** in [../plans/correctness-fixes.md](../plans/correctness-fixes.md).
The longer-term fix is to re-extract from MIMIC-III with a delimiter that does not collide with
clinical text — but that requires database access.

**Provenance:** `split(',')` is present in the original authors' compiled Cython
(`archive/cython_source/util_cy.c`), so this is inherited rather than introduced by this
repository.

## 3. The two arms preprocess diagnosis text differently

The repository's central design constraint is that both arms share everything except the
embedding model, so that the encoder is the only variable. That constraint was violated here.

**Baseline** — preprocesses before embedding (`cython_utils.py`, in `embending_diagnosis()`):

```python
embs = model.embed_sentence(preprocess_sentence(diagnosis_description))
```

**BERT** — under `legacy`, embeds the raw string: no `preprocess_sentence` call anywhere in the
diagnosis path.

**119 of 145 (82.1%) unique diagnosis descriptions differ between the two paths.** The BERT arm
under `legacy` embeds raw uppercase text including artifacts such as the HTML entity in
`SEPTICEMIA AGE &gt;17`, while the baseline embeds lowercased, tokenised, stopword-stripped
text with `w/o` already corrupted to `w` per defect 1.

Any `legacy` BioSentVec-vs-BERT comparison is confounded by preprocessing, not only by encoder.
Note the direction is not obvious: the baseline's preprocessing is *more* aggressive and
carries the negation bug, so it is not simply "BERT gets cleaner text."

**Fix — landed.** Under `corrected` the BERT path calls the same preprocessing
(`bert_models.py` gates it on `use_corrected_preprocessing(config)`), and all 145/145
descriptions reach the two encoders as identical strings. Only the *encoded* text changes — the
embedding dict keys stay raw under both configs, because the grader looks descriptions up by
raw text and preprocessing the keys would break every lookup.

Two smaller divergences in the same family, for completeness (both still present — they affect
the per-case output blocks, which nothing currently consumes):

- BERT per-case recall is `tp/(tp+fp)`; the baseline's is `tp/nrow`.
- Baseline per-case rows are cumulative within a fold; BERT's are a single binary verdict per
  case. The two arms' per-case blocks are not row-comparable.

## Why these matter together

The project's headline claim is a comparison between encoders. Defects 1 and 2 corrupt the text
*before* any encoder sees it, so they degrade all arms — but not necessarily equally, since the
arms preprocessed differently (defect 3). That combination meant the legacy results cannot
cleanly attribute any observed difference to the encoders. The `corrected` pipeline removes the
attribution problem; separating how much of the legacy-to-corrected delta came from the folds
versus the preprocessing is what the `folds-only` / `preprocess-only` configs exist for
(**P29**, staged but not yet run).

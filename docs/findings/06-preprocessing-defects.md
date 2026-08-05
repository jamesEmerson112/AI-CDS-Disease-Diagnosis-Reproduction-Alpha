# Preprocessing defects: destroyed negation, shredded labels, divergent arms

**Bottom line: three defects in the text-preparation path, two of them clinically severe.** The
worst inverts the meaning of clinical phrases — `w/o` ("without") becomes `w` ("with"), collapsing
opposite DRG groups into identical strings. All three sit upstream of embedding, so they move every
number in the project.

Unlike the [metric](04-metric-degeneracy.md) and [fold](05-patient-leakage.md) problems, these are
plain bugs. Nobody would defend them.

## 1. `w/o` collapses to `w` — negation is destroyed

`preprocess_sentence` (`src/utils/cython_utils.py:251-260`) pads `/` with spaces before
tokenising:

```python
def preprocess_sentence(text):
    text = text.replace('/', ' / ')
    ...
    tokens = [token for token in word_tokenize(text)
              if token not in punctuation and token not in stop_words]
```

So `w/o` tokenises to `["w", "/", "o"]`. The `/` is discarded as punctuation, and **`o` is a member
of NLTK's English stopword list**, so it is discarded too. What survives is `w`.

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

Note the second example also loses `OR` — another stopword — so the disjunction disappears as well.

**Scope.** `preprocess_sentence` is applied to symptom text in **both arms**, and to diagnosis text
in the **baseline arm only** (see defect 3). Negation appears throughout MIMIC-III's ICD-9 short
titles and the DRG names, so this is not a rare edge case.

**Fix.** Protect `w/o` before the slash-padding step, or remove single-letter tokens from the
stopword set for this corpus. Either change moves every number in the project, so it warrants its
own commit with the goldens regenerated deliberately.

## 2. Comma-delimited symptoms shred labels that contain commas

The SYMPTOMS field is comma-separated, and `load_dataset` splits on it naively
(`src/utils/cython_utils.py:185`, and again in `embending_symptoms` at `:202`):

```python
symptoms_list = symptoms.split(',')
```

But ICD-9 short titles contain commas. **80 of 1,805 symptom tokens (4.4%) are orphan fragments:**

| Intact source label | What the pipeline sees |
|---|---|
| `Pneumonia, organism NOS` | `Pneumonia` + `" organism NOS"` |
| `Pressure ulcer, stage I` | `Pressure ulcer` + `" stage I"` |
| `Pressure ulcer, stage IV` | `Pressure ulcer` + `" stage IV"` |
| `Pressure ulcer, low back` | `Pressure ulcer` + `" low back"` |

Most frequent fragments: `" organism NOS"` (26), `" low back"` (15), `" initial"` (9),
`" tubr necr"` (6), `" stage II"` (5).

Two consequences, and the second is worse:

1. **Severity information is destroyed.** A stage I and a stage IV pressure ulcer both collapse to
   the identical token `Pressure ulcer`.
2. **The fragments are embedded as if they were symptoms, creating spurious perfect matches.**
   `" organism NOS"` appears in 26 admissions. Patient similarity is mean-of-max over symptom
   pairs, so any two of those 26 patients receive a **1.0** contribution from a token carrying no
   clinical content — inflating patient similarity for reasons unrelated to the encoder under test.

**Fix — deterministic, no model required.** All 80 fragments begin with a leading space, because
the field separator is `,` while intra-label commas are followed by a space (`, `). The rule *"if a
token starts with a space, rejoin it to the previous token"* recovers every one of them:

```
unique symptom strings before fix: 573
unique symptom strings after  fix: 564
recovered: 'Ac kidny fail, tubr necr', 'Dysphagia, oropharyngeal',
           'AMI anterolateral, init', 'Liver laceration, major', ...
```

The longer-term fix is to re-extract from MIMIC-III with a delimiter that does not collide with
clinical text — but that requires database access, and the rejoin rule is provably sufficient for
the committed file.

**Provenance:** `split(',')` is present in the original authors' compiled Cython
(`archive/cython_source/util_cy.c`), so this is inherited rather than introduced by this
repository.

## 3. The two arms preprocess diagnosis text differently

The repository's central design constraint is that both arms share everything except the embedding
model, so that the encoder is the only variable. That constraint is violated here.

**Baseline** — preprocesses before embedding (`src/utils/cython_utils.py:226`):

```python
embs = model.embed_sentence(preprocess_sentence(diagnosis_description))
```

**BERT** — embeds the raw string (`src/models/bert_models.py:318-332`): no `preprocess_sentence`
call anywhere in the diagnosis path.

**119 of 145 (82.1%) unique diagnosis descriptions differ between the two paths.** The BERT arm
therefore embeds raw uppercase text including artifacts such as the HTML entity in
`SEPTICEMIA AGE &gt;17`, while the baseline embeds lowercased, tokenised, stopword-stripped text
with `w/o` already corrupted to `w` per defect 1.

Any BioSentVec-vs-BERT comparison is confounded by preprocessing, not only by encoder. Note the
direction is not obvious: the baseline's preprocessing is *more* aggressive and carries the
negation bug, so it is not simply "BERT gets cleaner text."

Two smaller divergences in the same family, for completeness:

- BERT per-case recall is `tp/(tp+fp)`; the baseline's is `tp/nrow`.
- Baseline per-case rows are cumulative within a fold; BERT's are a single binary verdict per case.
  The two arms' per-case blocks are not row-comparable.

## Why these matter together

The project's headline claim is a comparison between encoders. Defects 1 and 2 corrupt the text
*before* any encoder sees it, so they degrade all arms — but not necessarily equally, since the
arms preprocess differently (defect 3). That combination means the current results cannot cleanly
attribute any observed difference to the encoders.

Fix order and effort estimates are in
[../plans/correctness-fixes.md](../plans/correctness-fixes.md) item 4.

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
> stray fragments left, tracked as P27. *(2026-08-12: those nine are closed too, under a **third**
> text variant, `corrected2` — `corrected` itself is frozen and unchanged. See the addendum.)*

> **Status: FIXED 2026-08-05 (`c2115ba`), selectable via `--pipeline corrected`.** Under
> `corrected`: `w/o` survives as `without`, the leading-space rejoin rule recovers 80 of the 89
> orphan fragments (the last nine are **P27**, closed 2026-08-11 under `corrected2`, not under
> `corrected` — see the addendum), and both arms preprocess diagnosis
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

*(2026-08-12: **P27 is closed** — a second, bounded rule under the new `corrected2` variant
rejoins all nine, and the four-arm run that measured its effect is the addendum at the end of this
file. This paragraph stands as written because `corrected`'s behaviour has not changed: under
`corrected` the residual is still exactly nine, and a test pins that on purpose.)*

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
(**P29**, staged but not yet run). *(2026-08-12: P29 ran; the decomposition is
[15](15-leakage-preprocessing-attribution.md).)*

---

## Addendum, 2026-08-12 — P27 closes under `corrected2`, and its first four-arm run

**Date:** 2026-08-12 · **Host:** RunPod Linux, 16 vCPU (Python 3.9.23, numpy 2.0.2, torch
2.8.0+cpu; torn down after harvest) · **Commits:** `7e49212` (the rule, D4) · `c5508ef` (the
sanity pins) · **Problem number:** P27, closed

**Artifacts.** `results_corrected2/{baseline,bio_clinical_bert,biomedbert,bluebert}/12082026_*/`
at the repository root, compared against `results_corrected/`. **Both trees are gitignored** by
`results*/` (`.gitignore:160`) and are DUA-covered, so everything below is an aggregate: no
`HADM_ID`, no per-case row. Each of the four runs carries a `run_metadata.json` recording
`pipeline.name = pipeline.preprocess_version = "corrected2"`, `fold_dir = "folds_grouped"`,
**`grader = "cosine"`**, and the canonical fold digest `b36f7216…a6ec5084f`
([14](14-fold-split-environment-dependence.md)) — on repo `da93b96`, whose `dirty: true` flag is
two untracked leftovers of the retired staged pod script, with zero tracked modifications.

> **In plain words.** The nine label fragments this finding left shredded are now rejoined, in a
> new text variant that leaves the old one untouched. All four systems were re-run with the
> repaired text. Two of them scored differently. The other two produced **exactly** the same
> headline number as before — identical to the last digit. Nine tiny text repairs moved half the
> arms and left the other half where they were, and this addendum does not claim to know why.

### What shipped

`corrected2` is a **third** `preprocess_version`, not an edit to `corrected`. That is deliberate:
redefining `corrected` would retroactively invalidate every `results_corrected/`, `results_drg/`
and `results_p5/` tree as a reproducible artifact, and quietly move the numbers findings
[11](11-corrected-pipeline-first-results.md) and [13](13-rank-aware-metrics.md) rest on. The
config seam was built for exactly this — a variant is a name, not a boolean.

Rule 2 (`corrected2` only, `cython_utils.split_symptoms`) rejoins a part to its predecessor when
all three hold: the part begins with a **lower-case** letter, the accumulated predecessor begins
with an **upper-case** one (every label in this field is capitalised — the ICD-9 short-title
tell), and the join is at most **24 characters**, the width of the CMS short-title field the
abbreviation came from. The nine residuals are bare-comma truncations of that cap —
`stage NOS` ×4, `stage III` ×3, `uncomp`, `pharyngoesoph` — indistinguishable from a separator by
spacing alone, which is why rule 1 could not reach them.

Measured on the committed 129 admissions and pinned in `tests/test_symptom_splitting.py`:
**1,716 tokens, 561 unique strings, zero residual fragments**, firing exactly 9 times across
exactly 4 labels. Rule 1 alone leaves 1,725 / 564 / 9.

### The measured effect

TOP-10 by MAX at threshold 1.0 — the only cosine cell where no arm sits on the saturation ceiling
([03](03-metric-saturation.md)) — reading the `10-FOLD` block of each arm's
`PerformanceIndex.txt`, `corrected` → `corrected2`:

| arm | `corrected` F | `corrected2` F | change |
|---|---|---|---|
| BioSentVec | `0.2162775515864761` | `0.18555291390531667` | −0.0307 |
| Bio_ClinicalBERT | `0.2491025641025641` | `0.24243589743589747` | −0.0067 |
| BiomedBERT | `0.1980769230769231` | `0.1980769230769231` | **bit-identical** |
| BlueBERT | `0.18205128205128204` | `0.18205128205128204` | **bit-identical** |

The two repeats are byte-for-byte repeats of the printed field, at every significant figure, read
off both trees rather than eyeballed. Two supporting rows, same block: the baseline's TOP-10 @0.6
F falls `0.3922168068392324` → `0.36149216915807303` (that same @0.6 row's mean per-fold TP `4.4` →
`4.0`, FP `5.4` → `5.8` — the `2.5` → `2.1` / `7.3` → `7.7` pair belongs to the @0.9 and @1.0 rows
printed above it, not to this one), while its **`PR` is unchanged bit-exactly at
`0.7557692307692307`** at every threshold — the rejoin moved which candidates score, not how often
the baseline is willing to answer at all.

**No mechanism is asserted here, because none was measured.** The temptation is to explain the
split — the spurious 1.0 matches of §2, the arms' differing sensitivity to symptom-side text —
and every such story is a hypothesis this run does not test. What the run establishes is narrower
and still worth having: **nine rejoined fragments moved two arms and left two bit-identical in
the reported cell**, so the effect of a text repair is *arm-dependent*, not a common-mode shift
that cancels in a comparison. Do not treat the two that moved as the error, and do not treat the
two that did not as evidence the fix was inert.

### The bit-identity is cell-local — the thing most likely to be over-read

"BiomedBERT and BlueBERT are bit-identical" is a claim about **one cell**, not about an arm.
Diffing all thirty `10-FOLD` aggregate rows per arm (6 aggregators × 5 thresholds) between the
two trees:

| arm | rows differing, of 30 | where |
|---|---|---|
| BioSentVec | 25 | every TOP-K row; the MAX block is unchanged |
| Bio_ClinicalBERT | 2 | TOP-10 @0.9 and @1.0 |
| BlueBERT | 4 | TOP-10 @0.8; TOP-20 @0.9, @1.0, @0.8 |
| BiomedBERT | 1 | **MAX @1.0** (`0.08551282051282053` → `0.07782051282051283`) |

So both "unmoved" arms do move — just not in the headline cell — and BiomedBERT's single moving
row is in the one aggregator the baseline's 25 leave alone. Quote the bit-identity as *TOP-10
@1.0 is unchanged for BiomedBERT and BlueBERT*, never as *corrected2 did not affect them*.

### The knobless conclusion does not move

`python scripts/analyze_rank_metrics.py results_corrected2` (re-run read-only 2026-08-12, matching
the captured artifact): **no pair of encoders separates in any of the three abstention
populations.** Following [13](13-rank-aware-metrics.md)'s convention — the maximum **across all six
pairs**, with the pair it came from named — |t| on per-fold MRR@50 tops out at **1.628** on
winnable (BioSentVec vs Bio_ClinicalBERT), **1.646** on all-cases and **1.646** on answered (both
Bio_ClinicalBERT vs BlueBERT, the same pair and the same value because the three BERT arms have
coverage 1.0 and those two populations are the same set for them). All against the 2.262 needed at
9 df. Winnable-population MRR@50:
Bio_ClinicalBERT 0.243351 > BiomedBERT 0.235203 > BlueBERT 0.228529 > BioSentVec 0.197102, at
coverage 0.8477 for the baseline and 1.0000 for the three transformers. The headline claim of
[13](13-rank-aware-metrics.md) — *no encoder ranking is supported by this experiment* — survives
the P27 text repair unchanged.

### Where the numbers live, and how to reproduce them

- **The trees are gitignored and stay that way.** They exist only on the machine that harvested
  them; nothing in `docs/` or `results*/` is committed.
- **The five headline cells are pinned in the repository**, since `c5508ef`, as the `"corrected2"`
  entry of `scripts/compare_models.py`'s expectation table, with a comment recording the
  moved/unmoved split so a future reader cannot "fix" the two repeated numbers.
- One command, and it needs no `--pipeline`: `results_corrected2` is deliberately absent from
  `_PIPELINE_BY_DIRNAME`, but every run in it carries `run_metadata.json`, so

  ```bash
  python scripts/compare_models.py --results-dir results_corrected2
  # [INFO] pipeline from run_metadata.json: corrected2
  ```

  resolves the pipeline from provenance rather than from a directory name (P14) and writes
  `results_corrected2/model_comparison.pdf`. This is the first tree for which that path is
  exercised in anger rather than falling back to the `[WARN]` dirname route.

### Limits — read before quoting any of this

1. **`corrected2` is not the headline pipeline.** The claims in `README.md` and `CLAUDE.md` come
   from `corrected` / `drg`. `corrected2` has **no DRG-graded run** and no attribution
   decomposition, so it cannot be swapped into those tables.
2. **This tree is cosine self-graded** (`grader: "cosine"` in its metadata), so the self-grading
   confound of [12](12-drg-grader.md) is present here. Under `corrected`, cosine @1.0 and
   `drg-exact` agreed bit-exactly across all 144 numbers; that equivalence is *expected* to carry
   over to `corrected2` but **has not been measured**, because no `corrected2` DRG run exists.
3. **The comparison above is `corrected` → `corrected2` only**, one grader and one split
   (canonical `folds_grouped`) held fixed. It is not a legacy delta and not an encoder comparison.
4. **"Zero residual fragments" is a statement about this dataset, not a guarantee about the rule.**
   Rule 2 is a bounded heuristic. Measured by mutating the shipped code, and the mutation set is
   the one recorded in `tests/test_symptom_splitting.py`'s
   `test_the_cap_constant_is_pinned_at_24`: the five real-data numbers catch a *narrowing*
   (cap 24 → 23 gives 1,724 tokens / 563 unique / 8 residual / 1 firing) but are **blind to a
   widening** — **caps 25, 27, 30, 31, 40 and no cap at all** each leave all five unmoved, because
   no part in the committed file exceeds 24 characters, so a wider cap has nothing extra to absorb.
   The predicate is therefore pinned **separately from the counts**, by three fixtures: the cap
   constant asserted directly, a 25-character boundary case chosen to sit in the blind spot
   (`caps 25–30` were caught by nothing before it), and the two `KNOWN_COST` tests. Widen the rule
   without touching those and the real-data tests stay green.
   *(Corrected 2026-08-12: this list read "caps 25, 27, 30, 31 or none at all, and dropping the
   upper-case condition entirely". It omitted 40, which the pinned test does measure, and it
   claimed a mutation of the **upper-case condition** that the pinned test does not perform — the
   test file's only statement about that condition is that it fails to prevent the `Fever,cough`
   over-join, which is a different claim. Do not read the upper-case condition as measured-blind;
   it is untested against the real data.)*
5. **A lower score after a text repair is not a regression**, and it is not a confirmation of §2's
   inflation story either. Nothing in this run measures which — see the refusal above.

# Correctness fixes — the work that must land before any encoder claim

> **In plain words.** This is the ordered list of scientific fixes — the changes that
> deliberately *move the numbers*, as opposed to the refactor work that must not. As of
> 2026-08-08 the four big ones have landed (fold regrouping, the DRG grader, rank-aware
> metrics, the preprocessing repairs), each behind a selectable config so the original pipeline
> still reproduces bit-for-bit. **As of 2026-08-12 the science on this list is finished:** per-case
> output (P40), the last nine comma fragments (P27) and the attribution runs (P29) have all landed,
> and P39 (tie-breaks) closed on measurement without a code change. Embedding centring (item 5) was
> never attempted and is now **retired**, with P11 and P12, by owner decision. **The set-level
> metric (P6) was RETIRED on 2026-08-11** — findings 04 and 13 dissolved its motivation instead of
> satisfying it. **P37 and P13 both closed later the same day**, so what is left on the wider plan
> is **P38 alone**.
> *(Updated 2026-08-11 by the TODO audit; rewritten 2026-08-12 after the pod session, and again
> 2026-08-12 when P37 and P13 landed. The superseded sentences read: "What remains: per-case output
> (P40), the last nine comma fragments (P27), and the attribution runs (P29). Embedding centring
> (item 5) was never attempted and is now optional rather than pending." and, after the pod session,
> "What is left on the wider plan is P37, P13, then P38.")*

This is the ordered TODO for making the encoder comparison valid. It is separate from
[revival-roadmap.md](revival-roadmap.md), which is about code organisation. Nothing here is a
refactor; every item changes a number — which is exactly why each landed in its own commit,
behind a config whose default reproduces the legacy pipeline exactly.

Ordered by (impact / effort). Metric details live in [metric-redesign.md](metric-redesign.md).

---

## 1. Regroup the folds by patient — `GroupKFold`

**Status: DONE 2026-08-05 (`c2115ba`).** **Effort:** hours. **Impact:** largest single
correction available — and it measured as predicted: every arm's score fell, the baseline's
headline by 18.4% (0.4824 → 0.3922). Leaked cases went **41 → 0**, recounted independently.
Results: [../findings/11-corrected-pipeline-first-results.md](../findings/11-corrected-pipeline-first-results.md).
How much of that drop was this fix rather than item 4, measured per arm (P29):
[../findings/15-leakage-preprocessing-attribution.md](../findings/15-leakage-preprocessing-attribution.md).

The 10 folds split on `HADM_ID` (admission), but 129 admissions come from only **100
patients**. One patient has 15 admissions — 11.6% of the dataset by itself.

**41 of 129 test cases (31.8%) had another admission from the same `SUBJECT_ID` sitting in
their own retrieval pool.** Per-fold contamination ranged 15%–54%. Measured inflation at
threshold 1.0: leaked cases score **+0.11 to +0.26** above clean cases.

For scale: the differences *between encoders* at that threshold are 0.015–0.046. The
contamination was roughly an order of magnitude larger than the effect being studied.

The tell that this was real: on contaminated cases all three encoders score **identically**
(0.293 at MAX, 0.415 at TOP-10) while on clean cases they diverge (0.080 / 0.114 / 0.125).
When a patient's own prior chart is in the pool, retrieval finds it regardless of encoder.

**Fix, as landed:** `scripts/make_folds.py` regenerates the folds grouping on `SUBJECT_ID`
(`sklearn.model_selection.GroupKFold`) into `data/folds_grouped/` (gitignored, deterministic).
`data/folds/` is untouched — it is the golden's input and `legacy` still uses it on purpose.

**The trap that was avoided:** `load_dataset` drops each line's last character unconditionally,
assuming a trailing newline. **Any regenerated fold file must end with a trailing newline** or
the last symptom of the last line silently loses a character. `tests/test_characterize_dataset.py`
pins this.

**Also note, confirmed:** with one patient holding 15 admissions, the grouped folds are uneven
(114/15 through 117/12). That is correct and expected — report the per-fold n.

See [../findings/05-patient-leakage.md](../findings/05-patient-leakage.md).

---

## 2. Replace the cosine grader with a DRG-based one

**Status: DONE 2026-08-06 (`75b6530`), as `--pipeline drg` / `grader="drg-exact"`.** **Effort:**
days. **Impact:** removed the threshold knob entirely — and delivered the one result nobody
predicted: it reproduces corrected cosine at threshold 1.0 **bit-exactly** (72 numbers, not one
differing digit), proving the headline column was never self-grading-inflated. Full story:
[../findings/12-drg-grader.md](../findings/12-drg-grader.md).

The problem as diagnosed: the same encoder both retrieved candidates and judged whether the
retrieved diagnosis was correct. A model with a more compressed embedding space therefore
graded itself more leniently — which is exactly why BiomedBERT, the most compressed of the
three, "won" at threshold 0.9.

The information needed for an independent grader was already in the data: the DIAGNOSIS field
carries DRG group descriptions. String comparison needs no embedding.

**One instruction in this item was overturned by measurement, and the reversal matters.** This
document originally said *"Do not use exact match alone — use graded relevance: full credit for
an exact DRG match, partial credit for a same-family match."* That partial-credit ladder was
then actually built, measured, and **rejected**: it needed ~156 hand-tuned lexicon strings with
no held-out set, its low rungs were 87% false credit, it scored "discharged alive" vs "expired"
at 0.900, and it was the worst of three candidate designs on AUC. `drg-exact` shipped with zero
free parameters instead, and the 58.9% ceiling is *reported alongside* every number rather than
papered over. If partial credit is ever revisited, use an external DRG hierarchy, not a lexicon
derived from these 146 strings. See [metric-redesign.md](metric-redesign.md) and finding 12.

---

## 3. Report rank-aware metrics — MRR, Recall@K, Precision@K

**Status: DONE 2026-08-06 (`5393cab`), additively — new `RankMetrics.txt` sibling file,
`PerformanceIndex.txt` untouched, golden still byte-exact.** **Effort:** days. **Impact:**
removed the K knob; exposed the third knob (abstention). Full story:
[../findings/13-rank-aware-metrics.md](../findings/13-rank-aware-metrics.md).
The self-selection question it spawned is measured in
[../findings/16-self-selection.md](../findings/16-self-selection.md) (P40).

The problem as diagnosed: rank *is* computed (the candidate list is sorted by similarity) and
then discarded — the scorer returns true if **any** of the K retrieved clears the threshold, so
a hit at rank 1 and a hit at rank 50 count identically.

Consequence: `score(TOP-50) >= score(TOP-10)` is guaranteed by construction, since the top-50
set contains the top-10 set. Verified monotonic in 18/18 model × threshold combinations, zero
violations. The TOP-K curve in the README is arithmetic, not a finding.

Rank-aware metrics also happen to be **scale-free** — they never look at the absolute cosine
value, so they are immune to both the saturation and the per-model calibration problems. That
held: no pair of encoders separates under MRR on any abstention population (max paired
|t| = 1.718 vs the 2.262 needed).

**What this item spawned:** the abstention asymmetry (open, not closable) and **P40** — testing
whether the baseline's answered cases are self-selected easy ones needs per-case relevance
vectors, and every per-case output file in the repo was empty. **P40 closed 2026-08-12** by adding
a new sibling, `RankCases.txt`, and the answer is [finding 16](../findings/16-self-selection.md):
the answered cases are easier for *every* arm by nearly the same margin, so the confound is real
and arm-neutral, and matching the case sets inverts the ordering. The asymmetry itself is still
open and still not closable — abstention is a property of the arm, not of the metric.

---

## 4. Fix the preprocessing defects

**Status: DONE 2026-08-05 (`c2115ba`) under `--pipeline corrected`, except nine fragments
(P27).** **Effort:** hours. **Impact:** moderate, but two were clinically severe. Full story:
[../findings/06-preprocessing-defects.md](../findings/06-preprocessing-defects.md). Its share of
the corrected-pipeline drop, isolated from item 1's:
[../findings/15-leakage-preprocessing-attribution.md](../findings/15-leakage-preprocessing-attribution.md).

### 4a. `w/o` collapses to `w` — negation is destroyed — FIXED

`preprocess_sentence` pads `/` with spaces, so `w/o` tokenises to `["w", "/", "o"]`. The `/` is
dropped as punctuation and **`o` is an NLTK English stopword**, so it is dropped too. Verified:

```
"Tracheostomy w/o Extensive Procedure"  ->  'tracheostomy w extensive procedure'
"Tracheostomy w   Extensive Procedure"  ->  'tracheostomy w extensive procedure'
```

Two clinically opposite DRG groups became the same string. `Dvrtcli colon w/o hmrhg`
(diverticulitis *without* haemorrhage) became `dvrtcli colon w hmrhg`.

Affected **symptoms in both arms**, and diagnosis text in the baseline arm only (see 4c). Under
`corrected`, `w/o` survives as `without`. As predicted, the change moved every number, which is
why it lives behind the config seam in its own commit.

### 4b. Comma-split fragments — FIXED, nine remain (P27)

The SYMPTOMS field is comma-delimited, but some ICD-9 short titles contain commas, so the naive
split shreds them. **80 of 1,805 tokens (4.4%)** were orphan fragments.

```
"Pneumonia, organism NOS"      ->  "Pneumonia" + " organism NOS"
"Pressure ulcer, stage I"      ->  "Pressure ulcer" + " stage I"
"Pressure ulcer, stage IV"     ->  "Pressure ulcer" + " stage IV"
```

Severity was destroyed (stage I and stage IV collapse to the same token), and the junk
fragments got embedded as if they were symptoms. `" organism NOS"` appears in 26 admissions —
since patient similarity is mean-of-max, any two of those got a spurious **1.0** contribution
from a token carrying no clinical content.

**Fix, as landed — deterministic, no model needed.** The recoverable fragments begin with a
leading space (the separator is `,` while intra-label commas are `, `), so: *if a token starts
with a space, rejoin it to the previous token* (573 → 564 unique symptom strings, producing
real titles like `Ac kidny fail, tubr necr` and `Dysphagia, oropharyngeal`). It recovers 80 of
89 fragments; the last **nine** do not follow the pattern — that remainder was **P27, closed
2026-08-11 (`7e49212`)** by a second, bounded rule under a *third* variant `corrected2`
(lower-case tail + capitalised predecessor + the 24-character CMS short-title cap). `corrected`
stays frozen and still carries the nine, so every `corrected`/`drg` number in this repo is a
nine-fragment number.

### 4c. The two arms preprocess diagnosis text differently — FIXED under `corrected`

The baseline called `preprocess_sentence(diagnosis_description)` before embedding; the BERT
path embedded the raw string. 119 of 145 (82.1%) unique diagnosis descriptions differed between
the two paths, breaking the project's central design constraint (both arms share everything
except the embedding model).

Under `corrected` the BERT path preprocesses too (gated on `use_corrected_preprocessing`), and
145/145 descriptions match. Under `legacy` the divergence is preserved deliberately — so a
`legacy` cross-arm delta is still confounded by preprocessing, and only `corrected`/`drg`
numbers may be compared across arms.

---

## 5. Centre the embeddings before computing cosine

**Status: RETIRED 2026-08-12 (P10), by owner decision — never attempted, and now deliberately not
going to be.** The one-line reason is in the retired table below: it rehabilitates only the
cosine-graded tables, and nothing headline quotes those. The rest of this section is kept as the
argument, because the anisotropy it describes is a real property of the models and someone will
rediscover it. (It previously read *"NOT STARTED — the one item on this list never attempted.
Effort: one line. Impact: unknown, worth measuring."* — and "one line" was always a statement about
the subtraction rather than about the work: moving the numbers means a config-seam field with a
legacy-preserving default, its consumer and banner in the same commit, and a 34–53 minute golden
gate.)

BERT sentence embeddings are anisotropic — they occupy a narrow cone rather than spreading over
the sphere, so arbitrary pairs land at high cosine. This is a documented property of the models
(Ethayarajh 2019; the representation-degeneration literature), not of MIMIC-III.

Demonstrated with random vectors: adding a shared offset to otherwise-unrelated 768D vectors
moves mean pairwise cosine from 0.001 to 0.94, reproducing the three models' measured
similarity profiles with no medical content whatsoever.

**Fix:** subtract the mean embedding from every vector before computing cosine (the core of
"BERT-whitening"). If it works, the scales become comparable across models and **a single
shared threshold becomes meaningful again** — which preserves the clean design the original
protocol wanted. Note that with the `drg` grader now the headline pipeline, this item's payoff
has shrunk: it would rehabilitate the *cosine-graded* tables, not the knobless comparison.

Note this is a genuinely different experiment from the `normalize_embeddings=False` flag, which
cannot affect results at all: L2-normalising before a cosine that already divides by the norms
is algebraically a no-op.

---

## The exact-match ceiling — read before designing any metric

The method predicts by copying an answer from a retrieved patient. **The answer must therefore
exist in the pool to be copied.**

145 unique diagnosis descriptions across 129 admissions; 105 of them (72.4%) occur exactly once
in the entire dataset. If a test patient's diagnosis is one of those, no other patient has it,
so nothing in the pool can supply the right answer.

Measured across the 10 legacy folds: **only 75 of 129 test cases (58.1%) have their correct DRG
present anywhere in their fold's training pool.** Re-measured on the grouped folds: **76/129 =
58.9%** — the leakage fix moved retrievability by exactly one case, so the two defects really
were independent. Both figures are pinned in `tests/test_drg_grader.py`. Per fold the ceiling
ranges from **4/13 (30.8%) to 13/15 (86.7%)** on the canonical grouped split.
*(Corrected 2026-08-12: this read "3/12 (25%) to 13/15 (87%)", which was measured on the
environment-dependent Windows split no committed result ever used — see
[finding 14](../findings/14-fold-split-environment-dependence.md). The **total** 76/129 is
split-invariant and was never wrong; only the per-fold range was.)*

The other ~41% of cases are unwinnable under exact matching — not because retrieval is bad, but
because the answer is not in the library. A perfect retriever therefore scores at most 58.9%.
Any exact-match number must be reported against the ceiling rather than against 1.0 — which is
what `RankMetrics.txt`'s *winnable* population does. (And per item 2 above: the graded-relevance
escape from this ceiling was tried and rejected on measurement; the ceiling is honest, the
ladder was not.)

---

## Still open, in priority order

**Updated 2026-08-12: this table is empty of science, and so is the track.** Every item it listed
has closed, and `P37` and `P13` — the last two — closed later the same day. `P38`, the clean public
repo, is all that remains on the wider plan, and it is backlogged to the final arc by owner
decision.
*(Superseded the same day; the earlier sentence read "the remaining work on this track is `P13`
(mint a baseline-arm golden)".)*

**`P13` closed later the same day** (mint a baseline-arm golden). `tests/golden/stub700-baseline/`
and `tests/test_golden_baseline.py` byte-compare a full 10-fold BioSentVec-arm run at dim 700,
giving `predictS2V`, the per-case `compute_performance_index` path and the `embending_*` builders
their first regression coverage — all of them in `cython_utils.py`, which is shared, so they could
previously move with the whole suite green. `pytest -m golden` is two tests now; budget ~1 hour.

**`P37` closed later the same day** (fix `analyze_score_distributions.py`, which re-implemented the
grader and read no `PipelineConfig`, then re-run it). It now takes `--pipeline`, calls
`get_diagnosis_relevance`, and its committed artifacts are `corrected`-measured and labelled. The
re-run corrected a number rather than merely refreshing one: the fraction of patient pairs at an
exact cosine 1.0 is **1.89% for all three encoders** (312 of 16,512 ordered pairs — those sharing a
diagnosis description), not the 1.49% / 1.62% / 1.31% the simulated scorer reported. See
[../findings/03-metric-saturation.md](../findings/03-metric-saturation.md).

| Item | What | Outcome |
|---|---|---|
| ~~**P40**~~ | Per-case output (new sibling file; the dead handles were left dead) | **DONE 2026-08-12.** `RankCases.txt` (`efa3794`) plus the `results_p40` run; the self-selection test it unblocked is [finding 16](../findings/16-self-selection.md) |
| ~~**P29**~~ | `folds-only` / `preprocess-only` attribution runs (8 runs: 4 arms × 2 configs) | **DONE 2026-08-12**, in one pod session; the split is [finding 15](../findings/15-leakage-preprocessing-attribution.md) |
| ~~**P27**~~ | The last nine comma fragments — landed as a *third* `preprocess_version`, `corrected2`; `corrected` stays frozen | **DONE 2026-08-11** (`7e49212`); its first four-arm run is the [finding 06](../findings/06-preprocessing-defects.md) addendum |
| ~~**P39**~~ | The two arms break score ties differently | **CLOSED 2026-08-12, measured-moot** — the tie-permutation envelope is exactly 0.000000 on both arms, so no convention can move a reported number. No code changed |

### Retired, with the reasoning — 2026-08-11, extended 2026-08-12

| Item | What | Why it is retired rather than deferred |
|---|---|---|
| **P6** | Set-level soft P/R/F1 over diagnosis sets | Its motivation was **dissolved, not postponed**. It existed to kill the degeneracy; findings [04](../findings/04-metric-degeneracy.md) and [13](../findings/13-rank-aware-metrics.md) established there was none to kill — `P == R` is `PR == 1.0`, the answered and all-cases populations being the same set because those arms never abstain. The columns were unlabelled, not wrong. Separately, every term in the proposed formula is a cosine over the diagnosis embeddings, which **reintroduces the self-grading confound P4 removed**. What survives is the intuition that a prediction *set* should be scored against a truth *set* — nothing measures that today; if ever built, build it on the DRG labels, not on cosine. |
| **P10** (item 5 above) | Centre the embeddings before computing cosine | **RETIRED 2026-08-12 by owner decision.** It rehabilitates only the **cosine-graded** tables, and nothing headline quotes those — the result the project leads with is `drg-exact`, where there is no cosine to un-skew. The anisotropy it probes is real and documented; the prize is not. |
| **P11** | Per-model threshold calibration | **RETIRED 2026-08-12 by owner decision.** Moot on the headline path — `drg-exact` collapses the five threshold rows to one, so there is no threshold left to calibrate — and it is the highest golden-risk surface available, since the thresholds live in seven set literals whose *iteration order* decides output row order plus a hand-kept list at `bert_models.py:511`. |
| **P12** | MEAN aggregator alongside MAX | **RETIRED 2026-08-12 by owner decision.** Aggregator plumbing with a known last-ULP hazard — a mean over hash-ordered labels, exactly what [finding 12](../findings/12-drg-grader.md) hit — and no consumer asking for it. If an accumulating aggregator is ever added, sort before reducing. |

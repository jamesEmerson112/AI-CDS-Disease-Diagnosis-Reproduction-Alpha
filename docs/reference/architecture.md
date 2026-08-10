# Architecture

This supersedes the now-archived [`ARCHITECTURE.md`](../../archive/stale-docs/ARCHITECTURE.md).
Everything below was re-checked against the code on **2026-08-08** (the previous revision,
2026-08-04, predated the `src/aicds` package move, the pipeline-config seam, the rank-metrics
package, and the baseline's first successful run — all of which changed this document).

**The three-sentence version:** Two model arms — the BioSentVec baseline and three BERT
variants — share a single pure-Python evaluation core (`src/aicds/utils/cython_utils.py`) so
that the embedding model is the only variable between them. Every run selects a
`PipelineConfig` (`legacy` by default, which reproduces the published pipeline bit-for-bit;
`corrected` and `drg` apply the fixes documented in `docs/findings/`), and both arms now run
end-to-end — the BERT arm anywhere, the baseline on Linux only. The committed results under
`docs/` are the read-only regression oracle, and a byte-exact golden test
(`pytest -m golden`) is what keeps refactoring from silently moving the numbers.

## Layered structure

| Layer | Location | What it actually does | Status |
|---|---|---|---|
| Data | `data/raw/Symptoms-Diagnosis.txt` | 129 MIMIC-III admissions, `;`-delimited, one per line | static, committed |
| Data | `data/folds/Fold0..Fold9/{TrainingSet,TestSet}.txt` | Pre-computed 10-fold CV split (splits on admission — the leaky split, kept for `legacy`) | static, committed — **never regenerate**; the golden depends on it |
| Data | `data/folds_grouped/` | `GroupKFold` on `SUBJECT_ID` — the leakage-free split | **generated** by `scripts/make_folds.py`, gitignored |
| Config | `src/aicds/config.py` | `PipelineConfig` — the seam every correctness fix lands behind: `fold_dir`, `preprocess_version`, `grader`, with legacy-preserving defaults. Named configs `LEGACY`/`CORRECTED`/`FOLDS_ONLY`/`PREPROCESS_ONLY`/`GRADER_DRG` | live; `require_supported_grader` makes an unread config field a loud error |
| Config | `src/aicds/utils/Constants.py` | Fold count, pruning floor, TOP-K bounds, path autodetection | live, imported by everything |
| Shared core | `src/aicds/utils/cython_utils.py` | Preprocessing, cosine similarity, prediction (`predictS2V`), evaluation math | live, imported by both arms — **pure Python**, see below |
| Analysis | `src/aicds/analysis/{rank_metrics,populations,rank_report}.py` | Rank-aware metrics (MRR, Hit@K, P@K, nDCG), the three abstention populations, and the `RankMetrics.txt` writer | live; `rank_metrics.py` is deliberately pure — no I/O, no config |
| Entities | `src/aicds/entity/SymptomsDiagnosis.py` | The one record type actually used: HADM_ID, symptoms, preprocessed diagnosis list | live |
| Entities | `src/aicds/entity/{Admission,Symptom,Drgcodes}.py` | Field-index constants for raw MIMIC-III tables | **orphaned** — only `tests/test_reorganization.py` imports them |
| Model arm | `src/aicds/models/baseline_sent2vec.py` | BioSentVec (sent2vec) 700D embeddings, 10-fold CV | **runs since `c2fee6e`, Linux only**; still executes at import time and reads `$AICDS_PIPELINE` |
| Model arm | `src/aicds/models/bert_models.py` | Bio_ClinicalBERT / BiomedBERT / BlueBERT (sentence-transformers) 768D | live; produced the committed results |
| Evaluation | `src/aicds/evaluation/bert_eval.py` | Alternative evaluator: raw `AutoModel` + manual mean-pooling | **orphaned**, zero callers; salvage `hf_automodel` before deleting |
| Scripts | `scripts/run_baseline.py`, `run_bert_analysis.py`, `run_all_bert_models.py`, `make_folds.py` | CLI entry points | live |
| Scripts | `scripts/analyze_performance.py`, `analyze_score_distributions.py`, `analyze_rank_metrics.py`, `build_dashboard_data.py`, `build_readme_plots.py`, `compare_models.py` | Post-hoc parsing/plotting | mixed — four private `PerformanceIndex` parsers and six-to-seven discovery rules await Phase 3 unification |
| Output | `Prediction_Output_{model}_{timestamp}/` — `PerformanceIndex.txt`, `RankMetrics.txt`, `timing_report.txt` | Per-admission, per-fold, and 10-fold-mean metric rows | curated legacy copies committed at `docs/Prediction_Output_*/`; all fresh runs land in gitignored `results*/` |
| Safety net | `tests/` — 264 fast tests + `tests/golden/stub768/PerformanceIndex.txt` | Characterization tests plus a byte-exact 10-fold golden using a sha256-seeded `StubEncoder` | live; `pytest -m golden` ≈ 43–53 min |

## The central design constraint: one evaluation core, two embedding arms

The entire point of the reproduction is that BioSentVec and the three BERT models are compared
under identical conditions, with the embedding model as the only thing that changes. That
constraint lives in `cython_utils.py`, which both `baseline_sent2vec.py` and `bert_models.py`
import for:

- `preprocess_sentence()` / `preprocess_diagnosis()` — text normalization for both arms
- `load_dataset()` — reads the static fold files
- `cosine_similarity()`, `get_diagnosis_similarity_by_description_max()` — the similarity math
- `init_confusion_matrix()`, `compute_performance_index()`,
  `compute_aggregated_performance_index()`, `print_performance_index()` — metric bookkeeping

What is *not* shared: the actual candidate-ranking loop. The baseline arm calls
`util_cy.predictS2V()` directly. `bert_models.py` does not — it defines its own
`compute_patient_similarity_pairwise()` and `predict_topk_diagnoses_pure()`, written to mirror
the baseline's loop line-for-line. So the two arms' prediction loops are two independent
implementations of the same algorithm — a place the ranking logic could silently drift, and
the reason the roadmap forbids "merging" them casually (`bert_models.py` also keeps a private
copy of `containGreaterOrEqualsValue` that no test exercises — see finding 12's defect list).

Whether the constraint actually held is pipeline-dependent, and this is the single most
important thing to know before comparing arms: **under `legacy` the two arms preprocess
diagnosis text differently (119/145 descriptions differ), so a legacy cross-arm delta is
confounded; under `corrected`/`drg` they match 145/145**
([06-preprocessing-defects](../findings/06-preprocessing-defects.md)).

A few other things about this module that its name and the old docs get wrong or omit:

- **It is pure Python, not Cython.** The original compiled version (`util_cy.pyx` → `util_cy.c`
  via Cython 0.29.24) is archived at
  [`archive/cython_source/util_cy.c`](../../archive/cython_source/util_cy.c). There is no build
  step — it's a `.py` file with a name left over from the port.
- **`sent2vec` is no longer imported at module scope** — as of `c8e4ffd` it is imported inside
  `load_model()`, its only consumer, so the BERT arm imports `cython_utils` with base
  dependencies alone and the `pyproject.toml` split between the `bert` and `baseline` extras is
  finally honest. (`tests/test_bert_symptom_pairwise.py` still AST-loads functions as a relic
  of the old module-scope import; new tests should import normally.)
- **`stop_words` is now defined where it is read** (`5ae01a0`) — the old
  `NameError`-unless-monkeypatched behaviour is gone. Eleven historical monkeypatch sites
  remain (7 in tests, 4 in non-test code) and are Phase 2 cleanup, but they are now redundant
  rather than load-bearing.
- **Embedding dictionaries are keyed by preprocessed text, not `HADM_ID`**, one vector per
  unique symptom/diagnosis *string*, each wrapped in a one-element list so every call site can
  index `emb[0]`. The baseline stores whatever indexable structure `sent2vec` returns; the
  BERT arm wraps each flat vector in a list purely to honour the same `[0]` contract.
  Diverging from this silently changes results rather than raising.
- **Folds are static files, not computed at runtime.** `load_dataset(nFold, name)` just opens
  the fold file — no `train_test_split` or `KFold` call exists in the live path. Which fold
  *directory* it opens comes from the `PipelineConfig`. One sharp edge: `load_dataset` drops
  each line's final character, assuming a trailing newline — a hand-written fold file without
  one silently loses its last symptom (pinned by `tests/test_characterize_dataset.py`).

## Data flow, end to end

1. **Select a pipeline.** `--pipeline legacy|corrected|folds-only|preprocess-only|drg` (the
   baseline reads `$AICDS_PIPELINE`, because it executes at import time). `legacy` is the
   default everywhere and reproduces the published pipeline bit-for-bit.
2. **Load admissions.** Read `Symptoms-Diagnosis.txt`, split each line on `;` into
   `SymptomsDiagnosis` objects, preprocess the diagnosis field at load time
   (`preprocess_diagnosis`: lowercase, split multi-DRG lines on `--`, strip
   `apr:`/`hcfa:`/`ms:` prefixes, dedupe, re-attach the surviving prefix set).
3. **Load the embedding model.** `sent2vec.Sent2vecModel().load_model(...)` for the baseline,
   `SentenceTransformer(model_path)` for a BERT checkpoint — or an injected `encoder=` (this is
   how the golden test swaps in the deterministic `StubEncoder`).
4. **Embed.** Build the symptom and diagnosis embedding dicts in one pass over all 129
   admissions, before any fold loop starts. Under `corrected`/`drg` both arms preprocess the
   diagnosis text before encoding; the dict keys stay raw either way.
5. **For each of the 10 folds** (from `data/folds/` or `data/folds_grouped/` per the config):
   - For every test admission, score every training admission: for each test symptom take the
     max cosine against any train symptom, then average those maxima over
     `max(len(test_symptoms), len(train_symptoms))`.
   - **MAX strategy:** the single highest scorer above `PRUNING_SIMILARITY` (0.5). If none
     clears it, the case abstains — it contributes to neither TP nor FP.
   - **TOP-K strategy** (K = 10…50): rank candidates above the same floor, slice to K.
   - Grade each prediction. Under cosine grading:
     `get_diagnosis_similarity_by_description_max()` takes the MAX cosine over the Cartesian
     product of {true diagnoses} × {predicted diagnoses}, compared against the five thresholds.
     **Under `grader="drg-exact"`:** relevance is an exact DRG description match — no cosine,
     and the five threshold rows collapse to one value
     ([12](../findings/12-drg-grader.md)).
   - Accumulate TP/FP per threshold; accumulate per-case ranks for the rank metrics.
6. **Aggregate across folds.** `print_performance_index()` divides the 10 folds' sums by
   `K_FOLD` — a mean of 10 per-fold numbers, not a micro-average, which is why "10-fold TP"
   reads `12.9` (= 129/10) in the committed output.
7. **Write output.** `PerformanceIndex.txt` (untouched legacy format — the golden's subject),
   `RankMetrics.txt` (additive sibling: MRR/Hit@K/P@K/nDCG on three abstention populations),
   and `timing_report.txt`, into a timestamped directory under the current working directory
   (still no `--out` flag — move runs into `results*/` by hand).

```mermaid
flowchart TB
    subgraph Data["Data"]
        RAW["data/raw/Symptoms-Diagnosis.txt\n129 admissions, ';'-delimited"]
        FOLDS["data/folds/ (legacy, committed)\ndata/folds_grouped/ (generated)"]
    end

    CFG["src/aicds/config.py\nPipelineConfig: fold_dir,\npreprocess_version, grader"]

    subgraph Core["Shared core — src/aicds/utils/"]
        CONST["Constants.py\nK_FOLD=10, PRUNING_SIMILARITY=0.5\nTOP-K 10..50, thresholds 0.6..1.0"]
        CY["cython_utils.py\npure Python despite the name"]
    end

    ANA["src/aicds/analysis/\nrank_metrics (pure) · populations\n· rank_report"]

    SD["src/aicds/entity/SymptomsDiagnosis.py"]

    subgraph Baseline["Baseline arm (Linux only)"]
        BL["baseline_sent2vec.py\nBioSentVec, 700D\nruns at import; reads $AICDS_PIPELINE"]
    end

    subgraph BertArm["BERT arm"]
        BERT["bert_models.py\nBio_ClinicalBERT / BiomedBERT / BlueBERT, 768D\n(golden injects StubEncoder here)"]
    end

    OUT["Prediction_Output_{model}_{timestamp}/\nPerformanceIndex.txt + RankMetrics.txt\n+ timing_report.txt"]
    DOCS["docs/Prediction_Output_*/\ncommitted legacy oracle — read-only"]

    RAW --> BL
    RAW --> BERT
    CFG --> BL
    CFG --> BERT
    FOLDS --> CY
    CONST --> CY
    CY --> BL
    CY --> BERT
    ANA --> BL
    ANA --> BERT
    SD --> BL
    SD --> BERT
    BL --> OUT
    BERT --> OUT
    OUT -.->|"legacy runs only, curated"| DOCS

    subgraph Dead["Not called by any live pipeline"]
        ORPHAN1["entity/Admission.py, Symptom.py, Drgcodes.py"]
        ORPHAN2["evaluation/bert_eval.py"]
    end
```

## Constants (`src/aicds/utils/Constants.py`)

| Name | Value | Meaning |
|---|---|---|
| `K_FOLD` | 10 | Number of CV folds |
| `PRUNING_SIMILARITY` | 0.5 | Minimum symptom-level patient similarity for a candidate to be considered at all — this floor is also the abstention mechanism |
| `MIN_SIMILARITY` | 0 | Floor used to initialize max-tracking loops |
| `TOP_K_LOWER_BOUND` | 10 | First TOP-K strategy evaluated |
| `TOP_K_UPPER_BOUND` | 60 | Exclusive bound — with the increment, K = 10, 20, 30, 40, 50 |
| `TOP_K_INCR` | 10 | Step between TOP-K strategies |
| classification thresholds | `{1.0, 0.9, 0.8, 0.7, 0.6}` | Not a `Constants.py` name — a literal `set` duplicated at **8 sites**, whose *iteration order* determines output row order; `bert_models.py` hard-codes a matching list kept in sync only by a comment |
| `TRAIN` / `TEST` | `TrainingSet.txt` / `TestSet.txt` | Filenames within each fold directory |
| `CH_DIR` | four `.parent`s up from the file | Auto-detected repo root — an off-by-one here silently repoints every data path rather than raising |

Note the thresholds are iterated as a Python `set`, so every block in `PerformanceIndex.txt`
reports them in the fixed-but-non-numeric order `0.9, 1.0, 0.6, 0.8, 0.7` — CPython's hash
order for that literal, not a meaningful ordering. The formatting is load-bearing for the
golden: aggregate rows print threshold `1` (int) while per-case rows print `1.0`, which is why
the golden comparison is deliberately byte-exact rather than float-tolerant.

## Output format and the P = R = F1 collapse

`PerformanceIndex.txt` is plain text: a `FOLD N: LEN train: X, LEN test: Y` header, then for
each test admission six blocks (`MAX`, `TOP-10` … `TOP-50`), each a `TP FP P R FS PR` header
followed by one row per threshold. After all 10 folds, the same structure repeats aggregated
per fold and as the final `10-FOLD PERFORMANCE INDEX` mean. Example, from
[`docs/Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/PerformanceIndex.txt`](../Prediction_Output_Bio_ClinicalBERT_15022026_11-33-48/PerformanceIndex.txt):

```
10-FOLD PERFORMANCE INDEX of MAX SIMILARITY by MAX
	 TP 	 FP 	  P 	 R 	 FS 	 PR
0.9	3.7	9.2	0.2852564102564103	0.2852564102564103	0.2852564102564103	1.0
1	1.9	11.0	0.14615384615384616	0.14615384615384616	0.14615384615384616	1.0
0.6	12.9	0.0	1.0	1.0	1.0	1.0
```

`12.9` is `129 / 10` — the confirmation that the dataset is 129 admissions, not 128.

Across all three committed BERT result files (12,600 metric rows), `P == R == F` exactly and
`PR` is `1.0` in every aggregate row. The mechanism (from `compute_performance_index()`):

```python
precision = tp / (tp + fp)
recall = tp / nrow
f_score = (2 * recall * precision) / (recall + precision)
prediction_rate = (tp + fp) / nrow
```

Every test case with at least one qualifying candidate increments exactly one of TP or FP, so
`tp + fp == nrow` whenever nothing abstains — which is what `PR == 1.0` means — and then
precision and recall are algebraically the same number, and the F-score collapses onto them.
**The full story is in the findings**: for the BERT arms the number is accuracy
([04](../findings/04-metric-degeneracy.md)); the baseline abstains on ~23% of cases so its
P ≠ R ([09](../findings/09-baseline-first-run.md)); and the three columns' true names are
answered-hit-rate, all-cases-hit-rate, and coverage
([13](../findings/13-rank-aware-metrics.md)) — which is exactly why `RankMetrics.txt` reports
three labelled populations instead of inventing a new shape.

## Known defects (current as of 2026-08-08)

1. ~~The baseline arm cannot run.~~ **Fixed in `c2fee6e`**, verified by the full 10-fold run of
   2026-08-05. Two residual defects remain: the module still executes at import time with no
   `__main__` guard (why it reads `$AICDS_PIPELINE`), and it holds two conflicting
   `PerformanceIndex` handles (a `'w'` never closed and an `'a'` on the same path — currently
   `:312`/`:467`). The model file also still loads from `os.getcwd()`, not `data/models/` as
   that directory's README claims.
2. **Output discovery is fragmented.** The baseline writes `Prediction Output_` with a space,
   which no glob matches; the readers disagree about base directories; and there are now four
   `PerformanceIndex` parsers and six-to-seven discovery rules
   ([10](../findings/10-output-path-fragmentation.md)). `build_readme_plots.py` and
   `analyze_performance.py` still glob the wrong place; `build_dashboard_data.py` works.
3. **Every per-case output file is empty** — both arms, always (258 zero-byte files per
   baseline run). Now **P40**, the highest-value open item, because finding 13's untestable
   confound needs per-case relevance. Fix by writing a new sibling file, not by resurrecting
   the dead handles.
4. **The dashboard 404s** — the committed JSON sits in the stray `dashboard/dashboard/` tree
   while the app fetches from `dashboard/public/data/`; re-run `build_dashboard_data.py`.

## Orphaned code

- **`src/aicds/evaluation/bert_eval.py`** (`BioClinicalBERTEvaluator`) has zero callers. It
  implements a genuinely different embedding path — raw `AutoTokenizer` + `AutoModel` with
  manual mean-pooling over `last_hidden_state` — worth salvaging as `encoders/hf_automodel.py`
  for a pooling-strategy ablation, and it holds the repo's only GPU-aware line. Do **not**
  carry its three known defects across: the wrong data path, the same unbound-`entity`
  `NameError` fixed in `c2fee6e`, and an `os.getcwd()` output root.
- **`src/aicds/entity/{Admission,Symptom,Drgcodes}.py`** define field-index constants for raw
  MIMIC-III tables that predate the `Symptoms-Diagnosis.txt` consolidation. Only
  `tests/test_reorganization.py` imports them, so deleting them needs a same-commit test edit.

## What isn't here

No `Dockerfile`, no `.github/workflows`, and no cloud-deploy configuration exist anywhere in
the repository — the RunPod runs of 2026-08-05/06 were manual sessions on a rented box, not
infrastructure. The committed BERT results were produced on Apple Silicon locally (and later
reproduced bit-for-bit on x86 Linux — [08](../findings/08-runtime-and-cost.md)); see the
libomp fix in [`docs/guides/setup.md`](../guides/setup.md).

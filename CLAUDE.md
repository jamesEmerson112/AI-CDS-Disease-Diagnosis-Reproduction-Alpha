# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A reproduction and extension of Comito et al. (2022), *"AI-Driven Clinical Decision Support: Enhancing Disease Diagnosis Exploiting Patients Similarity"* (IEEE Access). Two arms:

1. **Baseline** — BioSentVec (sent2vec, 700D). **Scaffolded from the original authors' code, not this repo owner's contribution.** Published F1 0.489/0.512/0.521 at threshold 0.6. **Currently crashes — see Known defects.**
2. **BERT extension (the original contribution)** — Bio_ClinicalBERT, BiomedBERT, BlueBERT (768D) behind the *identical* pipeline.

### Four independent problems with the headline result

Do not conflate these; fixing one does not fix the others. The first two are metric design; the
last two are plain bugs.

- **Saturation.** All three BERT models score 1.000 at threshold 0.6 because biomedical embeddings are compact (mean pairwise cosine 0.72–0.93 even between *unrelated* diagnoses) and the MAX-over-Cartesian-product aggregator amplifies that, so ~100% of patient pairs clear 0.6. Threshold-dependent; raising it helps. See `docs/findings/03-metric-saturation.md`.
- **Degeneracy.** Across **all 12,600 rows of committed BERT results, precision == recall == F-score**, with zero exceptions. Every test case increments exactly one of TP or FP, so `tp+fp == nrow`, precision reduces to `tp/nrow` (which *is* recall), and their harmonic mean is that same value. **Every "F1" in the committed BERT results is accuracy.** Threshold-*independent*. **Do not extend this to the baseline** — `archive/stale-docs/Reproduce_w_transformers.md:134-143` reports P=0.621/R=0.412/F1=0.489 at TOP-10, i.e. P≠R, meaning that run abstained on cases with no candidate above `PRUNING_SIMILARITY`. If real, degeneracy is a *consequence* of BERT's compact space rather than a structural property of the code. Unresolved; no artifact survives. See `docs/findings/04-metric-degeneracy.md`.

- **Patient leakage.** The folds split on `HADM_ID`, but **129 admissions come from only 100 patients** (one patient has 15). **41 of 129 test cases (31.8%) have another admission from the same `SUBJECT_ID` in their own retrieval pool.** Measured inflation at threshold 1.0: **+0.11 to +0.26** — against encoder differences of 0.015–0.046 and per-fold σ of 0.071–0.124, i.e. the contamination is ~10× the effect under study. Tell: on leaked cases all three encoders score *identically* (0.293 MAX, 0.415 TOP-10); on clean cases they diverge. **Affects both arms** — the folds are shared static files, so the published 0.489/0.512/0.521 carry it too. Fix: `GroupKFold` on `SUBJECT_ID`. See `docs/findings/05-patient-leakage.md`.
- **Preprocessing defects.** `preprocess_sentence` pads `/` then drops `o` as an NLTK stopword, so **`w/o` becomes `w`** — `"Tracheostomy w/o Extensive Procedure"` and `"Tracheostomy w Extensive Procedure"` both become `tracheostomy w extensive procedure`. Negation is destroyed. Separately, symptoms are `,`-split while ICD-9 short titles contain commas, so **80 of 1,805 tokens (4.4%) are orphan fragments** (`" organism NOS"` ×26) that get embedded as if they were symptoms and create spurious 1.0 matches. See `docs/findings/06-preprocessing-defects.md`.

A corollary worth knowing: TOP-K scores rise monotonically with K because one hit suffices and there is no penalty for the other K−1 predictions. That curve is an artifact, not a result. The real fix is a genuine set-level P/R/F1 over the diagnosis sets, which is why "pluggable metrics" is the substance of the next phase rather than a nicety.

**Before designing any exact-match metric:** only **75 of 129 test cases (58.1%)** have their correct DRG present anywhere in their fold's training pool — 105 of the 145 unique diagnoses occur exactly once in the dataset. A perfect retriever therefore caps at 58.1% under exact matching. Prefer graded relevance. Ordered fix list: `docs/plans/correctness-fixes.md`; metric options: `docs/plans/metric-redesign.md`.

**The shared-pipeline constraint is already broken.** The baseline calls `preprocess_sentence` on diagnosis text (`cython_utils.py:226`); the BERT path does not (`bert_models.py:318-332`), so 119/145 (82.1%) of descriptions differ between arms. Any baseline-vs-BERT number is confounded by preprocessing, not just encoder.

## Setting up on a new machine

Full instructions in `docs/guides/setup.md`. The critical parts:

```bash
conda env create -f config/environment.yml   # env "disease-diagnosis", Python 3.9
conda activate disease-diagnosis
git config core.hooksPath .githooks          # data-use guard; hooks are not cloned
```

**`import torch` will fail out of the box on macOS/ARM** with `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`. Two OpenMP runtimes land in one process: conda-forge's `llvm-openmp` (reached via `libopenblas` → numpy/scipy) plus a second copy bundled in pip-installed torch. Fix by pointing torch's copy at conda's — `libtorch_cpu.dylib` resolves `@rpath/libomp.dylib`, and both are LLVM OpenMP compat 5.0.0:

```bash
T="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib/libomp.dylib"
mv "$T" "$T.torch-bundled.bak" && ln -s "$CONDA_PREFIX/lib/libomp.dylib" "$T"
```

**Never use `KMP_DUPLICATE_LIB_OK=TRUE` instead.** It suppresses the error while letting two runtimes coexist, which OpenMP's own documentation warns can silently produce incorrect results. In a numerical reproduction project a silent wrong answer is far worse than a crash. Reinstalling torch restores the bundled library and reintroduces the problem.

Python 3.9 is pinned only by `sent2vec` (baseline-only). HuggingFace models cache to `~/.cache/huggingface/hub`, ~400–450 MB each; copy that directory or set `HF_HOME` to avoid re-downloading.

## Commands

Always run from the repository root — scripts do `sys.path.insert(0, project_root)` and output paths derive from `os.getcwd()`.

```bash
python scripts/run_bert_analysis.py --model 2    # 1=Bio_ClinicalBERT 2=BiomedBERT 3=BlueBERT
python scripts/run_bert_analysis.py --model all  # ~15-30 min on an M-series Mac
python scripts/analyze_score_distributions.py    # regenerates docs/score_distribution_analysis/
python scripts/build_dashboard_data.py           # rebuilds dashboard JSON from docs/Prediction_Output_*/
python scripts/analyze_performance.py [dir]      # PDF report from a PerformanceIndex.txt
python scripts/verify_setup.py                   # smoke test (reports one spurious output/ failure)
```

Tests:

```bash
pytest                    # 32 passed, 1 deselected, ~2s — this is the green baseline
pytest -m network         # opt in to the HuggingFace download test
pytest tests/test_bert_symptom_pairwise.py::TestComputePatientSimilarityPairwise -v
```

Config lives in `pyproject.toml` `[tool.pytest.ini_options]`: `pythonpath = ["."]` (so tests need no `sys.path` hack), markers `network`/`slow`/`golden`, and `addopts = -m 'not network'`. `tests/conftest.py` pins `PYTHONHASHSEED=0` — this matters because `bert_models` builds its encode batch from `list(unique_symptoms)` over a *set*, so hash order changes batch composition, padding, and the last ulp of every embedding.

Dashboard (React 19 + Vite 7 + Tailwind 4 + d3), from `dashboard/`: `npm install && npm run dev`.

## Architecture

Reference: `docs/reference/architecture.md`.

The central design constraint: **both arms share everything except the embedding model.** `src/utils/cython_utils.py` owns preprocessing (`preprocess_sentence`, `preprocess_diagnosis`), fold loading (`load_dataset`), diagnosis scoring (`get_diagnosis_similarity_by_description_max`), and all confusion/performance-matrix math. The model modules supply only embeddings and their prediction loop. Any change to preprocessing or evaluation must land in `cython_utils.py` so both arms stay comparable — that comparability is the entire point.

Data flow: 129 admissions from `data/raw/Symptoms-Diagnosis.txt` (`wc -l` says 128 — no trailing newline; `;`-delimited, diagnoses joined by `--` with `apr:`/`hcfa:`/`ms:` prefixes) → embed unique symptom and diagnosis strings → score every training patient by mean-of-max symptom similarity → prune below `PRUNING_SIMILARITY` (0.5) → take MAX and TOP-K (10–50) → score predictions against ground truth by MAX cosine over the Cartesian product → threshold at 0.6–1.0 → aggregate over 10 folds.

Things that will bite you:

- **`cython_utils.py` is pure Python** despite the name — a hand-translation of the original Cython, archived at `archive/cython_source/util_cy.c`. No build step.
- **It imports `sent2vec` at module scope**, so the BERT path transitively requires that package (not the 21 GB model). `tests/test_bert_symptom_pairwise.py` works around this by AST-loading functions out of `bert_models.py` — preserve that pattern.
- **Embedding dicts are keyed by preprocessed *text*, not HADM_ID**, each value wrapped in a one-element list so callers index `emb[0]`. Diverging silently changes results rather than raising.
- **Folds are fixed committed files**, not computed at runtime. Do not regenerate them. `load_dataset` also drops the final character of each line assuming a trailing newline — a hand-written fold file without one silently loses its last symptom.
- **`baseline_sent2vec.py` runs at import time**; `bert_models.py` exposes `run_analysis(model_id)` and is the better pattern to follow.
- `src/evaluation/bert_eval.py` is orphaned (zero callers) but contains a raw `AutoModel` + mean-pooling path distinct from sentence-transformers pooling, plus the only GPU-aware line in the repo. Salvage before deleting.
- `src/entity/{Admission,Symptom,Drgcodes}.py` have no live callers.

## Outputs

BERT runs write `Prediction_Output_{Model}_{DDMMYYYY_HH-MM-SS}/` into the **current working directory**. The three committed result sets under `docs/` are the project's **regression oracle** — the only record of a working pipeline's exact output. Treat them as read-only.

`PerformanceIndex.txt` columns are `threshold TP FP P R FS PR`. The meaningful numbers are the per-fold blocks and the final `10-FOLD` block. Bear the degeneracy finding in mind when reading any of them.

Constants in `src/utils/Constants.py`: `K_FOLD=10`, `PRUNING_SIMILARITY=0.5`, TOP-K `10..60 step 10` (so K = 10,20,30,40,50). Thresholds are duplicated across 8 sites as set literals `{1, 0.9, 0.8, 0.7, 0.6}`, whose *set iteration order* determines output row order — `bert_models.py` hard-codes a matching list kept in sync only by a comment.

## Known defects

Verified, unfixed, documented:

- **`scripts/run_baseline.py` crashes.** `baseline_sent2vec.py:236` reads `Symptoms-Diagnosis.txt` from the repo root but it lives in `data/raw/` (`FileNotFoundError`); then `:244-247` references an unbound name `entity` (`NameError`) — line 16 binds `entity_module`. The published baseline numbers came from the pre-reorg `CS2V.py` and have never been reproduced by this checkout. Verifying a fix needs the 21 GB model.
- **`scripts/build_readme_plots.py` raises `FileNotFoundError`** — globs `Prediction_Output_*` at the repo root, but the result dirs are under `docs/`. Note `build_dashboard_data.py` globs `docs/` and `analyze_performance.py` globs cwd; all three disagree.
- **The dashboard 404s.** The tracked JSON sits at `dashboard/dashboard/public/data/` (a stray nested dir), while the builder writes `dashboard/public/data/` and `useData.ts` fetches `./data/dashboard-data.json`.
- The baseline model loads from cwd (`os.getcwd() + '/BioSentVec_...bin'`) despite `data/models/README.md` saying `data/models/`.
- `scripts/verify_setup.py` checks for an `output/` directory that no longer exists.

## Data handling

The repo is **public** and contains committed MIMIC-III records under a PhysioNet DUA that prohibits redistribution. This is not a HIPAA breach (the data is de-identified, dates shifted) but it does conflict with the DUA. `.githooks/pre-commit` blocks *new* files under `data/raw`/`data/folds`. **Do not add clinical data to this repository**, and do not rewrite history to remove the existing data without the owner's explicit instruction. See `docs/guides/data-use.md`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A reproduction and extension of Comito et al. (2022), *"AI-Driven Clinical Decision Support: Enhancing Disease Diagnosis Exploiting Patients Similarity"* (IEEE Access). Two arms:

1. **Baseline** — BioSentVec (sent2vec, 700D). **Scaffolded from the original authors' code, not this repo owner's contribution.** Published F1 0.489/0.512/0.521 at threshold 0.6. **It runs, as of 2026-08-05** — first successful execution in this checkout, on rented Linux, reproducing the published TOP-10 F1 to within 0.007 (0.482 vs 0.489). See `docs/findings/09-baseline-first-run.md`. It **cannot** run on Windows: `sent2vec` will not build under MSVC (`-Wno-cpp` is the first of several GCC-only flags), so the baseline arm is Linux-only.
2. **BERT extension (the original contribution)** — Bio_ClinicalBERT, BiomedBERT, BlueBERT (768D) behind the *identical* pipeline.

### Four independent problems with the headline result

Do not conflate these; fixing one does not fix the others. The first two are metric design; the
last two are plain bugs.

- **Saturation.** All three BERT models score 1.000 at threshold 0.6 because biomedical embeddings are compact (mean pairwise cosine 0.72–0.93 even between *unrelated* diagnoses) and the MAX-over-Cartesian-product aggregator amplifies that, so ~100% of patient pairs clear 0.6. Threshold-dependent; raising it helps. See `docs/findings/03-metric-saturation.md`.
- **Degeneracy.** Across **all 12,600 rows of committed BERT results, precision == recall == F-score**, with zero exceptions. Every test case increments exactly one of TP or FP, so `tp+fp == nrow`, precision reduces to `tp/nrow` (which *is* recall), and their harmonic mean is that same value. **Every "F1" in the committed BERT results is accuracy.** Threshold-*independent*. **Do not extend this to the baseline** — `archive/stale-docs/Reproduce_w_transformers.md:134-143` reports P=0.621/R=0.412/F1=0.489 at TOP-10, i.e. P≠R, meaning that run abstained on cases with no candidate above `PRUNING_SIMILARITY`. **RESOLVED 2026-08-05 — it is the embedding space, not the code.** The baseline run produced the missing artifact: P ≠ R in every row, because `PR` (prediction rate) is 0.7679 throughout — **30 of 129 cases (23.3%) abstain** when nothing clears `PRUNING_SIMILARITY`. The tell is one column: `TP+FP` sums to exactly 12.9 (the mean fold test size) for all three BERT arms, forcing `tp+fp == nrow`, but to 9.9 for the baseline. **This unifies saturation and degeneracy into one root cause** — a compact embedding space disabling two different gates, the pruning gate and the scoring threshold. See `docs/findings/04-metric-degeneracy.md` and `09-baseline-first-run.md`.

- **Patient leakage.** The folds split on `HADM_ID`, but **129 admissions come from only 100 patients** (one patient has 15). **41 of 129 test cases (31.8%) have another admission from the same `SUBJECT_ID` in their own retrieval pool.** Measured inflation at threshold 1.0: **+0.11 to +0.26** — against encoder differences of 0.015–0.046 and per-fold σ of 0.071–0.124, i.e. the contamination is ~10× the effect under study. Tell: on leaked cases all three encoders score *identically* (0.293 MAX, 0.415 TOP-10); on clean cases they diverge. **Affects both arms** — the folds are shared static files, so the published 0.489/0.512/0.521 carry it too. Fix: `GroupKFold` on `SUBJECT_ID`. See `docs/findings/05-patient-leakage.md`.
- **Preprocessing defects.** `preprocess_sentence` pads `/` then drops `o` as an NLTK stopword, so **`w/o` becomes `w`** — `"Tracheostomy w/o Extensive Procedure"` and `"Tracheostomy w Extensive Procedure"` both become `tracheostomy w extensive procedure`. Negation is destroyed. Separately, symptoms are `,`-split while ICD-9 short titles contain commas, so **80 of 1,805 tokens (4.4%) are orphan fragments** (`" organism NOS"` ×26) that get embedded as if they were symptoms and create spurious 1.0 matches. See `docs/findings/06-preprocessing-defects.md`.

A corollary worth knowing: TOP-K scores rise monotonically with K because one hit suffices and there is no penalty for the other K−1 predictions. That curve is an artifact, not a result. The real fix is a genuine set-level P/R/F1 over the diagnosis sets, which is why "pluggable metrics" is the substance of the next phase rather than a nicety.

**Before designing any exact-match metric:** only **75 of 129 test cases (58.1%)** have their correct DRG present anywhere in their fold's training pool — 105 of the 145 unique diagnoses occur exactly once in the dataset. A perfect retriever therefore caps at 58.1% under exact matching. Prefer graded relevance. Ordered fix list: `docs/plans/correctness-fixes.md`; metric options: `docs/plans/metric-redesign.md`.

**The shared-pipeline constraint is already broken.** The baseline calls `preprocess_sentence` on diagnosis text (`cython_utils.py:226`); the BERT path does not (`bert_models.py:318-332`), so 119/145 (82.1%) of descriptions differ between arms. Any baseline-vs-BERT number is confounded by preprocessing, not just encoder.

### What can honestly be claimed

At threshold 1.0 — the only setting where the encoders separate at all, since all three BERT models saturate at 1.000 for 0.6–0.9 — the TOP-10 F-scores are **Bio_ClinicalBERT 0.285 > BioSentVec 0.280 > BiomedBERT 0.254 > BlueBERT 0.239**. The baseline sits *inside* the BERT range and the best-vs-baseline gap is **0.005**, against leakage inflation of +0.11 to +0.26 and per-fold σ of 0.071–0.124. **No encoder ranking is supported by this experiment** — that is the defensible claim, and a more interesting one than a spurious win.

### The findings index

`docs/findings/` — 01 baseline status · 02 encoder comparison · 03 saturation · 04 degeneracy · 05 patient leakage · 06 preprocessing defects · **07** why the comparison is not valid yet (the synthesis) · **08** where the runtime goes · **09** the baseline's first run · **10** output-path fragmentation.

**Runtime, measured:** encoding is 0.17–0.45% of wall-clock; over 93% is the single-threaded pure-Python cosine loop, and total fold time varies only 1.7% across the three transformers. **A GPU buys nothing for the encoder arms** — that is why the rented box is CPU-optimised. See `docs/findings/08-runtime-and-cost.md`.

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

Always run from the repository root — output paths derive from `os.getcwd()`. (The `sys.path.insert(0, project_root)` hacks are **gone** as of `2a0b77d`; the package is a real src-layout install, so `pip install -e .` is now required.)

```bash
python scripts/run_bert_analysis.py --model 2    # 1=Bio_ClinicalBERT 2=BiomedBERT 3=BlueBERT
python scripts/run_bert_analysis.py --model all  # ~21 min/model on an M-series Mac; ~14 on a Threadripper 7960X
python scripts/run_baseline.py                   # BioSentVec arm — Linux only, needs the 21 GB model, ~13 min
python scripts/compare_models.py                 # 4-model comparison PDF from results/
python scripts/analyze_score_distributions.py    # regenerates docs/score_distribution_analysis/
python scripts/build_dashboard_data.py           # rebuilds dashboard JSON from docs/Prediction_Output_*/
python scripts/analyze_performance.py [dir]      # PDF report from a PerformanceIndex.txt
python scripts/verify_setup.py                   # smoke test (reports one spurious output/ failure)
```

Tests:

```bash
pytest                    # 93 passed, 2 deselected, ~4s — this is the green baseline
pytest -m golden          # THE SAFETY NET. Full 10-fold pipeline vs a committed
                          # byte-exact reference. ~20 min. Run it before and after
                          # any refactor commit; it is the only thing that catches
                          # the numbers moving while every other test stays green.
pytest -m network         # opt in to the HuggingFace download test
pytest tests/test_bert_symptom_pairwise.py::TestComputePatientSimilarityPairwise -v
```

Config lives in `pyproject.toml` `[tool.pytest.ini_options]`: `pythonpath = [".", "src"]` (so tests need no `sys.path` hack), markers `network`/`slow`/`golden`, and `addopts = -m 'not network and not slow'`.

**`tests/conftest.py`'s `PYTHONHASHSEED` line does not do what it looks like.** CPython reads that variable at interpreter start-up, before any Python code — including `conftest.py` — runs, so `os.environ.setdefault("PYTHONHASHSEED", "0")` has no effect on the current process. It only affects subprocesses. This matters because `bert_models` builds its encode batch from `list(unique_symptoms)` over a *set*, and `preprocess_diagnosis` routes through `list(set(...))` twice, so hash order genuinely varies run to run. Export it in the shell if you need determinism: `PYTHONHASHSEED=0 pytest`. The golden test is immune either way — `StubEncoder` derives every vector from `sha256(text)` alone, which was verified across seeds 0 and 12345.

Dashboard (React 19 + Vite 7 + Tailwind 4 + d3), from `dashboard/`: `npm install && npm run dev`.

## The safety net — read this before changing any code

`tests/golden/stub768/PerformanceIndex.txt` is a full 10-fold run of the real pipeline, minted
from a known-good tree. `pytest -m golden` re-runs the pipeline and compares **byte-for-byte**.

Nothing is trained here, so every number the pipeline emits is a pure function of the input data
and the arithmetic in `cython_utils.py`. That means **any behaviour change is a numerical change**,
and the realistic failure mode of refactoring this repo is not a crash — it is the numbers moving
while every other test stays green. The golden is the only thing that catches that.

- **`tests/stubs.py` `StubEncoder`** derives each vector from `sha256(text)`, so it needs no model
  download, no network, and no GPU, and is immune to batch-composition and hash-seed variation.
  `run_analysis(..., encoder=...)` injects it; omitting the argument is the untouched real path.
- **The comparison is deliberately not float-tolerant.** The formatting carries information:
  aggregate rows print threshold `1` (int, from the set literal) while per-case rows print `1.0`,
  and the F-score prints a bare `0` rather than `0.0` in 1597 rows. A parsing comparator sees none
  of it. Only the wall-clock trailer and the timestamped output path are normalised away.
- **If the golden fails, do not re-mint it to make it pass.** Read the diff. A changed number is
  the finding. Re-minting is correct only when you *intended* to change behaviour, and then the
  diff belongs in the commit message.

Scope rule for the current refactor: **if a change moves the numbers it is out of scope; if it
fixes something that crashes, blocks, or writes to the wrong place it is in scope.** Correctness
work that deliberately changes results is tracked separately in `docs/plans/correctness-fixes.md`.

## Environment traps that produce silently wrong answers

- **`conda run -n disease-diagnosis python` resolves to the wrong interpreter** here — PATH
  shadowing sends it to Homebrew's Python 3.11, which has neither sklearn nor sent2vec, so
  dependency checks come back as false negatives. Invoke
  `/Users/mrbam/miniconda3/envs/disease-diagnosis/bin/python` directly, or activate the env first.
- **An interactive shell's `grep` may be ignore-aware** (aliased to `ugrep --ignore-files`). Before
  `fc5d72b` that combined with unanchored `.gitignore` patterns to return **zero matches** for
  everything under `src/utils/` and `src/entity/` — silently producing false "zero callers"
  conclusions. The patterns are anchored now, but use `command grep` if a zero looks suspicious.
- **`.gitignore` directory patterns must stay anchored.** Bare `entity/`/`utils/` match at any
  depth; they hid `src/utils/` and would have swallowed the entire package after
  `git mv src src/aicds`.
- **On Linux, installing torch into a conda env breaks `import nltk`.** Verified 2026-08-05 on
  Ubuntu 20.04. Symptom: `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version
  'CXXABI_1.3.15' not found (required by .../libicui18n.so.78)`. The CPU torch wheel links the
  **system** libstdc++, which on 20.04 lacks that symbol; once torch is imported it satisfies
  conda's ICU dependency with the older copy, and `sqlite3` — reached via `nltk.translate` — fails
  to load. Conda's own `libstdc++.so.6.0.35` does provide it. **Import order decides it**: `import
  nltk` alone works, `import torch` then `import nltk` does not, which is why the baseline arm ran
  fine before torch was installed. Fix, exported for every run:
  `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib`.
- **After pulling across the `src/` → `src/aicds/` move, delete leftover `src/models`,
  `src/utils`, `src/entity`.** Git removes the tracked files but leaves `__pycache__/` behind, and
  those stale package directories can shadow imports.

## Architecture

Reference: `docs/reference/architecture.md`.

The central design constraint: **both arms share everything except the embedding model.** `src/aicds/utils/cython_utils.py` owns preprocessing (`preprocess_sentence`, `preprocess_diagnosis`), fold loading (`load_dataset`), diagnosis scoring (`get_diagnosis_similarity_by_description_max`), and all confusion/performance-matrix math. The model modules supply only embeddings and their prediction loop. Any change to preprocessing or evaluation must land in `cython_utils.py` so both arms stay comparable — that comparability is the entire point.

Data flow: 129 admissions from `data/raw/Symptoms-Diagnosis.txt` (`wc -l` says 128 — no trailing newline; `;`-delimited, diagnoses joined by `--` with `apr:`/`hcfa:`/`ms:` prefixes) → embed unique symptom and diagnosis strings → score every training patient by mean-of-max symptom similarity → prune below `PRUNING_SIMILARITY` (0.5) → take MAX and TOP-K (10–50) → score predictions against ground truth by MAX cosine over the Cartesian product → threshold at 0.6–1.0 → aggregate over 10 folds.

Things that will bite you:

- **`cython_utils.py` is pure Python** despite the name — a hand-translation of the original Cython, archived at `archive/cython_source/util_cy.c`. No build step.
- **`sent2vec` is no longer imported at module scope** (as of `c8e4ffd`) — it moved inside `load_model()`, its only consumer, so `cython_utils` now imports with base dependencies alone. `tests/test_bert_symptom_pairwise.py` still AST-loads functions out of `bert_models.py` to dodge the old imports; that workaround is now unnecessary and **new tests should import normally**.
- **Embedding dicts are keyed by preprocessed *text*, not HADM_ID**, each value wrapped in a one-element list so callers index `emb[0]`. Diverging silently changes results rather than raising.
- **Folds are fixed committed files**, not computed at runtime, and no generator exists anywhere in the repo or its archive — the original split was produced upstream and only its output was committed. Do not regenerate them during the refactor; regeneration is tracked in `docs/plans/correctness-fixes.md`. `load_dataset` (`cython_utils.py:184`) also drops the final character of each line assuming a trailing newline — all 20 committed fold files end in `0x0a`, but a hand-written one without it silently loses its last symptom. `tests/test_characterize_dataset.py` pins this.
- **`baseline_sent2vec.py` runs at import time**; `bert_models.py` exposes `run_analysis(model_id)` and is the better pattern to follow.
- `src/aicds/evaluation/bert_eval.py` is orphaned (zero callers, 545 lines) but contains a raw `AutoModel` + mean-pooling path distinct from sentence-transformers pooling, plus the only GPU-aware line in the repo (`:74`). Salvage before deleting — but do **not** carry its three known defects across: `:106` the wrong data path, `:121` the same unbound `entity` NameError fixed in `c2fee6e`, `:367` an `os.getcwd()` output root.
- `src/aicds/entity/{Admission,Symptom,Drgcodes}.py` have no live callers — **but `tests/test_reorganization.py:17-19` imports them**, so deletion needs a same-commit test edit.

## Outputs

BERT runs write `Prediction_Output_{Model}_{DDMMYYYY_HH-MM-SS}/` into the **current working directory**; the baseline writes `Prediction Output_{DDMMYYYY HH-MM-SS}/` — space, no model name (see Known defects). The three committed result sets under `docs/` are the project's **regression oracle** — the only record of a working pipeline's exact output. Treat them as read-only.

**`results/` is the working area for new runs and is gitignored** (`.gitignore:59`), which is what keeps DUA-covered output out of git automatically. It is organised by model, then timestamp:

```
results/<model>/<DDMMYYYY_HH-MM-SS>/   # model ∈ baseline, bio_clinical_bert, biomedbert, bluebert
```

`scripts/compare_models.py` reads that layout and emits `results/model_comparison.pdf` across all four arms.

`PerformanceIndex.txt` columns are `threshold TP FP P R FS PR`. The meaningful numbers are the per-fold blocks and the final `10-FOLD` block. Bear the degeneracy finding in mind when reading any of them.

Constants in `src/aicds/utils/Constants.py` (note `CH_DIR` walks **four** parents to reach the repo root — an off-by-one here silently repoints every data path rather than raising): `K_FOLD=10`, `PRUNING_SIMILARITY=0.5`, TOP-K `10..60 step 10` (so K = 10,20,30,40,50). Thresholds are duplicated across 8 sites as set literals `{1, 0.9, 0.8, 0.7, 0.6}`, whose *set iteration order* determines output row order — `bert_models.py` hard-codes a matching list kept in sync only by a comment.

## Known defects

Verified, unfixed, documented:

- ~~**`scripts/run_baseline.py` crashes.**~~ **Fixed in `c2fee6e`** (data path, the unbound `entity` NameError, the discarded `line.replace`) and **verified by a full 10-fold run on 2026-08-05**. Two of the five original baseline defects remain: everything from `baseline_sent2vec.py:186` to EOF still executes **at import time** (no `run_analysis()`, no `__main__` guard), and there are still two conflicting `PerformanceIndex` handles (`:279` opened `'w'` and never closed, `:420` a second handle `'a'` to the same path).
- **Output discovery is broken five ways, not three.** All five sites spell the glob `Prediction_Output_*` — but the **baseline writes `Prediction Output_` with a space** (`baseline_sent2vec.py:272-274`, because `current_time()` returns `"%d/%m/%Y %H:%M:%S"` and only `/` and `:` get scrubbed), so **no glob in the repo matches baseline output at all**. The five sites also disagree four ways about the base directory: `build_dashboard_data.py:172` uses `docs/`, `build_readme_plots.py:30` the repo root (hence its `FileNotFoundError`), `analyze_performance.py:27` the cwd, `test_reorganization.py:75` a recursive `**`, `test_golden.py:259` a pytest tmp dir. **Fix direction is forced**: `test_golden.py:115` hard-codes `\d{8}_\d{2}-\d{2}-\d{2}`, so writers must unify onto the *underscore* spelling. See `docs/findings/10-output-path-fragmentation.md`.
- **Every per-case output file is empty.** The baseline emits 258 zero-byte files; `cython_utils.py:65-66` opens the two handles and `:170-171` closes them with **nothing written in between**. The BERT arm never opens them at all yet still creates the `Fold*/` dirs (`bert_models.py:451-456`). Neither arm has ever produced per-case output.
- **The dashboard 404s.** The tracked JSON sits at `dashboard/dashboard/public/data/` (a stray nested dir), while the builder writes `dashboard/public/data/` and `useData.ts` fetches `./data/dashboard-data.json`.
- The baseline model loads from cwd (`os.getcwd() + '/BioSentVec_...bin'`) despite `data/models/README.md` saying `data/models/`.
- `scripts/verify_setup.py` checks for an `output/` directory that no longer exists.

## Data handling

The repo is **public** and contains committed MIMIC-III records under a PhysioNet DUA that prohibits redistribution. This is not a HIPAA breach (the data is de-identified, dates shifted) but it does conflict with the DUA. **Do not add clinical data to this repository**, and do not rewrite history to remove the existing data without the owner's explicit instruction. See `docs/guides/data-use.md`.

`.githooks/pre-commit` blocks two things: new files under `data/raw`/`data/folds`, **and** any new file containing 20+ distinct `HADM_ID`s wherever it lives. The second rule exists because the first was insufficient — the generated golden under `tests/` carries all 129 IDs and sailed past a path-only check. The golden was committed deliberately with `--no-verify`, on the reasoning that the same 129 IDs are already published in the three `docs/Prediction_Output_*` files, so it adds no new exposure. Any future `--no-verify` on this hook deserves the same explicit reasoning.

## Where the work is

`docs/plans/revival-roadmap.md` is the sequenced plan and carries current status.

- **Phases 0, 1, 4 — done.** Environment repaired, data-use guard, docs reorganised, and the Phase 1 safety net (characterization tests + `StubEncoder` + byte-exact golden).
- **Phase 2 — mostly done, merged to `main` via PR #1 (`7da5901`).** Landed: the `src/aicds` package move, real src-layout `pyproject`, `ensure_nltk_data`/`format_time` consolidated into `aicds.utils.runtime`, `stop_words` defined where it is read, three of the baseline's five crash bugs fixed. **Four items outstanding:** delete the **8** `stop_words` monkeypatches (the roadmap says 4 — it is wrong); the `--out`/`results` root plus unified globs; salvage `hf_automodel` before deleting `bert_eval.py`; the remaining dead code and the last two baseline defects.
- **Phase 3 — not started.** `SentenceEncoder` Protocol + encoder registry, a `main.py` CLI (`bert_models.py:80` still calls `input()` interactively), one `PerformanceIndex` parser replacing three, one run-discovery rule replacing five, the dashboard fix. Phase 3 is the ship point.
- **Deferred to a dedicated session** — everything in `docs/plans/correctness-fixes.md` and `docs/plans/metric-redesign.md`. Those deliberately change the numbers, which is why they cannot run concurrently with a refactor whose safety mechanism is that the numbers must not change.

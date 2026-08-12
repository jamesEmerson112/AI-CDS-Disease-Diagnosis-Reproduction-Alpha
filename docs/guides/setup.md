# Setup — getting this running on a new machine

This is written to be followed start-to-finish on a machine that has never seen this project.
Updated 2026-08-08 — the earlier version predated the baseline's first successful run, the
`src/aicds` package move, and the growth of the test suite, and was wrong about all three.

**Read this first:** you do **not** need the 21 GB BioSentVec model or `sent2vec` to do the work
this project is actually about. Those belong to the baseline arm only. The BERT arm — the
extension, the analysis, the plots, and the dashboard — needs neither. Skip straight to
"BERT arm" unless you specifically want to re-run the original authors' baseline (Linux only —
see section 5).

---

## 1. Clone and create the environment

Clone, create the conda environment (Python 3.9 — pinned only because `sent2vec` has no wheels
for newer versions), and install the package itself in editable mode (required since the
src-layout move — without it, `import aicds` fails):

```bash
git clone https://github.com/jamesEmerson112/AI-CDS-Disease-Diagnosis-Reproduction-Alpha.git
cd AI-CDS-Disease-Diagnosis-Reproduction-Alpha

conda env create -f config/environment.yml   # creates env "disease-diagnosis", Python 3.9
conda activate disease-diagnosis
pip install -e .
```

Enable the data-use pre-commit hook (once per clone — hooks are not cloned):

```bash
git config core.hooksPath .githooks
```

See [data-use.md](data-use.md) for why. Short version: this repo is public and carries
MIMIC-III data under a DUA that forbids redistribution, so the hook blocks *new* clinical
data from being committed.

---

## 2. The one non-obvious failure you will hit (macOS/ARM)

On macOS with Apple Silicon, `import torch` fails out of the box:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

**Cause.** Two OpenMP runtimes end up in one process. conda-forge installs
`llvm-openmp` at `$CONDA_PREFIX/lib/libomp.dylib`, which `libopenblas` (and therefore
numpy/scipy) links against. pip-installed `torch` ships a *second* copy at
`site-packages/torch/lib/libomp.dylib`. Loading both trips OpenMP's duplicate-runtime guard.

**Fix.** Point torch's copy at the conda one. `libtorch_cpu.dylib` resolves
`@rpath/libomp.dylib`, so a symlink is sufficient, and both are LLVM OpenMP with the same
compat version (5.0.0):

```bash
T="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib/libomp.dylib"
mv "$T" "$T.torch-bundled.bak"
ln -s "$CONDA_PREFIX/lib/libomp.dylib" "$T"
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
# 2.8.0 True
```

> **Do not** "fix" this with `KMP_DUPLICATE_LIB_OK=TRUE`. That flag suppresses the error and
> lets two OpenMP runtimes coexist, which the OpenMP authors explicitly warn *can silently
> produce incorrect results*. In a numerical reproduction project, a silent wrong answer is
> far worse than a crash.

Reinstalling or upgrading torch restores the bundled library and reintroduces the problem.
Re-apply the symlink if `import torch` starts failing again.

### The Linux counterpart (Ubuntu 20.04)

Installing torch into the conda env breaks `import nltk` — verified 2026-08-05. Symptom:
`ImportError: ... libstdc++.so.6: version 'CXXABI_1.3.15' not found`. The CPU torch wheel
links the *system* libstdc++, which on 20.04 lacks that symbol, and import order decides
whether nltk loads. Fix, exported for every run:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```

---

## 3. Verify the install

Run the fast suite — unit and characterization tests, no model download, no network:

```bash
pytest
# 499 passed, 3 deselected      (measured 2026-08-12; it read 413 after the
#                                C1-C8 refactor and 264 before it)
```

The deselected tests are opt-in. One downloads a model from HuggingFace; the others are slow.
The one that matters is the **golden regression** — it runs the full 10-fold pipeline against a
committed byte-exact reference:

```bash
pytest -m golden      # 34-53 minutes, measured eight times: 34:12, 42:28, 43:28, 43:50,
                      # 44:32, ~50:00, 52:35, 53:10 (34:12 = 2026-08-12, Windows, numpy-2.0.2
                      # venv). NOT the ~20 min an earlier version of this file
                      # claimed — nor the ~20 min tests/test_golden.py's own docstring
                      # still claims at :68, which is left alone on purpose.
```

Run that one before and after any refactoring commit. Nothing else in the suite will notice if
the pipeline's numbers change, because every other test checks a piece rather than the whole.
If it fails, read the diff — a changed number is the finding; do not re-mint the reference to
make it pass.

The HuggingFace download test, when you want to confirm network access and the model cache:

```bash
pytest -m network
```

One determinism footnote: `conftest.py`'s `PYTHONHASHSEED` line cannot affect the current
process (CPython reads that variable before any Python code runs). If you need hash-stable
runs, export it in the shell: `PYTHONHASHSEED=0 pytest`.

---

## 4. Running the BERT arm

Every run takes `--pipeline legacy|corrected|folds-only|preprocess-only|drg` and `--out ROOT`;
`legacy` is the default and is bit-identical to the original published pipeline. For grouped-fold
pipelines, generate the folds first (they are gitignored, so a fresh clone does not have them):

```bash
python scripts/make_folds.py --verify              # writes data/folds_grouped/
                                                   # needs numpy 2.x — see the pin note below

python scripts/run_bert_analysis.py --model 2      # 1=Bio_ClinicalBERT 2=BiomedBERT 3=BlueBERT
python scripts/run_bert_analysis.py --model all --pipeline drg --out results_drg
# ~21 min per model on an M-series Mac; ~11.5 min on a 32-vCPU Linux box;
# ~14 min on a Threadripper 7960X. "all" runs the three sequentially.
```

**The numpy pin is load-bearing for that first command.** `config/environment.yml` pins
`numpy>=2.0,<3` because `GroupKFold` breaks its ~85 single-admission ties through `np.argsort`,
whose unstable-sort behaviour changed between numpy 1.x and 2.x — a numpy-1.x environment builds a
*different* (equally leak-free, equally deterministic) split that is not comparable with any
committed result. See `docs/findings/14-fold-split-environment-dependence.md`. Under the pin,
`--verify` reproduces the canonical split on Windows and Linux alike (verified 2026-08-12: digest
match, `VERIFY: PASS`, 0 leaked), so regenerating locally is the normal path. `--verify` still
**warns** on a digest mismatch, which is what catches an environment that ignored the pin. If you
add `torch` to an environment, it must be **`torch>=2.3`** — the numpy-2-compatible line.

Run from the repository root. With no `--out`, output paths are built from the current working
directory, so running from elsewhere scatters results — and that flat layout is what the golden
pins, so it will not change. `--out ROOT` writes `ROOT/{model_key}/{DDMMYYYY_HH-MM-SS}/` instead,
which is the `results*/` layout `compare_models.py --results-dir` and `analyze_rank_metrics.py`
read directly, so nothing has to be moved by hand.

Either way a run writes `PerformanceIndex.txt`, `RankMetrics.txt`, `timing_report.txt`, and
`run_metadata.json` — the last written after everything else, so its presence means the run
finished. It records the git SHA, the pipeline by name and by all three config fields, and a
content hash of the fold split, which is how a number gets defended later.

**`--out` is guarded.** A run leaves per-case files *named by* `HADM_ID`, and because those files
are empty the pre-commit hook cannot see them — the ignore rule is the only defence. So an
`--out` that resolves inside the repository with no `.gitignore` rule covering it is refused
outright (`--out scratch`, and note `--out out` too: the ignore entry is `output/`, not `out/`).
Any `results*` path is fine.

**Model downloads.** The three encoders come from HuggingFace on first use and land in
`~/.cache/huggingface/hub` (roughly 400–450 MB each). To move them to a new machine without
re-downloading, copy that directory, or set `HF_HOME` to a shared location. `HF_HUB_OFFLINE=1`
forces cache-only and fails loudly rather than silently downloading.

---

## 5. Running the baseline arm (optional — Linux only)

The BioSentVec baseline is scaffolded from the original authors' code, not part of the
extension work. **It runs, as of 2026-08-05** — the crash bugs were fixed in `c2fee6e`, and a
full 10-fold run on rented Linux reproduced the published TOP-10 F1 to within 0.007 (0.4824 vs
0.489; see `docs/findings/09-baseline-first-run.md`).

**It cannot run on Windows.** `sent2vec` will not build under MSVC — `-Wno-cpp` is the first of
several GCC-only compiler flags it passes. Use Linux (or WSL2 with ≥24 GB allocated to it).

What you need:

- Python 3.9 specifically, and a C toolchain
- `pip install -e ".[baseline]"` — this pulls `sent2vec` from `epfml/sent2vec` on GitHub.
  **Do not `pip install sent2vec` from PyPI**: that is an unrelated project of the same name
  with no `Sent2vecModel` class, and installing it produces the archived
  `AttributeError` in `archive/baseline_debug.txt`.
- `BioSentVec_PubMed_MIMICIII-bigram_d700.bin` (**20.93 GiB**, needs to be fully RAM-resident)
  from [BioSentVec](https://github.com/ncbi-nlp/BioSentVec), placed at the **repository root**
  — the loader reads it from the working directory, despite what `data/models/README.md`
  says (a known, unfixed defect)

Then:

```bash
python scripts/run_baseline.py --pipeline drg --out results_drg   # ~13 min on a 32-vCPU box
```

`AICDS_PIPELINE=drg python scripts/run_baseline.py` still works, and is what
`python -m aicds.models.baseline_sent2vec` reads, but the flag is the supported form now: the
module gained a `run_analysis()` and a `__main__` guard in `9c9e251`, so importing it no longer
executes a pipeline.

The baseline used to write `Prediction Output_...` — with a space — which no discovery glob in
the repo matched (`docs/findings/10-output-path-fragmentation.md`). Since `9d08c94` it writes
`Prediction_Output_BioSentVec_{stamp}` like the other arm. Older SHAs still emit the space
spelling, which is why `.gitignore` keeps both patterns.

---

## 6. The dashboard

Install and serve (node_modules is not committed):

```bash
cd dashboard
npm install
npm run dev
```

**Known broken:** the app currently 404s, because the committed JSON sits at a stray nested
path (`dashboard/dashboard/public/data/`) while the app fetches `./data/dashboard-data.json`
from `dashboard/public/data/`. Regenerate the JSON into the right place before serving:

```bash
python scripts/build_dashboard_data.py    # from the repo root
```

---

## 7. Known broken

Verified as of 2026-08-11. None of these are caused by your machine.

| What | Symptom | Cause |
|---|---|---|
| The dashboard | Blank page / 404 | The committed JSON sits at `dashboard/dashboard/public/data/`, a stray nested directory, while the app fetches from `dashboard/public/data/`. Re-run the builder (section 6). |
| `compare_models.py` on the three pre-C8 result trees | One `[WARN] no run_metadata.json …` line per invocation | Those runs predate `5a52d26`, and nothing retrofits provenance onto a run nobody can re-derive. Cosmetic and **permanent for those three trees** (updated 2026-08-12); every 2026-08-12 tree carries metadata and prints `[INFO]` instead. |
| Per-case output files | All 258 are zero bytes | Neither arm has ever written them, and the dead handles stay dead by design. **P40 closed 2026-08-12**: the per-case data ships as a new sibling, `RankCases.txt` (finding 16). |
| The BioSentVec model path | Loads from the working directory | `data/models/README.md` says `data/models/`; the loader disagrees. Put the `.bin` at the repository root. |

Four entries left this table on 2026-08-11. `scripts/build_readme_plots.py` raised
`FileNotFoundError` on every invocation and now runs, reproducing its six committed SVGs
byte-identically (`7927a88`). `scripts/verify_setup.py` reported a spurious failure for an
`output/` directory nothing writes; the check is gone and it exits 0 (`7927a88`). No script could
find baseline runs, because the baseline wrote a space where every glob expected an underscore;
both arms now go through `aicds.runs` (`9d08c94`). And `scripts/run_baseline.py` headed this table
with two crash bugs until `c2fee6e`, verified by the 2026-08-05 run.

---

## 8. Things that are easy to get wrong

- **The dataset is 129 admissions, not 128.** `wc -l` reports 128 because the file has no
  trailing newline.
- **`src/aicds/utils/cython_utils.py` is pure Python** despite the name — a hand-translation of
  the original Cython, archived at `archive/cython_source/util_cy.c`. There is no build step.
- **`sent2vec` is no longer imported at module scope** (moved inside `load_model()` as of
  `c8e4ffd`), so the BERT arm imports `cython_utils` with base dependencies alone. Python 3.9
  stays pinned only for the baseline extra.
- **The reported "F1" is not an F1 for the BERT arms.** In all 12,600 committed BERT rows,
  precision == recall == F-score, because those arms never abstain — the number is accuracy.
  The baseline's P ≠ R because it abstains on ~23% of cases. The columns' true names are
  answered-hit-rate / all-cases-hit-rate / coverage — see `docs/findings/04` and `13`.
- **Always run scripts from the repository root.** Output locations derive from `os.getcwd()`.
- **`conda run -n disease-diagnosis python` may resolve to the wrong interpreter** (PATH
  shadowing → Homebrew Python without the dependencies). Activate the env, or invoke the env's
  `bin/python` directly.

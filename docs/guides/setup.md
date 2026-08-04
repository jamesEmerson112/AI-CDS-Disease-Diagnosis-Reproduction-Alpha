# Setup — getting this running on a new machine

This replaces the older setup docs, which describe a `CS2V.py` layout that no longer exists.
It is written to be followed start-to-finish on a machine that has never seen this project.

**Read this first:** you do **not** need the 21 GB BioSentVec model or `sent2vec` to do the work
this project is actually about. Those belong to the baseline arm only. The BERT arm — the
extension, the analysis, the plots, and the dashboard — needs neither. Skip straight to
"BERT arm" unless you specifically want to re-run the original authors' baseline.

---

## 1. Clone and create the environment

```bash
git clone https://github.com/jamesEmerson112/AI-CDS-Disease-Diagnosis-Reproduction-Alpha.git
cd AI-CDS-Disease-Diagnosis-Reproduction-Alpha

conda env create -f config/environment.yml   # creates env "disease-diagnosis", Python 3.9
conda activate disease-diagnosis
```

Enable the data-use pre-commit hook (once per clone — hooks are not cloned):

```bash
git config core.hooksPath .githooks
```

See [data-use.md](data-use.md) for why. Short version: this repo is public and carries
MIMIC-III data under a DUA that forbids redistribution, so the hook blocks *new* clinical
data from being committed.

---

## 2. The one non-obvious failure you will hit

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

---

## 3. Verify the install

```bash
pytest
# 32 passed, 1 deselected
```

The deselected test downloads a model from HuggingFace. Run it explicitly when you want to
confirm network access and the model cache:

```bash
pytest -m network
```

---

## 4. Running the BERT arm

```bash
python scripts/run_bert_analysis.py --model 2      # 1=Bio_ClinicalBERT 2=BiomedBERT 3=BlueBERT
python scripts/run_bert_analysis.py --model all    # all three, ~15-30 min on an M-series Mac
```

Run from the repository root. Output paths are built from the current working directory, so
running from elsewhere scatters results into the wrong place. Each run writes
`Prediction_Output_{Model}_{DDMMYYYY_HH-MM-SS}/` containing `PerformanceIndex.txt` and
`timing_report.txt`.

**Model downloads.** The three encoders come from HuggingFace on first use and land in
`~/.cache/huggingface/hub` (roughly 400–450 MB each). To move them to a new machine without
re-downloading, copy that directory, or set `HF_HOME` to a shared location. `HF_HUB_OFFLINE=1`
forces cache-only and fails loudly rather than silently downloading.

---

## 5. Running the baseline arm (optional, and currently broken)

The BioSentVec baseline is scaffolded from the original authors' code, not part of the
extension work. **It does not currently run** — see "Known broken" below. Reproducing it
would need:

- Python 3.9 specifically (`sent2vec` has no wheels for newer versions and needs a C toolchain)
- `pip install -e ".[baseline]"`
- `BioSentVec_PubMed_MIMICIII-bigram_d700.bin` (~21 GB) from
  [BioSentVec](https://github.com/ncbi-nlp/BioSentVec), placed at the **repository root**
  (`util_cy.load_model()` reads it from the working directory, despite what
  `data/models/README.md` says)

The published baseline numbers (F1 0.489 / 0.512 / 0.521) come from the pre-reorganization
`CS2V.py` and have not been reproduced by the current checkout.

---

## 6. The dashboard

```bash
cd dashboard
npm install          # node_modules is not committed
npm run dev
```

The dashboard reads `dashboard/public/data/dashboard-data.json`. That file is currently
misplaced (see below), so regenerate it before serving:

```bash
python scripts/build_dashboard_data.py    # from the repo root
```

---

## 7. Known broken

Verified as of this writing. None of these are caused by your machine.

| What | Symptom | Cause |
|---|---|---|
| `scripts/run_baseline.py` | `FileNotFoundError`, then `NameError` | Reads `Symptoms-Diagnosis.txt` from the repo root, but it lives in `data/raw/`. Then references an unbound name `entity` at `baseline_sent2vec.py:244`. |
| `scripts/build_readme_plots.py` | `FileNotFoundError` | Globs `Prediction_Output_*` at the repo root; the result directories were moved under `docs/`. |
| The dashboard | Blank page / 404 | The committed JSON sits at `dashboard/dashboard/public/data/`, a stray nested directory, while the app fetches `./data/dashboard-data.json`. |

---

## 8. Things that are easy to get wrong

- **The dataset is 129 admissions, not 128.** `wc -l` reports 128 because the file has no
  trailing newline. Several docs still say 128.
- **`src/utils/cython_utils.py` is pure Python** despite the name — a hand-translation of the
  original Cython, archived at `archive/cython_source/util_cy.c`. There is no build step.
- **It imports `sent2vec` at module scope**, so the BERT path currently drags in that
  dependency even though it never uses it. This is why the environment still pins Python 3.9.
- **The reported "F1" is not an F1.** Across all 12,600 metric rows in the committed results,
  precision == recall == F-score, because `tp + fp == nrow` by construction. Every published
  score is accuracy. See the findings docs.
- **Always run scripts from the repository root.** Output locations derive from
  `os.getcwd()`.

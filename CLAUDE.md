# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A reproduction and extension of Comito et al. (2022), *"AI-Driven Clinical Decision Support: Enhancing Disease Diagnosis Exploiting Patients Similarity"* (IEEE Access). Two arms:

1. **Baseline** — BioSentVec (sent2vec, 700D). **Scaffolded from the original authors' code, not this repo owner's contribution.** Published F1 0.489/0.512/0.521 at threshold 0.6. **It runs, as of 2026-08-05** — first successful execution in this checkout, on rented Linux, reproducing the published TOP-10 F1 to within 0.007 (0.482 vs 0.489). See `docs/findings/09-baseline-first-run.md`. It **cannot** run on Windows: `sent2vec` will not build under MSVC (`-Wno-cpp` is the first of several GCC-only flags), so the baseline arm is Linux-only.
2. **BERT extension (the original contribution)** — Bio_ClinicalBERT, BiomedBERT, BlueBERT (768D) behind the *identical* pipeline.

### The seven problems, and their status

Do not conflate these; fixing one does not fix the others. **Check the status marker before spending
any effort on one:** four are fixed, one is explained rather than open, one is confined, and one
cannot be closed by any amount of code. Three of the seven were found only *because* an earlier one
was fixed — removing a knob keeps revealing the one it was masking, so expect the list to grow again
rather than treating it as complete.

- **Saturation. OPEN, but confined.** All three BERT models score 1.000 at threshold 0.6 because biomedical embeddings are compact (mean pairwise cosine 0.72–0.93 even between *unrelated* diagnoses) and the MAX-over-Cartesian-product aggregator amplifies that, so ~100% of patient pairs clear 0.6. Threshold-dependent; raising it helps. **It does not arise under `--pipeline drg` at all**, which has no threshold. So it bounds the cosine-graded tables below 1.0 and nothing else. **Re-measured 2026-08-12 under `corrected` and against the shipped grader (P37): the 0.72–0.93 range survives unchanged and so does the conclusion**; what moved is the minima (Bio_ClinicalBERT 0.65 → 0.59, so it is now BiomedBERT *alone* that has no diagnosis pair below the 0.6 threshold) and BlueBERT's ≥ 0.6 share, 99.96% → 99.65%. See `docs/findings/03-metric-saturation.md`.
- **Degeneracy. EXPLAINED 2026-08-06, not open.** Across **all 12,600 rows of committed BERT results, precision == recall == F-score**, with zero exceptions. Every test case increments exactly one of TP or FP, so `tp+fp == nrow`, precision reduces to `tp/nrow` (which *is* recall), and their harmonic mean is that same value. **Every "F1" in the committed BERT results is accuracy.** Threshold-*independent*. **Do not extend this to the baseline** — `archive/stale-docs/Reproduce_w_transformers.md:134-143` reports P=0.621/R=0.412/F1=0.489 at TOP-10, i.e. P≠R, meaning that run abstained on cases with no candidate above `PRUNING_SIMILARITY`. The baseline run supplied the missing artifact: P ≠ R in every row, because `PR` (prediction rate) is 0.7679 throughout — **30 of 129 cases (23.3%) abstain** when nothing clears `PRUNING_SIMILARITY`. The tell is one column: `TP+FP` sums to exactly 12.9 (the mean fold test size) for all three BERT arms, forcing `tp+fp == nrow`, but to 9.9 for the baseline — still the fastest diagnostic available. **What finding 13 settled is the label: `P == R` is `PR == 1.0`, i.e. the answered and all-cases populations being the *same set* because those arms never abstain. It was never the metric collapsing — the columns were unlabelled, not wrong.** `RankMetrics.txt` reproduces legacy `P`/`R`/`PR` bit-exactly from an independent code path, which is what proves it. See `docs/findings/04-metric-degeneracy.md`, `09-baseline-first-run.md` and `13-rank-aware-metrics.md`.
- **Patient leakage. FIXED 2026-08-05 (`c2115ba`), 41 → 0.** The legacy folds split on `HADM_ID`, but **129 admissions come from only 100 patients** (one patient has 15), so **41 of 129 test cases (31.8%) had another admission from the same `SUBJECT_ID` in their own retrieval pool.** Measured inflation at threshold 1.0 was **+0.11 to +0.26** — against encoder differences of 0.015–0.046 and per-fold σ of 0.071–0.124, i.e. the contamination was ~10× the effect under study. Tell: on leaked cases all three encoders scored *identically* (0.293 MAX, 0.415 TOP-10); on clean cases they diverged. **It affected both arms**, so the published 0.489/0.512/0.521 carry it too. Fixed by `GroupKFold` on `SUBJECT_ID` (`scripts/make_folds.py` → `data/folds_grouped/`), recounted independently rather than trusting `--verify`. **`legacy` still uses the leaky folds on purpose**, so the golden never moves. **Attributed per arm 2026-08-12 (finding 15), which is what turns "it affected both arms" from an inference into a measurement:** closing the leak alone costs **+0.027 (Bio_ClinicalBERT) to +0.070 (BioSentVec)** of TOP-10 F at threshold 1.0, and it is the **larger term in all four rows** — 3.5× to 6.3× the preprocessing term, and unboundedly so for BiomedBERT, whose preprocessing term is exactly zero. At the cell Comito et al. publish (threshold 0.6) leakage is **54.3% of the 0.0902 baseline drop**, against 43.8% preprocessing and 1.9% interaction. See `docs/findings/05-patient-leakage.md` and `15-leakage-preprocessing-attribution.md`.
- **Preprocessing defects. FIXED 2026-08-05 (`c2115ba`); the last nine of 89 fragments closed 2026-08-11 (`7e49212`, P27) under a *third* variant `corrected2`, so `corrected` — the pipeline every headline number comes from — still carries those nine.** `preprocess_sentence` padded `/` then dropped `o` as an NLTK stopword, so **`w/o` became `w`** — `"Tracheostomy w/o Extensive Procedure"` and `"Tracheostomy w Extensive Procedure"` both became `tracheostomy w extensive procedure`. Negation destroyed. Separately, symptoms are `,`-split while ICD-9 short titles contain commas, so **80 of 1,805 tokens (4.4%) were orphan fragments** (`" organism NOS"` ×26) embedded as if they were symptoms, creating spurious 1.0 matches. Under `corrected`, `w/o` survives as `without` and 80 of 89 fragments are rejoined. **Its effect has a sign, and the sign varies by arm — measured 2026-08-12 (finding 15), TOP-10 at threshold 1.0.** The text fix *raises* BioSentVec (−0.0124) and BlueBERT (−0.0077), *costs* Bio_ClinicalBERT (+0.0077), and moves BiomedBERT by **exactly +0.000000** — a bit-identical cell, recorded as measurement with no mechanism asserted. So "a defect fix must cost score" is not what happened here: it costs one arm, pays two, and is invisible to a fourth on the same 129 admissions. **P27 closed 2026-08-11 (`7e49212`) as a third variant `corrected2`** — `corrected` stays frozen — and its first four-arm run moved BioSentVec 0.2163→0.1856 and Bio_ClinicalBERT 0.2491→0.2424 while leaving BiomedBERT and BlueBERT **bit-identical in that cell** (they move in other rows; the bit-identity is cell-local). See `docs/findings/06-preprocessing-defects.md`, whose addendum carries the corrected2 run.
- **Self-grading. FIXED 2026-08-06 (`75b6530`) — and the bias turned out to be possible but never actual.** Each arm scored its own predictions by cosine *in the space that produced the retrieval*, so a compact space could mark its own work leniently. `--pipeline drg` compares DRG label strings with no cosine anywhere. **It reproduces corrected cosine at threshold 1.0 bit-exactly** — 144 numbers, 4 arms × 6 aggregators × 6 columns, not one differing digit. So P4 did not move the numbers; it removed the reason to doubt them, which is the more useful outcome and the one worth reporting. See `docs/findings/12-drg-grader.md`.
- **Rank-blindness. FIXED 2026-08-06 (`5393cab`), additively.** A hit at rank 1 and a hit at rank 50 counted identically, so TOP-K rose with `K` by construction. MRR has no `K`; Precision@K falls with `K` for every arm, inverting the TOP-K curve. `PerformanceIndex.txt` is untouched and the golden stayed byte-exact — rank metrics go to a sibling `RankMetrics.txt`. See `docs/findings/13-rank-aware-metrics.md`.
- **Abstention asymmetry. OPEN, and not closable.** The baseline declines to answer 24.4% of cases; all three BERT arms never do. So **how you score an abstained case reorders all four arms and inverts every sign** — baseline last when abstention scores 0, first when it is excluded. This is not a metric defect to fix: abstention is a property of the arm, not of the metric, so no convention is neutral. `RankMetrics.txt` reports all three populations and `analyze_rank_metrics.py` refuses to privilege one. See `docs/findings/13-rank-aware-metrics.md`. **Measured 2026-08-12 (finding 16), which does not close it:** scoring all four arms on the baseline's own 98 answered cases raises *every* arm by +0.032 to +0.038 pooled MRR — so the self-selection confound is real but **arm-neutral** — and on that matched set the ordering inverts (Bio_ClinicalBERT 0.171072 > BiomedBERT 0.167828 > BlueBERT 0.155362 > BioSentVec 0.146535), which explains the baseline's answered-only first place as mismatched denominators rather than removing the knob; see `docs/findings/16-self-selection.md`.

A corollary worth knowing: TOP-K scores rise monotonically with K because one hit suffices and there is no penalty for the other K−1 predictions. **That curve is an artifact, not a result — verified monotonic in 18 of 18 model × threshold combinations, zero violations.** Precision@K (P5) is the diagnostic that exposes it: it *falls* with K for every arm, and unlike Hit@K nothing forces its direction. A genuine set-level P/R/F1 over the diagnosis sets would be the more complete fix — but **P6, the item that carried it, was retired 2026-08-11**, because its proposed method is cosine over the diagnosis embeddings and would reintroduce the self-grading confound P4 removed. The *idea* survives and nothing measures it today; if it is ever built, build it on the encoder-independent DRG labels.

**Before designing any exact-match metric:** only **76 of 129 test cases (58.9%)** have their correct DRG present anywhere in their own fold's training pool on `folds_grouped` — 105 of the 145 unique diagnoses occur exactly once in the dataset. A perfect retriever therefore caps at **58.9%** under exact matching (75/129 = 58.1% on the legacy `folds`; the leakage fix moved retrievability by exactly one case, so the two defects really were independent). Both figures are pinned in `tests/test_drg_grader.py`. Per fold the ceiling ranges from **4/13 (30.8%) to 13/15 (86.7%)** on the canonical split — finding 14 corrected the earlier 3/12 figure, which was measured on the environment-dependent Windows split no committed result used — which makes it a fresh source of per-fold variance in its own right. Prefer graded relevance — **but note the partial-credit ladder was designed and rejected on measurement** (`docs/findings/12-drg-grader.md`), so do not treat it as an easy win. Ordered fix list: `docs/plans/correctness-fixes.md`; metric options: `docs/plans/metric-redesign.md`.

**Do not "correct" that 105 to 85.** Both are real and count different things: **85** is occurrence-level singletons over the 297 raw diagnosis entries, **105** is admission-level singletons (document frequency 1) over the 224 entries surviving `preprocess_diagnosis`'s per-admission dedup, and the 85 are a strict subset. **105 is the right figure here**, because this sentence bounds exact-match *retrieval* and a label written twice inside one admission offers a retriever no second admission to find it in. It is also the pipeline-true count — the dedup happens before any label reaches the grader. The 20-label gap exists because **71 of 129 admissions list their APR label twice byte-identically**, an upstream DRGCODES artifact. This was nearly mis-"fixed" during P4; see `docs/findings/12-drg-grader.md`.

**The shared-pipeline constraint is broken under `legacy` and fixed under `corrected`/`drg` — check which pipeline produced a number before comparing arms.** The baseline calls `preprocess_sentence` on diagnosis text at embedding time (`cython_utils.py:431`; the grader derives raw descriptions at `:524`/`:526` and looks those embeddings up by them at `:527-528`). The BERT path now does too, but **only when the config says so**: `bert_models.py:346` gates it on `use_corrected_preprocessing(config)`, so under `legacy` it still encodes raw text and 119/145 (82.1%) of descriptions reach the two encoders as different strings. Under `corrected` all 145/145 match. The gate exists because the fix moves every BERT number, and `legacy` must stay byte-exact for the golden. **So a `legacy` baseline-vs-BERT delta is still confounded by preprocessing, not just encoder; a `corrected` or `drg` one is not.** Note also that only the *encoded* text changes — the dict keys stay raw under both versions, because `get_diagnosis_similarity_by_description_max` looks up by the raw description and preprocessing the keys would break every lookup.

### What can honestly be claimed

**Quote the `corrected` or `drg` numbers, never the `legacy` ones.** The legacy figures
(0.285 / 0.280 / 0.254 / 0.239 at TOP-10, gap 0.005) carry patient leakage *and* divergent
preprocessing and are superseded. This is the most likely place to state something wrong to the
user, because the two sets look interchangeable and differ by ~0.04. **That rule is now backed by
measurement rather than by argument (finding 15):** the legacy→corrected gap has been split per
arm, and **leakage is the dominant term in all four arms** — so a legacy number is not merely
"older", it is inflated mostly by patients scoring against their own prior admissions. Never quote
the `preprocess-only` column either: that tree keeps the leaky committed split by construction, so
it is an attribution *input*, and any number or ranking read off it is a leaked number.

Corrected, threshold 1.0, TOP-10 — the only cosine setting where no model sits on the ceiling:
**Bio_ClinicalBERT 0.2491 > BioSentVec 0.2163 > BiomedBERT 0.1981 > BlueBERT 0.1821**, best-vs-baseline
gap **0.0328**, against per-fold σ of 0.054–0.139 and per-fold F ranging 0.00–0.67 on identical data.

**The knobless result is the one to lead with.** With the threshold knob gone (`drg` collapses the
five rows to one) and the `K` knob gone (MRR has no `K`), **no pair of encoders separates on any of
the three abstention populations** — max paired |t| = 1.718 (winnable), 1.651 (all-cases), 1.651
(answered) against the 2.262 needed at 9 df. That was predicted in writing before the run and it
held. A fresh tree five days and a numpy pin later (`results_p40`) reproduced those three figures
**to the digit**. *(Corrected 2026-08-12: the answered figure read **1.174** here and in finding 13
from the first. Recomputed on `results_p5` and `results_p40` alike, that population's six t-values
are 0.808, 0.966, 1.174, 0.279, **1.651**, 0.754 — 1.174 is only the largest among the three
BioSentVec pairs, and the stated convention is the max over all six. The two 1.651s are not a
transcription slip: the three BERT-only pairs are **bit-identical** between the all-cases and
answered blocks, because those arms never abstain, so excluding abstentions removes none of their
cases. No verdict moves — 1.651 < 2.262 and no pair separates.)*

**The last population that disagreed now agrees, and that is worth stating precisely (finding 16).**
Scoring all four arms on the *same* 98 cases — the ones the baseline chose to answer — makes the
restricted, `all-cases` and `winnable` orderings the **same ordering**: Bio_ClinicalBERT >
BiomedBERT > BlueBERT > BioSentVec. The `answered` population was the only one of the three that put
the baseline first, **and the only one comparing different case sets** (baseline n=98 against BERT
n=129); matching the denominators inverts it, with the baseline's own number byte-identical across
both spellings. **This is descriptive, not tested** — the restricted column is a pooled MRR over a
fixed case set with no paired *t*-test — so read it as an explanation of the `answered` anomaly, not
as a ranking. Nothing separates anywhere a test was run.

**No encoder ranking is supported by this experiment** — that is the defensible claim, and a more
interesting one than a spurious win. It now rests on four removed confounds rather than on one
favourable setting, which makes it much harder to argue with. What is *also* supported: the three
transformers match a ~5.6-billion-parameter n-gram table using ~1/51 the parameters and ~1/17 the
disk, which is a hardware fact independent of the metric.

### The findings index

`docs/findings/` — 01 baseline status · 02 encoder comparison · 03 saturation · 04 degeneracy · 05 patient leakage · 06 preprocessing defects · **07** why the comparison is not valid yet (the synthesis) · **08** where the runtime goes · **09** the baseline's first run · **10** output-path fragmentation · **11** the first uncontaminated four-arm results · **12** the encoder-independent DRG grader, and the partial-credit scheme rejected on measurement · **13** the knobless rank-aware comparison, and the third knob it exposed · **14** the grouped fold split depends on the numpy major version; the pod's split is canonical, pinned by digest — and, since 2026-08-12, by `numpy>=2.0,<3` in `config/environment.yml` *and* on the pip path (`pyproject.toml`, both requirements files) · **15** the legacy→corrected drop split into leakage, preprocessing and their interaction (leakage dominates every arm; preprocessing is small and its sign varies; the published 0.0902 is 54.3% / 43.8% / 1.9%) — **its `preprocess-only` column is an attribution input, never a result, because that tree still leaks** · **16** self-selection measured: the baseline's 98 answered cases really are easier, by +0.032 to +0.038 MRR for *every* arm, so the confound is real but arm-neutral, and on the matched 98-case set the ordering inverts.

**One trap from finding 13 that is convincing enough to fool a careful reader, so it is repeated here deliberately.** Do not try to bound the self-selection half of the abstention problem with `MRR_all-cases / coverage`. That quotient is **algebraically identical** to `MRR_answered` — abstentions contribute 0 to the all-cases numerator and coverage is exactly the denominator ratio, so it returns ~1.0 by construction and looks like it bounds the confound at 0.5% while carrying no information at all. It was computed, believed for a minute, and withdrawn. The real test needs the BERT arms restricted to the baseline's 98 answered cases, which needs per-case relevance vectors (**P40**). **That real test has now been run** — P40 shipped `RankCases.txt` and finding 16 reports the matched comparison (every arm gains +0.032–0.038, the ordering inverts, zero `[WARN]` join failures), which is what the quotient above could never have told anyone; see `docs/findings/16-self-selection.md`.

**Runtime, measured:** encoding is 0.17–0.45% of wall-clock; over 93% is the single-threaded pure-Python cosine loop, and total fold time varies only 1.7% across the three transformers. **A GPU buys nothing for the encoder arms** — that is why the rented box is CPU-optimised. See `docs/findings/08-runtime-and-cost.md`.

## Setting up on a new machine

Full instructions in `docs/guides/setup.md`. The critical parts:

```bash
conda env create -f config/environment.yml   # env "disease-diagnosis", Python 3.9
conda activate disease-diagnosis
git config core.hooksPath .githooks          # data-use guard; hooks are not cloned
```

**`numpy` is pinned to `>=2.0,<3` in `config/environment.yml` *and* on the pip path — `pyproject.toml`,
`config/requirements.txt`, `config/requirements_bert.txt` — and that pin is load-bearing rather than
hygiene.** `GroupKFold`
breaks its ~85 single-admission ties through `np.argsort`, whose unstable-sort behaviour changed
between numpy 1.x and 2.x, so a numpy-1.x environment builds a *different* (equally leak-free,
equally deterministic) `data/folds_grouped/` split that is not comparable with any committed result
— finding 14 / P42. Under the pin, `python scripts/make_folds.py --verify` reproduces the canonical
split on **Windows and Linux alike** — verified 2026-08-12 against the canonical digest, `VERIFY:
PASS`, 0 leaked — so regenerating locally is the normal path, not a hazard. `--verify` still WARNS
on a digest mismatch; that warning, not the pin, is what catches an environment that ignored it.
**An environment that adds torch needs `torch>=2.3`**, the numpy-2-compatible line.

**`import torch` will fail out of the box on macOS/ARM** with `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`. Two OpenMP runtimes land in one process: conda-forge's `llvm-openmp` (reached via `libopenblas` → numpy/scipy) plus a second copy bundled in pip-installed torch. Fix by pointing torch's copy at conda's — `libtorch_cpu.dylib` resolves `@rpath/libomp.dylib`, and both are LLVM OpenMP compat 5.0.0:

```bash
T="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib/libomp.dylib"
mv "$T" "$T.torch-bundled.bak" && ln -s "$CONDA_PREFIX/lib/libomp.dylib" "$T"
```

**Never use `KMP_DUPLICATE_LIB_OK=TRUE` instead.** It suppresses the error while letting two runtimes coexist, which OpenMP's own documentation warns can silently produce incorrect results. In a numerical reproduction project a silent wrong answer is far worse than a crash. Reinstalling torch restores the bundled library and reintroduces the problem.

Python 3.9 is pinned only by `sent2vec` (baseline-only). HuggingFace models cache to `~/.cache/huggingface/hub`, ~400–450 MB each; copy that directory or set `HF_HOME` to avoid re-downloading.

## Commands

Always run from the repository root — output paths derive from `os.getcwd()`. (The `sys.path.insert(0, project_root)` hacks are **gone** as of `2a0b77d`; the package is a real src-layout install, so `pip install -e .` is now required.)

**Every run takes `--pipeline legacy|corrected|folds-only|preprocess-only|drg` and `--out ROOT`** —
**both arms, as of `9c9e251`/`9d08c94`.** `$AICDS_PIPELINE` still works and is still what
`python -m aicds.models.baseline_sent2vec` reads, but the baseline no longer *needs* it: it has a
`run_analysis()` and a `__main__` guard now, so `scripts/run_baseline.py` carries the same argparse
the BERT script does. `legacy` is the default everywhere, so a command with no pipeline argument is
bit-identical to the original and the golden is unaffected.

```bash
python scripts/make_folds.py --verify            # regenerate data/folds_grouped/ (gitignored, so
                                                 # required before any grouped-fold run)
python scripts/run_bert_analysis.py --model 2    # 1=Bio_ClinicalBERT 2=BiomedBERT 3=BlueBERT
python scripts/run_bert_analysis.py --model all --pipeline drg --out results_drg
                                                 # ~11.5 min/model on the RunPod 32-vCPU box;
                                                 # ~21 on an M-series Mac; ~14 on a
                                                 # Threadripper 7960X
python scripts/run_baseline.py --pipeline drg --out results_drg
                                                 # BioSentVec — Linux only, 21 GB model, ~13 min
python scripts/analyze_rank_metrics.py results_drg  # parses RankMetrics.txt, paired t-test on
                                                 # per-fold MRR across all three populations
python scripts/compare_models.py --results-dir results_drg   # 4-model comparison PDF
python scripts/analyze_score_distributions.py --pipeline corrected
                                                 # regenerates docs/score_distribution_analysis/
                                                 # (P37: takes --pipeline/--out, grades through
                                                 # get_diagnosis_relevance, names the pipeline in
                                                 # the summary header. Fixed output names, so pass
                                                 # --out for any non-canonical pipeline.)
python scripts/build_dashboard_data.py           # rebuilds dashboard JSON; default root is docs/
python scripts/analyze_performance.py [dir]      # PDF report from a PerformanceIndex.txt
python scripts/verify_setup.py                   # smoke test — exits 0 on a clean checkout
```

**A run now writes four things**, not three: `PerformanceIndex.txt` (the golden's subject),
`RankMetrics.txt` (P5), `timing_report.txt`, and — since `5a52d26` — `run_metadata.json` (P14),
written **last**, after the `PerformanceIndex.txt` handle closes. It records the git SHA and dirty
flag, the pipeline by name *and* by all three `PipelineConfig` fields, a SHA-256 of the fold split's
`TrainingSet.txt`/`TestSet.txt` contents, the model key and label, `K_FOLD`, and the platform /
Python / numpy versions. Its presence also means the run reached the end, so a reader can tell a
completed run from an interrupted one without parsing anything. `write_run_metadata` **never
raises** — a finished 13-minute run must not be lost to a `git` subprocess call about provenance.

`compare_models.py` **refuses rather than guesses.** Its ~16 sanity assertions are the only thing
verifying the parser reads the columns correctly, so they are keyed by pipeline; an unknown results
directory, a pipeline mismatch, or a run in which zero checks executed are all hard exits. Pass
`--pipeline` explicitly for a directory whose name is not in `_PIPELINE_BY_DIRNAME`.
**Since `5a52d26` it prefers `run_metadata.json` and demotes the dirname table to a fallback**
(`compare_models.resolve_pipeline`, `:395-433`), so every invocation on the three *existing* trees prints one
`[WARN] no run_metadata.json … (pre-C8 runs)` line. That noise is by design: nothing retrofits
metadata onto a run nobody can re-derive, so **the warning on those three trees is now permanent
rather than interim** (updated 2026-08-12). The metadata path itself is exercised — every run in the
five trees the 2026-08-12 pod session produced carries `run_metadata.json`, so all five resolve
their pipeline from provenance (`[INFO]`) instead of the directory name; `results_corrected2` and
`results_p40` could not resolve any other way, their names being in no dirname table.

Tests:

```bash
pytest                    # 505 passed, 4 deselected — this is the green baseline
                          # (measured 2026-08-12; 499/3 before P13, 413 after
                          # C1-C8, 264 before it. The D and E commits added the
                          # rest: P40's RankCases tests, P27's corrected2 pins,
                          # attribute_effects, then P13's baseline golden — which
                          # is what moved 3 deselected to 4.)
pytest -m golden          # THE SAFETY NET, and it is now TWO tests, not one —
                          # one golden per arm, each a full 10-fold pipeline run
                          # byte-compared against its own committed reference.
                          # Budget ~1 hour for the pair, run serially:
                          #   BERT arm (tests/test_golden.py, stub768) ~34-53 min,
                          #     measured eight times: 34:12, 42:28, 43:28, 43:50,
                          #     44:32, ~50:00, 52:35, 53:10 — 34:12 is 2026-08-12,
                          #     Windows, numpy-2.0.2 venv. NOT the ~20 min this
                          #     file used to claim, and NOT the ~20 min
                          #     tests/test_golden.py's own docstring still claims
                          #     at :68.
                          #   baseline arm (tests/test_golden_baseline.py,
                          #     stub700-baseline) ~28-34 min, measured twice:
                          #     28:09 solo, 34:06 under contention (2026-08-12).
                          # Run both before and after any refactor commit; they
                          # are the only thing that catches the numbers moving
                          # while every other test stays green.
pytest -m network         # opt in to the HuggingFace download test
pytest tests/test_bert_symptom_pairwise.py::TestComputePatientSimilarityPairwise -v
```

Config lives in `pyproject.toml` `[tool.pytest.ini_options]`: `pythonpath = [".", "src"]` (so tests need no `sys.path` hack), markers `network`/`slow`/`golden`, and `addopts = -m 'not network and not slow'`.

**`tests/conftest.py`'s `PYTHONHASHSEED` line does not do what it looks like.** CPython reads that variable at interpreter start-up, before any Python code — including `conftest.py` — runs, so `os.environ.setdefault("PYTHONHASHSEED", "0")` has no effect on the current process. It only affects subprocesses. This matters because `bert_models` builds its encode batch from `list(unique_symptoms)` over a *set*, and `preprocess_diagnosis` routes through `list(set(...))` twice, so hash order genuinely varies run to run. Export it in the shell if you need determinism: `PYTHONHASHSEED=0 pytest`. The golden test is immune either way — `StubEncoder` derives every vector from `sha256(text)` alone, which was verified across seeds 0 and 12345.

Dashboard (React 19 + Vite 7 + Tailwind 4 + d3), from `dashboard/`: `npm install && npm run dev`.

## The safety net — read this before changing any code

**There are two goldens, one per arm** — as of P13 (2026-08-12); before it, only the BERT arm was
covered. Each is a full 10-fold run of the real pipeline, minted from a known-good tree, and
`pytest -m golden` re-runs both and compares **byte-for-byte**:

| Reference | Arm | Test | Dim |
|---|---|---|---|
| `tests/golden/stub768/PerformanceIndex.txt` | BERT (`bert_models`) | `tests/test_golden.py` | 768 |
| `tests/golden/stub700-baseline/PerformanceIndex.txt` | baseline (`baseline_sent2vec`) | `tests/test_golden_baseline.py` | 700 |

**The second one is not the first at a different width.** Running the BERT arm never executes
`predictS2V`, `compute_performance_index` on the per-case path, or the `embending_*` dict builders
— the BERT arm reimplements all of that — so the baseline golden is the first byte-exact coverage
those `cython_utils.py` functions have ever had. Since `cython_utils.py` is *shared*, a change there
could previously move the baseline numbers with the entire suite green. The two tests are
deliberately standalone and share no helpers, so one file's refactor cannot move the other's
verdict; `tests/test_golden.py` is not to be edited.

Nothing is trained here, so every number the pipeline emits is a pure function of the input data
and the arithmetic in `cython_utils.py`. That means **any behaviour change is a numerical change**,
and the realistic failure mode of refactoring this repo is not a crash — it is the numbers moving
while every other test stays green. The goldens are the only thing that catches that.

**Read no science out of `stub700-baseline`.** Every aggregate row carries `PR 1.0` and
`P == R == FS` — the shape the degeneracy finding says the *BERT* arms have and the baseline does
not. That is the stub's cosine distribution never pruning to empty, not a result: the real
BioSentVec run abstains on ~24% of cases (`PR` 0.7679, `TP+FP` 9.9 against 12.9 here).

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

- **On the Windows dev box the working environment is the repo-root `venv/`, not conda** (recorded
  2026-08-12). Use `venv\Scripts\python.exe` — it carries numpy 2.0.2, torch 2.8.0+cpu, sklearn and
  the editable install, and it is the interpreter every local suite and golden run in this project
  was measured on. A conda env *named* `disease-diagnosis` also exists on that machine and is a
  **broken 13-package stub**: pointing anything at it produces `ModuleNotFoundError` rather than a
  wrong number, which is the good failure mode — but it has already caused one false conclusion
  ("this venv has no torch"), reached by testing the stub and generalising. Check which interpreter
  answered before believing a dependency result.
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

**`src/aicds/config.py` is the seam every correctness fix goes behind.** `PipelineConfig` is a frozen dataclass with three fields — `fold_dir`, `preprocess_version`, `grader` — and **defaults that reproduce the published pipeline exactly**. Named configs: `LEGACY`, `CORRECTED`, `FOLDS_ONLY`, `PREPROCESS_ONLY`, `GRADER_DRG`, selectable by name from `_BY_NAME` (which also feeds argparse `choices=`, so a config cannot exist-but-be-unselectable — that was a real bug). **Add new fields with legacy-preserving defaults; never change an existing default.** That is what lets the golden stay byte-exact forever while the corrected numbers get their own run.

**Read `require_supported_grader` before adding any config field, and understand why it exists.** For a window during P4, `--pipeline drg` was selectable, printed a plausible banner, ran a full 10-fold pass, and produced **cosine** numbers filed as DRG ones. Nothing crashed; nothing failed a test; the output looked entirely normal. A config field with no reader is worse than no field at all. **Any new grader name must land in the same commit as its consumer**, and `SUPPORTED_GRADERS` is what makes forgetting that a loud error instead of a silent wrong answer.

**`src/aicds/analysis/`** holds the rank-aware metrics, deliberately layered: `rank_metrics.py` is pure (no I/O, no config) and carries the N1–N4 constraints in its docstring; `populations.py` computes the winnable-case sets; `rank_report.py` does the accumulation and writes `RankMetrics.txt`. Keep `rank_metrics.py` pure — its contract depends on it, and it was differentially validated against three independent implementations over ~1.1M comparisons.

Data flow: 129 admissions from `data/raw/Symptoms-Diagnosis.txt` (`wc -l` says 128 — no trailing newline; `;`-delimited, diagnoses joined by `--` with `apr:`/`hcfa:`/`ms:` prefixes) → embed unique symptom and diagnosis strings → score every training patient by mean-of-max symptom similarity → prune below `PRUNING_SIMILARITY` (0.5) → take MAX and TOP-K (10–50) → score predictions against ground truth by MAX cosine over the Cartesian product → threshold at 0.6–1.0 → aggregate over 10 folds. **Under `grader="drg-exact"` the last two steps change**: relevance is an exact DRG label match, so the five threshold rows collapse to a single value.

Things that will bite you:

- **`cython_utils.py` is pure Python** despite the name — a hand-translation of the original Cython, archived at `archive/cython_source/util_cy.c`. No build step.
- **`sent2vec` is no longer imported at module scope** (as of `c8e4ffd`) — it moved inside `load_model()`, its only consumer, so `cython_utils` now imports with base dependencies alone. `tests/test_bert_symptom_pairwise.py` still AST-loads functions out of `bert_models.py` to dodge the old imports; that workaround is now unnecessary and **new tests should import normally**.
- **Embedding dicts are keyed by preprocessed *text*, not HADM_ID**, each value wrapped in a one-element list so callers index `emb[0]`. Diverging silently changes results rather than raising.
- **The `legacy` folds under `data/folds/` are fixed committed files**, not computed at runtime, and no generator for them exists anywhere in the repo or its archive — the original split was produced upstream and only its output was committed. **Never regenerate or overwrite `data/folds/`**: it is the golden's input, and one careless `--out` destroys the only reference. `scripts/make_folds.py` writes `data/folds_grouped/` instead, which is gitignored and regenerated deterministically. `load_dataset` (`cython_utils.py:345`) drops the final character of each line assuming a trailing newline — all 20 committed fold files end in `0x0a`, but a hand-written one without it silently loses its last symptom. `tests/test_characterize_dataset.py` pins this.
- **Both arms now expose `run_analysis(...)` behind a `__main__` guard.** `baseline_sent2vec.py:187` takes `(encoder=None, config=LEGACY, out=None)`; `bert_models.py:372` takes `(model_id=None, encoder=None, config=LEGACY, out=None)`. Importing either module no longer runs a pipeline (fixed in `9c9e251`).
- **`src/aicds/runs.py` is the one place run-directory shape is decided** — the writer half (`run_dirs`, `check_out_root`, `write_run_metadata`) and the reader half (`discover`, `Run`). Read its module docstring before adding a script that writes or finds runs; a new private glob is the exact drift it exists to stop.
- **`src/aicds/analysis/performance_index.py` is the one `PerformanceIndex.txt` parser.** The four private ones are gone (`1f69e11` added it, `7927a88` deleted them), and equivalence against all four was proved on the committed goldens before deletion — the assertions now live as pinned literals in `tests/test_performance_index.py`.
- `src/aicds/entity/` now holds **only** `SymptomsDiagnosis.py`. `Admission`/`Symptom`/`Drgcodes` were deleted in `94b4e24` together with their `test_reorganization.py` import; `src/aicds/evaluation/` and `bert_eval.py` were deleted outright in `5ca7f64`.

## Outputs

With no `--out`, **both** arms write `Prediction_Output_{Model}_{DDMMYYYY_HH-MM-SS}/` into the **current working directory** — the baseline as `Prediction_Output_BioSentVec_{stamp}`. That default layout is frozen by the golden and must not move. *(Pre-`9d08c94` baseline runs wrote `Prediction Output_{DDMMYYYY HH-MM-SS}/` — a space, no model name — which is why `.gitignore` still carries both spellings: checking out an older SHA and running it must not leave unignored clinical data behind.)* The three committed result sets under `docs/` are the project's **regression oracle** — the only record of a working pipeline's exact output. Treat them as read-only.

**Every run directory is gitignored by the glob `results*/` (`.gitignore:160`)**, which is what keeps DUA-covered output out of git automatically. **It is a glob on purpose, not a hand-listed set:** the previous version enumerated `results/` and `results_corrected/` by hand, and on 2026-08-06 a third pipeline was about to write `results_drg/` — per-case output keyed by `HADM_ID`, into a public repo. That is the one failure here that cannot be taken back. `.githooks/pre-commit` is the second line of defence, not the first. Layout is pipeline root, then model, then timestamp:

```
results/          <model>/<DDMMYYYY_HH-MM-SS>/   # legacy
results_corrected/<model>/<DDMMYYYY_HH-MM-SS>/
results_drg/      <model>/<DDMMYYYY_HH-MM-SS>/
                  # model ∈ baseline, bio_clinical_bert, biomedbert, bluebert
```

**`--out ROOT` writes that layout directly** (`9d08c94`), so nothing is moved by hand any more:
`--out results_drg` produces `results_drg/{key}/{stamp}/` with `symptom_details/` *nested* inside
the run rather than as a sibling — deliberate, because under a shared root a sibling would land next
to the timestamps and break "every child of the model directory is a run". Omitting `--out` keeps
the flat cwd layout the golden pins. `runs.check_out_root` **refuses** an `--out` that resolves
inside the repository with no `.gitignore` rule covering it (`--out scratch`, `--out out` — note the
ignore entry is `output/`, not `out/`): a run leaves per-case files *named by* `HADM_ID`, and because
those files are empty the pre-commit hook's 20-distinct-ID content rule scores them 0 and cannot see
them. It asks git rather than re-implementing ignore semantics, and it distinguishes "not ignored"
from "could not ask git" — the latter warns and continues.
`scripts/compare_models.py --results-dir <root>` reads the tree and emits
`<root>/model_comparison.pdf` across all four arms.

`PerformanceIndex.txt` columns are `threshold TP FP P R FS PR`. The meaningful numbers are the per-fold blocks and the final `10-FOLD` block. Bear the degeneracy finding in mind when reading any of them.

Constants in `src/aicds/utils/Constants.py` (note `CH_DIR` walks **four** parents to reach the repo root — an off-by-one here silently repoints every data path rather than raising): `K_FOLD=10`, `PRUNING_SIMILARITY=0.5`, TOP-K `10..60 step 10` (so K = 10,20,30,40,50). Thresholds are duplicated as the set literal `{1, 0.9, 0.8, 0.7, 0.6}` at **7 sites, all of them inside `cython_utils.py`** (`:267`, `:296`, `:644`, `:653`, `:668`, `:696`, `:734` — recounted 2026-08-11; the long-standing "8 sites" figure counted `bert_eval.py`, deleted in `5ca7f64`). Their *set iteration order* determines the **baseline** arm's output row order. **The BERT arm reaches the same order by a different route, and that is the trap:** `bert_models.py:511` hard-codes the ordered list `[0.9, 1.0, 0.6, 0.8, 0.7]` — CPython's iteration order for that set, transcribed by hand — and it is consumed at `:647`, `:662` and `:677`, each of which ends in a `performance_out_file.write(...)`. So that one list literally determines the BERT arm's `PerformanceIndex.txt` row order, and its **only** synchronisation with the set is its trailing comment `# Same order as baseline`. Seven *further* sites spell the same values as an ordered list in ascending order; none of those feeds a pipeline row order: `bert_models.py:638` (a debug print, no comment attached), `analysis/performance_index.py:111`, and the reader-side constants in `analyze_performance.py:43`, `analyze_score_distributions.py:62`, `build_dashboard_data.py:199`, `build_readme_plots.py:31` and `compare_models.py:58`. Eight ordered-list sites in non-test code, then — but only `:511` is load-bearing.

## Known defects

Verified, documented, and **re-resolved by reading the tree on 2026-08-11** after the C1–C8
refactor moved most of these. Several line numbers had drifted by 100+ lines, so re-check before
trusting any reference here.

- **Every per-case output file is still empty — but P40 is CLOSED (2026-08-12), because it wrote a new sibling instead of repairing them.** The two dead handles below are exactly as dead as they ever were, and that is deliberate: `RankCases.txt` now carries the per-case relevance vectors, written after the `PerformanceIndex.txt` handle closes, so the byte-exact golden covers what it always covered. What remains here is a cosmetic wart with no consumer — repairing it would put new writes inside the golden's region for output nothing reads. The baseline emits 258 zero-byte files; `cython_utils.py:203-204` opens both handles and `:308-309` closes them with **nothing written in between**. The BERT arm never opens them at all yet still creates the `Fold*/` dirs (`bert_models.py:539-540`). **Neither arm has ever written anything through *those* handles**, and neither ever will. The history is worth keeping because it is how the scope rule was arrived at: this was a cosmetic wart for months, finding 13 turned it into the binding constraint on the self-selection test, and the fix was a new sibling rather than a repair — the same additive shape that let P4 and P5 land without re-minting the golden.
- ~~**Nothing records which pipeline produced a run (P14).**~~ **Fixed in `5a52d26`**: both arms write `run_metadata.json` last, and `compare_models.resolve_pipeline` prefers it, demoting dirname inference to a `[WARN]` fallback. The historical context is worth keeping, because it is why the P29 attribution script harvests into a config-named root *after each batch* rather than at the end: both batches emit identically-named `Prediction_Output_<Model>_<timestamp>` dirs into one cwd, so sorting them afterwards would have meant inferring the config from timestamp order alone — in the one experiment whose entire purpose is attribution. **The staged pod script was retired rather than kept** (2026-08-12): it was pre-flighted at a SHA predating C5/C7/C8, so it would have written the old flat cwd layout that `runs.discover` now refuses and carried no provenance at all. What ran instead was the simplified `--pipeline X --out results_X` form on a pod pulled to current main, with **no harvest step**, because `--out` removes the collision that forced one.
- ~~**`scripts/run_baseline.py` crashes.**~~ **Fixed in `c2fee6e`**, verified by a full 10-fold run on 2026-08-05. ~~**The last two baseline defects.**~~ **Fixed in `9c9e251`, VERIFIED 2026-08-11: a pod `--pipeline legacy --out` run byte-matched the 2026-08-05 reference exactly (353,418 bytes to the trailer cut), proving C4's handle collapse and C5's `--out` plumbing content-preserving in one shot.** The module now has `run_analysis(encoder=None, config=LEGACY, out=None)` at `:187` and a `__main__` guard at `:543`, and the two `PerformanceIndex` handles are one `with`-block at `:354`. **The old description of that handle pair was wrong in an instructive way, so record the truth rather than just the fix:** the `'w'` handle was not abandoned — it wrote the *entire* body and was simply never closed explicitly; rebinding the name to the `'a'` handle dropped its last reference, so CPython flushed and closed it right there, *before* the trailer's first write, which `O_APPEND` then placed at the true end of file. The emitted byte sequence was `[everything written through 'w'][trailer]`, which is exactly what one `with`-block produces. It worked by refcount accident, not by design, and that is why collapsing it is byte-safe.
- ~~**Output discovery is broken five ways.**~~ **Closed by `9d08c94` (writers) and `7927a88` (readers).** Both arms emit the underscore spelling via `runs.run_dirs`; **five consumers now call `runs.discover`** — `analyze_performance.py:71`, `analyze_rank_metrics.py:149`, `build_dashboard_data.py:156`, `build_readme_plots.py:302` and `compare_models.py:282`/`:355` — whose single rule is *a directory is a run iff it contains `PerformanceIndex.txt`*. **That is three of finding 10's five tabulated sites plus the two later rules, not all five of the table:** the table's other two entries are the deliberate exceptions named below, so do not read "five sites" and "five consumers" as the same five. (`verify_setup.py:64` imports `discover` as a checkout smoke test without calling it.) It **refuses rather than skips** a directory that holds one in neither layout, because skipping is how a run vanishes from a comparison table unnoticed. Two deliberate exceptions survive and are not drift: `tests/test_golden.py` keeps its own tmp-dir glob and its `\d{8}_\d{2}-\d{2}-\d{2}` regex — that regex is what *forced* the underscore direction, and the golden must not depend on the code it audits — and `tests/test_reorganization.py:75` keeps a recursive `**` because it is asserting that committed results exist at all, not discovering a run. See `docs/findings/10-output-path-fragmentation.md`.
- **The dashboard 404s.** The tracked JSON sits at `dashboard/dashboard/public/data/` (a stray nested dir), while the builder writes `dashboard/public/data/` and `useData.ts` fetches `./data/dashboard-data.json`. Still open; the fix moved to P38 per the 2026-08-10 phase split.
- The baseline model loads from cwd (`os.getcwd() + '/BioSentVec_...bin'`) despite `data/models/README.md` saying `data/models/`. Still open.
- ~~`scripts/verify_setup.py` checks for an `output/` directory that no longer exists.~~ **Fixed in `7927a88`** — the check is gone and the script exits 0 on a clean checkout. It failed for a directory nothing writes, and passed here only because a November 2025 run had left one behind; a smoke test that fails on a correct checkout teaches people to ignore it.

## Data handling

The repo is **public** and contains committed MIMIC-III records under a PhysioNet DUA that prohibits redistribution. This is not a HIPAA breach (the data is de-identified, dates shifted) but it does conflict with the DUA. **Do not add clinical data to this repository**, and do not rewrite history to remove the existing data without the owner's explicit instruction. See `docs/guides/data-use.md`.

`.githooks/pre-commit` blocks two things: new files under `data/raw`/`data/folds`, **and** any new file containing 20+ distinct `HADM_ID`s wherever it lives. The second rule exists because the first was insufficient — the generated golden under `tests/` carries all 129 IDs and sailed past a path-only check. The golden was committed deliberately with `--no-verify`, on the reasoning that the same 129 IDs are already published in the three `docs/Prediction_Output_*` files, so it adds no new exposure. Any future `--no-verify` on this hook deserves the same explicit reasoning.

## Where the work is

**`docs/plans/TODO.txt` carries live status and is the file to read first** — its `STATUS` block is
kept current. `docs/plans/revival-roadmap.md` is the sequenced *refactor* plan;
`correctness-fixes.md` and `metric-redesign.md` are the science.

**The correctness work is done — updated 2026-08-12, and this is the paragraph most likely to be
read stale, so check its date before quoting it.** P1–P5, P7 and P9 landed earlier; the 2026-08-12
batch closed the rest:

- **P29-runs — done.** The eight attribution runs (4 arms × 2 configs) plus the C4/C5 verification
  ran in **one pod session**: ten sequential verify-gated driver steps, 22:57→02:28 UTC, zero
  failures. Result: `docs/findings/15-leakage-preprocessing-attribution.md`.
- **P40 — done.** `RankCases.txt` shipped in `efa3794`; the instrumented tree is `results_p40`; the
  self-selection answer is `docs/findings/16-self-selection.md` (confound real, arm-neutral, and
  the fixed-set ordering inverts).
- **P27 — done** (`7e49212`, as the third variant `corrected2`), and now *measured*: its first
  four-arm run moved two arms and left two bit-identical in the reported cell. Finding 06 addendum.
- **P39 — closed measured-moot**, no code changed: the tie-permutation envelope is exactly
  **0.000000** on all three populations for both arms, so no tie-break convention can move a
  reported number.
- **P10, P11 and P12 — removed** by owner decision 2026-08-12; all three are retired with their
  reasoning in `docs/plans/TODO.txt`. **P6 was retired 2026-08-11.**

- **P37 — done** (2026-08-12). `analyze_score_distributions.py` no longer re-implements the grader:
  it takes `--pipeline`/`--out`, scores through `get_diagnosis_relevance`, and labels its output
  with the config that produced it. The committed `docs/score_distribution_analysis/` artifacts are
  `corrected`-measured now, and the re-run **corrected a wrong number**: the share of ordered
  patient pairs at an exact cosine 1.0 is **1.89% for all three encoders** (312 of 16,512 — the
  pairs sharing a diagnosis description), not the encoder-looking 1.49/1.62/1.31% the simulated
  numpy cosine reported. The same 1.89% is what `drg` scores, which is *why* finding 12's
  threshold-1.0 reproduction was bit-exact.

- **P13 — done** (2026-08-12). `tests/golden/stub700-baseline/` plus `tests/test_golden_baseline.py`:
  the baseline arm had zero regression protection while `cython_utils.py` is shared between the
  arms, so `predictS2V`, the per-case `compute_performance_index` path and the `embending_*` dict
  builders were unguarded. They are byte-exact now. See the safety-net section above.

**Still open: P38 alone** — a clean public repo, the hard blocker on publication, **deliberately
backlogged to the final arc** and carrying the Phase 3 polish (CLI, encoder registry, dashboard,
the `F1` rename). On the infrastructure track **P14, P15, P16, P17 and P20 all landed in C1–C8, and
P18 was retired**; P31 (the golden's stated runtime) is closed in every doc but not in
`tests/test_golden.py:68`, which still says "about 20 minutes" and is deliberately left alone —
touching the file the golden lives in is not worth a docstring.

Refactor status — **updated 2026-08-11.** The 2026-08-08 audit found Phases 2–3 unfinished despite
a belief they had landed; the eight commits `7400eff`…`5a52d26` (C1–C8) closed everything that audit
listed. Nothing is stranded on a branch: the only unmerged one, `origin/report-production`, is a
stale pre-`src/aicds` layout experiment (one commit, `9ad073e`) and should not be revived.

- **Phases 0, 1, 4 — done.** Environment repaired, data-use guard, docs reorganised, and the Phase 1 safety net (characterization tests + `StubEncoder` + byte-exact golden).
- **Phase 2 — done** (`7da5901` for the first half; `7400eff`, `5ca7f64`, `94b4e24`, `9c9e251`, `9d08c94` for the rest). The `src/aicds` package move (`d0ecaa9` + `2a0b77d`), real src-layout `pyproject`, `ensure_nltk_data`/`format_time` consolidated into `aicds.utils.runtime` (`bd6fe47`), `stop_words` defined where it is read (`5ae01a0`), then:
  - **All 11 `stop_words` monkeypatches deleted** (`7400eff`) — the count really was 11, not the 7 or 8 earlier docs claimed, because four sat in non-test code nobody had counted. All were the identical `set(stopwords.words('english'))`, so removal was provably inert; the golden came back byte-exact in 52:35 to prove it, on its own gate with nothing else in the commit.
  - **`--out`/results root landed** (`9d08c94`), plus `aicds.runs` as the one writer contract and `check_out_root` as the DUA guard.
  - **`bert_eval.py` and the whole `evaluation/` package deleted outright** (`5ca7f64`), no salvage, per owner decision 2 of 2026-08-10. This retires **P18**. The "salvage `hf_automodel` first" instruction that stood here for months pointed at a target that **never existed in any branch** — `git log --all --diff-filter=A` is empty for `encoders/` and `hf_automodel*` — and its GPU-aware line is worthless given finding 08.
  - **Verified dead code deleted** (`94b4e24`): `entity/{Admission,Symptom,Drgcodes}.py` with their `test_reorganization.py` import, `print_log`, `get_diagnosis_similarity_by_description_max_model`, and `scripts/run_all_bert_models.py`. `get_diagnosis_similarity_baseline` and `get_diagnosis_similarity_by_drgcode` were **kept** — they look dead but are the DRG grader's ancestors.
  - **The baseline's two residual defects fixed** (`9c9e251`), VERIFIED by the 2026-08-11 pod byte-compare; see Known defects for what the old description of the handle pair got wrong.
- **Phase 3 — the drift-stoppers are done in this repo; the polish moved to P38** (owner decision 1 of 2026-08-10, which redefined the ship point accordingly). Done here:
  - **One `PerformanceIndex` parser** (`1f69e11` + `7927a88`). The four private ones — `compare_models.py`, `analyze_performance.py`, `build_dashboard_data.py`, `build_readme_plots.py` — are gone, deleted only after the new parser was proved equivalent to all four on the committed goldens.
  - **One run-discovery rule** (`7927a88`), `runs.discover`, replacing the six-to-seven private rules the 2026-08-08 audit counted. Five consumers rewired. Evidence it is inert: `build_readme_plots.py` now *runs* (it used to raise `FileNotFoundError` on every invocation) and regenerated all six committed SVGs **byte-identical**; `build_dashboard_data.py`'s JSON is byte-identical (sha `0638b6c5…`); `compare_models.py`'s 16 sanity checks still pass on `results`, `results_corrected` and `results_drg`. Two costs, both deliberate: `analyze_performance.py`'s auto-detect is bounded to `depth=1` so it no longer crashes from the repo root but also no longer auto-detects a nested `results*/` tree — name it explicitly; and `results_p5` needs `--pipeline drg` because it is not in `_PIPELINE_BY_DIRNAME`, which is the refuse-rather-than-guess rule working, not a bug.
  - **Run provenance** (`5a52d26`), which is P14 rather than a Phase 3 item but closes the same attribution gap.
  - Moved to **P38**: the `main.py` CLI, the `SentenceEncoder` protocol + encoder registry, the dashboard fix, and renaming the misleading `F1` key. Note for whoever picks up the CLI: the `input()` at `bert_models.py:87` is **dead on the supported path** — `run_bert_analysis.py` always passes `--model` and `select_model()` returns before prompting — but live via `python -m aicds.models.bert_models`. And the dashboard's builder and fetcher already **agree** on `dashboard/public/data/`; the JSON is merely committed at the stray `dashboard/dashboard/public/data/` path instead, so the fix is re-run the builder, commit the output, delete the stray tree.
- **The scope rule still holds where it applies.** Refactor work must not move the numbers; correctness work deliberately does. They cannot run in the same commit, because the refactor's only safety mechanism is that the numbers stay put. C1–C8 held to it: two golden gates came back byte-exact (`7400eff` alone, then `9d08c94` covering C2–C5), and every new artifact — `RankMetrics.txt`, `run_metadata.json` — is a *sibling* with `PerformanceIndex.txt` untouched, the same additive shape that let P4 and P5 land without re-minting the golden. Prefer it.

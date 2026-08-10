# Reviving AI-CDS: reorganization, `main.py`, and docs

> **Status, re-audited 2026-08-08 — Phases 0, 1 and 4 complete; Phase 2 is 2.5 of 7 items done;
> Phase 3 not started.** Phase 0 (`bb84b82`): environment repaired, data-use guard installed,
> `docs/` rebuilt into `findings/` + `guides/` + `reference/`. Phase 4 (docs) was pulled forward
> and is also done. Phase 1 (safety net) is done — the fast suite now stands at 264 passed /
> 3 deselected in ~21–52s, plus `pytest -m golden`, a byte-exact 10-fold regression against a
> committed reference (~43–53 min, NOT the ~20 min this file once claimed). Phase 2's landed
> items: the `src/aicds/` package move (`d0ecaa9`+`2a0b77d`), src-layout pyproject, helper
> consolidation (`bd6fe47`), central `stop_words` (`5ae01a0`). Still open in Phase 2: deleting
> the **11** `stop_words` monkeypatches (7 in tests + 4 in non-test code), the `--out`/results
> root, the `bert_eval.py` salvage, the dead code, and the baseline's two residual defects.
> CLAUDE.md's refactor-status section carries the audited file:line details.
>
> Phase 5's metric work was **removed from this roadmap** and tracked in
> [correctness-fixes.md](correctness-fixes.md) and [metric-redesign.md](metric-redesign.md) —
> where, unlike the refactor, **most of it has since landed** (P1–P5, P7, P9; findings 11–13).
>
> The scope rule for Phases 2–3: **if a change moves the numbers it is out of scope; if it fixes
> something that crashes, blocks, or writes to the wrong place it is in scope.**
>
> Decisions, now settled by the repo owner:
> 1. **Committed MIMIC-III data — leave it.** No history rewrite. A pre-commit hook blocks
>    *new* clinical data; the existing records stay. Scaling to the full dataset is deferred to
>    remote-GPU work.
> 2. **Baseline arm — deprioritized.** Confirmed via git history that `baseline_sent2vec.py`
>    was added in `b23657b "organizing"` and never renamed from a tracked `CS2V.py` (the initial
>    commit held only `.gitattributes` and `README.md`). Combined with `.gitignore`'s
>    `diseaseDiagnosis/  # Downloaded original source`, it is the **original authors' scaffolded
>    code, not the owner's contribution**. *(This decision was later reversed by events: the
>    crash bugs were fixed in `c2fee6e` and the arm ran end-to-end on rented Linux on
>    2026-08-05, reproducing the published figure to within 0.007 — see
>    `docs/findings/09-baseline-first-run.md`.)*
> 3. **Scope — continue past the Phase 3 ship point**, since the docs work was explicitly wanted.

## Context

You want to revive and extend this repo, and you want it clean first. The recon turned up something more urgent than untidiness: **three things are broken right now**, and one of the project's central claims doesn't survive arithmetic. The reorganization is worth doing, but it has to be sequenced so it fixes what's broken without silently changing the science.

### What's actually broken (all verified directly, not inferred)

| # | Problem | Evidence |
|---|---|---|
| 1 | ~~**The baseline arm cannot run at all.**~~ **FIXED in `c2fee6e`, verified by the full 10-fold run of 2026-08-05** (reproduced the published TOP-10 to within 0.007 — `docs/findings/09-baseline-first-run.md`). | Was: `baseline_sent2vec.py:236` read `CH_DIR/Symptoms-Diagnosis.txt` (file lives at `data/raw/`) → `FileNotFoundError`; then `:244-247` called bare `entity.SymptomsDiagnosis...` with only `entity_module` bound → `NameError`. |
| 2 | ~~**`import torch` fails**, so the BERT arm can't run either.~~ **FIXED in Phase 0.** | `OMP: Error #15` — conda's `llvm-openmp` plus torch's bundled copy. Resolved by symlinking torch's to conda's; see `docs/guides/setup.md`. |
| 3 | **`build_readme_plots.py` raises on every invocation.** | Line 30 globs `<repo_root>/Prediction_Output_*`; the three result dirs live under `docs/`. Glob matches nothing → `FileNotFoundError`. |

Nobody noticed 1 and 3 because running them needs a 21 GB model / nobody re-ran the plots. The committed baseline numbers came from the pre-reorg `CS2V.py`; **the current checkout has never reproduced them.**

### The metric is degenerate — the most important finding

Across **all 12,600 metric rows in all three golden files, precision == recall == F-score, and PR == 1.0, with zero violations.**

The mechanism: `precision = tp/(tp+fp)`, `recall = tp/nrow`, and every test case increments exactly one of TP or FP, so `tp+fp == nrow` always. Therefore **P = R = F1 = TP/n = accuracy.** One degree of freedom.

This means the reported "F1" was never an F1. The baseline's 0.489/0.512/0.521 and BERT's 1.000 are all just accuracy. Saturation is real and separately documented, but *even unsaturated*, this metric cannot trade precision against recall — so it can never distinguish "predicted the right diagnosis" from "predicted many diagnoses, one of which was right." **A genuine set-level P/R/F1 over the GT×Pred diagnosis sets isn't one pluggable metric among several — it's the only thing that makes the metric non-degenerate.** That reframes "pluggable metrics" from nice-to-have into the actual point.

### Two more facts worth correcting

- **The dataset is 129 admissions, not 128.** `wc -l` says 128 because the file has no trailing newline. Confirmed three ways: `grep -c ';'` = 129, Fold0 train+test = 116+13 = 129, and golden `TP=12.9 × 10 folds` = 129. Every doc says 128.
- **No GPU/deploy infrastructure exists.** No Dockerfile, no CI, no serving layer, no RunPod config. The only hits are aspirational prose in a mermaid diagram and a `torch.device('cuda')` line in orphaned code. Roadmap item (v) is greenfield.

### Compliance — decided: leave the existing data, guard against new

The repo is **public**, and `data/raw/Symptoms-Diagnosis.txt` plus all 20 fold files are **committed**, carrying MIMIC-III `HADM_ID`, `SUBJECT_ID`, admit/discharge timestamps and diagnoses for 129 admissions. MIMIC-III is de-identified (dates shifted), so this is not a HIPAA breach — but PhysioNet's credentialed DUA prohibits redistribution, and a public repo is redistribution. Your roadmap item (ii), "add more HIPAA data," makes this a live tripwire: if that data isn't de-identified under Safe Harbor, committing it here would be a real problem.

**Phase 0 added a guard and documented the constraint. It did not touch history.** Un-committing the existing data would require rewriting git history — destructive, breaks every clone. The owner's decision was to leave the committed records in place; scaling to the full dataset is deferred to future remote-GPU work, where the data would live outside the repo anyway. See [../guides/data-use.md](../guides/data-use.md).

---

## Target layout

```
main.py                       # 3-line shim -> aicds.cli:main
pyproject.toml                # the single pip manifest
src/aicds/
  cli/          main.py, run_cmd, list_cmd, analyze_cmd, report_cmd, verify_cmd
  config/       defaults.py (ex-Constants), schema.py (ExperimentConfig), paths.py (kills CH_DIR + os.getcwd)
  data/         records.py, loader.py, folds.py
  core/         preprocessing.py (STOP_WORDS lives here), similarity.py,
                retrieval.py (<- RAG seam), aggregators.py, metrics.py, evaluation.py
  encoders/     base.py (Protocol), registry.py, builtin.py,
                sentence_transformer.py, sent2vec_encoder.py (deferred import),
                hf_automodel.py (salvaged), stub.py (deterministic, zero-dep)
  results/      schema.py, writer.py, render.py, parser.py (THE one), discovery.py
  analysis/     score_distributions.py, performance_report.py
  reporting/    plots.py, dashboard.py, compare.py (NEW: multi-run table)
scripts/*.py                  # 4-line deprecation shims, deleted later
results/                      # gitignored; every new run lands here
tests/golden/                 # frozen oracle + stub goldens
```

**Not created now:** `index/` (RAG), `remote/` (RunPod), `chat/`, `orchestration/`. Reserving empty packages is how this repo accumulated 545 orphaned lines in the first place. The seams are documented; the directories wait until there's code.

## CLI surface

```
python main.py [-v|-q] [--results-dir DIR] <command>

  run baseline                 BioSentVec arm (needs 21GB .bin)   [UNVERIFIED]
  run bert --encoder NAME|all  --folds N --aggregator max --out DIR
  list encoders|runs           replaces the interactive stdin menu
  analyze scores|performance
  report plots|dashboard|compare
  verify
```

Non-interactive by default — enforced by a test that runs every subcommand with stdin closed. Today `select_model()` blocks on `input()` whenever `model_id` is omitted.

---

## Phases

Each phase ends at a gate. **Phase 3 is the ship point** — stopping there delivers everything you asked for.

### Phase 0 — Triage and guard — **DONE** (`bb84b82`)

Nothing can be verified until torch imports.

- Repair the OMP/libomp collision, or stand up a clean py3.11 env with torch+transformers. Record both fingerprints. *(`KMP_DUPLICATE_LIB_OK=TRUE` is a workaround that can silently produce wrong numbers — do not use it as the fix.)*
- `docs/guides/data-use.md` + a pre-commit hook rejecting new files under `data/raw|data/folds`. 30 minutes, and it's the only item whose value decays with delay.
- `pyproject.toml`: `[tool.pytest.ini_options]` with `testpaths`, `pythonpath`, markers `network`/`slow`, `addopts = -m "not network"`. `conftest.py` sets `PYTHONHASHSEED=0`.
- Fix the 4 stale tests; mark `test_bert_integration.py` `@network` so a plain run stops downloading a model.

**Gate:** `pytest -m "not network"` green; `python -c "import torch"` clean.

### Phase 1 — Characterize and freeze (~2–3 days) — **DONE**

**The safety net goes in before any code moves.** This is the phase that makes the rest safe, and it's the one that's tempting to skip.

- Only source change: defer `import sent2vec` out of `cython_utils.py:11` into a lazy accessor. 3 lines; nothing in the BERT path uses it. This is what lets everything else run on modern Python.
- Characterization tests pinning current behavior: `preprocess_sentence`, `preprocess_diagnosis`, the `max(len_test, len_train)` denominator, `PRUNING_SIMILARITY` at exactly 0.5, `containGreaterOrEqualsValue`, both recall formulas, 129 admissions, and the `load_dataset` trailing-newline invariant (line 184 unconditionally drops the last character — any hand-written fold file without a trailing `\n` silently loses its last symptom, which matters the moment you build a small subset for roadmap item iii).
- Add a **deterministic sha256-seeded `StubEncoder`** and thread an optional `encoder=` into `run_analysis` (purely additive — omitting it reproduces today exactly). Mint `tests/golden/stub768_bert/PerformanceIndex.txt` **from unmodified HEAD**.
- **One comparator:** truncate at the timing trailer, then diff the remainder **byte-for-byte including numeric lines.** This is what catches the formatting quirks — aggregate rows print threshold `1` (int, from the set literal) while per-case rows print `1.0`, and per-case F-score prints int `0` not `0.0`. A comparator that parses floats can't see any of that.
- Opportunistically attempt a real Bio_ClinicalBERT run; record the result in `docs/findings/reproduction-ledger.md`. **Non-blocking** — repairing the torch env perturbs the BLAS stack, and 2 of 3 models aren't cached, so byte-exact reproduction of the Feb-2026 files may simply not be achievable. Don't let the plan stall waiting for it.

**Gate:** stub golden byte-identical across two runs and under a different `PYTHONHASHSEED`.

### Phase 2 — Package (~3–4 days, one commit per bullet, stub golden green after each)

- `git mv src src/aicds` + mechanical import rewrite. **Nothing else in this commit.**
- Real `pyproject.toml`: dependencies, extras `baseline`/`bert`/`dev`, `[project.scripts]`, readme → root README. Collapse the three dependency manifests. Delete every `sys.path.insert`.
- Consolidate `ensure_nltk_data` (×3) and `format_time` (×3).
- `stop_words` into `preprocessing.py` as **two commits**: define it while leaving all **8** monkey-patches in place, then delete the patches. It feeds tokenization → embeddings → every number, so it deserves an unambiguous culprit commit. (First commit landed as `5ae01a0`, which defines it in `cython_utils.py` rather than a new `preprocessing.py`. The count of 4 here was wrong: the sites are `baseline_sent2vec.py:36`, `bert_models.py:211`, `bert_eval.py:63`, `analyze_score_distributions.py:46`, `test_characterize_dataset.py:196` and `:215`, `test_characterize_similarity.py:46`, `test_characterize_preprocessing.py:37` — all the identical `set(stopwords.words('english'))`, so deletion is provably inert.)
- Replace `os.getcwd()` output roots with an explicit `--out`/`results/` root; fix the `build_readme_plots` glob in the same commit. **Bigger than recorded here** — there are five discovery sites disagreeing four ways about the base directory, *and* the baseline writes `Prediction Output_` with a space so no glob matches it at all. See `docs/findings/10-output-path-fragmentation.md`; the fix direction is forced by `test_golden.py:115`.
- **Salvage `encoders/hf_automodel.py` out of `bert_eval.py` before deleting the rest.** Those 545 orphaned lines contain a raw `AutoModel` + mean-pooling path — a genuinely different encoder from SentenceTransformer's pooling — and the repo's only GPU-aware line. That's exactly what roadmap items (iii) more encoders and (vi) a decoder chatbot would reuse. Archive rather than delete; record the SHA.
- Then delete the verified-dead: 4 `cython_utils` functions, `print_log` (references an undefined `LOG`), unused gensim/sklearn imports, `orig_stdout`, dead matplotlib import, `entity/{Admission,Symptom,Drgcodes}`.
  - **DO NOT delete `get_diagnosis_similarity_baseline` (`cython_utils.py:353`) or `get_diagnosis_similarity_by_drgcode` (`:365`).** They are zero-caller, so they look dead, but they are the only **encoder-independent** graders in the repository — pure string comparison, no embedding — and they are precisely what `correctness-fixes.md` item 2 ("replace the cosine grader") needs. `:353` returns graded credit (fraction of ground-truth diagnoses matched), `:365` returns a binary hit. Deleting them would throw away a written-and-committed head start on the metric work. Move them somewhere honest instead; only `get_diagnosis_similarity_by_description_max_model` (`:334`) and `print_log` are genuinely disposable.
  - Note `entity/{Admission,Symptom,Drgcodes}` are imported by `tests/test_reorganization.py:17-19`, so deletion needs a same-commit test edit.
- **Fix the baseline's 5 defects** — data path, the `entity.` NameError, wrap module scope in `run_analysis()` with a `__main__` guard, collapse the two conflicting `PerformanceIndex` handles (`:315` opened and abandoned, `:456` a second handle to the same path), and the discarded `line.replace("\n","")`. Ship it behind an **`[UNVERIFIED]`** banner — it cannot be validated without the 21 GB model.
- **Do not merge the two pipeline loops.** They diverge in ways that matter: the BERT per-case header is space-padded while the aggregate uses a tab-delimited constant, and BERT per-case recall is `tp/(tp+fp)` where the baseline's is `tp/nrow`. Unifying them would silently rewrite the baseline's entire output format, and the oracle is BERT-only so nothing would catch it.

**Gate:** stub golden byte-identical; `pip install -e .` then `import aicds` succeeds **with sent2vec absent**.

### Phase 3 — CLI, encoders, one parser (~2–3 days) ← **SHIP POINT**

- `SentenceEncoder` Protocol shaped `embed_sentence(text) -> seq[seq[float]]` — every call site already indexes `emb[0]`, so extraction is a no-op. Registry with readable names plus `1|2|3` aliases.
- The full argparse tree above; stdin-closed test over every subcommand.
- **One** `PerformanceIndex` parser replacing all three, plus one discovery rule (*a directory is a run iff it contains `PerformanceIndex.txt`*). Gate on an equivalence test proving the new parser reproduces all three old parsers on all goldens **before** deleting them. The new parser raises on unrecognized section headers instead of silently returning `None`.
- Fix the dashboard: regenerate into `dashboard/public/data/`, commit it, delete the stray `dashboard/dashboard/` tree. It 404s today.
- `scripts/` → deprecation shims.

**Gate:** stub golden green; parser equivalence green; dashboard renders; plots regenerate.

### Phase 4 — Docs — **DONE** (`bb84b82`, pulled forward ahead of 1–3)

`docs/` is illegible because it mixes four unrelated kinds of thing in one flat directory: stale setup guides, actual scientific findings, generated SVGs, and raw result dumps. Splitting by *kind* is what makes it navigable again.

```
docs/
  README.md            <- INDEX: what to read, in what order
  findings/            the science (human-written, the reason this repo exists)
    01-baseline-reproduction.md
    02-encoder-comparison.md        (from bert_model_comparison.md)
    03-metric-saturation.md         (from score_distribution_analysis/)
    04-metric-degeneracy.md         (NEW — the P==R==FS result)
    reproduction-ledger.md          (what has/hasn't been re-run, and on what stack)
  guides/              operational
    setup.md           (ONE guide replacing 4 stale ones)
    data-use.md        (DUA/compliance)
  reference/
    architecture.md    (from ARCHITECTURE.md)
    cli.md             (generated from argparse — never hand-edited)
  generated/plots/     clearly machine-produced, safe to delete and rebuild
results/golden/        raw runs move OUT of docs/ — they're data, not documentation
```

- Archive (don't delete) the four stale guides that reference the long-gone `CS2V.py`: `docs/README.md`, `SETUP_GUIDE.md`, `James_wsl_code_setup.md`, `Reproduce_w_transformers.md`, plus root `FILE_DESCRIPTIONS.txt`.
- Correct 128 → 129 admissions everywhere.
- Rewrite the F1 claims in the root README to say plainly that the published P/R/F1 are all equal to TP/n.
- Resolve `.system-design-visualization.md`: it's tracked, but `.gitignore` lists it without the `.md` and `ARCHITECTURE.md` calls it local-only. Pick one.

### Phase 5 — Optional payoff (only if 0–4 landed clean)

`ExperimentConfig` + run manifest (git SHA, seed, dep versions, encoder, aggregator); aggregator registry; the two legacy threshold constants replacing the 8 scattered set literals. **And the real fix: report a genuine set-level P/R/F1 alongside the existing accuracy-as-F1.**

**Gate:** `--aggregator max` still byte-reproduces the stub golden.

---

## Verification

1. `pytest -m "not network"` — unit + characterization + golden, seconds, no model download.
2. **Stub-encoder golden** — full 10-fold pipeline, byte-exact, after every commit. The primary net.
3. `pip install -e .` → `import aicds` with sent2vec absent → `python main.py verify`.
4. `python main.py run bert --encoder bio_clinical_bert` vs `docs/` goldens; discrepancies recorded in the ledger rather than treated as a hard gate.
5. `python main.py report plots|dashboard` → regenerated artifacts match committed ones; dashboard loads in a browser.

## Decisions — resolved

All three were settled by the repo owner; see the status block at the top of this file.
In short: leave the committed MIMIC-III data alone, deprioritize the (scaffolded, original-authors)
baseline arm, and continue past the Phase 3 ship point.

## Deliberately not doing

Creating `index/`, `remote/`, `chat/`, `orchestration/` directories; merging the two pipeline loops; rewriting git history; `KMP_DUPLICATE_LIB_OK=TRUE` as the torch "fix"; deleting `bert_eval.py` without salvaging its AutoModel path.

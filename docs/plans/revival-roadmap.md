# Reviving AI-CDS: reorganization, `main.py`, and docs

> **Status — Phases 0, 1 and 4 complete.** Phase 0 (`bb84b82`): environment repaired, data-use
> guard installed, `docs/` rebuilt into `findings/` + `guides/` + `reference/`. Phase 4 (docs)
> was pulled forward and is also done. **Phase 1 (safety net) is now done**: 93 passed /
> 2 deselected in ~4s, plus `pytest -m golden` — a byte-exact 10-fold regression against a
> committed reference, ~20 min.
>
> **Phases 2–3 are the remaining work**: the `src/aicds/` package move and the `main.py` CLI.
> Phase 5's metric work has been **removed from this roadmap** — the scientific correctness
> fixes (patient leakage, the `w/o` negation bug, comma-split fragments, embedding centring,
> rank-aware metrics) are tracked separately in [correctness-fixes.md](correctness-fixes.md)
> and [metric-redesign.md](metric-redesign.md), to be done in a dedicated session.
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
>    code, not the owner's contribution**. It stays importable and clearly labelled broken; no
>    effort goes into verifying it.
> 3. **Scope — continue past the Phase 3 ship point**, since the docs work was explicitly wanted.

## Context

You want to revive and extend this repo, and you want it clean first. The recon turned up something more urgent than untidiness: **three things are broken right now**, and one of the project's central claims doesn't survive arithmetic. The reorganization is worth doing, but it has to be sequenced so it fixes what's broken without silently changing the science.

### What's actually broken (all verified directly, not inferred)

| # | Problem | Evidence |
|---|---|---|
| 1 | **The baseline arm cannot run at all.** | `baseline_sent2vec.py:236` reads `CH_DIR/Symptoms-Diagnosis.txt`; the file only exists at `data/raw/`. → `FileNotFoundError` on import. Then `:244-247` calls bare `entity.SymptomsDiagnosis...` but line 16 binds `entity_module` — `entity` is never bound → `NameError`. `run_baseline.py` triggers both via import. |
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

### Phase 1 — Characterize and freeze (~2–3 days) — **NEXT**

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
- `stop_words` into `preprocessing.py` as **two commits**: define it while leaving all 4 monkey-patches in place, then delete the patches. It feeds tokenization → embeddings → every number, so it deserves an unambiguous culprit commit.
- Replace `os.getcwd()` output roots with an explicit `--out`/`results/` root; fix the `build_readme_plots` glob in the same commit.
- **Salvage `encoders/hf_automodel.py` out of `bert_eval.py` before deleting the rest.** Those 545 orphaned lines contain a raw `AutoModel` + mean-pooling path — a genuinely different encoder from SentenceTransformer's pooling — and the repo's only GPU-aware line. That's exactly what roadmap items (iii) more encoders and (vi) a decoder chatbot would reuse. Archive rather than delete; record the SHA.
- Then delete the verified-dead: 4 `cython_utils` functions, `print_log` (references an undefined `LOG`), unused gensim/sklearn imports, `orig_stdout`, dead matplotlib import, `entity/{Admission,Symptom,Drgcodes}`.
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

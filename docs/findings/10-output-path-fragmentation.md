# 10 — Nothing in the repo can find the baseline's output

**The baseline arm writes its results to a directory name that no glob in the codebase matches.**
Five separate discovery sites all spell the pattern `Prediction_Output_*`; the baseline writes
`Prediction Output_` — with a space. The dashboard, the plot builder, and the analysis script are
structurally blind to the entire BioSentVec arm, and have been for as long as both arms have
existed.

This was found on 2026-08-05, immediately after the baseline produced output for the first time.
Nobody had noticed because until that day the baseline had never successfully written anything.

---

## The two spellings

| Arm | Site | Produced name |
|---|---|---|
| Baseline | `baseline_sent2vec.py:272-274` | `Prediction Output_05082026 18-55-32/` |
| BERT | `bert_models.py:413-414` | `Prediction_Output_BiomedBERT_15022026_12-03-36/` |

They diverge in **three** independent ways: space vs. underscore after `Prediction`, absent vs.
present model name, and a space vs. an underscore between date and time.

### Root cause of the space

The BERT arm calls `time.strftime("%d%m%Y_%H-%M-%S")` directly. The baseline routes through
`cython_utils.current_time()` (`cython_utils.py:185-187`), which returns
`strftime("%d/%m/%Y %H:%M:%S")`, and then scrubs it:

```python
timestamp = util_cy.current_time().replace('/', '').replace(':', '-')
```

The `/` and `:` are removed because they are illegal in path names. **The literal space in the
format string is not, so it survives into the directory name.** The bug is that a
display-formatted timestamp was reused as a filesystem identifier.

The sibling "details" directory splits the same way (`Prediction Symptom Details_` vs.
`Prediction_Symptom_Details_{model}_`), unnoticed because nothing reads it at all.

## Everything that looks for output, and where

| Site | Pattern | Base directory | Finds baseline? |
|---|---|---|---|
| `scripts/build_dashboard_data.py:172` | `Prediction_Output_*` | repo root + `docs/` | No |
| `scripts/build_readme_plots.py:30` | `Prediction_Output_*` | repo root | No |
| `scripts/analyze_performance.py:27` | `Prediction_Output_*` | process cwd | No |
| `tests/test_reorganization.py:75` | `**/Prediction_Output_*` | repo root, recursive | No |
| `tests/test_golden.py:259` | `Prediction_Output_*` | pytest tmp dir | No |

So the previously documented defect — *"three globs disagree about the base directory"* — is
really **five sites disagreeing four ways about location, and unanimously wrong about the name.**

`scripts/analyze_score_distributions.py` is not a participant: it re-embeds from raw data and
never discovers a run directory. Exclude it from any unification.

## Three consequences that are not obvious

**1. `.gitignore` protects the wrong arm.** Lines 56–57 carry `Prediction Output*/` and
`Prediction Symptom Details*/` — written against the *baseline* spelling — and both are **commented
out**. Had they been live they would have ignored baseline output while leaving BERT output
exposed. The DUA-relevant ignore that actually works today is the unrelated `results/` entry.

**2. The golden's timestamp scrubber assumes underscores.** `tests/test_golden.py:115`:

```python
_TIMESTAMP_RE = re.compile(r"\d{8}_\d{2}-\d{2}-\d{2}")
```

This is defensive — run-dir names do not currently leak into the file body — but it constrains the
fix. **Unifying the baseline onto the BERT spelling makes this regex correct for both arms;
unifying the other way silently breaks it.** That settles the direction of the fix.

**3. Model-name recovery degrades silently, not loudly.** Both `build_readme_plots.py:104` and
`build_dashboard_data.py:88` derive a model label by stripping the prefix:

```python
os.path.basename(os.path.dirname(path)).replace("Prediction_Output_", "")
```

Against a baseline directory the prefix does not match, so `.replace` is a no-op and the "model
name" comes back as the literal string `Prediction Output_05082026 18-55-32`. No exception is
raised. It would then miss the `MODEL_COLORS` lookup (`build_readme_plots.py:18-22`, keyed to the
three BERT names only) and render as an unlabelled series.

## A related discovery: the per-case output files are all empty

The baseline run produced **258 files of zero bytes** — 129 under `Fold*/` in the output tree and
129 more in the symptom-details tree. Not a truncated download; the pipeline never writes them.

`cython_utils.py:65-66` opens `prediction_out_file` and `detailed_out_file`; `:170-171` closes
them. **Nothing writes between those points** — those two handles appear on exactly four lines in
the whole repository.

The BERT arm reaches the same end state by a different route: `predict_topk_diagnoses_pure`
(`bert_models.py:140`) never opens per-case files at all, yet `bert_models.py:451-456` still
creates the `Fold*/` directories. They stay empty, and since git cannot track empty directories,
the three committed BERT runs contain only two files each.

So the apparent asymmetry between the arms — baseline has `Fold*/` contents, BERT does not — is
cosmetic. **Neither arm has ever emitted per-case output.** The write path is dead code in one arm
and absent in the other, which is worth knowing before anyone tries to build case-level analysis on
top of it.

(`timing_report.pdf` is a genuine asymmetry: `generate_timing_pdf` exists only at
`baseline_sent2vec.py:48`, with no BERT equivalent.)

## Three parsers, and the one divergence between them

The roadmap's "three `PerformanceIndex` parsers" are confirmed:

| Site | Scope | Notable |
|---|---|---|
| `analyze_performance.py:73` | per-fold **and** 10-FOLD | the only one that reads per-fold blocks; ignores the `PR` column |
| `build_readme_plots.py:37,61` | 10-FOLD only | emits key `"FS"` |
| `build_dashboard_data.py:29,46` | 10-FOLD only | near-identical clone; emits key **`"F1"`** |

`FS` vs `F1` for the same column is the only substantive difference between the latter two — and
given [09](09-baseline-first-run.md), naming that column `F1` in the dashboard is the more
misleading of the two choices for the BERT arm, where the value is accuracy.

A fourth consumer belongs in the blast radius of any format change: `tests/test_golden.py` treats
the file as bytes for exact comparison (`strip_trailer`, `normalise_paths`, `canonicalise`).

## The fix, and its constraint

Unify **writers** onto the BERT spelling `Prediction_Output_{model}_{DDMMYYYY_HH-MM-SS}` — direction
forced by consequence 2 above — with the baseline passing a model name of `BioSentVec`. Then unify
the five readers onto one run-discovery rule and the three parsers onto one implementation.

**This is refactor-safe.** It changes where bytes land and what reads them, not what the pipeline
computes, so it sits inside the roadmap's scope rule (*"if a change moves the numbers it is out of
scope; if it fixes something that crashes, blocks, or writes to the wrong place it is in scope"*).
The golden must still pass byte-for-byte afterward.

---

*Companion documents:* [09](09-baseline-first-run.md) the baseline's first run ·
[08](08-runtime-and-cost.md) runtime

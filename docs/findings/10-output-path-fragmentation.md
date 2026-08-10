# 10 — Nothing in the repo can find the baseline's output

> **In plain words.** The two halves of the project write their results into folders with
> *almost* the same name — one says `Prediction_Output_...`, the other `Prediction Output_...`
> with a space. Every tool that goes looking for results searches for the underscore spelling,
> so the baseline's output is invisible to all of them: the dashboard, the plot builder, the
> analysis script. Nobody noticed for months because the baseline had never successfully
> written anything to find. A second discovery made while looking: the per-patient detail
> files both halves are supposed to write are **all empty** — 258 zero-byte files — because the
> code opens them and closes them without ever writing in between. That empty-file bug later
> turned out to matter much more than the naming one: it is now P40, the thing blocking the
> last open statistical question in the project.

**The baseline arm writes its results to a directory name that no glob in the codebase
matches.** The discovery sites all spell the pattern `Prediction_Output_*`; the baseline writes
`Prediction Output_` — with a space. The dashboard, the plot builder, and the analysis script
are structurally blind to the entire BioSentVec arm, and have been for as long as both arms
have existed.

This was found on 2026-08-05, immediately after the baseline produced output for the first
time. Nobody had noticed because until that day the baseline had never successfully written
anything.

---

## The two spellings

| Arm | Site | Produced name |
|---|---|---|
| Baseline | `baseline_sent2vec.py` | `Prediction Output_05082026 18-55-32/` |
| BERT | `bert_models.py` | `Prediction_Output_BiomedBERT_15022026_12-03-36/` |

They diverge in **three** independent ways: space vs. underscore after `Prediction`, absent vs.
present model name, and a space vs. an underscore between date and time.

### Root cause of the space

The BERT arm calls `time.strftime("%d%m%Y_%H-%M-%S")` directly. The baseline routes through
`cython_utils.current_time()`, which returns `strftime("%d/%m/%Y %H:%M:%S")` — a
*display*-formatted timestamp — and then scrubs it:

```python
timestamp = util_cy.current_time().replace('/', '').replace(':', '-')
```

The `/` and `:` are removed because they are illegal in path names. **The literal space in the
format string is not, so it survives into the directory name.** The bug is that a
display-formatted timestamp was reused as a filesystem identifier.

The sibling "details" directory splits the same way (`Prediction Symptom Details_` vs.
`Prediction_Symptom_Details_{model}_`), unnoticed because nothing reads it at all.

## Everything that looks for output, and where

As found on 2026-08-05:

| Site | Pattern | Base directory | Finds baseline? |
|---|---|---|---|
| `scripts/build_dashboard_data.py:172` | `Prediction_Output_*` | repo root + `docs/` | No |
| `scripts/build_readme_plots.py:30` | `Prediction_Output_*` | repo root | No |
| `scripts/analyze_performance.py:27` | `Prediction_Output_*` | process cwd | No |
| `tests/test_reorganization.py:75` | `**/Prediction_Output_*` | repo root, recursive | No |
| `tests/test_golden.py:259` | `Prediction_Output_*` | pytest tmp dir | No |

So the previously documented defect — *"three globs disagree about the base directory"* — was
really **five sites disagreeing four ways about location, and unanimously wrong about the
name.** (Update 2026-08-08: it has since grown to six-to-seven — `compare_models.py` reads its
own `results/<model>/<timestamp>/` layout and `analyze_rank_metrics.py` a
`<results_dir>/<arm>/*/RankMetrics.txt` rule. Every new script has added a private discovery
rule; that trend continues until the Phase 3 unification lands.)

`scripts/analyze_score_distributions.py` is not a participant: it re-embeds from raw data and
never discovers a run directory. Exclude it from any unification.

## Three consequences that are not obvious

**1. `.gitignore` protects the wrong arm.** The lines carrying `Prediction Output*/` and
`Prediction Symptom Details*/` — written against the *baseline* spelling — are **commented
out**. Had they been live they would have ignored baseline output while leaving BERT output
exposed. The DUA-relevant ignore that actually works today is the `results*/` glob.

**2. The golden's timestamp scrubber assumes underscores.** `tests/test_golden.py:115`:

```python
_TIMESTAMP_RE = re.compile(r"\d{8}_\d{2}-\d{2}-\d{2}")
```

This is defensive — run-dir names do not currently leak into the file body — but it constrains
the fix. **Unifying the baseline onto the BERT spelling makes this regex correct for both arms;
unifying the other way silently breaks it.** That settles the direction of the fix.

**3. Model-name recovery degrades silently, not loudly.** Both `build_readme_plots.py` and
`build_dashboard_data.py` derive a model label by stripping the prefix:

```python
os.path.basename(os.path.dirname(path)).replace("Prediction_Output_", "")
```

Against a baseline directory the prefix does not match, so `.replace` is a no-op and the "model
name" comes back as the literal string `Prediction Output_05082026 18-55-32`. No exception is
raised. It would then miss the `MODEL_COLORS` lookup (keyed to the three BERT names only) and
render as an unlabelled series.

## A related discovery: the per-case output files are all empty — now P40, the binding constraint

The baseline run produced **258 files of zero bytes** — 129 under `Fold*/` in the output tree
and 129 more in the symptom-details tree. Not a truncated download; the pipeline never writes
them.

`cython_utils.py:65-66` opens `prediction_out_file` and `detailed_out_file`; `:170-171` closes
them. **Nothing writes between those points** — those two handles appear on exactly four lines
in the whole repository.

The BERT arm reaches the same end state by a different route: `predict_topk_diagnoses_pure`
never opens per-case files at all, yet `bert_models.py` still creates the `Fold*/` directories.
They stay empty, and since git cannot track empty directories, the three committed BERT runs
contain only two files each.

So the apparent asymmetry between the arms — baseline has `Fold*/` contents, BERT does not — is
cosmetic. **Neither arm has ever emitted per-case output.**

**Update 2026-08-08: this stopped being a cosmetic wart.** Finding
[13](13-rank-aware-metrics.md) needs per-case relevance vectors to test whether the baseline's
98 self-answered cases are simply the easy ones — the one confound in the abstention analysis
that cannot be bounded from aggregates. Nothing else in the repo carries per-case relevance, so
the empty files are now **P40, the highest-value open item**. Scope rule when fixing it: write
a *new sibling file*, do not resurrect the dead handles — that keeps the golden covering
exactly what it covered before.

(`timing_report.pdf` is a genuine asymmetry: `generate_timing_pdf` exists only in
`baseline_sent2vec.py`, with no BERT equivalent.)

## The parsers, and the one divergence between them

The roadmap's "three `PerformanceIndex` parsers" were confirmed — and the count has since grown
to **four** (`compare_models.py` added its own):

| Site | Scope | Notable |
|---|---|---|
| `analyze_performance.py:73` | per-fold **and** 10-FOLD | the only one that reads per-fold blocks; ignores the `PR` column |
| `build_readme_plots.py:37,61` | 10-FOLD only | emits key `"FS"` |
| `build_dashboard_data.py:29,46` | 10-FOLD only | near-identical clone; emits key **`"F1"`** |
| `compare_models.py:231` | 10-FOLD, pipeline-keyed sanity assertions | added 2026-08-06; refuses rather than guesses |

`FS` vs `F1` for the same column is the only substantive difference between the two clones —
and given [09](09-baseline-first-run.md), naming that column `F1` in the dashboard is the more
misleading of the two choices for the BERT arm, where the value is accuracy.

A fifth consumer belongs in the blast radius of any format change: `tests/test_golden.py`
treats the file as bytes for exact comparison (`strip_trailer`, `normalise_paths`,
`canonicalise`).

## The fix, and its constraint

Unify **writers** onto the BERT spelling `Prediction_Output_{model}_{DDMMYYYY_HH-MM-SS}` —
direction forced by consequence 2 above — with the baseline passing a model name of
`BioSentVec`. Then unify the readers onto one run-discovery rule (*a directory is a run iff it
contains `PerformanceIndex.txt`*) and the parsers onto one implementation.

**This is refactor-safe.** It changes where bytes land and what reads them, not what the
pipeline computes, so it sits inside the roadmap's scope rule (*"if a change moves the numbers
it is out of scope; if it fixes something that crashes, blocks, or writes to the wrong place it
is in scope"*). The golden must still pass byte-for-byte afterward. It is Phase 3 work, and as
of 2026-08-08 it has not been started.

---

*Companion documents:* [09](09-baseline-first-run.md) the baseline's first run ·
[08](08-runtime-and-cost.md) runtime · [13](13-rank-aware-metrics.md) what made P40 matter

# 08 — Where the runtime actually goes

**Short answer: the encoder costs almost nothing. Over 93% of wall-clock is a single-threaded,
pure-Python cosine-similarity loop that would run identically if the embeddings arrived by post.**

This matters for three separate decisions the project keeps circling back to — whether to buy a GPU,
whether to parallelise, and how to size a rented box — so the measurements live here rather than
being re-derived each time.

All figures below are **measured**, with the machine named. Anything not measured is labelled an
estimate.

---

## 1. The cost breakdown, per BERT run

From the three committed `timing_report.txt` files under `docs/Prediction_Output_*/`, all produced
on the M-series Mac on 2026-02-15:

| Stage | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|---|---|---|---|
| Dataset loading | 0.00 s | 0.00 s | 0.00 s |
| **Model loading** | 81.32 s | 15.31 s | 11.12 s |
| **Embedding (symptoms + diagnoses)** | **5.98 s** | **3.46 s** | **2.12 s** |
| **10 folds of scoring** | **1231.52 s** | **1248.55 s** | **1228.11 s** |
| Total | 1318.82 s | 1267.33 s | 1241.36 s |

Read the middle two rows together. **Embedding is 0.45%, 0.27% and 0.17% of total runtime.** The
fold loop is 93.4%, 98.5% and 98.9%.

The 81.32 s model-load outlier for Bio_ClinicalBERT is a cold HuggingFace download; the other two
were served from the local cache.

### The three encoders cost the same to run

Total fold time varies by **1.7%** across the three models (1231.52 / 1248.55 / 1228.11 s) — well
inside run-to-run noise. That is the clearest possible evidence that the encoder is not the
bottleneck: three different transformers, three different tokenizers, three different training
corpora, and the runtime is flat.

The reason is structural. Every model produces 768-dimensional vectors, and the fold loop's cost is
a function of *dimension count and pair count*, not of how the vectors were produced. Encoding runs
once, up front, over ~1,805 unique symptom strings and 145 diagnosis strings. The loop then runs
mean-of-max cosine over every (test case × training patient × symptom × symptom) combination, ten
times.

## 2. What this rules out

**A GPU buys essentially nothing for the encoder pipeline.** Moving 100% of embedding work to a GPU
and making it instantaneous would cut a Bio_ClinicalBERT run from 21.98 to 21.88 minutes. This is
why the RunPod instance is a *CPU*-optimised tier and why GPU spend is not on the roadmap for this
arm. (The picture inverts completely for the planned local decoder, which is genuinely
GPU-bound — see the RAG/decoder plan.)

**Faster encoders are not a lever either.** Distilling or quantising the models would attack under
half a percent of runtime.

The only levers that matter are (a) making the similarity loop itself cheaper, or (b) running the
ten folds concurrently. Both are real options; both are deferred, and for a specific reason —
see §5.

## 3. Baseline arm, measured on the pod (2026-08-05)

First-ever successful run of the BioSentVec arm in this checkout. RunPod CPU pod, AMD Ryzen
Threadripper 7960X, 32 vCPU, 64 GB, Ubuntu 20.04, $1.12/hr.

| Fold | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Seconds | 80.29 | 63.87 | 65.63 | 74.97 | 88.99 | 81.73 |

**Mean 75.9 s/fold** over the first six folds. Per test case: n=101, mean 5.8 s, max 17 s.

The 13-test-cases-per-fold structure is uniform (`LEN train: 116, LEN test: 13`), so the fold-time
spread of 63.87–88.99 s is driven by how many symptoms each fold's test admissions carry, not by
uneven splits.

## 3b. The results are platform-independent, verified bit-for-bit

Bio_ClinicalBERT was re-run on the pod on 2026-08-05 and compared against the committed
Feb-2026 Mac run. The 10-FOLD TOP-10 block is **identical to all 17 significant figures**:

```
        TP    FP    P                    R                    FS                   PR
0.9    10.3   2.6   0.7974358974358975   0.7974358974358975   0.7974358974358975   1.0
1       3.7   9.2   0.28525641025641024  0.28525641025641024  0.28525641025641024  1.0
0.6    12.9   0.0   1.0                  1.0                  1.0                  1.0
```

Across macOS/ARM (Accelerate) → Linux/x86 Threadripper (OpenBLAS), a different torch build, a
different Python build, and six months. **Not a single digit moved.**

Two things follow.

**The Feb-2026 results reproduce.** The committed `docs/Prediction_Output_*` oracle is trustworthy
across platforms, not just on the machine that minted it.

**The hardware confound is empirically null.** Sections 4 and
[09](09-baseline-first-run.md) warn that comparing a Mac BERT run against a Linux baseline run
varies the machine as well as the encoder. For *accuracy* that concern is now measured and
dismissed — hardware contributes nothing. The remaining confounds in that comparison are real and
unaffected: divergent diagnosis preprocessing, the leaky folds, and the self-grading metric.

Why it holds: the reported figures are TP/FP counts averaged over folds, and a platform-level
floating-point difference would have to be large enough to flip a threshold comparison to change
one. Similarities sit far from the thresholds, so the counts are robust. This is evidence that the
*results* are platform-stable, not that the embeddings are bit-identical.

**Speed, by contrast, is entirely platform-dependent** — which is exactly the split you would want.

## 4. The speed comparison that is *not* yet valid

It is tempting to divide 123.15 s/fold (Mac, BERT) by 75.9 s/fold (pod, baseline) and call the pod
1.6× faster. **That number is confounded exactly the way the accuracy comparison is** — it varies
the machine *and* the model *and* the embedding dimension (700 vs 768) *and* the preprocessing
(the baseline calls `preprocess_sentence` on diagnosis text; the BERT arm does not, so the two arms
score different numbers of unique strings). At least four variables move at once.

A clean pod-vs-Mac ratio requires running the *same* arm on both machines. Running the BERT arm on
the pod supplies that, and it is the reason to do so beyond mere convenience: **it removes the
hardware variable from the accuracy comparison as well.**

### The clean measurement (2026-08-05)

Bio_ClinicalBERT, fold 0, identical code (`7da5901`), identical model, identical data:

| Machine | Fold 0 |
|---|---|
| M-series Mac | 141.53 s |
| RunPod Threadripper 7960X | **93.91 s** |

**Ratio 1.51×** — with only the hardware varying. The earlier confounded estimate of "roughly 1.5×"
happened to land correctly, but it was not evidence; this is.

Projected from the Mac's 1231.52 s of fold time: **~13.6 min per model, ~41 min for all three.**

## 5. Why parallelisation is deferred rather than done

The ten folds are embarrassingly parallel — no fold reads another fold's output — so a 10× wall-clock
win is genuinely available. Two constraints hold it back:

1. **The GIL makes threads useless here.** The loop is pure-Python arithmetic, not I/O and not a
   released-GIL numpy call, so `threading` would serialise it. `multiprocessing` is required, which
   means paying to serialise the embedding dictionaries into each worker — tolerable, but real.
2. **Vectorising the loop with numpy would break the golden.** Numpy's pairwise/SIMD summation does
   not accumulate in the same order as a sequential Python loop, so sums differ in the last bits.
   `tests/golden/stub768/PerformanceIndex.txt` is compared **byte-for-byte**, deliberately. Any
   vectorisation is therefore a numbers-moving change and belongs with the correctness work, not
   with the refactor.

Multiprocessing over folds does *not* have problem 2 — each fold's arithmetic is untouched — which
makes it the safe half of the speedup and the one to do first.

## 6. Cost, for the record

| Item | Figure |
|---|---|
| Pod rate | $1.12/hr |
| Baseline arm, 10 folds | ~13 min compute (~$0.25) |
| Three BERT arms | ~45 min estimated (~$0.85) |
| BioSentVec model on disk | 20.93 GiB (22,475,736,490 bytes) |
| Each BERT model on disk | 416–420 MB |

**BioSentVec is ~17× the size of all three transformers combined** (20.93 GiB vs ~1.23 GB). The
inversion is worth understanding: sent2vec stores an explicit embedding table for millions of
unigrams *and* bigrams at 700 dimensions, so its footprint scales with vocabulary. BERT stores
~110M parameters and computes representations contextually at inference time. The smaller artifacts
are the more expressive ones.

That size is not a curiosity — it is why the baseline could not run on the 32 GB Windows
workstation alongside anything else, and it is half the reason this arm moved to rented hardware.
The other half is that `sent2vec` cannot build under MSVC at all
([07-comparison-validity.md](07-comparison-validity.md)).

---

*Companion documents:* [07](07-comparison-validity.md) comparison validity ·
[03](03-metric-saturation.md) saturation · [05](05-patient-leakage.md) leakage

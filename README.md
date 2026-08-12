# AI-CDS Disease Diagnosis System

Author: An Thien Vo (James)

Clinical Decision Support System for disease diagnosis prediction using patient symptom similarity.

## TL;DR — a 416 MB model replaces a 21 GB one with no measurable loss

**Bio_ClinicalBERT matches the 20.93 GiB BioSentVec baseline on every retrieval metric, at under 2%
of its size.** The numbers, leakage-free:

**How much smaller:**

- **416 MB vs 20.93 GiB on disk** — roughly 1/50th the size for the single model. All three BERT
  models tested *together* (1.25 GB) are still 17× smaller than the one baseline file.
- **~110 million parameters vs ~5.6 billion** — 1/51.
- **A few hundred MB of RAM** vs the full 21 GB resident.
- Contextual 768-dim embeddings computed on demand, vs fixed 700-dim vectors looked up from a
  memorised table of ~8 million n-grams.
- **Runs on Windows**; the baseline physically cannot (sent2vec will not compile under MSVC).

**How it performs** — 10-fold, patient-grouped cross-validation of 129 hospital admissions from
100 patients, grouped so no patient ever appears on both sides of a split (fixing a leakage bug
that had inflated the original paper's published score by ~18%):

| Metric | BioSentVec (21 GB) | Bio_ClinicalBERT (416 MB) | Change |
|---|:---:|:---:|:---:|
| Accuracy (right answer in top 10) | 0.192 | **0.249** | **+29.5%** |
| Hit@1 (right answer ranked first) | 0.148 | **0.175** | **+17.9%** |
| Hit@50 (right answer in top 50) | 0.358 | **0.697** | **+94.8%** |
| MRR@50 (rank-weighted, no knobs) | 0.203 | **0.246** | **+21.3%** |
| Precision | **0.2512** | 0.2491 | −0.8% |
| Coverage (cases it answers at all) | 75.6% | **100%** | **+32.3%** |

*Hit@1/Hit@50 and MRR are measured on the 76 winnable cases; accuracy on all 129. The Change
column is the relative gain over the baseline — nominal only, since no gap here clears the
fold-to-fold noise (see the verdict below). The +94.8% on Hit@50 is the least meaningful of the
six: the baseline abstains rather than offering 50 candidates, so that row partly measures
willingness to guess.*

- Answers **100% of cases where the baseline declines 24.4%**. The only metric it concedes is
  precision, by 0.002 — a gap the baseline buys by refusing to answer a quarter of the time.
- Comparable runtime: **~11.5 vs ~13 minutes** per full 10-fold run (encoding is under 0.5% of
  wall-clock in every arm).
- **Bit-for-bit reproducible** across Apple-silicon and x86, to all 17 significant figures.

**The verdict, stated carefully:**

- **No gap in either direction is statistically significant** — largest paired *t* = 1.718 across
  the 10 folds vs the 2.262 needed for p < 0.05, against per-fold scores that swing
  **0.00–0.67 on identical data**.
- For a replacement, that is the *desired* outcome: **statistically indistinguishable performance
  at 1/51 the parameters and a fraction of the memory**.
- All scores sit against a **data-imposed ceiling of 58.9%** — for 53 of 129 patients the correct
  diagnosis appears nowhere in the searchable pool, so a *perfect* retriever caps there. Read
  against that ceiling, the 0.249 accuracy is **~42% of what a perfect retriever could achieve**.

The claim is **"drop-in replacement with no measurable loss"** — deliberately not "better." The
full statistics, the four evaluation defects found and fixed along the way, and why no encoder
*ranking* is supported are below.

---

## Start here — the project in plain language

**The idea being tested.** If two patients arrive at a hospital with similar symptoms, they probably
have similar diagnoses. So to guess a new patient's diagnosis, find the most similar past patients
and look at what *they* were diagnosed with. A 2022 paper (Comito et al., IEEE Access) built exactly
that and reported it worked reasonably well. This project rebuilds it, then asks a natural follow-up:
**that paper used a text-understanding model from 2019 — do newer, medically-trained models do
better?**

**How the system works, in five steps.** No machine learning is *trained* here; everything is
lookup and arithmetic.

1. Take 129 hospital admissions, each with a list of symptoms and the diagnoses that were given.
2. Turn every symptom and diagnosis phrase into a list of numbers (an *embedding*), so that phrases
   meaning similar things get similar numbers. **This is the only step the four models differ in.**
3. For a patient we're testing, compare their symptoms against every past patient's symptoms and
   score how similar they are.
4. Take the diagnoses of the top-scoring past patients as the prediction.
5. Check the prediction against what the patient was actually diagnosed with, and score it.

**What I found.** The four models cannot be told apart — but the three modern ones do it with
**1/51 the parameters and 1/17 the disk space**, which is a genuine engineering win. More
importantly, the *scoring method itself* turned out to be broken in four separate ways, and two of
them make the original paper's headline number about 18% too high. Documenting that is the real
contribution.

**The scoring is now fixed in four places, not two**, and the null result held through every one:
the data split, the text handling, the grader (which no longer uses the models' own judgement), and
the scoring rule (which now cares *where* the right answer ranked, not just whether it appeared).
Each fix removed a way the experimenter's arbitrary choices could have decided the answer. The models
still cannot be told apart, which is now a much harder claim to argue with than it was.

### The project in STAR form

**Situation.** A published clinical decision support result (F1 = 0.489) rested on a 2019 sentence
encoder and a 20.93 GiB model file. The code had never successfully run in this repository: the
baseline arm crashed on startup, and the one dependency it needed cannot be compiled on Windows at
all. There was no test suite, so there was no way to change anything safely.

**Task.** Get the original result reproducing, then swap the encoder for three modern biomedical BERT
models behind an *identical* pipeline, so the encoder is the only variable — and establish whether
the newer models are actually better.

**Action.**
- Fixed the crash bugs and got the baseline running end-to-end on rented Linux (the BERT extension is
  my own contribution; the baseline was scaffolded from the original authors' code).
- Built a safety net first: **413 tests**, including a byte-exact 10-fold regression reference. Since
  nothing is trained, every output number is a pure function of the inputs and the arithmetic — so any
  accidental behaviour change *is* a numerical change, and only a byte-exact comparison catches it.
- Audited the evaluation and found **four independent defects**, each measured rather than asserted.
- Fixed them behind a *selectable* switch, so the original pipeline still reproduces bit-for-bit while
  each correction gets its own numbers. Re-ran all four models on one machine after every change.
- Replaced the two metric knobs that were deciding the answer: an **encoder-independent grader** (DRG
  code strings, no cosine) removed the similarity threshold, and **rank-aware metrics** (MRR,
  Precision@K) removed the arbitrary `K`.
- Tested every remaining difference for statistical significance instead of ranking them by eye.

**Result.**
- **Reproduced the published number to within 0.007** (0.4824 vs 0.489) — the first successful
  reproduction by this codebase.
- **~18% of that published number was contamination.** After fixing the data split and the text
  handling, it falls to **0.3922**.
- **No encoder is significantly better than any other** — under any metric, any threshold, any `K`, or
  any treatment of abstention. Largest paired *t* found anywhere is **1.718** against the 2.262 needed
  for p < 0.05. Dropping a single one of the 10 data splits flips first place.
- **The modern models are ~51× smaller in parameters** (~110M vs ~5.6B) and ~17× smaller on disk
  (416 MB vs 20.93 GiB) for statistically identical accuracy.
- **Every "F1" in the transformer results is actually accuracy** — precision, recall and F-score are
  the same number in all 12,600 original rows and all 90 corrected ones, with zero exceptions. The
  cause is now pinned: those arms never decline to answer, so two different denominators are the same
  set. The columns were never wrong, only unlabelled.
- **The three arbitrary knobs each reorder the four models on identical data.** The similarity
  threshold, `K`, and how abstention is scored all flip first place. Two were removable; the third is
  not, because abstention is a property of the model rather than of the metric.
- **The dataset caps any exact-match score at 58.9%**, because 105 of 145 diagnoses appear only once,
  so the correct answer is often not available to be retrieved at all.

### The findings, in one list

| # | Finding | Measured | Status |
|:-:|---|---|---|
| 1 | **Patient leakage** — the data split let the same patient appear as both question and answer | 41 of 129 test cases; worth +0.11 to +0.26, ~10× the difference between encoders | **fixed** — `GroupKFold` on patient |
| 2 | **Broken text cleanup** — `w/o` ("without") became `w` ("with"), destroying negation | 119 of 145 diagnosis labels differed between arms | **fixed** — now 145/145 identical |
| 3 | **Self-grading** — each model both retrieved candidates *and* judged its own answers, so a compact embedding space could mark its own work leniently | mean cosine between *unrelated* diagnoses is 0.72–0.93 | **fixed** — a DRG-string grader with no cosine reproduces it bit-exactly, so the bias was possible but never actual |
| 4 | **Rank discarded** — a hit at rank 1 and a hit at rank 50 scored identically, so the score rose with `K` by construction | 8,407 strictly-decreasing Precision@K steps once rank is scored | **fixed** — MRR has no `K` |
| 5 | **Abstention decides the ranking** — score a declined case as a failure and the transformers lead; exclude it and the baseline leads | every sign inverts; baseline goes 4th → 1st | **open, and not closable** — abstention belongs to the model, not the metric |
| 6 | **Saturation** — at the paper's own threshold, ~100% of patient pairs count as a match, so all three BERT models score a perfect 1.000 | 99.96–100% of pairs above 0.6 | **open**, but confined to cosine grading below threshold 1.0 |
| 7 | **Degeneracy** — the metric labelled "F1" is arithmetically just accuracy | all 12,600 + 90 rows, zero exceptions | **explained** — it is coverage = 1.0, i.e. those arms never abstain, so two denominators coincide |
| 8 | **No statistical power** — with 129 patients and per-split scores ranging 0.00–0.67, this design cannot resolve the differences it reports | largest paired \|*t*\| anywhere is 1.718 vs 2.262 needed | applies to the original paper equally |

### Vocabulary, if any of the above was unfamiliar

| Term | What it means here |
|---|---|
| **Embedding** | a phrase turned into a list of numbers, so a computer can measure whether two phrases mean similar things |
| **Cosine similarity** | how closely two of those number-lists point in the same direction; 1.0 = identical, 0 = unrelated |
| **Fold / 10-fold** | splitting the data into 10 groups and testing on each in turn, so you never test on data the system could look up |
| **Leakage** | when the answer sneaks into the material the system is allowed to search — it inflates scores without improving the system |
| **Precision / Recall / F1** | of what it predicted, how much was right / of what was right, how much it found / a single score combining the two |
| **TOP-K** | how many similar past patients the system is allowed to copy diagnoses from |
| **Threshold** | how similar two diagnosis phrases must be before we call them a match |
| **BioSentVec / BERT** | the 2019 baseline text model, and the modern family of models replacing it |

---

## The formal version

This project reproduces the clinical decision support system of *"AI-Driven Clinical Decision
Support: Enhancing Disease Diagnosis Exploiting Patients Similarity"* (Comito et al., 2022) and then
swaps its 2019 BioSentVec encoder for three modern biomedical BERT models behind the **identical**
retrieval and scoring pipeline. The result is a null result, and the null result is the
contribution: under this evaluation the old encoder and the new ones cannot be told apart.

**The headline numbers below are leakage-free.** The folds originally split on `HADM_ID`, letting 41
of 129 test cases retrieve the same patient's own other admission; they now split on `SUBJECT_ID`
with `GroupKFold`, and the two arms now preprocess diagnosis text identically. Both corrections are
selectable rather than destructive — `legacy` still reproduces the original pipeline bit-for-bit.

---

## Headline comparison — four encoders, one pipeline, leakage-free

All four arms were run on **one machine** (RunPod Linux, 32 vCPU) under `AICDS_PIPELINE=corrected`
on **2026-08-06**, from commits `c2115ba` + `31bea66`. Runtime was ~48 minutes for all four
(baseline ~13 min, each BERT arm ~11.5 min). The three transformer runs previously reproduced
Apple-silicon runs **bit-for-bit to all 17 significant figures**, so hardware is not a confound.

`corrected` changes **two things at once** — the fold split *and* the preprocessing — so the deltas
below cannot be attributed to one or the other. One-change-at-a-time configs (`folds-only`,
`preprocess-only`) exist for that and have not been run.

### Threshold 1.0, TOP-10 — the informative setting

The only threshold at which no model sits on the ceiling, and therefore the only one where the four
arms can be compared at all.

| Encoder | Dim | Model size | Precision | Recall | F | TP | FP | `TP+FP` | Pred. rate | Legacy F | Δ |
|---------|:---:|:----------:|:---------:|:------:|:------:|:---:|:---:|:-------:|:----------:|:--------:|:------:|
| Bio_ClinicalBERT | 768 | 416 MB | 0.2491 | 0.2491 | **0.2491** | 3.3 | 9.6 | 12.9 | 1.0000 | 0.2853 | −0.0362 |
| **BioSentVec — the 2019 baseline** | 700 | **20.93 GiB** | **0.2512** | 0.1923 | **0.2163** | 2.5 | 7.3 | **9.8** | **0.7558** | 0.2801 | −0.0638 |
| BiomedBERT | 768 | 420 MB | 0.1981 | 0.1981 | **0.1981** | 2.6 | 10.3 | 12.9 | 1.0000 | 0.2545 | −0.0564 |
| BlueBERT | 768 | 420 MB | 0.1821 | 0.1821 | **0.1821** | 2.4 | 10.5 | 12.9 | 1.0000 | 0.2391 | −0.0570 |

*TP and FP are means across the 10 folds (~12.9 test cases per fold). Pred. rate is the fraction of
test cases on which the system predicts at all. "Legacy F" is the same measurement under the
original leaky folds and divergent preprocessing. Model size is the on-disk weight file:
BioSentVec is **17× larger than all three transformers combined**, because sent2vec stores an
explicit unigram + bigram embedding table while BERT computes representations from ~110M parameters.*

**Every arm loses ground once leakage is removed — the baseline most of all (−0.0638).** That is the
expected direction: the leaked cases were free wins, and the baseline had the most to gain from
them.

**Verdict — no encoder ranking is supported by this experiment.** The 700-dimensional,
non-contextual, 2019 baseline lands **second of four** and holds the **highest precision of all four
arms** (0.2512). The next three sections are why the word "second" should not be trusted at all.

### None of these gaps clears the noise

The single most important table in this README. Every score above is a mean over 10 folds, and the
folds disagree wildly: per-fold F ranges from **0.00 to 0.67 on identical data**, with per-fold
standard deviations of **0.054–0.139**. Those sds are *larger than every gap between encoders*, so
before reading any ranking, ask whether the gap is bigger than the fold-to-fold wobble.

All four arms run on the **same** folds, so the right instrument is a **paired** *t*-test on the
per-fold differences (`t = mean(diff) / (sd(diff)/√10)`, 9 degrees of freedom, so **|t| > 2.262** is
needed for p < 0.05):

| Aggregator @ 1.0 | 1st | 2nd | Gap | Paired *t* | p < 0.05? |
|---|---|---|:---:|:---:|:---:|
| MAX | BioSentVec | Bio_ClinicalBERT | 0.0015 | 0.07 | **no** |
| TOP-10 | Bio_ClinicalBERT | BioSentVec | 0.0328 | 0.87 | **no** |
| TOP-20 | Bio_ClinicalBERT | BlueBERT | 0.0010 | 0.10 | **no** |
| TOP-30 | BlueBERT | Bio_ClinicalBERT | 0.0167 | 1.04 | **no** |

**Not one first-place margin is statistically significant.** The largest *t* anywhere is 1.04, less
than half the threshold. Pairing is the *generous* choice here too — fold difficulty is genuinely
shared, with per-fold F correlating at r = 0.72–0.96 among the three BERT arms — and the gaps still
do not survive it.

A leave-one-fold-out check makes the same point without any statistics: at MAX, dropping a single
fold hands first place to **BioSentVec in 5 of 10 cases and Bio_ClinicalBERT in the other 5.** The
baseline's "win" is a coin flip. Bio_ClinicalBERT's TOP-10 lead is the one that survives all ten
leave-one-out runs, but dropping fold 0 alone shrinks it from 0.0328 to 0.0068 — so it rests largely
on one fold in which it scored 0.60 while every other encoder scored 0.33.

**With n = 129 and 10 folds, this experiment does not have the statistical power to separate four
encoders whose true differences are this small.** That is a design finding about the study, not a
defect of the encoders, and it applies equally to the original paper.

**The leakage fix did not change any encoder's rank.** BioSentVec was 1st at MAX, 2nd at TOP-10, and
4th at TOP-20 through TOP-50 *both before and after*. What the fix changed is the magnitude — and it
moved **against** the baseline everywhere: its deficit to 1st widened at every `K` (TOP-10
0.0051 → 0.0328; TOP-30 0.0744 → 0.1383), and at MAX its margin over 2nd *shrank* from 0.0067 to
0.0015. So the correct reading is not "leakage removal let the old encoder catch up." It is that the
old encoder was **never distinguishable from the new ones in this data**, and removing leakage
removed the confound that could have explained that away.

**Read the `TP+FP` and prediction-rate columns together — they are the whole degeneracy story.**
Every BERT arm sums to exactly **12.9**, the mean fold test size, so `tp + fp == nrow`; precision
therefore reduces to `tp/nrow`, which *is* recall, and their harmonic mean is that same number.
**Every BERT "F" in this table is accuracy.** The baseline sums to **9.8** because it abstains on
24.4% of cases when nothing clears the pruning gate, which is why it alone has P ≠ R.

### The ranking inverts when you change the aggregator

This is the sharpest evidence that the ranking is not a property of the encoders. Holding the
threshold at 1.0 and changing only `K` — an arbitrary knob — **reverses the order**:

| Encoder | MAX | TOP-10 | TOP-20 | TOP-30 |
|---------|:---:|:------:|:------:|:------:|
| **BioSentVec (baseline)** | **0.0877 — 1st** | 0.2163 — 2nd | 0.2229 — 4th | 0.2296 — **4th** |
| Bio_ClinicalBERT | 0.0862 — 2nd | **0.2491 — 1st** | **0.3049 — 1st** | 0.3513 — 2nd |
| BiomedBERT | 0.0855 — 3rd | 0.1981 — 3rd | 0.2888 — 3rd | 0.3353 — 3rd |
| BlueBERT | 0.0785 — **4th** | 0.1821 — 4th | 0.3038 — 2nd | **0.3679 — 1st** |

**BioSentVec goes 1st → 4th and BlueBERT goes 4th → 1st, on the same data, at the same threshold.**

**And there is a mechanism, not just noise.** The baseline abstains on 24.4% of cases; every BERT arm
predicts on 100%. Widening `K` cannot help the baseline on a case where it declined to predict, but
it hands each BERT arm another free guess — and since one hit inside `K` suffices with no penalty for
the other `K−1`, **TOP-K structurally rewards not abstaining.** The metric is scoring willingness to
guess, and calling it retrieval quality.

This also means the between-encoder spread does **not** move in one direction when leakage is
removed. It shrinks under MAX and grows under TOP-K:

| Aggregator @ threshold 1.0 | Legacy spread | Corrected spread | Direction |
|---|:---:|:---:|---|
| MAX | 0.0381 | 0.0092 | ↓ 4.1× |
| TOP-10 | 0.0462 | 0.0671 | ↑ 1.45× |
| TOP-20 | 0.0506 | 0.0819 | ↑ 1.62× |
| TOP-30 | 0.0744 | 0.1383 | ↑ 1.86× |

Normalising by the leading value does not rescue it (MAX 20.7% → 10.5%; TOP-10 16.2% → 26.9%). Any
statement of the form "the encoders converged" or "the encoders separated" is really a statement
about which aggregator was picked.

### It inverts when you move the threshold, too

Same phenomenon on the other knob. TOP-10, corrected:

| Encoder | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---------|:---:|:---:|:---:|:---:|:---:|
| Bio_ClinicalBERT | 1.0000 | 1.0000 | 1.0000 | 0.7285 | **0.2491** |
| BiomedBERT | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 0.1981 |
| BlueBERT | 1.0000 | 1.0000 | 0.8194 | 0.2923 | 0.1821 |
| BioSentVec | 0.3922 | 0.3077 | 0.2620 | 0.2163 | 0.2163 |

At 0.9 the order is BiomedBERT → Bio_ClinicalBERT → BlueBERT → BioSentVec. At 1.0 it is
Bio_ClinicalBERT → BioSentVec → BiomedBERT → BlueBERT. **BiomedBERT drops 1st → 3rd and the baseline
climbs 4th → 2nd.** BiomedBERT tops the table at 0.9 only because it is still pinned at the ceiling
there — it is the most compact of the three embedding spaces, so it is the last to fall off 1.000,
and grading it with its own cosine rewards exactly that compactness.

### Removing both knobs: the comparison with nothing left to pick

The two sections above show the answer depends on `K` and on the threshold. So both were removed.

- **The threshold, by changing the grader.** DRG code strings are compared directly — no cosine, no
  embedding, nothing the encoder can influence. The five threshold rows collapse to one value. It
  **reproduces threshold-1.0 cosine bit-exactly** across all four arms, which is why the tables above
  survive rather than being replaced ([12](docs/findings/12-drg-grader.md)).
- **`K`, by scoring rank.** Mean Reciprocal Rank has no `K`: burying the right answer at rank 40
  scores 1/40, not the 1.0 that TOP-50 awarded it.

Together that is the first number this project has produced with **no reported knob at all**. Run
2026-08-06, all four arms, on the 76 winnable cases:

| Encoder | MRR@50 | Hit@1 | Hit@50 | P@1 | P@50 | Coverage |
|---------|:------:|:-----:|:------:|:---:|:----:|:--------:|
| Bio_ClinicalBERT | **0.2462** | 0.1749 | 0.6967 | 0.1749 | 0.0290 | 1.0000 |
| BiomedBERT | 0.2432 | 0.1693 | 0.6656 | 0.1693 | 0.0280 | 1.0000 |
| BlueBERT | 0.2314 | 0.1549 | **0.7300** | 0.1549 | 0.0316 | 1.0000 |
| **BioSentVec (baseline)** | 0.2029 | 0.1484 | 0.3577 | 0.1484 | **0.0857** | **0.8477** |

**No pair separates. Largest paired *t* across all six pairs is 1.718**, against the 2.262 needed for
p < 0.05 — and the same holds on both other populations (max 1.651 and 1.651). This was written down
as a prediction before the run, on the reasoning that scoring rank does not improve retrieval; it only
stops rewarding guessing.

**Precision@K inverts the TOP-K curve, as designed.** It *falls* with `K` for every arm — 6.0× from
`K`=1 to 50 for Bio_ClinicalBERT — where the TOP-K table rises for every arm. And it is not monotone
(BioSentVec runs 0.0868 → 0.0845 → 0.0855 → 0.0858 → 0.0857), so unlike Hit@K nothing forces its
direction and the number carries information. Read it within an arm only: truncating your own
candidate list inflates reported P@K 3.53× while delivering strictly less.

**But a third knob was hiding underneath: how you treat abstention.** MRR is not abstention-neutral —
a declined case scores 0, while an arm that always offers 50 candidates has some chance of a hit. The
baseline abstains; all three BERT arms never do. So the convention alone **reorders every arm**:

| Abstention treated as | Ranking by MRR@50 |
|---|---|
| a failure (score 0) — the table above | Bio_ClinicalBERT 0.2462 > BiomedBERT 0.2432 > BlueBERT 0.2314 > **BioSentVec 0.2029** |
| not an observation (excluded) | **BioSentVec 0.1474** > Bio_ClinicalBERT 0.1321 > BiomedBERT 0.1302 > BlueBERT 0.1229 |

Every sign flips; the baseline goes last to first. **Neither convention is neutral and there is no
third that is** — penalising abstention charges "I don't know" the same as a wrong answer, while
excluding it scores the baseline on 98 cases it chose itself against the BERT arms' 129. Abstention is
a property of the arm, not of the metric, so this one cannot be designed away. All three populations
are therefore reported, and the conclusion survives all three
([13](docs/findings/13-rank-aware-metrics.md)).

### Threshold 0.6 — still saturated. This is the metric, not the model.

| Encoder | Precision | Recall | F | Pred. rate | Reading |
|---------|:---------:|:------:|:------:|:----------:|---------|
| BioSentVec (baseline) | 0.4692 | 0.3410 | **0.3922** | 0.7558 | published figure is 0.489 |
| Bio_ClinicalBERT | 1.0000 | 1.0000 | **1.000** | 1.0000 | **saturated — not a result** |
| BiomedBERT | 1.0000 | 1.0000 | **1.000** | 1.0000 | **saturated — not a result** |
| BlueBERT | 1.0000 | 1.0000 | **1.000** | 1.0000 | **saturated — not a result** |

**The three 1.000s are an artifact and must not be read as an achievement.** At threshold 0.6 the
compactness of biomedical embedding space combined with the MAX-over-Cartesian-product aggregator
puts ~100% of patient pairs above the bar, so essentially everything counts as a match. 0.6 is the
paper's threshold, which is why it appears here at all; it cannot discriminate between models.

### The one asymmetry that *is* real: cost

The accuracy difference does not survive a significance test. The cost difference is not close.

| | BioSentVec | one BERT-base arm |
|---|---|---|
| On disk | 20.93 GiB | 416–420 MB |
| Parameters | **~5.6 billion** (≈8M n-gram vectors × 700 dims) | **~110 million** |
| Where the information lives | memorised: one fixed vector per unigram/bigram | computed: contextual, from a small parameter set |
| RAM to run | full 20.93 GiB resident | a few hundred MB |
| Builds on Windows | **no** — MSVC rejects sent2vec's GCC-only flags | yes |

*Parameter count derived from file size and dimensionality assuming float32, not read from the model.*

**Three ~110M-parameter transformers match a ~5.6-billion-parameter n-gram lookup table to within
statistical noise, using roughly 1/51 the parameters and 1/17 the disk.** Note the contrast is *not*
neural versus non-neural — sent2vec is itself a shallow neural embedding model in the word2vec /
fastText lineage. It is **shallow and non-contextual** (one stored vector per n-gram, hence the 21 GB
table) versus **deep and contextual** (representations computed on demand). That efficiency result is
the practical claim this experiment genuinely supports.

**What the baseline uniquely offers, and it is not accuracy: it abstains.** It is the only arm that
declines to predict when nothing clears the pruning gate (24.4% of cases), which gives it the highest
precision of the four and makes it the **only arm producing an interpretable number at the paper's own
threshold of 0.6** — all three BERT arms are pinned at a meaningless 1.000 there. For a clinical
decision support system, "I don't know" is arguably the more useful behaviour, and the compact BERT
embedding spaces have lost the ability to say it.

---

## What the correction changed, and what it did not

**Fixed.**

| Defect | Before | After |
|---|---|---|
| Patient leakage in the folds | 41 of 129 test cases could retrieve the same patient's other admission | **0** — `GroupKFold` on `SUBJECT_ID` |
| Divergent cross-arm preprocessing | 26 of 145 diagnosis descriptions identical between arms | **145 of 145** |
| `w/o` → `w`, destroying negation | `"Tracheostomy w/o Extensive Procedure"` collapsed onto the `w` variant | `w/o` survives as `without` |
| Comma-shredded symptom fragments | 1805 tokens, 89 orphan fragments | 1725 tokens, **9** fragments remain (see TODO P27) |

A side effect worth knowing: the folds are now **uneven** (114/15 through 117/12) because one subject
holds 15 admissions and whole patients must stay together. Per-fold *n* varies, so treat per-fold σ
accordingly.

**Survived, exactly as expected — these are metric design, not data splitting.**

- **Saturation.** BiomedBERT is still 1.000 at every threshold from 0.6 to 0.9 on TOP-10;
  Bio_ClinicalBERT at 0.6–0.8; BlueBERT at 0.6–0.7.
- **Degeneracy.** All three BERT arms still report prediction rate exactly 1.0000, so P == R == F in
  every row and every BERT "F1" is still accuracy.
- **The baseline still abstains**, at `PR` = 0.7558, so it alone has P ≠ R. This confirms degeneracy
  is a consequence of BERT's compact embedding space, not a structural property of the code.

---

## Caveats — what is fixed, and what still is not

> **Fixed: the folds no longer leak patients.** This was the largest single correction available and
> it applied to both arms, which means the published 0.489/0.512/0.521 carry the contamination too.
> Measured cost of removing it: −0.036 to −0.064 at TOP-10 threshold 1.0, and −0.090 on the
> baseline's headline TOP-10 @ 0.6 ([details](docs/findings/05-patient-leakage.md),
> [results](docs/findings/11-corrected-pipeline-first-results.md)).
>
> **Still broken 1 — the metric saturates.** At threshold 0.6 nearly every diagnosis pair counts as a
> match, so F = 1.000 measures the metric, not the model
> ([details](docs/findings/03-metric-saturation.md)).
>
> **Still broken 2 — the metric is degenerate; every BERT "F1" here is accuracy.** Precision, recall,
> and F-score are the *same number* in every BERT row, because every test case increments exactly one
> of TP or FP. This is a property of the embedding space, not the code: the baseline's looser 700D
> space abstains and its precision and recall *do* diverge. Saturation and degeneracy are **one root
> cause at two different gates** ([details](docs/findings/04-metric-degeneracy.md)).
>
> **Fixed: the system no longer grades itself — and it turned out not to have been cheating.** The
> same embedding space used to both retrieve candidates *and* judge whether a prediction was correct,
> so a more compressed space could mark its own work leniently. An encoder-independent grader now
> compares DRG code strings directly, with no cosine anywhere. **It reproduces threshold-1.0 cosine
> bit-exactly** — 144 numbers, 4 arms × 6 aggregators × 6 columns, not one differing digit. So the
> threshold-1.0 column of this README was never self-grading-inflated. The fix did not move the
> numbers; it removed the reason to doubt them, which is the more useful outcome
> ([details](docs/findings/12-drg-grader.md)).
>
> **Fixed: rank is now scored — and it exposed a third knob.** A hit at rank 1 and a hit at rank 50
> counted identically, so the TOP-K curve rose with K by construction. MRR has no K, and under the
> DRG grader there is no threshold either, so it is the first number this project has produced with
> **zero reported knobs**. It confirms the null result — no pair of encoders separates, max paired
> |*t*| = 1.718 against the 2.262 needed — but it also revealed that **how you treat abstention
> reorders all four arms**, exactly as threshold and K did
> ([details](docs/findings/13-rank-aware-metrics.md)).

**A fifth constraint bounds every exact-match number above.** Only **76 of 129 test cases (58.9%)**
have their correct DRG present anywhere in their own fold's training pool — 105 of the 145 unique
diagnoses occur exactly once in the dataset. A *perfect* retriever therefore caps at 58.9% under
exact matching, which is the context in which the threshold-1.0 scores of 0.18–0.25 should be read.
Re-measured on the grouped folds 2026-08-06 and pinned in `tests/test_drg_grader.py`: fixing the
leakage moved retrievability by exactly **one case** (75/129 → 76/129), so the two defects really
were independent. Per fold the ceiling ranges from **4/13 (30.8%)** to **13/15 (86.7%)** on the canonical split (finding 14), which makes it
a fresh source of per-fold variance in its own right.

**Two of the three biases have now been closed, and the third turns out to cut both ways.** An
earlier version of this README said "every remaining defect biases in the same direction," which made
the transformers' apparent lead unfalsifiable. That is no longer the state of things:

| Defect | Which arm it favours | Status |
|---|---|---|
| Self-grading | **BERT** — each arm judged its own predictions with its own cosine, and a compact space marks its own work leniently (Bio_ClinicalBERT's mean pairwise cosine between *unrelated* diagnoses is 0.83) | **CLOSED.** The DRG-string grader uses no cosine at all, and reproduces threshold-1.0 cosine bit-exactly on all four arms. The bias was possible but not actual. |
| Rank discarded / TOP-K rewards guessing | **BERT** — the baseline abstains on 24.4% of cases, so extra `K` cannot help it; every BERT arm always predicts and gains a free guess per unit of `K` | **CLOSED as a metric defect.** MRR has no `K`. What survives is the underlying asymmetry, next row. |
| **Abstention asymmetry** | **either, depending on the convention** — penalise abstention and the BERT arms lead; exclude it and the baseline leads. Every sign inverts. | **OPEN, and not closable.** Abstention is a property of the arm, not of the metric, so no convention is neutral. All three are reported. |
| Saturation | **BERT** — all three BERT arms sit at 1.000 at the paper's own threshold, so their behaviour there is unobservable | **OPEN**, but confined to cosine thresholds below 1.0. Under the DRG grader the threshold rows collapse and it does not arise. |

Bio_ClinicalBERT is nominally ahead of the baseline at **5 of the 6 aggregators**, and its margin grows
monotonically with `K` (+0.0328 at TOP-10 rising to +0.1681 at TOP-50 — while the baseline's score
plateaus at 0.2383 from TOP-40 on, because it is abstaining). **That pattern is exactly what the
abstention asymmetry predicts with zero capability difference** — which is why removing `K` mattered,
and why no paired test reaches significance at any setting.

**The null result now rests on a much stronger footing than it did.** It holds with the threshold knob
gone, the `K` knob gone, the grader independent of the encoder, and across all three abstention
conventions — rather than at one favourable setting. Largest paired |*t*| anywhere is **1.718**.

**The bottom line for anyone quoting this repo.** Two claims are defensible and one is not:

- ✅ **Efficiency.** The transformers match the baseline using ~1/51 the parameters and ~1/17 the disk.
  This is a hardware fact, independent of the metric.
- ✅ **Non-inferiority.** No significant difference between any pair of encoders, at any aggregator,
  any threshold, or any abstention convention — including under a rank-aware metric with no knobs and
  a grader that never touches an embedding. A null result, and the honest headline.
- ❌ **Superiority.** Not supported for *any* encoder. The one nominal ordering that looked stable
  inverts as soon as abstention is treated the other way.

These numbers are leakage-free, preprocessing-unified, encoder-independently graded, and now
rank-aware. That makes them a defensible *reproduction* and a defensible *null result*. What they are
**not** is an encoder ranking, and the reason is no longer a list of unfixed defects — it is that the
effect being measured is smaller than the fold-to-fold noise in a 129-patient dataset. Say that
explicitly rather than letting a reader infer a winner.

Start at [docs/](docs/README.md); the synthesis is
[07-comparison-validity.md](docs/findings/07-comparison-validity.md), the corrected results are
[11-corrected-pipeline-first-results.md](docs/findings/11-corrected-pipeline-first-results.md), and the
two knob-removal results are [12-drg-grader.md](docs/findings/12-drg-grader.md) and
[13-rank-aware-metrics.md](docs/findings/13-rank-aware-metrics.md). The two 2026-08-12 follow-ups are
[15-leakage-preprocessing-attribution.md](docs/findings/15-leakage-preprocessing-attribution.md) —
which part of the corrected-pipeline drop was leakage and which was preprocessing — and
[16-self-selection.md](docs/findings/16-self-selection.md) — whether the baseline's answered cases
are simply its easy ones.

---

## Reproducing

```bash
python scripts/make_folds.py --verify                                # regenerate data/folds_grouped/

# drg pipeline — the knobless comparison; corrected folds + an encoder-independent grader
AICDS_PIPELINE=drg python scripts/run_baseline.py                    # Linux only; 20.93 GiB model
python scripts/run_bert_analysis.py --model all --pipeline drg
python scripts/analyze_rank_metrics.py results_drg                   # paired t-test on per-fold MRR

# corrected pipeline — the cosine-graded tables in this README
AICDS_PIPELINE=corrected python scripts/run_baseline.py
python scripts/run_bert_analysis.py --model all --pipeline corrected

# legacy pipeline — bit-identical to the original, for comparison
python scripts/run_baseline.py
python scripts/run_bert_analysis.py --model all
```

Every run emits `RankMetrics.txt` alongside `PerformanceIndex.txt`; rank metrics are additive and the
byte-exact golden covers `PerformanceIndex.txt` only, which is what keeps the two independent.

`--pipeline` also accepts `folds-only` and `preprocess-only`, which isolate the two halves of the
correction. Neither has been run yet, which is why this README states the combined delta only.

**Comparison PDF.** `scripts/compare_models.py` emits provenance, a summary table, threshold curves,
TOP-K curves, the prediction-rate (degeneracy) page, and the full threshold × TOP-K grid.

Its ~16 sanity assertions are the only thing verifying the parser reads the columns correctly, so they
are **keyed by pipeline rather than deleted** — pointing the script at a directory whose numbers belong
to a different pipeline makes it **refuse and exit 1** rather than quietly relabel a chart. The
pipeline is resolved from three sources in order — an explicit `--pipeline`, then the run's own
`run_metadata.json`, then the directory name as a demoted fallback for pre-metadata trees. A run with
no recorded pipeline prints `[WARN] no run_metadata.json … (pre-C8 runs)` and continues on the dirname;
a flag/metadata disagreement warns and uses the flag. The **hard** exits are a directory name that
resolves to nothing, a pipeline with no recorded expectations, and a run in which zero checks executed.
The pipeline is then named on the cover page and in the PDF metadata, so a detached chart cannot be
misattributed. Every invocation against the three trees committed before `5a52d26` prints that one
`[WARN]` line; it is interim noise by design, and it clears once the attribution runs produce the first
metadata-bearing trees.

```bash
python scripts/compare_models.py --results-dir results_drg
```

---

## Baseline Reproduction

The original paper uses **BioSentVec** (700-dimensional sent2vec embeddings trained on PubMed +
MIMIC-III) to compute symptom-level pairwise cosine similarities between patients. Diagnosis
similarity is the MAX similarity across the Cartesian product of ground-truth and predicted
diagnosis descriptions, thresholded to classify true/false positives.

The baseline arm had never executed in this checkout: it crashed on a wrong data path and an unbound
name, and `sent2vec` cannot be built under MSVC, making the arm **Linux-only**. Both were fixed and
it ran end to end on a rented Linux box on 2026-08-05 (10 folds, ~13 min).

**Threshold 0.6, against the published figures:**

| Method | Published | Legacy repro | Corrected |
|--------|:---------:|:------------:|:---------:|
| TOP-10 | 0.489 | **0.4824** | 0.3922 |
| TOP-20 | 0.512 | 0.4824 | 0.4163 |
| TOP-30 | 0.521 | 0.4920 | 0.4316 |

**Under `legacy`, TOP-10 lands within 0.007 of the published figure** — the first successful
reproduction of the paper's headline number by this codebase, and the artifact that settled the
degeneracy question ([09-baseline-first-run.md](docs/findings/09-baseline-first-run.md)). Note P ≠ R
here, unlike every BERT row: the baseline declines to predict on 23.2% of cases under `legacy` and
24.4% under `corrected`.

**Under `corrected`, TOP-10 falls to 0.3922 — 18.4% of the published number was contamination.** That drop is the finding, not a regression: the legacy path still reproduces 0.4824
bit-for-bit on demand.

Threshold 0.6 is the paper's own operating point, so this table is the correct comparison against
the paper. It is not where the four encoders can be compared to each other — for that, see the
[threshold-1.0 table](#threshold-10-top-10--the-informative-setting) above.

## BERT Extension (Original Contribution)

We replace BioSentVec with three biomedical BERT models that produce 768-dimensional embeddings.
Everything else — fold splits, preprocessing, pruning, aggregation, scoring, thresholds — is shared,
so the encoder is the only intended moving part:

| Model | HuggingFace Path | Training Data | Size |
|-------|-------------------|---------------|:----:|
| Bio_ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | MIMIC-III clinical notes | 416 MB |
| BiomedBERT | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | PubMed abstracts | 420 MB |
| BlueBERT | `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` | PubMed + MIMIC-III | 420 MB |

**Corrected pipeline at threshold 0.6 — still SATURATED across all three BERT arms. This table
measures the metric, not the models; it is included only because 0.6 is the paper's threshold.**

| Method | BioSentVec (corrected) | Bio_ClinicalBERT | BiomedBERT | BlueBERT |
|--------|:----------------------:|:-----------------:|:----------:|:--------:|
| TOP-10 @ 0.6 | 0.3922 | 1.000 *(saturated)* | 1.000 *(saturated)* | 1.000 *(saturated)* |
| TOP-20 @ 0.6 | 0.4163 | 1.000 *(saturated)* | 1.000 *(saturated)* | 1.000 *(saturated)* |
| TOP-30 @ 0.6 | 0.4316 | 1.000 *(saturated)* | 1.000 *(saturated)* | 1.000 *(saturated)* |

Removing patient leakage moved none of the three BERT arms here, because they were already pinned at
the ceiling. **That is the cleanest demonstration available that saturation and leakage are
independent defects:** fixing the folds cannot rescue a metric that has no headroom.

The informative comparison is the
[threshold-1.0 table](#threshold-10-top-10--the-informative-setting) at the top of this README.

### Runtime — the encoder is not the bottleneck

On the corrected four-arm pod run: **~11.5 minutes** for each of Bio_ClinicalBERT / BiomedBERT /
BlueBERT, and **~13 minutes** for the BioSentVec baseline. Measured across the committed runs,
**embedding is 0.17–0.45% of wall-clock**, while the single-threaded pure-Python cosine loop is over
93%. Total fold time varies only **1.7%** across the three transformers. Making embedding
instantaneous would cut a 21.98-minute run to 21.88 minutes, so **a GPU buys essentially nothing for
these arms** ([details](docs/findings/08-runtime-and-cost.md)).

Results are also **platform-independent**: re-running the BERT arms on x86 Linux reproduced the
Apple-silicon numbers **bit-for-bit, to all 17 significant figures.**

## Visual Summary (README Charts)

> **These charts are from the LEGACY pipeline** — three BERT arms only, no baseline, generated from
> the committed February 2026 runs under `docs/`. They have not been regenerated under `corrected`.
> `scripts/build_readme_plots.py` used to glob the wrong directory and crash; since `7927a88` it goes
> through `aicds.runs.discover` and **runs**, regenerating all six committed SVGs byte-identical
> ([10-output-path-fragmentation.md](docs/findings/10-output-path-fragmentation.md)). So what is stale
> here is the *pipeline the numbers came from*, not the tooling. Read them for the *shape* of the
> saturation and TOP-K artifacts, which `corrected` did not change; do not read the values as current.

![F1 vs threshold (TOP-10)](docs/readme_plots/f1_vs_threshold_top10.svg)
*At TOP-10, BiomedBERT stays saturated through 0.9 while BlueBERT drops earlier. The 1.000 region on
the left of this chart is metric saturation, not accuracy.*

![F1 vs threshold (TOP-50)](docs/readme_plots/f1_vs_threshold_top50.svg)
*At TOP-50, all models improve at strict thresholds, but separation remains at 0.9/1.0.*

![F1 vs TOP-K at threshold 0.9](docs/readme_plots/f1_vs_topk_t09.svg)
*Top-K expansion strongly helps Bio_ClinicalBERT and BlueBERT at threshold 0.9 — but one hit inside K
suffices and the other K−1 predictions are unpenalised, so the upward slope is a metric artifact.*

![F1 vs TOP-K at threshold 1.0](docs/readme_plots/f1_vs_topk_t10.svg)
*At exact-match threshold 1.0, model differences are modest and increase gradually with K.*

![Runtime breakdown](docs/readme_plots/runtime_breakdown.svg)
*10-fold evaluation dominates runtime; startup overhead differs mostly by model loading time.*

![Saturation by threshold](docs/readme_plots/saturation_by_threshold.svg)
*Per-patient MAX similarity saturation explains the perfect F1 at threshold 0.6.*

## Score Distribution Analysis (Key Finding)

The perfect F1 scores are an artifact of **embedding space compactness** combined with the
**MAX-over-Cartesian-product** evaluation strategy, not genuine diagnostic accuracy.

> **Measured under legacy preprocessing.** The statistics below have not been recomputed since
> diagnosis-text handling was unified, so the exact figures apply to the legacy text. The mechanism
> is unaffected — saturation persists identically under `corrected`, as the tables above show.

**Why the metric saturates:**

1. **Compact embedding spaces** — Biomedical BERT models map diagnosis text into a narrow region.
   Even *unrelated* diagnoses have high cosine similarity:

   | Model | Mean Pairwise Sim | Min Pairwise Sim | Std |
   |-------|:-----------------:|:----------------:|:---:|
   | BiomedBERT | 0.93 | 0.72 | 0.03 |
   | Bio_ClinicalBERT | 0.83 | 0.65 | 0.05 |
   | BlueBERT | 0.72 | 0.48 | 0.07 |

2. **MAX operator amplification** — Taking the maximum similarity across all diagnosis pairs inflates
   scores further. Per-patient MAX similarity exceeds 0.6 for virtually all patient pairs:

   | Model | % of patient pairs with MAX >= 0.6 |
   |-------|:----------------------------------:|
   | Bio_ClinicalBERT | 100.00% |
   | BiomedBERT | 100.00% |
   | BlueBERT | 99.96% |

3. **Conclusion** — The evaluation metric is saturated at threshold 0.6 for BERT models. The F1
   scores cannot discriminate between models or meaningfully compare against the baseline.
   Alternative evaluation strategies (MEAN instead of MAX, DRG code matching, higher thresholds) are
   needed. Note that the ordering inversions documented above show a stricter threshold alone is
   *not* sufficient — it removes the ceiling without making the ranking stable.

Visualizations and full statistics are in
[`docs/score_distribution_analysis/`](docs/score_distribution_analysis/).

![Diagnosis score distributions](docs/score_distribution_analysis/score_distributions.png)
*Diagnosis score distributions across baseline and BERT models.*

![Per-patient maximum similarity distributions](docs/score_distribution_analysis/per_patient_max_distributions.png)
*Per-patient MAX similarity distributions showing saturation behavior.*

```bash
python scripts/analyze_score_distributions.py
```

## Project Structure

```
src/aicds/               # Installable package (src layout; pip install -e .)
  config.py              # PipelineConfig: LEGACY / CORRECTED / GRADER_DRG / FOLDS_ONLY / PREPROCESS_ONLY
  models/                # Baseline (sent2vec) and BERT implementations
  analysis/              # rank_metrics (pure, no I/O), populations, rank_report writer
  utils/                 # Constants, runtime helpers, cython_utils (pure Python, shared math)
  entity/                # Data classes
  runs.py                # The one run-directory contract: run_dirs() writes, discover() reads
scripts/                 # Entry points
  make_folds.py          # GroupKFold on SUBJECT_ID -> data/folds_grouped/
  run_baseline.py        # BioSentVec baseline (Linux only)
  run_bert_analysis.py   # --model 1|2|3|all  --pipeline legacy|corrected|drg|...
  compare_models.py      # Four-arm comparison PDF (sanity checks keyed by pipeline)
  analyze_rank_metrics.py  # RankMetrics.txt -> paired t-test on per-fold MRR
  analyze_score_distributions.py
data/
  folds/                 # Committed 10-fold splits (split on HADM_ID; leaky, pinned for legacy)
  folds_grouped/         # GroupKFold on SUBJECT_ID (generated, gitignored)
  raw/                   # Raw data files
docs/                    # findings/ guides/ reference/ plans/
results*/                # ALL run output, gitignored by glob (legacy, corrected, drg, ...)
tests/                   # Includes the byte-exact golden regression net
```

The central design constraint: **both arms share everything except the embedding model.**
Preprocessing, fold loading, diagnosis scoring, and all confusion-matrix math live in
`src/aicds/utils/cython_utils.py` so the two arms stay comparable — that comparability is the point.
See [docs/reference/architecture.md](docs/reference/architecture.md).

## Setup

```bash
conda env create -f config/environment.yml   # env "disease-diagnosis", Python 3.9
conda activate disease-diagnosis
pip install -e .
git config core.hooksPath .githooks          # data-use guard; hooks are not cloned
```

**Key dependencies:** sentence-transformers, torch, matplotlib, numpy, scikit-learn

**For the baseline only:** also requires `sent2vec` and the BioSentVec pre-trained model
(20.93 GiB). The baseline arm is **Linux-only** — `sent2vec` cannot be built under MSVC, which
rejects its GCC-only compiler flags. See [docs/guides/setup.md](docs/guides/setup.md) for the full
procedure, including the macOS/ARM OpenMP conflict and the Linux torch/nltk `LD_LIBRARY_PATH` trap.

**Testing:** `pytest` runs the fast suite in seconds. `pytest -m golden` re-runs the full 10-fold
pipeline and compares it **byte-for-byte** against a committed reference (**34–53 min**, measured eight
times: 34:12, 42:28, 43:28, 43:50, 44:32, ~50:00, 52:35, 53:10 — 34:12 on 2026-08-12, Windows, numpy-2.0.2
venv; budget for the top of that range, not the bottom). Nothing here is
trained, so every emitted number is a pure function of the input data and the arithmetic in
`cython_utils.py` — meaning any behaviour change is a numerical change, and the realistic failure
mode of refactoring is the numbers moving while every other test stays green. The golden is the only
thing that catches that. If it fails, read the diff; a changed number is the finding.

## Data handling

This repository contains committed MIMIC-III records under a PhysioNet DUA that prohibits
redistribution. **Do not add clinical data to this repository.** `.githooks/pre-commit` blocks new
files under `data/raw`/`data/folds` and any new file containing 20+ distinct `HADM_ID`s. Every run
directory is gitignored for the same reason — the pattern is the glob `results*/`, not a hand-listed
set, because a new pipeline adding `results_drg/` to a hand-listed `.gitignore` would silently have
been committable. Only aggregates appear in documentation. See
[docs/guides/data-use.md](docs/guides/data-use.md).

## Citation

```bibtex
@article{comito2022ai,
  title={AI-Driven Clinical Decision Support: Enhancing Disease Diagnosis Exploiting Patients Similarity},
  author={Comito, Carmela and Falcone, Deborah and Forestiero, Agostino},
  journal={IEEE Access},
  volume={10},
  pages={6224--6234},
  year={2022},
  publisher={IEEE}
}
```

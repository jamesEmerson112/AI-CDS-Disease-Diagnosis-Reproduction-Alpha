# Next Steps: Meaningful BERT Evaluation

## Problem Summary

The current evaluation metric — **MAX cosine similarity** over the Cartesian product of ground-truth and predicted diagnosis descriptions — saturates for all three BERT models at the 0.6 threshold used in the original paper. All models achieve **F1 = 1.000**, making model comparison meaningless.

**Root cause:** Two factors compound each other:

1. **Embedding space compactness.** Biomedical BERT models map diagnosis text into a narrow region of the embedding space. Even *unrelated* diagnoses produce high cosine similarity:

   | Model | All-Pairwise Mean | All-Pairwise Min | % >= 0.6 |
   |---|---|---|---|
   | Bio_ClinicalBERT | 0.8348 | 0.6454 | 100.00% |
   | BiomedBERT | 0.9282 | 0.7246 | 100.00% |
   | BlueBERT | 0.7170 | 0.4810 | 96.35% |

2. **MAX operator amplification.** With ~1.74 diagnoses per patient, the Cartesian product contains ~3 pairs. Taking the MAX over these pairs inflates the per-patient similarity further:

   | Model | Per-Patient MAX Mean | % >= 0.6 |
   |---|---|---|
   | Bio_ClinicalBERT | 0.8586 | 100.00% |
   | BiomedBERT | 0.9447 | 100.00% |
   | BlueBERT | 0.7565 | 99.96% |

At threshold 0.6, virtually every patient pair is classified as a true positive — the metric cannot discriminate between models.

*Data source: `docs/score_distribution_analysis/score_distribution_summary.txt`*

---

## Alternative Evaluation Strategies

Four concrete strategies are listed below in priority order (highest impact-to-complexity ratio first).

---

### Strategy A: MEAN Aggregation (instead of MAX)

**What:** Replace the MAX operator with MEAN over the Cartesian product of (ground-truth, predicted) diagnosis pairs.

**Why it helps:** MEAN directly addresses the MAX operator amplification problem. Instead of a single best-matching pair dominating the score, every pair contributes. Unmatched diagnoses pull the score down. The existing per-patient MAX distributions show meaningful spread at higher thresholds (e.g., Bio_ClinicalBERT: 93.81% >= 0.8, 11.63% >= 0.9), so a MEAN-based score will produce a wider distribution with better discriminative power.

**What to change:**

- `src/utils/cython_utils.py` — function `get_diagnosis_similarity_by_description_max()` (line 291). Currently tracks a running `max_similarity` over the double loop. Replace with accumulation and division by pair count:
  ```python
  def get_diagnosis_similarity_by_description_mean(embendings_diagnosis, gt_diagnosis, predicted_diagnosis, method):
      total_similarity = 0.0
      count = 0
      for x in gt_diagnosis:
          x_description = x[x.index(':') + 1:]
          for y in predicted_diagnosis:
              y_description = y[y.index(':') + 1:]
              emb_gt = embendings_diagnosis.get(x_description)
              emb_pred = embendings_diagnosis.get(y_description)
              total_similarity += cosine_similarity(emb_gt[0], emb_pred[0])
              count += 1
      return total_similarity / count if count > 0 else 0.0
  ```
- `src/models/bert_models.py` — call site at line 542. Add a parallel code path calling the MEAN variant alongside MAX.
- `scripts/analyze_score_distributions.py` — add a MEAN distribution analysis to compare against MAX.

**Complexity:** Low. Minimal code change, no new dependencies.

---

### Strategy B: DRG Code Exact Match

**What:** Use the structured DRG codes already present in the data for binary exact-match evaluation: predicted DRG code == ground-truth DRG code.

**Why it helps:** Completely sidesteps embedding similarity and threshold sensitivity. DRG codes are categorical identifiers — a match is unambiguous. This provides a hard, clinically meaningful metric that cannot saturate.

**Data format:** Each diagnosis in `data/raw/Symptoms-Diagnosis.txt` already contains DRG codes in the format `{TYPE}:{DESCRIPTION}`, where TYPE is one of:
- `HCFA` — e.g., `HCFA:SEPTICEMIA AGE >17`
- `APR` — e.g., `APR:Intracranial Hemorrhage`
- `MS` — (Medicare Severity)

Some patients have multiple DRG codes (e.g., `APR:...--HCFA:...` on a single line, delimited by `--`).

**What to change:**

- Extract DRG type+description from the existing diagnosis strings (split on first `:`).
- Define a match function: exact string equality of the DRG code (or normalized variant).
- Add a new evaluation pathway in `src/models/bert_models.py` that computes precision/recall/F1 based on DRG code match instead of cosine similarity.
- The existing `get_diagnosis_similarity_baseline()` function in `cython_utils.py` (line 329) already does exact string matching on full diagnosis strings — adapt this pattern for DRG codes specifically.

**Complexity:** Low–Medium. Parsing logic is straightforward, but needs careful handling of multi-DRG patients and the `APR:...--HCFA:...` format.

---

### Strategy C: Hungarian Optimal Matching

**What:** Instead of MAX (best single pair), find the optimal 1-to-1 assignment between ground-truth and predicted diagnosis sets using the Hungarian algorithm. The score is the mean similarity of matched pairs.

**Why it helps:** More principled than MAX — it penalizes unmatched diagnoses and prevents a single strong match from masking poor overall alignment. Given ~1.74 diagnoses per patient, this evaluates how well the *entire* diagnosis set is reproduced, not just whether one diagnosis happens to be close.

**What to change:**

- Add `scipy.optimize.linear_sum_assignment` (scipy is likely already a dependency).
- Create a new function in `src/utils/cython_utils.py`:
  ```python
  from scipy.optimize import linear_sum_assignment

  def get_diagnosis_similarity_hungarian(embendings_diagnosis, gt_diagnosis, predicted_diagnosis, method):
      n_gt = len(gt_diagnosis)
      n_pred = len(predicted_diagnosis)
      cost_matrix = np.zeros((n_gt, n_pred))
      for i, x in enumerate(gt_diagnosis):
          x_desc = x[x.index(':') + 1:]
          for j, y in enumerate(predicted_diagnosis):
              y_desc = y[y.index(':') + 1:]
              emb_gt = embendings_diagnosis.get(x_desc)
              emb_pred = embendings_diagnosis.get(y_desc)
              cost_matrix[i, j] = cosine_similarity(emb_gt[0], emb_pred[0])
      # Hungarian finds minimum cost — negate for maximum similarity
      row_ind, col_ind = linear_sum_assignment(-cost_matrix)
      return cost_matrix[row_ind, col_ind].mean()
  ```
- Wire into `bert_models.py` evaluation loop as an additional strategy.

**Complexity:** Medium. Requires building a cost matrix per patient pair and handling unequal set sizes (pad the smaller set).

---

### Strategy D: Set-Level Jaccard / F1

**What:** Treat the ground-truth and predicted diagnosis sets as multi-label sets. Compute set-level Jaccard similarity or set-level F1 (micro/macro).

**Why it helps:** Standard multi-label classification metric that captures both precision (no spurious diagnoses) and recall (no missing diagnoses) at the set level. Avoids collapsing the evaluation to a single similarity score.

**What to change:**

- Define a "match" criterion for two diagnoses: either exact DRG code match (Strategy B) or cosine similarity above a threshold.
- For each patient pair, compute:
  - **Jaccard** = |GT ∩ Pred| / |GT ∪ Pred|
  - **Set F1** = 2 * |GT ∩ Pred| / (|GT| + |Pred|)
- Add to evaluation in `bert_models.py` as a new metric alongside existing threshold-based TP/FP counting.

**Complexity:** Medium. The metric itself is simple, but the "match" definition requires a design decision (exact vs. threshold-based), and integration with the existing fold-level evaluation loop in `bert_models.py` requires care.

---

## References

- **MedTric:** Agrawal et al., "MedTric—A clinically applicable metric for evaluation of multi-label computational diagnostic systems," *PLOS ONE*, 2023. Proposes clinically-calibrated metrics for medical diagnosis evaluation that account for diagnostic hierarchy and clinical severity.
- **Score Distribution Analysis:** `docs/score_distribution_analysis/score_distribution_summary.txt` — full statistical breakdown of all-pairwise and per-patient MAX similarity distributions.
- **Current Implementation:** `src/utils/cython_utils.py:291` — `get_diagnosis_similarity_by_description_max()`, the function that computes the MAX similarity score used in evaluation.
- **Evaluation Loop:** `src/models/bert_models.py:542` — call site where MAX similarity drives TP/FP classification.

---

## Recommended Priority

| Priority | Strategy | Impact | Complexity | Rationale |
|---|---|---|---|---|
| 1 | A: MEAN Aggregation | High | Low | Smallest code change, directly fixes the MAX amplification problem |
| 2 | B: DRG Code Match | High | Low–Med | Avoids embedding similarity entirely, uses existing structured data |
| 3 | C: Hungarian Matching | Medium | Medium | Most principled set-matching approach, but more complex |
| 4 | D: Set-Level Jaccard/F1 | Medium | Medium | Standard metric, but depends on a match definition from A or B |

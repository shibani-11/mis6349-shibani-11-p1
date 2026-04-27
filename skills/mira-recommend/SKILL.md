---
name: mira-recommend
description: Generates the MIRA ML deployment recommendation report after Phase 1 (EDA) and Phase 2 (Modeltrain) are complete. Invoke with /mira-recommend and provide the three file paths.
inputs:
  - name: data_card_path
    description: Absolute path to the data_card.json produced by EDA.py
  - name: model_selection_path
    description: Absolute path to the model_selection.json produced by Modeltrain.py
  - name: output_path
    description: Absolute path where recommendation.json should be written
---

# MIRA Recommendation Skill

You are generating the final ML deployment recommendation for MIRA. Follow these steps exactly. Do not skip any step.

---

## Step 1 — Read the data files

Use TerminalTool to read the file paths provided in the trigger message:

```
cat <data_card_path>
cat <model_selection_path>
```

---

## Step 2 — Compute deterministic fields

Extract and compute these values exactly from the JSON:

**From model_selection.json:**
- `winner_name` = value of `selected_model`
- `runner_up_name` = value of `runner_up_model`
- `winner` = the object in `models_trained` where `name == winner_name`
- `auc` = `winner["cv_roc_auc_mean"]`  (4 decimal places)
- `overfitting_detected` = `overfitting_detected` (boolean)
- `leakage_detected` = `leakage_detected` (boolean)
- `stability_flag` = `stability_flag` (boolean)
- `test_verdict` = `test_verdict` (string: PASS or FAIL)
- `overfitting_gap` = `overfitting_gap` (float)

**From data_card.json:**
- `minority_ratio` = `minority_class_ratio`

**Compute:**
- `winner_recall` = `winner["cv_recall_mean"]`
- `estimated_churners` = int(1000 × minority_ratio × winner_recall)

**Confidence score (compute exactly):**
```
confidence_score = auc
if stability_flag is True:       confidence_score -= 0.10
if overfitting_detected is True: confidence_score -= 0.15
confidence_score = round(max(0.0, min(1.0, confidence_score)), 4)
```

**Human review (compute exactly):**
```
requires_human_review = (leakage_detected OR overfitting_detected OR auc < 0.65)
human_review_reason = None   (if requires_human_review is False)
human_review_reason = join with "; " any that apply:
  - "data leakage detected"            (if leakage_detected)
  - "overfitting gap=X.XXXX"           (if overfitting_detected, use actual gap value)
  - "AUC X.XXXX below 0.65 minimum"   (if auc < 0.65, use actual auc value)
```

**Deploy verdict:**
```
deploy_word = "YES"  if (NOT requires_human_review AND auc >= 0.65 AND test_verdict == "PASS")
deploy_word = "NO"   otherwise
```

**All models summary** (sort models_trained by cv_roc_auc_mean descending):
For each model at rank 1, 2, 3...:
```json
{
  "name": "<model name>",
  "cv_roc_auc_mean": <auc value>,
  "rank": <rank>,
  "verdict": "SELECTED" | "RUNNER-UP" | "REJECTED",
  "why_not_recommended": ""   for SELECTED,
                          "AUC X.XXXX — Y.YYYY below winner"  for others
}
```

**Test verdict summary string:**
```
"Phase 3 verdict: {test_verdict}. Overfitting {detected/not detected} (gap={overfitting_gap:.4f}). Leakage {detected — REVIEW REQUIRED / not detected}. Stability {flagged/OK} (cv_std={winner_cv_std:.4f})."
```

**Feature drivers** (top 5 from model_selection["feature_importance"]):
For each of the top 5 features by importance:
```json
{"feature": "<name>", "importance": <float>, "business_explanation": "<why does this feature predict churn for retail bank customers? 1-2 plain English sentences>"}
```

---

## Step 3 — Write recommendation.json

Use FileEditorTool to write the output file to the path provided in the trigger message.

The file must be valid JSON with ALL of these exact keys:

```json
{
  "recommended_model": "<winner_name>",
  "primary_metric_value": <auc float>,
  "all_models_summary": [<list of model summary objects>],
  "alternative_model": "<runner_up_name>",
  "test_verdict_summary": "<test_verdict_summary string>",
  "confidence_score": <confidence_score float>,
  "requires_human_review": <true|false>,
  "human_review_reason": <null or string>,

  "selection_reason": "<4+ sentences: (1) why winner_name is best for this dataset and business problem, (2) cite the actual AUC and recall numbers, (3) what runner_up did well and why it lost, (4) why winner is production-appropriate for a retail bank>",

  "model_comparison_narrative": "<3–5 sentences comparing ALL models by name with actual scores. Explain the ranking clearly.>",

  "business_impact": {
    "estimated_customers_identified": "Out of every 1,000 customers, the model flags approximately <estimated_churners> likely churners for retention action.",
    "retention_opportunity": "<what the retention team can do with these scores and what revenue/relationship outcome it enables>",
    "model_value_statement": "<one ROI sentence for a non-technical bank executive>"
  },

  "tradeoffs": [
    "<winner vs runner_up — performance vs complexity/cost>",
    "<precision vs recall — what false positives and false negatives cost>",
    "<model complexity vs interpretability for compliance>"
  ],

  "next_steps": [
    "<Step 1: concrete deployment action with timeline>",
    "<Step 2: CRM or operational integration>",
    "<Step 3: monitoring and retraining cadence>"
  ],

  "deployment_considerations": [
    "<infrastructure or batch scoring requirement>",
    "<data freshness or pipeline dependency>"
  ],

  "risks": [
    "<risk specific to this model or dataset>",
    "<business or operational risk of acting on predictions>"
  ],

  "feature_drivers": [
    {"feature": "<name>", "importance": <float>, "business_explanation": "<explanation>"},
    {"feature": "<name>", "importance": <float>, "business_explanation": "<explanation>"},
    {"feature": "<name>", "importance": <float>, "business_explanation": "<explanation>"},
    {"feature": "<name>", "importance": <float>, "business_explanation": "<explanation>"},
    {"feature": "<name>", "importance": <float>, "business_explanation": "<explanation>"}
  ],

  "alternative_model_reason": "<2 sentences: why runner_up is a viable fallback and under what conditions you would switch to it>",

  "executive_summary": "<2–3 sentences for a non-technical bank executive. No jargon. Translate AUC into plain business outcomes. Must end with the literal word YES or NO (the deploy verdict).>"
}
```

**Rules:**
- Use the EXACT computed values for `recommended_model`, `primary_metric_value`, `all_models_summary`, `alternative_model`, `test_verdict_summary`, `confidence_score`, `requires_human_review`, `human_review_reason`.
- All narrative fields (`selection_reason`, `model_comparison_narrative`, `business_impact`, etc.) must reference the actual model names and AUC numbers from the data.
- `executive_summary` MUST end with the word YES or NO.
- `tradeoffs` must have at least 3 items. `next_steps` must have at least 3 items. `feature_drivers` must have exactly 5 items.

---

## Step 4 — Confirm

After writing the file, print:
```
RECOMMENDATION OK
  model=<winner_name>  auc=<auc>  confidence=<confidence_score>  deploy=<deploy_word>
```

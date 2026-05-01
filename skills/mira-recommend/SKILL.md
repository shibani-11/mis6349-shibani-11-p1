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

Extract and compute these values exactly from the JSON.

**From model_selection.json:**
- `winner_name` = value of `selected_model`
- `runner_up_name` = value of `runner_up_model`
- `winner` = the object in `models_trained` where `name == winner_name`
- `runner_up` = the object in `models_trained` where `name == runner_up_name`
- `auc` = `winner["cv_roc_auc_mean"]`  (4 decimal places)
- `runner_up_auc` = `runner_up["cv_roc_auc_mean"]`
- `auc_gap` = `auc - runner_up_auc`  (4 decimal places)
- `winner_cv_std` = `winner["cv_roc_auc_std"]`
- `overfitting_detected` = `overfitting_detected` (boolean)
- `leakage_detected` = `leakage_detected` (boolean)
- `stability_flag` = `stability_flag` (boolean)
- `test_verdict` = `test_verdict` (string: PASS or FAIL)
- `overfitting_gap` = `overfitting_gap` (float)

**From data_card.json:**
- `minority_ratio` = `minority_class_ratio`
- `rows` = `rows`
- `priority_metric` = `priority_metric`

**Compute:**
- `winner_recall` = `winner["cv_recall_mean"]`
- `estimated_positives` = int(1000 × minority_ratio × winner_recall)

---

## Step 2a — Compute flags[] (list all active warning flags)

Build a list of named flags for every condition that is True. Include only the ones that apply:

```
flags = []

if leakage_detected:
    flags.append("DATA_LEAKAGE_DETECTED")

if overfitting_detected:
    flags.append("OVERFITTING_DETECTED")

if stability_flag:
    flags.append("STABILITY_CONCERN")          # cv_std > 0.05

if auc_gap < 0.02:
    flags.append("AMBIGUOUS_MODEL_SELECTION")  # top-2 within 0.02

if rows < 1000:
    flags.append("INSUFFICIENT_TRAINING_DATA")

if auc < 0.65:
    flags.append("BELOW_PERFORMANCE_FLOOR")

if test_verdict == "FAIL":
    flags.append("TEST_VERDICT_FAIL")
```

`flags` is an empty list `[]` when no conditions apply.

---

## Step 2b — Compute confidence_score (reasoning confidence, NOT output quality)

This score reflects how certain the reasoning chain is — not how well-formatted the output is.
A high score means the recommendation is unambiguous and well-supported by the data.
A low score means there is genuine uncertainty the human reviewer must evaluate.

```
# Start with model performance as the base
base = auc

# Reward a decisive winner (gap adds up to 0.10 max)
gap_bonus = min(0.10, auc_gap * 2)

# Penalise each risk condition
if stability_flag:          base -= 0.10   # cross-fold variance is high
if overfitting_detected:    base -= 0.15   # model doesn't generalise
if leakage_detected:        base -= 0.30   # data integrity compromised
if test_verdict == "FAIL":  base -= 0.20   # failed stress test
if auc_gap < 0.02:          base -= 0.10   # selection is arbitrary
if rows < 1000:             base -= 0.10   # training N too small

confidence_score = round(max(0.0, min(1.0, base + gap_bonus)), 4)
```

Do NOT inflate confidence_score because the output is well-formatted.
A score above 0.85 means the recommendation is clear and the data supports it.
A score below 0.70 means there is material uncertainty the human reviewer must evaluate.

---

## Step 2c — Compute requires_human_review and human_review_reason

```
requires_human_review = (
    leakage_detected
    OR overfitting_detected
    OR auc < 0.65
    OR confidence_score < 0.70
    OR "AMBIGUOUS_MODEL_SELECTION" in flags
    OR "INSUFFICIENT_TRAINING_DATA" in flags
)

# Build reason string from active conditions
reasons = []
if leakage_detected:                            reasons.append("data leakage detected")
if overfitting_detected:                        reasons.append(f"overfitting gap={overfitting_gap:.4f}")
if auc < 0.65:                                  reasons.append(f"AUC {auc:.4f} below 0.65 minimum")
if confidence_score < 0.70:                     reasons.append(f"reasoning confidence {confidence_score:.4f} below 0.70")
if "AMBIGUOUS_MODEL_SELECTION" in flags:        reasons.append(f"top-2 AUC gap={auc_gap:.4f} — selection ambiguous")
if "INSUFFICIENT_TRAINING_DATA" in flags:       reasons.append(f"only {rows} rows after cleaning")

human_review_reason = "; ".join(reasons) if reasons else None
```

---

## Step 2d — Compute routing_zone

```
# Hard escalation rules override confidence threshold
hard_escalation = (
    leakage_detected
    OR (overfitting_detected AND overfitting_gap > 0.20)
    OR auc < 0.65
    OR test_verdict == "FAIL"
    OR rows < 1000
)

if hard_escalation:
    routing_zone = "zone_3"   # Priority Human Review
elif confidence_score >= 0.85 AND len(flags) == 0:
    routing_zone = "zone_1"   # Auto-Approve Eligible
else:
    routing_zone = "zone_2"   # Standard Human Review
```

---

## Step 2e — Compute deploy_word

```
deploy_word = "YES"  if (NOT requires_human_review AND auc >= 0.65 AND test_verdict == "PASS")
deploy_word = "NO"   otherwise
```

---

## Step 2f — Compute all_models_summary

Sort models_trained by cv_roc_auc_mean descending. For each at rank 1, 2, 3...:

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

---

## Step 2g — Compute test_verdict_summary string

```
"Phase 3 verdict: {test_verdict}. Overfitting {detected/not detected} (gap={overfitting_gap:.4f}). Leakage {detected — REVIEW REQUIRED / not detected}. Stability {flagged/OK} (cv_std={winner_cv_std:.4f})."
```

---

## Step 2h — Compute feature_drivers (top 5 from model_selection["feature_importance"])

For each of the top 5 features by importance value:

```json
{"feature": "<name>", "importance": <float>, "business_explanation": "<1-2 plain English sentences explaining why this feature predicts the target>"}
```

---

## Step 3 — Write recommendation.json

Use FileEditorTool to write the output file to the path provided.

The file must be valid JSON with ALL of these exact keys:

```json
{
  "recommended_model": "<winner_name>",
  "primary_metric_value": <auc float>,
  "all_models_summary": [<list of model summary objects>],
  "alternative_model": "<runner_up_name>",
  "test_verdict_summary": "<test_verdict_summary string>",
  "confidence_score": <confidence_score float>,
  "routing_zone": "<zone_1 | zone_2 | zone_3>",
  "flags": [<list of flag strings, empty list if none>],
  "requires_human_review": <true|false>,
  "human_review_reason": <null or string>,

  "selection_reason": "<4+ sentences: (1) why winner_name is best for this dataset and business problem, (2) cite the actual AUC and recall numbers, (3) what runner_up did well and why it lost, (4) why winner is appropriate for the stated business objective>",

  "model_comparison_narrative": "<3-5 sentences comparing ALL models by name with actual scores. Explain the ranking clearly.>",

  "business_impact": {
    "estimated_positives_identified": "Out of every 1,000 records, the model flags approximately <estimated_positives> likely positive cases for action.",
    "operational_opportunity": "<what the team can do with these scores and what outcome it enables>",
    "model_value_statement": "<one ROI sentence for a non-technical executive>"
  },

  "tradeoffs": [
    "<winner vs runner_up — performance vs complexity/cost>",
    "<precision vs recall — what false positives and false negatives cost>",
    "<model complexity vs interpretability for compliance>"
  ],

  "next_steps": [
    "<Step 1: concrete deployment action with timeline>",
    "<Step 2: operational or integration action>",
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

  "executive_summary": "<2-3 sentences for a non-technical executive. No jargon. Translate AUC into plain business outcomes. Must end with the literal word YES or NO (the deploy verdict).>"
}
```

**Rules:**
- Use EXACT computed values for: `recommended_model`, `primary_metric_value`, `all_models_summary`, `alternative_model`, `test_verdict_summary`, `confidence_score`, `routing_zone`, `flags`, `requires_human_review`, `human_review_reason`.
- All narrative fields must reference actual model names and metric numbers from the data.
- `executive_summary` MUST end with YES or NO.
- `tradeoffs` must have at least 3 items. `next_steps` must have at least 3 items. `feature_drivers` must have exactly 5 items.
- `flags` must be a JSON array — use `[]` when no flags apply, never `null`.
- `routing_zone` must be exactly `"zone_1"`, `"zone_2"`, or `"zone_3"`.

---

## Step 4 — Confirm

After writing the file, print:

```
RECOMMENDATION OK
  model=<winner_name>  auc=<auc>  confidence=<confidence_score>  zone=<routing_zone>  deploy=<deploy_word>
  flags=<flags list or 'none'>
```

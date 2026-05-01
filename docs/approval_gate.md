# Approval Gate Design — MIRA ML Recommendation Agent
## Session 6 Deliverable · MIS6349 · Shibani Kumar

---

## 1. Agent Output Schema (DECIDING Version)

The recommendation output schema adds `confidence_score`, `routing_zone`, and `flags[]`
to the base Project 1 schema. These three fields drive all routing decisions.

```json
{
  "recommended_model": "string",
  "primary_metric_value": "float",
  "confidence_score": "float [0.0–1.0] — reasoning certainty, NOT output quality",
  "routing_zone": "zone_1 | zone_2 | zone_3",
  "flags": ["DATA_LEAKAGE_DETECTED", "OVERFITTING_DETECTED", ...],
  "requires_human_review": "bool",
  "human_review_reason": "string or null",
  "all_models_summary": "[...]",
  "selection_reason": "string",
  "model_comparison_narrative": "string",
  "business_impact": "{...}",
  "tradeoffs": "[...]",
  "next_steps": "[...]",
  "deployment_considerations": "[...]",
  "risks": "[...]",
  "feature_drivers": "[...]",
  "alternative_model": "string",
  "alternative_model_reason": "string",
  "test_verdict_summary": "string",
  "executive_summary": "string — must end with YES or NO"
}
```

**Confidence scoring instruction (in system prompt):**
```
After completing your reasoning, assign a confidence_score between 0 and 1.
This score must reflect reasoning certainty — not output format quality:
  - How decisive was the model selection? (clear gap = higher, tied models = lower)
  - How complete and clean was the training data? (clean data = higher, heavy imputation = lower)
  - Did the model pass all stress tests? (PASS = higher, FAIL = lower)
  - How stable were the cross-validation results? (low cv_std = higher)

A score above 0.85 means the recommendation is clear and well-supported.
A score below 0.70 means there is material uncertainty the human reviewer must evaluate.
Do NOT inflate this score because the output is well-formatted.
```

---

## 2. Three Routing Zones

### Zone 1 — Auto-Approve Eligible

**All conditions must be true:**
- `confidence_score >= 0.85`
- `flags = []` (no warning flags active)
- No hard escalation rules triggered
- `test_verdict = PASS`
- `winner_auc >= 0.75`
- AUC gap between winner and runner-up >= 0.03

**Behaviour:** Pipeline auto-proceeds to evals. No CLI pause. Run is logged and flagged for periodic audit sampling.

**Auto-approve is a business decision:** Zone 1 means *eligible* for auto-approve. The 0.85 threshold is the starting point — it should be calibrated against real run data once 50+ runs exist.

---

### Zone 2 — Standard Human Review (24h window)

**Any one condition is enough:**
- `confidence_score` between 0.70 and 0.84
- Any soft flag: `STABILITY_CONCERN`, `AMBIGUOUS_MODEL_SELECTION`, `HEAVY_IMPUTATION`, `FEATURE_IMPORTANCE_CONCENTRATION`
- Overfitting gap between 0.07 and 0.10 (watch zone)
- `METRIC_OBJECTIVE_MISMATCH` rule triggered

**What the human sees:** recommended model, AUC, confidence score, active flags, review reason, executive summary, full recommendation file path.

**Human must:** type `yes` or `no`, select an override category, and provide a brief rationale.

---

### Zone 3 — Priority Human Review (immediate)

**Any hard escalation rule OR:**
- `confidence_score < 0.70`
- `test_verdict = FAIL`
- `leakage_detected = True`
- `overfitting_gap > 0.20` (severe)
- `winner_auc < 0.65`
- `rows < 1000` after cleaning

**What the human sees:** everything in Zone 2 PLUS the full list of escalation rules with their severity and detail, top 3 feature drivers, and full test verdict summary.

**Human must:** type `yes` or `no`, select an override category, and provide a rationale. The override is logged with all zone details for audit.

---

## 3. Hard Escalation Rules (minimum 3, one per category)

### Category 1: Data Quality — DATA_LEAKAGE_DETECTED

```
Rule Name   : DATA_LEAKAGE_DETECTED
Trigger     : leakage_detected = True in model_selection.json
              (winner CV AUC > 0.99 after EDA already removed obvious leaky columns)
Detection   : Modeltrain.py computes and writes leakage_detected;
              escalation_rules.py reads it before the gate
Routing     : Zone 3 — Priority Review
Override    : Human confirms the data pipeline investigation is complete
              and no leakage source remains in the feature set
Logging     : leakage_detected flag, winner_auc, all escalation rule details
```

### Category 2: Model Quality — OVERFITTING_DETECTED

```
Rule Name   : OVERFITTING_DETECTED / SEVERE_OVERFITTING
Trigger     : overfitting_gap > 0.10 → Zone 2 (medium)
              overfitting_gap > 0.20 → Zone 3 (severe)
Detection   : Modeltrain.py computes train_score - cv_auc_mean for winner;
              escalation_rules.py checks threshold
Routing     : Zone 2 (gap 0.10-0.20) or Zone 3 (gap > 0.20)
Override    : Human confirms generalisation risk is acceptable for this deployment
Logging     : overfitting_gap, train_score, val_score, winner_name
```

### Category 3: Semantic Ambiguity — AMBIGUOUS_MODEL_SELECTION

```
Rule Name   : AMBIGUOUS_MODEL_SELECTION
Trigger     : AUC gap between winner and runner-up < 0.02
              (difference is within statistical noise of 5-fold CV)
Detection   : escalation_rules.py computes gap from models_trained list
Routing     : Zone 2 — Standard Review
Override    : Human selects model based on operational criteria —
              cost, interpretability, team familiarity, deployment infrastructure
Logging     : winner_auc, runner_up_auc, gap, human_selection_rationale
```

### Additional Rules (implemented in agent/escalation_rules.py):

| Rule | Category | Trigger | Zone |
|------|----------|---------|------|
| INSUFFICIENT_TRAINING_DATA | Data Quality | rows < 1,000 after cleaning | 3 |
| HEAVY_IMPUTATION | Data Quality | any column > 30% imputed | 2 |
| TEST_VERDICT_FAIL | Model Quality | test_verdict = FAIL | 3 |
| BELOW_PERFORMANCE_FLOOR | Model Quality | winner AUC < 0.65 | 3 |
| METRIC_OBJECTIVE_MISMATCH | Semantic Ambiguity | priority=recall but winner recall < 0.60 | 2 |
| SEVERE_IMBALANCE_UNHANDLED | Semantic Ambiguity | minority < 10% and no balancing applied | 3 |
| FEATURE_IMPORTANCE_CONCENTRATION | Adversarial Signal | top feature > 60% of total importance | 2 |

---

## 4. Override Category Vocabulary

Defined upfront to enable pattern analysis across runs. Free-text rationale is captured separately.

| Code | Category | When to Use |
|------|----------|-------------|
| 1 | `PERFORMANCE_ACCEPTABLE` | AUC is below threshold but human judges it acceptable for this specific business context |
| 2 | `BUSINESS_PRIORITY` | Human chose a different model or accepted the risk based on operational factors — cost, interpretability, team familiarity |
| 3 | `DATA_QUALITY_RESOLVED` | The flagged data issue is understood and confirmed not to affect the deployment decision |
| 4 | `DOMAIN_KNOWLEDGE` | Human applied business domain expertise the agent could not infer from the dataset alone |
| 5 | `METRIC_MISMATCH` | Human overrides the inferred evaluation metric — the actual business priority is different |
| 6 | `RISK_ACCEPTED` | Human explicitly acknowledges the risk (overfitting, instability, imbalance) and proceeds |
| 7 | `AGENT_ERROR` | Agent reasoning was factually wrong — flag for system prompt review |

**Why this matters:** After 30+ runs, if 40% of overrides are `DATA_QUALITY_RESOLVED`, that points to EDA cleaning being too aggressive. If 35% are `BUSINESS_PRIORITY`, the agent's operational awareness needs improvement. The category vocabulary turns individual human decisions into improvement signals.

---

## 5. Override Log Schema

Written to `logs/overrides/{run_id}_override.json` after every Zone 2 or Zone 3 decision.

```json
{
  "run_id": "run_a1b2c3d4",
  "timestamp": "2026-05-01T14:22:00Z",
  "routing_zone": "zone_2",
  "escalation_rules_triggered": ["AMBIGUOUS_MODEL_SELECTION"],
  "agent_recommendation": {
    "recommended_model": "LightGBM",
    "primary_metric_value": 0.812,
    "confidence_score": 0.74,
    "flags": ["AMBIGUOUS_MODEL_SELECTION"],
    "requires_human_review": true,
    "human_review_reason": "top-2 AUC gap=0.007 — selection ambiguous"
  },
  "human_decision": "APPROVED",
  "override_category": "BUSINESS_PRIORITY",
  "human_rationale": "LightGBM is acceptable — team has existing deployment tooling",
  "review_duration_seconds": 94.3,
  "agent_version": "MIRA v0.5.0",
  "prompt_version": "mira_agent_v0_5_0.md"
}
```

**Critical fields:**
- `review_duration_seconds`: 8 seconds = rubber stamp. 90 seconds = engaged review.
- `override_category`: enables systematic pattern analysis across runs.
- `agent_version` + `prompt_version`: enables before/after comparison when the system prompt changes.
- `escalation_rules_triggered`: shows which rules fire most — potential for rule calibration.

---

## 6. Rubber Stamp Prevention

Per Session 6: *"A rubber stamp approval process creates the appearance of accountability without the substance of it."*

Three design choices in MIRA's gate prevent rubber stamping:

1. **Zone 3 shows reasoning, not just decision.** The human sees the full escalation rule detail, top features, and test verdict — not just "yes/no on this model." The recommendation file path is shown for deeper review.

2. **Override category is mandatory.** The human cannot just type `yes` — they must categorise the override and provide a rationale. This creates friction that is proportional to the risk zone.

3. **Review duration is logged.** An 8-second review on a Zone 3 case with a leakage flag is a rubber stamp. The override log captures this for audit.

---

*Shibani Kumar · MIS6349 · Session 6 Deliverable*
*Implementation: agent/escalation_rules.py, agent/main.py (hitl_approval_gate), skills/mira-recommend/SKILL.md*

# Evaluation Report — MIRA ML Recommendation Agent
## Session 4 / Session 8 Scorecard · MIS6349 · Shibani Kumar

---

## Part 1: 4-Dimension Evaluation (Session 4 Carry-Forward)

### Dimension 1: Accuracy

**Definition:** Percentage of runs where MIRA recommended the correct model — the one with the highest cross-validated score on the inferred priority metric — with a complete, coherent justification.

| Target | ≥ 80% correct recommendations across 10+ varied inputs |
|--------|-------------------------------------------------------|
| Method | Manual review of recommendation.json vs. model_selection.json for each run |

| Run ID | Dataset | Winner (Ground Truth) | MIRA Recommended | Correct? | Notes |
|--------|---------|----------------------|-----------------|---------|-------|
| TBD | | | | | Run and fill |
| TBD | | | | | |
| TBD | | | | | |
| TBD | | | | | |
| TBD | | | | | |
| TBD | | | | | |
| TBD | | | | | |
| TBD | | | | | |
| TBD | | | | | |
| TBD | | | | | |

**Accuracy result:** TBD / 10 = TBD%

---

### Dimension 2: Latency

**Definition:** Total wall-clock time from `MIRAAgent().run()` start to recommendation.json being written. Captured as `duration_seconds` in `logs/runs/{run_id}_run.json`.

| Target | p50 < 600s for a full 3-phase run |
|--------|----------------------------------|

| Run ID | Duration (s) | Phase 1 | Phase 2 | Phase 3 | Notes |
|--------|-------------|---------|---------|---------|-------|
| TBD | | | | | |

**Latency results:**
- p50: TBD
- p95: TBD
- Slowest run: TBD (reason: TBD)

---

### Dimension 3: Cost per Run

**Definition:** LLM token usage per run. Tracked via LiteLLM usage fields in EDA.py (metric inference call) and accumulated across agent iterations.

| Target | Baseline established and tracked per run |
|--------|----------------------------------------|

| Run ID | EDA tokens (prompt + completion) | Agent iterations | Total tokens (est.) | Estimated cost ($) |
|--------|----------------------------------|-----------------|--------------------|--------------------|
| TBD | | | | |

**Cost result:** TBD per run (baseline)

---

### Dimension 4: Failure Rate

**Hard failure:** run completed but recommendation.json was not produced (schema missing, agent timed out, script crashed).

**Silent failure:** recommendation.json produced but unit tests or behavior evals failed — the output looks correct but contains errors.

| Target | Hard failure rate < 5% · Silent failure rate < 10% |
|--------|---------------------------------------------------|

| Metric | Count | Total runs | Rate |
|--------|-------|------------|------|
| Hard failures | TBD | TBD | TBD% |
| Silent failures | TBD | TBD | TBD% |
| Complete successes | TBD | TBD | TBD% |

---

## Part 2: Project 2 Scenario Scorecard (Session 8)

### Zone 1: Happy Path (target: ≥10 cases, ≥90% pass rate, latency <600s)

| Case ID | Scenario | Pass? | Latency | Notes |
|---------|----------|-------|---------|-------|
| case_01 | clear_winner | TBD | TBD | |
| case_11 | small_dataset_happy_path | TBD | TBD | |
| case_12 | recall_metric_happy_path | TBD | TBD | |
| case_13 | gradient_boosting_winner | TBD | TBD | |
| case_14 | logistic_regression_winner | TBD | TBD | |
| case_15 | f1_metric_happy_path | TBD | TBD | |
| case_16 | high_imbalance_correctly_handled | TBD | TBD | |
| case_17 | large_dataset_happy_path | TBD | TBD | |
| case_18 | moderate_auc_acceptable | TBD | TBD | |
| case_19 | precision_metric_happy_path | TBD | TBD | |

**Zone 1 pass rate:** TBD / 10 = TBD% (target: ≥90%)

---

### Zone 2: Exceptions (target: ≥8 cases, ≥80% pass rate, latency <900s)

| Case ID | Scenario | Pass? | HITL triggered? | Notes |
|---------|----------|-------|----------------|-------|
| case_02 | poor_performance | TBD | TBD | |
| case_04 | data_leakage_detected | TBD | TBD | |
| case_07 | overfitting_top_model | TBD | TBD | |
| case_08 | missing_required_fields | TBD | TBD | |
| case_20 | all_models_high_variance | TBD | TBD | |
| case_21 | severe_class_imbalance_unhandled | TBD | TBD | |
| case_22 | moderate_overfitting_borderline | TBD | TBD | |
| case_23 | feature_concentration_risk | TBD | TBD | |

**Zone 2 pass rate:** TBD / 8 = TBD% (target: ≥80%)

---

### Zone 3: Edge Cases (target: ≥6 cases, ≥70% pass rate, latency <900s)

| Case ID | Scenario | Pass? | Notes |
|---------|----------|-------|-------|
| case_03 | severe_class_imbalance | TBD | |
| case_05 | equal_models | TBD | |
| case_06 | jargon_in_summary | TBD | |
| case_09 | wrong_model_recommended | TBD | |
| case_10 | no_deployment_verdict | TBD | |
| case_24 | boundary_auc_at_floor | TBD | |

**Zone 3 pass rate:** TBD / 6 = TBD% (target: ≥70%)

---

### Zone 4: Adversarial (target: ≥4 cases, 100% pass — zero successful attacks)

| Case ID | Scenario | Attack succeeded? | Notes |
|---------|----------|-----------------|-------|
| case_25 | prompt_injection_in_selection_reason | TBD | |
| case_26 | impossible_metric_values | TBD | |
| case_27 | contradictory_recommendation | TBD | |
| case_28 | false_high_confidence_gray_zone | TBD | |

**Zone 4 result:** TBD / 4 successful injections (target: 0)

---

## Part 3: LLM-as-Judge Validation

Per Project 2 spec: run LLM judge on ≥10 cases, validate against human review to establish judge accuracy before trusting at scale.

| Case ID | LLM Judge Score | Human Score | Agree? |
|---------|----------------|-------------|--------|
| TBD | | | |

**Judge accuracy:** TBD% agreement with human review (target: validate before scaling)

---

## Part 4: Production Bar Self-Assessment

| Item | Status | Evidence |
|------|--------|---------|
| Prompts versioned and immutable | ✅ | v0.1.0 through v0.5.0 in /prompts/ |
| Errors handled (retry → validate → escalate) | ✅ | Push loop + schema validation + HITL gate |
| Every run logged with prompt version | ✅ | logs/runs/{run_id}_run.json |
| max_iterations set | ✅ | 40 in build_agent_input() |
| Output validated before downstream use | ✅ | REQUIRED_KEYS + 18 unit tests |
| 3+ passing tests | ✅ | 18 unit tests in evals/unit_tests.py |
| Confidence threshold routing implemented | ✅ | Three-zone gate in agent/main.py |
| Override logging active | ✅ | logs/overrides/{run_id}_override.json |

---

## Part 5: Key Findings

*Fill in after Session 8 evaluation runs.*

**One adversarial test that changed the architecture:**
TBD — run case_25 through case_28 and document the result that required an architecture change.

**One finding the demo would never show:**
TBD — run systematic evaluation and document the failure mode that only appeared at scale.

**Playbook entry:**
TBD — one structural lesson from the evaluation, formatted as actionable guidance for the next engineer.

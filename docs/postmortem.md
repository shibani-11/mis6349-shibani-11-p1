# Post-Mortem: Pre-Mortem vs Actual Comparison
## MIRA ML Recommendation Agent · MIS6349 · Shibani Kumar

---

## Pre-Mortem vs Actual (Session 4 reckoning)

*Fill in "Actual Frequency" and "Detected?" after running the evaluation set.*

| Failure Mode (Predicted) | Likelihood | Actual Frequency | Detected by? | Notes |
|--------------------------|-----------|------------------|--------------|-------|
| Dataset file cannot be read | High | TBD | EDA.py exception | |
| Missing or invalid model evaluation metrics | Medium | TBD | REQUIRED_KEYS check | |
| Incorrect recommendation due to metric tradeoffs | Medium | TBD | Unit test UT-02 | |
| Agent produces non-JSON output | Low | TBD | Schema validator | |
| Network/API timeout (LLM call) | Low | TBD | LiteLLM exception | |
| Invalid input format | Medium | TBD | Pydantic AgentInput | |
| Empty dataset | Low | TBD | EDA.py shape check | |
| All models have equal metrics | Low | TBD | HITL gate ambiguity flag | |
| Agent hangs on mira-recommend skill | Medium | TBD | Push loop + missing recommendation.json | Added after first test run |
| Confidence score inflated (output quality vs reasoning) | Medium | TBD | Override log review_duration | Added after Session 6 |

---

## Failure Modes That Appeared That Were NOT Predicted

*Fill in after systematic evaluation.*

| Failure Mode | Zone | How Discovered | Architecture Change Made |
|-------------|------|---------------|------------------------|
| TBD | | | |

**Expected discovery pattern (based on Session 6 war room stories):**
- Agent may score high confidence on tied-model cases (ambiguous selection but confident output) → now caught by `AMBIGUOUS_MODEL_SELECTION` flag
- Reviewer may approve Zone 3 cases in under 10 seconds → tracked by `review_duration_seconds` in override log
- Adversarial inputs in business_problem field may influence metric inference in EDA.py → LiteLLM call result validated against VALID_METRICS list

---

## Structural Changes Triggered by Evaluation

| Change | Triggered by | Version |
|--------|-------------|---------|
| Confidence score changed from AUC-based to reasoning-based | Session 6 framework — output confidence ≠ reasoning confidence | v0.5.0 |
| Three-zone HITL gate added | Session 6 approval gate design | v0.5.0 |
| `flags[]` field added to recommendation schema | Session 6 DECIDING mode schema requirement | v0.5.0 |
| Hard escalation rules extracted to `agent/escalation_rules.py` | Need for structured, categorised rules (not ad-hoc checks) | v0.5.0 |
| Override logging added | Session 6 override logging as feedback infrastructure | v0.5.0 |
| TBD — from adversarial test results | case_25/26/27/28 — fill after running | TBD |

---

## Session 4 ObservationÂ³ Retrospective

**INWARD (what I assumed that the build proved wrong):**

TBD — fill after evaluation. Expected: assumed the confidence score being AUC-based was sufficient; Session 6 revealed it was measuring the wrong thing.

**UPWARD (architectural decision that shaped the outcome):**

TBD — fill after evaluation. Expected: the decision to use a pre-built script approach (EDA.py + Modeltrain.py) rather than letting the agent write Python was the most consequential architectural choice — it made outputs deterministic and removed hallucination risk from the data layer.

**OUTWARD (what the next engineer should know before starting):**

TBD — fill after evaluation. Expected: design the confidence threshold for reasoning certainty, not output quality, before writing a single line of gate code. Setting the threshold after implementation creates a false sense of safety.

---

## Playbook Entry

*One paragraph, second-person, actionable — for the shared class playbook.*

TBD — fill after evaluation. Format: "When building an ML recommendation agent with a HITL gate, define your hard escalation rules before your confidence thresholds. Escalation rules are deductive — they come from domain requirements. Thresholds are empirical — they come from calibration data. Designing them in the wrong order means your thresholds will be set to accommodate your rules instead of your data."

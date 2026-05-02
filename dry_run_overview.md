# MIRA — Agent Design Overview
**Model Intelligence & Recommendation Agent**

---

## 01 · Agent Definition

MIRA is an autonomous ML recommendation agent. It takes a business problem described in plain English and a dataset, runs a full machine-learning pipeline end-to-end, and produces a deployment recommendation — including which model to deploy, why, what risks exist, and what to do next.

A built-in human review gate ensures no decision is fully automated when uncertainty is high.

> **In the project:**
> - `agent/mira_agent.py` — `MIRAAgent` class is the core agent; called with an `AgentInput` object and runs all three phases
> - `schemas/input_schema.py` — defines `AgentInput` (dataset path, target column, business problem, run ID)
> - Produces four structured outputs per run: `data_card.json`, `model_selection.json`, `recommendation.json`, `eval_report.json`
> - The churn demo run (`run_bc710c55`) recommended **LightGBM** with recall=0.4811, AUC=0.863

---

## 02 · Run — How the Agent Acts
### Tooling, MCP, Functions & APIs

| Component | Detail |
|---|---|
| **Tool Use** | Python subprocess calls, file I/O, structured JSON outputs per phase |
| **MCP** | Model Context Protocol — standardised interface for tool invocation and result passing between phases |
| **Function Calling** | LLM function calls to infer evaluation metric from business problem text |
| **API** | OpenAI / LiteLLM API for LLM-powered reasoning (gpt-4o-mini by default) |
| **AgentSkill** | Versioned prompt skill `mira-recommend` driving the Phase 3 recommendation |

> **In the project:**
> - **Tool Use:** `app.py` calls `subprocess.run(["python", "scripts/EDA.py", ...])` and `scripts/Modeltrain.py` as tools; outputs written to `processed/`
> - **MCP:** LiteLLM acts as the unified API layer — same call interface regardless of which LLM provider is used
> - **Function Calling:** `scripts/EDA.py → infer_priority_metric()` sends the business problem to gpt-4o-mini and gets back the right metric (recall, AUC, F1, precision)
> - **API + Token Tracking:** `response.usage.prompt_tokens` / `completion_tokens` captured; cost logged in `_run.json` under `cost_tracking` (e.g. `eda_cost_usd: 0.000039`)
> - **AgentSkill:** defined in `prompts/SKILL.md` as `mira-recommend`; injected into Phase 3 system prompt

---

## 03 · Reason — How the Agent Thinks
### Context Engineering

- **Structured context injection** — Data card + model selection JSON passed as context to Phase 3 LLM
- **Versioned prompts** — v0.1.0 → v0.5.0 changelog tracked in `prompts/README.md`
- **DECIDING mode** — Agent infers confidence score, routing zone, and flags before recommending
- **Three-zone HITL gate** — Zone 1 auto-proceeds · Zone 2 standard review · Zone 3 priority escalation
- **Self-correction** — `run_with_validation` attempts one correction before escalating to human review

> **In the project:**
> - **Context injection:** `mira_agent.py` loads `data_card.json` + `model_selection.json` and builds the LLM prompt from their contents — the agent "knows" the data profile and all 5 model scores before reasoning
> - **Versioned prompts:** `prompts/README.md` documents 5 versions; v0.5.0 added `confidence_score`, `routing_zone`, and `flags[]` to the output schema
> - **DECIDING mode:** Every recommendation includes `confidence_score` (0–1), `routing_zone` (zone_1/2/3), and `flags[]` (e.g. `AMBIGUOUS_SELECTION`, `LOW_CONFIDENCE`)
> - **HITL gate:** `agent/escalation_rules.py` → `evaluate_escalation_rules()` checks 7 hard rules; triggers Zone 3 on critical violations
> - **Self-correction:** `agent/runner.py → run_with_validation()` calls the agent, validates output, and if it fails sends one correction prompt before raising `OutputValidationError`

---

## 04 · Data Sources
### Structured & Unstructured Input

**Unstructured**
- PDF documents
- MS Word (.docx) files
- Plain-text business problem descriptions
- LLM reads and reasons over these directly

**Structured**
- CSV files
- Excel (.xls / .xlsx)
- Database tables (via export)
- Profiled, cleaned, and encoded in Phase 1 (EDA)

> **In the project:**
> - **Unstructured — Business problem text:** `infer_priority_metric()` in `EDA.py` sends the free-text business problem to the LLM; for the churn dataset it correctly inferred **recall** as the priority metric with the reason: *"missing a customer at risk of closing their account leads to revenue loss"*
> - **Structured — File loading:** `EDA.py` dispatches on file extension — `pd.read_csv()` for `.csv`, `pd.read_excel()` for `.xls/.xlsx`; raises `ValueError` for unsupported formats
> - **Cleaning pipeline:** IQR outlier capping on `CreditScore`, `Age`, `NumOfProducts`; dropped ID columns (`CustomerId`, `RowNumber`, `Surname`); StandardScaler on all 10 numeric features
> - **Output:** cleaned data saved to `processed/run_xxx_cleaned.csv` and handed to Phase 2

---

## 05 · Markup Language
### Output Formatting

| Format | Used For |
|---|---|
| **JSON** | All inter-phase outputs — data_card, model_selection, recommendation, eval_report |
| **Markdown** | Executive summary, narratives, human-readable findings |
| **HTML / CSS** | Streamlit UI rendered via `st.markdown(unsafe_allow_html=True)` |
| **Pydantic Schema** | 20-key output schema enforced by `agent/validator.py` before any result is accepted |

> **In the project:**
> - **JSON:** Every phase writes a typed JSON file — e.g. `run_bc710c55_data_card.json` has `rows`, `class_distribution`, `priority_metric`, `cleaning_log`, `eda_llm_tokens`
> - **Markdown:** `genai_narrative` field in `data_card.json` and `model_selection.json` is LLM-generated plain-English narrative; `executive_summary` in recommendation must contain YES or NO deployment verdict
> - **HTML/CSS:** `app.py` renders the six-step homepage, pipeline nodes, HITL zone banners, and recommendation cards all via raw HTML injected into Streamlit
> - **Schema enforcement:** `agent/validator.py` defines `REQUIRED_RECOMMENDATION_KEYS` (20 keys including `confidence_score`, `routing_zone`, `flags`, `requires_human_review`, `executive_summary`); `validate_output()` is called before the result is accepted

---

## 06 · Knowledge Graph
### Connect with Arya — Document Graph Creation

- Ingest unstructured documents (PDF, Word) as nodes
- Extract entities and relationships to build a knowledge graph
- MIRA queries the graph to enrich context before reasoning
- Enables cross-document linking — e.g. policy docs informing model constraints
- **Next step:** Integrate Arya's graph layer as a retrieval tool in Phase 3 context

> **In the project (current state + integration plan):**
> - Currently, MIRA accepts the business problem as free text — the LLM reasons over it directly
> - **Integration point:** Arya's knowledge graph would be called as a tool in Phase 3 — before writing the recommendation, MIRA retrieves relevant graph nodes (e.g. company policy on model risk, historical decisions on similar datasets)
> - Graph output would be appended to the Phase 3 context window alongside `data_card.json` and `model_selection.json`
> - Entity types to extract from documents: model names, metric thresholds, business rules, risk flags, stakeholder names

---

## 07 · Evaluations

| Metric | Value |
|---|---|
| Golden Dataset Cases | 30 (10 happy path + 10 exception + 10 risk) |
| Unit Tests | 18 tests — 100% passing (`evals/unit_tests.py`) |
| Test Files | 4 files in `tests/` — 25+ tests total |
| Eval Dimensions | 7 (behaviour, quality, system, unit tests, HITL gate, production checklist, LLM judge) |

---

### Happy Path — 10 Cases
*File: `tests/test_happy_path.py` (9 tests) + `evals/golden_dataset.json`*

- Valid input, clean data, clear business problem description
- Agent produces full recommendation with all 20 required keys
- Confidence ≥ 0.85 → Zone 1 auto-approve eligible
- Production checklist passes all critical items
- Executive summary contains explicit YES / NO deployment verdict

> **In the project:**
> - `test_happy_path.py` checks: valid recommendation passes `validate_output()`, all 20 required keys present, `confidence_score` in [0,1], `routing_zone` is valid, `flags` is a list, `executive_summary` contains verdict, `requires_human_review` is bool
> - `evals/unit_tests.py` — 18 tests covering `TestPhase4RequiredKeys`, `TestPhase4NoJargon`, `TestPhase4DeploymentVerdict`, `TestPhase4ConfidenceScore`, `TestPhase4HumanReviewFlag` — all passing in `run_bc710c55`
> - `eval_report.json` behaviour score for data_card, model_testing phases: **100%**

---

### Exception Path — 10 Cases
*File: `tests/test_edge_cases.py` (11 tests) + `tests/test_retry.py` (7 tests)*

- Missing required output keys → validation fails, error raised
- Confidence value out of range [0, 1] → schema rejected
- Invalid `routing_zone` value → `OutputValidationError` raised
- API timeout → `run_with_retry` (3 attempts, exponential backoff: 1s, 2s, 4s)
- Persistent failure → `run_with_fallback` → escalated to HITL human review

> **In the project:**
> - `agent/runner.py → run_with_retry()`: catches `APIRateLimitError` and `APITimeoutError`; retries with `time.sleep(2**attempt)`; re-raises on 3rd failure
> - `run_with_validation()`: validates output after each call; sends one self-correction prompt if validation fails; raises `OutputValidationError` if still invalid
> - `run_with_fallback()`: if primary agent fails, calls `_route_to_human_review()` which returns `{"status": "NEEDS_REVIEW", "confidence_score": 0.0, "flags": ["ESCALATED"]}`
> - `test_edge_cases.py` covers: non-dict output, empty dict, unsupported file format, blank target column, blank business problem

---

### Risk-Based — 10 Cases *(Pre-mortem)*
*File: `evals/golden_dataset.json` + `agent/escalation_rules.py`*

| # | Risk Scenario |
|---|---|
| 1 | Top-2 models within 0.003 AUC — ambiguous selection flagged |
| 2 | Overfitting gap > 0.10 threshold breached |
| 3 | Class imbalance not handled — recall figures inflated |
| 4 | Data leakage: AUC suspiciously close to 1.0 |
| 5 | CV instability: std > 0.05 across folds |
| 6 | Rubber-stamp review: Zone 3 approved in < 8 seconds |
| 7 | Thin business justification: < 50 chars in rationale |
| 8 | Missing required fields in final recommendation JSON |
| 9 | LLM returns wrong metric for the given business context |
| 10 | Human rejects recommendation — override log written and audited |

> **In the project:**
> - **Risk 1:** `hitl_gate` in `eval_report.json` flagged `ambiguous_model_selection` (LightGBM vs Gradient Boosting differ by 0.0022 AUC) in `run_bc710c55` — correctly identified
> - **Risk 2:** `escalation_rules.py` → `OVERFITTING_RISK` rule triggers if `overfitting_gap > 0.10`; Random Forest (gap=0.1485) was rejected for this reason
> - **Risk 4:** `model_selection.py` leakage check — if `best_auc > 0.99` → `leakage_detected = True`, test verdict FAIL
> - **Risk 5:** stability check — if `cv_std > 0.05` → `stability_flag = True`
> - **Risk 6 & 7:** `write_override_log()` in `agent/main.py` records `review_duration_seconds` and `human_rationale`; short reviews and thin rationale are flagged in the HITL risk score
> - **Risk 10:** rejected runs write an override log to `logs/overrides/`; `hitl_approved = False` skips eval and flags the run in the UI

---

*MIRA v0.5.0 · Prompt v0.5.0 (DECIDING Mode) · Three-Zone HITL Gate · 7-Layer Eval System*

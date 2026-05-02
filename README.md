# MIRA - Model Intelligence & Recommendation Agent

MIRA is a fully autonomous multi-phase ML agent. Given a dataset and a business problem, it profiles the data, trains and stress-tests candidate models, and delivers a structured deployment recommendation — with a built-in three-zone human approval gate before anything is finalized.

---

## Architecture Answers

### 1. What exactly does this agent do?

MIRA is a fully autonomous 3-phase ML agent. Given any tabular dataset and a plain-English business problem, it:

1. **Profiles the data** and infers the right evaluation metric from the business problem using an LLM function call
2. **Trains 5 models** with 5-fold cross-validation and runs three stress tests (overfitting, leakage, stability) on every candidate
3. **Reasons over the results** using a versioned LLM AgentSkill and writes a structured deployment recommendation with confidence score, routing zone, flags, and a plain-English executive summary

A three-zone HITL gate then routes the result: Zone 1 auto-proceeds, Zone 2 pauses for standard review, Zone 3 escalates for priority human review. Every decision is logged with rationale and review duration for audit.

---

### 2. Input and Output

**Input** — configured via Streamlit UI or `AgentInput` directly:

```python
AgentInput(
    dataset_path="data/raw/Churn_Modelling.csv",
    target_column="Exited",
    business_problem="Identify customers likely to churn so retention can intervene early.",
    task_type="auto",
    max_models=5,
    max_iterations=40,
)
```

Supported formats: `.csv`, `.xlsx`, `.xls`

**Output** — four files written to `processed/`:

| File | Written by | Key contents |
|---|---|---|
| `{run_id}_data_card.json` | Phase 1 — EDA.py | rows, features, class distribution, imbalance flag, missing values, quality issues, priority metric + reason, LLM token usage |
| `{run_id}_model_selection.json` | Phase 2 — Modeltrain.py | per-model CV scores (AUC, F1, recall, precision), overfitting gap, leakage flag, stability flag, PASS/FAIL verdict, feature importance |
| `{run_id}_recommendation.json` | Phase 3 — MIRAAgent | recommended model, selection reason, confidence score, routing zone, flags, business impact, next steps, risks, executive summary with YES/NO verdict |
| `{run_id}_eval_report.json` | Eval Runner | scores across all 7 eval layers, HITL risk score, production readiness verdict |

---

### 3. What tools are used and why?

**Phase 1 & 2 — Subprocess scripts**

Phases 1 and 2 run as independent Python subprocesses called from the orchestrator or Streamlit UI. This keeps them isolated, fast, and independently testable.

| Script | What it does |
|---|---|
| `scripts/EDA.py` | Loads and cleans the dataset, profiles features, calls gpt-4o-mini to infer the priority metric from the business problem text, writes `data_card.json` |
| `scripts/Modeltrain.py` | Trains 5 models with 5-fold stratified CV, runs overfitting / leakage / stability stress tests, writes `model_selection.json` |

**Phase 3 — OpenHands Agent**

Phase 3 runs `MIRAAgent` in `agent/mira_agent.py` using the OpenHands SDK.

| Tool | What it does |
|---|---|
| `TerminalTool` | Executes commands and reads phase outputs |
| `FileEditorTool` | Reads phase 1 & 2 JSON files, writes `recommendation.json` |
| `TaskTrackerTool` | Signals run completion, terminates the agent loop |
| `mira-recommend` AgentSkill | Versioned prompt skill (`prompts/SKILL.md`) injected into Phase 3 context |

**LLM & Routing**

| Library | Role |
|---|---|
| `openhands-sdk` | Agent runtime for Phase 3 (`LLM`, `Agent`, `Skill`) |
| `litellm` | Routes LLM calls to any provider without code changes |
| `openai` | OpenAI API client (used via LiteLLM) |
| `python-dotenv` | Loads `LLM_API_KEY` and `LLM_MODEL` from `.env` |

**Data & ML**

| Library | Role |
|---|---|
| `pandas` | Dataset loading and profiling |
| `numpy` | Numerical operations |
| `openpyxl` | Excel format support |
| `scikit-learn` | Logistic Regression, Random Forest, cross-validation, preprocessing |
| `xgboost` | XGBoost model training |
| `lightgbm` | LightGBM model training |
| `imbalanced-learn` | SMOTE and resampling strategies (available, not applied by default) |

**Validation & Utilities**

| Library | Role |
|---|---|
| `pydantic >= 2.0` | `AgentInput` schema, validates all inputs at the boundary |
| `uuid` | Generates unique `run_id` for each run |
| `streamlit` | Web UI for upload, live run progress, HITL gate, and results |

---

### 4. Success Criteria

MIRA self-evaluates after every approved run across 7 layers:

| Layer | What it checks | Pass bar |
|---|---|---|
| Behavior Evals | Required fields present, key values computed in each phase | ≥ 70% per phase |
| Quality Eval | Best model ≥ 0.65 AUC, stress tests passed, recommendation produced | ≥ 70% |
| System Eval | All phases completed, no errors, agents tracked | ≥ 70% |
| Unit Tests | 18 deterministic checks on recommendation schema (keys, types, ranges) | 100% |
| HITL Gate | 7 weighted risk factors: leakage, overfitting, low confidence, ambiguous selection | Risk score < 5 |
| Production Checklist | 7 items; 3 critical (leakage, overfitting, AUC floor) block deployment if failed | ≥ 6/7 + all critical |
| LLM Judge | Independent LLM grades reasoning quality and schema compliance | Reported separately |

Overall pass = mean of Layers 1–4 + 6. Each run leaves a complete audit trail in `processed/{run_id}_eval_report.json`.

---

### 5. What are the three most likely ways it fails?

**1. Phase 3 produces a recommendation with missing required fields**

Detection: `agent/validator.py → validate_output()` checks all 20 required keys before the result is accepted. The Production Checklist item `all_required_fields_present` flags any that are `None`, empty strings, or empty lists.

Response: `agent/runner.py → run_with_validation()` sends one self-correction prompt if validation fails. If it still fails, `run_with_fallback()` routes to `_route_to_human_review()` with `confidence_score: 0.0` and `flags: ["ESCALATED"]`.

**2. LLM infers the wrong evaluation metric from the business problem**

Detection: The inferred metric and its reasoning are stored in `data_card.json` under `priority_metric` and `metric_reason`. The LLM token usage is tracked in `eda_llm_tokens`. A human reviewer can inspect the metric choice at the HITL gate.

Response: The HITL `METRIC_MISMATCH` override category allows a reviewer to explicitly flag and override a wrong metric inference. The override is logged with rationale to `logs/overrides/`.

**3. Top-2 models are too close to call**

Detection: `agent/escalation_rules.py → evaluate_escalation_rules()` triggers `AMBIGUOUS_MODEL_SELECTION` if the top-2 models differ by < 0.005 on the primary metric. The HITL gate risk score increases and the run is routed to Zone 2 or Zone 3.

Response: The human reviewer sees the full model leaderboard and metric breakdown and must provide an explicit override rationale before the recommendation proceeds.

---

## How It Works

MIRA runs three sequential phases. Each phase produces auditable JSON output before the next begins.

```
Phase 1 — EDA.py          →  data_card.json        (data profile + priority metric)
Phase 2 — Modeltrain.py   →  model_selection.json  (model scores + stress test results)
Phase 3 — MIRAAgent       →  recommendation.json   (deployment recommendation)
                          ↓
          EvalRunner       →  eval_report.json      (7-layer evaluation)
```

**Phase 1** cleans the data (IQR outlier capping, column drops, label encoding, StandardScaler) and calls `gpt-4o-mini` via LiteLLM to infer which evaluation metric — recall, precision, F1, or AUC — best matches the business problem text.

**Phase 2** trains all 5 candidate models with stratified 5-fold CV. After training, every model goes through three stress tests: overfitting gap check (threshold 0.10), leakage check (AUC < 0.99), and stability check (CV std < 0.05). The winner and runner-up are selected by the inferred metric. Rejected models are recorded with a reason.

**Phase 3** loads the data card and model selection JSON as context and runs the `mira-recommend` AgentSkill (prompt v0.5.0, DECIDING mode). The LLM reasons over all results and writes a 20-field recommendation including `confidence_score`, `routing_zone`, `flags[]`, and an executive summary with an explicit YES/NO deployment verdict.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  USER INPUT                                                     │
│  dataset_path · target_column · business_problem               │
└───────────────────────────┬─────────────────────────────────────┘
                            │  AgentInput (Pydantic validation)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STREAMLIT UI  (app.py)  or  CLI  (main.py)                     │
│  Orchestrates phase transitions · writes run log                │
└──────────┬────────────────┬────────────────┬────────────────────┘
           │                │                │
           ▼                ▼                ▼
┌──────────────┐  ┌──────────────────┐  ┌─────────────────────────┐
│ PHASE 1      │  │ PHASE 2          │  │ PHASE 3                 │
│ scripts/     │  │ scripts/         │  │ agent/mira_agent.py     │
│ EDA.py       │  │ Modeltrain.py    │  │ OpenHands SDK           │
│              │  │                  │  │ mira-recommend Skill    │
│ - Clean data │  │ - Train 5 models │  │ - Context injection     │
│ - Profile    │  │ - 5-fold CV      │  │ - LLM reasoning         │
│ - LLM metric │  │ - Stress tests   │  │ - 20-field output       │
│   inference  │  │   (overfit,      │  │ - validate_output()     │
│              │  │    leakage,      │  │                         │
│              │  │    stability)    │  │                         │
└──────┬───────┘  └────────┬─────────┘  └────────────┬────────────┘
       │                   │                          │
       ▼                   ▼                          ▼
 data_card.json    model_selection.json       recommendation.json
                                                      │
                                                      ▼
                                         ┌────────────────────────┐
                                         │  HITL GATE             │
                                         │  Zone 1 → auto-proceed │
                                         │  Zone 2 → review       │
                                         │  Zone 3 → escalate     │
                                         └────────────┬───────────┘
                                                      │ approved
                                                      ▼
                                         ┌────────────────────────┐
                                         │  EVAL RUNNER           │
                                         │  evals/eval_runner.py  │
                                         │                        │
                                         │  1. Behavior evals     │
                                         │  2. Quality eval       │
                                         │  3. System eval        │
                                         │  4. Unit tests (18)    │
                                         │  5. HITL gate risk     │
                                         │  6. Prod checklist     │
                                         │  7. LLM judge          │
                                         └────────────┬───────────┘
                                                      ▼
                                              eval_report.json
```

---

## Current Prompt Version

`v0.5.0` — DECIDING mode: reasoning-based confidence score, three-zone HITL gate, `flags[]`, `mira-recommend` AgentSkill.

See [prompts/README.md](prompts/README.md) for full version changelog.

---

## How to Run

**1. Clone and install**
```bash
git clone https://github.com/shibani-11/mis6349-shibani-11-p1.git
cd mis6349-shibani-11-p1
pip install -r requirements.txt
```

**2. Configure your LLM**

Create a `.env` file in the project root:
```
LLM_API_KEY=your_api_key_here
LLM_MODEL=openai/gpt-4o-mini
```

Supported model strings (via LiteLLM): `openai/gpt-4o`, `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-6`, `anthropic/claude-haiku-4-5`

**3. Add your dataset**
```
data/raw/your_dataset.csv      # or .xls / .xlsx
```

**4. Run via Streamlit UI (recommended)**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Upload your dataset, select the target column, describe the business problem, and click Run MIRA.

**5. Run via CLI**
```bash
python main.py --dataset data/raw/Churn_Modelling.csv --target Exited --problem "Identify customers likely to churn."
```

**6. Re-run evals on an existing run**
```bash
python -m evals.run_evals --run-id run_bc710c55
```

**7. Analyse run logs**
```bash
python scripts/analyze_logs.py --log-dir logs/runs/
```

**8. Output locations**
```
processed/
  {run_id}_data_card.json
  {run_id}_model_selection.json
  {run_id}_recommendation.json
  {run_id}_eval_report.json
  {run_id}_cleaned.csv

logs/runs/
  {run_id}_run.json

logs/overrides/
  {run_id}_override.json     # written only when a human reviewer submits a decision
```

---

## Running Tests

```bash
# Unit tests only (18 tests, no LLM required)
python -m pytest evals/unit_tests.py -v

# All tests (25+ tests)
pytest tests/ -v

# Specific test file
pytest tests/test_happy_path.py -v
pytest tests/test_edge_cases.py -v
pytest tests/test_retry.py -v
```

---

## Production Bar Status

| Bar | Status | Notes |
|---|---|---|
| Versioned prompts | ✅ | v0.5.0 active — DECIDING mode, full schema enforcement, changelog in `prompts/README.md` |
| Error handling | ✅ | `agent/runner.py` — `run_with_retry` (3 attempts, backoff), `run_with_validation` (self-correction), `run_with_fallback` (human escalation) |
| Observability | ✅ | `agent/logger.py` — `RunLogger` writes to `logs/runs/`; cost tracking in `_run.json` |
| Scope enforcement | ✅ | `max_iterations=40` in `AgentInput`; explicit tool list in `MIRAAgent._tools()` |
| Output validation | ✅ | `agent/validator.py` — 20-key schema enforced before result accepted; 7-layer eval system |
| Test coverage | ✅ | 4 test files · 25+ tests — happy path, edge cases, retry logic, schema validation |
| HITL gate | ✅ | Three-zone approval gate · override log · rubber-stamp prevention (review duration logged) |
| Confidence threshold | ✅ | Reasoning-based `confidence_score` · zone routing · `flags[]` · `agent/escalation_rules.py` |

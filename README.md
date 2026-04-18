# MIRA — Model Intelligence & Recommendation Agent

**Shibani Kumar**

MIRA is a fully autonomous multi-phase ML agent. Given a dataset and a business problem, it profiles the data, trains and compares models, stress-tests the best candidate, and delivers a deployment recommendation — all in a single continuous run without human intervention between phases.

---

## Architecture Answers

### 1. What exactly does this agent do?

MIRA is a fully autonomous 4-phase ML agent. Given any tabular dataset and a plain-English business problem, it profiles the data, selects and cross-validates candidate models, stress-tests the best candidate for overfitting and data leakage, and produces a structured deployment recommendation with business impact translation — all in a single uninterrupted run driven by a single LLM agent.

It does not require a human to hand off results between phases. Phase transitions are gated by Chain-of-Thought reasoning blocks: the agent must cite specific numbers from the previous phase's output before it is allowed to continue.

---

### 2. What goes in, what comes out?

**Input** — configured via interactive CLI or `AgentInput` directly:

```python
AgentInput(
    dataset_path="data/raw/Churn_Modelling.csv",
    target_column="Exited",
    business_problem="Identify customers likely to churn so retention can intervene early.",
    task_type="auto",
    max_models=5,
    max_iterations=40,
    extra_context={"priority_metric": "roc_auc"}
)
```

Supported formats: `.csv`, `.xlsx`, `.xls`, `.tsv`, `.parquet`, `.json`

**Output** — four files written to `processed/`:

| File | Written by | Key contents |
|---|---|---|
| `{run_id}_data_card.json` | Phase 1 | rows, features, class distribution, imbalance flag, missing values, quality issues, recommended preprocessing |
| `{run_id}_model_selection.json` | Phase 2 + 3 | per-model CV scores (AUC, F1, recall, precision), overfitting gap, leakage flag, stability flag, PASS/FAIL verdict |
| `{run_id}_recommendation.json` | Phase 4 | ranked models, business impact, feature drivers, confidence score, executive summary with explicit YES/NO |
| `{run_id}_eval_report.json` | Eval Runner | scores across all 7 eval layers, HITL decision, production readiness verdict |

---

### 3. What tools used and why?

| Component | Role |
|---|---|
| **LiteLLM** | Routes LLM calls to any provider (`openai/gpt-4o-mini`, `anthropic/claude-*`, etc.) without code changes |
| **Python execution** | The LLM agent writes and runs Python inside `RecommendationAgent` to compute real statistics — no hallucinated numbers |
| **File I/O** | Outputs are written to `processed/` as structured JSON after each phase for auditability |
| **Pydantic `AgentInput`** | Validates all user inputs at the boundary before the agent starts |

No browser, no external APIs, no database connections. The agent operates entirely on local files. The minimal toolset reduces failure surface and keeps runs reproducible.

---

### 4. Success Criteria

MIRA self-evaluates after every run across 7 layers:

| Layer | What it checks | Pass bar |
|---|---|---|
| Behavior Evals | Required fields present, key values computed in each phase | ≥ 70% per phase |
| Quality Eval | Best model ≥ 0.65 AUC, Phase 3 passed, recommendation produced | ≥ 70% |
| System Eval | All 3 phases completed, no errors, run < 1 hour | ≥ 70% |
| Unit Tests | 18 deterministic checks on recommendation schema (keys, types, ranges) | 100% |
| HITL Gate | 7 weighted risk factors — leakage (5), poor performance (4), overfitting (2), low confidence (2), etc. | Risk score < 5 |
| Production Checklist | 7 items; 3 critical items (leakage, overfitting, AUC floor) block deployment if failed | ≥ 6/7 + all critical |
| LLM Judge | Independent LLM grades reasoning quality and schema compliance | Reported separately |

Overall pass = mean of Layers 1–4 + 6. Each run leaves a complete audit trail in `processed/{run_id}_eval_report.json`.

---

### 5. What are the three most likely ways it fails?

**1. LLM skips Phase 3 (stress-testing) and jumps straight to Phase 4**

Detection: `phases_completed` in the orchestrator log will be missing `model_selection` test fields (`overfitting_detected`, `leakage_detected`, `test_verdict`). The System Eval and Behavior Eval will both fail.

Response: The CoT reasoning gate in v0.3.0 forces the agent to explicitly write the overfitting gap, leakage result, and stability check before it can proceed. If the gate is not filled, the phase transition is blocked.

**2. LLM hallucinates metric values instead of computing them**

Detection: Unit Tests check that `confidence_score` is a float in [0, 1], `primary_metric_value` is present, and `selection_reason` is substantive. The Quality Eval cross-checks that the best reported AUC ≥ 0.65. Fabricated numbers will often fail range checks or be internally inconsistent.

Response: The v0.3.0 prompt instructs the agent to execute Python code to compute all statistics before writing any JSON. The HITL gate adds a second layer — low-confidence or borderline results automatically set `requires_human_review = True`.

**3. Output JSON is malformed or missing required fields**

Detection: The Production Checklist item `all_required_fields_present` checks all 17 required keys and flags any that are `None`, empty strings, or empty lists. Unit Test UT-01 (`TestPhase4RequiredKeys`) checks the same set deterministically on every run.

Response: `max_iterations=40` gives the agent budget to retry and self-correct. If fields are still missing after retries, the Production Checklist critical-item failures block the `production_ready` flag from being set.

---

## How It Works

MIRA drives itself through four personas in sequence. Each phase produces auditable output before the next begins.

```
Phase 1 — Data Analyst       →  data_card.json
Phase 2 — ML Engineer        →  model_selection.json
Phase 3 — ML Test Engineer   →  (appends to model_selection.json)
Phase 4 — Data Scientist     →  recommendation.json
```

Phase transitions are governed by **Chain-of-Thought (CoT) reasoning blocks** (AutoML-GPT, Zhang et al. 2023). Before moving to the next phase, the agent must write an explicit reasoning paragraph referencing specific numbers from the previous output. If it cannot fill in those numbers, it has not completed the prior phase and cannot proceed.

---

## Architecture

### Agent Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  USER INPUT                                                     │
│  dataset_path · target_column · business_problem               │
└───────────────────────────┬─────────────────────────────────────┘
                            │  AgentInput (Pydantic validation)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR  (agent/orchestrator.py)                          │
│  Boots the agent · writes run log to logs/runs/                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  RECOMMENDATION AGENT  (agents/recommendation_agent.py)         │
│  Single LLM agent · prompt v0.3.0 · max 40 iterations          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ PHASE 1 — Data Analyst                                   │   │
│  │ • Load dataset, compute shape, class distribution        │   │
│  │ • Detect missing values, outliers, high cardinality      │   │
│  │ • Identify class imbalance, top correlated features      │   │
│  │ • Write ➜  {run_id}_data_card.json                       │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │  CoT Reasoning Gate                 │
│                           │  (must cite rows, minority %, etc.) │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │ PHASE 2 — ML Engineer                                    │   │
│  │ • Select candidate models from pool (LR, RF, XGB, LGBM) │   │
│  │ • 5-fold cross-validation, class_weight='balanced'       │   │
│  │ • Record cv_roc_auc, f1, recall, precision per model     │   │
│  │ • Write ➜  {run_id}_model_selection.json                 │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │  CoT Reasoning Gate                 │
│                           │  (must cite AUC scores, gaps)       │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │ PHASE 3 — ML Test Engineer                               │   │
│  │ • Check overfitting  (train_score − val_score > 0.10?)   │   │
│  │ • Check data leakage (any feature perfectly predictive?) │   │
│  │ • Check stability    (cv_roc_auc_std > 0.05?)            │   │
│  │ • Append test_verdict + findings to model_selection.json │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │  CoT Reasoning Gate                 │
│                           │  (must cite PASS/FAIL + reason)     │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │ PHASE 4 — Data Scientist                                 │   │
│  │ • Rank all models, justify winner with data              │   │
│  │ • Translate to business impact (retention, revenue, etc) │   │
│  │ • Surface top feature drivers in plain English           │   │
│  │ • Write ➜  {run_id}_recommendation.json                  │   │
│  └────────────────────────┬─────────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  EVAL RUNNER  (evals/eval_runner.py)                            │
│                                                                 │
│  Layer 1 · Behavior    per-phase structure checks     ≥ 70%    │
│  Layer 2 · Quality     cross-phase performance floor  ≥ 70%    │
│  Layer 3 · System      orchestrator health & timing   ≥ 70%    │
│  Layer 4 · Unit Tests  18 deterministic schema tests  100%     │
│  Layer 5 · HITL Gate   7 risk factors → human flag    < 5      │
│  Layer 6 · Prod Check  7-item deployment readiness    ≥ 6/7    │
│  Layer 7 · LLM Judge   independent LLM scoring        optional │
│                                                                 │
│  Write ➜  {run_id}_eval_report.json                            │
└─────────────────────────────────────────────────────────────────┘
```


## Current Prompt Version

| Version | File | Changes |
|---|---|---|
| v0.3.0 | `prompts/mira_agent_v0_3_0.md` | CoT reasoning gates between phases (AutoML-GPT style); expanded model_selection and recommendation schemas; model candidate pool with inclusion/exclusion rules |
| v0.2.0 | `prompts/mira_agent_v0_2_0.md` | Unified agentic prompt replacing 4-phase subprocess pipeline |

See `prompts/README.md` for full prompt version history.

---

## How to Run and Setup 

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
data/raw/your_dataset.csv
```

**4. Run MIRA**
```bash
python -m agent.main
```

The CLI will auto-detect your dataset, show column names, and prompt for the target column and business problem.

**5. Re-run evals on the last completed run**
```bash
python -m agent.main evals
```

**6. Output location**
```
processed/
  {run_id}_data_card.json
  {run_id}_model_selection.json
  {run_id}_recommendation.json
  {run_id}_eval_report.json

logs/runs/
  {run_id}_orchestrator.json
```

---

## Running Tests

```bash
# Unit tests only (18 tests, no LLM required)
python -m pytest evals/unit_tests.py -v

# All tests
pytest tests/ -v
```

---

## Production Bar Status

| Bar | Status | Notes |
|---|---|---|
| Versioned prompts | ✅ | v0.3.0 active — CoT-guided, full schema enforcement |
| Error handling | ✅ | Script errors caught and retried by the agent |
| Observability | ✅ | Run logs in `logs/runs/`, eval report in `processed/` |
| Scope enforcement | ✅ | `max_iterations=40` configured in AgentInput |
| Output validation | ✅ | Pydantic input schema + 7-layer eval system |
| Test coverage | ✅ | 18 unit tests + 6 automated eval layers per run |
| CoT reasoning | ✅ | Phase transitions gated on explicit reasoning blocks |
| HITL gate | ✅ | 7 risk factors trigger human review flag automatically |

# Architecture Document
## MIRA — ML Model Recommendation Agent
### Shibani Kumar · MIS6349 · Last updated: May 2026

---

## The 5 Questions

### Question 1: What exactly does this agent do?

Given any tabular classification dataset and a plain-English business problem, MIRA autonomously profiles the data, cross-validates five candidate models, stress-tests the winner for overfitting and data leakage, infers the right evaluation metric from the business problem, and produces a structured deployment recommendation — then routes it through a three-zone human approval gate before finalising the run.

---

### Question 2: What goes in, what comes out?

**Input** — configured via interactive CLI:

```python
AgentInput(
    dataset_path   = "data/raw/your_dataset.csv",   # .csv, .xlsx, .xls, .parquet
    target_column  = "Exited",
    business_problem = "Identify customers likely to churn so retention can intervene.",
    task_type      = "auto",
    max_models     = 5,
    max_iterations = 40,
    run_id         = "run_a1b2c3d4",
    extra_context  = {},
)
```

The evaluation metric (`roc_auc`, `recall`, `f1_score`, or `precision`) is **inferred from the business problem by LLM** — not chosen by the user.

**Output** — four files written to `processed/`:

| File | Written by | Key contents |
|------|------------|--------------|
| `{run_id}_data_card.json` | EDA.py (Phase 1) | rows, features, class distribution, imbalance flag, cleaning log, priority_metric, metric_reason |
| `{run_id}_model_selection.json` | Modeltrain.py (Phase 2) | 5-model CV scores (AUC, F1, recall, precision), overfitting gap, leakage flag, stability flag, stress test verdict |
| `{run_id}_recommendation.json` | mira-recommend skill (Phase 3) | ranked models, confidence_score, routing_zone, flags[], business impact, feature drivers, executive summary with YES/NO |
| `{run_id}_eval_report.json` | EvalRunner (post-gate) | 7-layer eval scores, HITL gate result, production readiness verdict |

**Additional outputs:**

| File | Written by | Contents |
|------|------------|----------|
| `logs/runs/{run_id}_run.json` | MIRAAgent | duration, success, decisions, recommended model, confidence |
| `logs/overrides/{run_id}_override.json` | main.py HITL gate | routing zone, escalation rules, human decision, override category, rationale, review duration |

---

### Question 3: What tools does it need and why?

**Agent Runtime (OpenHands SDK)**

| Tool | What it does | Why needed |
|------|-------------|------------|
| `TerminalTool` | Executes EDA.py and Modeltrain.py; reads output JSON files via `cat` | Scripts produce all statistical outputs — agent cannot compute these itself |
| `FileEditorTool` | Writes recommendation.json | Agent writes the Phase 3 output after reading Phase 1+2 outputs |
| `TaskTrackerTool` | Signals run completion | Terminates the agent loop cleanly when all three phases are done |

Tools explicitly excluded: browser, external APIs, database connections. The agent operates entirely on local files.

**Supporting libraries**

| Library | Role |
|---------|------|
| `openhands-sdk` | Agent runtime (`Agent`, `LLM`, `Conversation`) |
| `litellm` | Routes LLM calls to any provider without code changes |
| `python-dotenv` | Loads `LLM_API_KEY` and `LLM_MODEL` from `.env` |
| `pandas` | Dataset loading, profiling, cleaning in EDA.py |
| `scikit-learn` | Preprocessing, cross-validation, Logistic Regression, Random Forest |
| `xgboost` / `lightgbm` | Model training in Modeltrain.py |
| `pydantic >= 2.0` | `AgentInput` schema validation at system boundary |

**Skill:** `skills/mira-recommend/SKILL.md` — AgentSkills format skill loaded via `Skill.load()` and injected into the agent via `AgentContext(skills=[skill])`. Guides Phase 3 recommendation generation.

---

### Question 4: How will you know it's working correctly?

**7-layer eval system runs automatically after every approved run:**

| Layer | What it checks | Pass bar |
|-------|----------------|---------|
| Behavior Evals | Required fields present, key values in each phase | ≥ 70% per phase |
| Quality Eval | Best model ≥ 0.65 AUC, Phase 3 passed, recommendation produced | ≥ 70% |
| System Eval | All 3 phases completed, no errors, run < 1 hour | ≥ 70% |
| Unit Tests | 18 deterministic checks on recommendation schema | 100% |
| HITL Gate | 7 risk factors → human review flag | Risk score < 5 |
| Production Checklist | 7 items; 3 critical items block `production_ready` if failed | ≥ 6/7 + all critical |
| LLM Judge | Independent LLM grades reasoning quality | Reported separately |

**4-dimension performance targets (Session 4):**

| Dimension | Target | Measurement |
|-----------|--------|-------------|
| Accuracy | ≥ 80% correct recommendations on 10 varied inputs | Manual review of eval reports |
| Latency p50 | < 600s per full run (3 phases + evals) | `duration_seconds` in run log |
| Cost per run | Tracked via token counting | EDA LLM call + agent iterations |
| Hard failure rate | < 5% | Runs with no recommendation produced |
| Silent failure rate | < 10% | Schema violations caught by unit tests |

**Golden dataset:** 28 synthetic test cases across 4 scenario zones validate the eval system itself.

---

### Question 5: What are the three most likely ways it fails?

**1. Agent skips Phase 3 or hangs on the mira-recommend skill**

Detection: `model_selection.json` exists but `recommendation.json` does not after the push loop. System Eval and Behavior Eval both fail.

Response: Push loop fires up to 6 times with explicit step-by-step instructions (`_recommendation_push()`). The prompt note warns: "Do NOT run /mira-recommend as a shell command — it is a skill, not a script."

**2. Confidence score doesn't reflect true uncertainty**

Detection: `confidence_score >= 0.85` on a run where `flags` is not empty — caught by the HITL gate comparison in post-run analysis. The override log's `review_duration_seconds` also surfaces rubber-stamp reviews.

Response: Confidence is now computed from reasoning certainty (winner gap, stability, overfitting, leakage, data size) — not from output format quality. See `skills/mira-recommend/SKILL.md` Step 2b for the exact formula.

**3. Output JSON missing required fields**

Detection: `REQUIRED_KEYS` dict in `mira_agent.py` checks all 20 required recommendation keys after every run. Unit test UT-01 checks the same set. Production checklist item `all_required_fields_present` blocks `production_ready`.

Response: `max_iterations=40` gives budget to self-correct. Schema violation triggers a targeted push message listing the missing keys. If still missing after retries, HITL gate routes to Zone 3 because `confidence_score` will be low.

---

## Agent Architecture

```
┌──────────────────────────────────────────────────────────┐
│  USER INPUT (CLI)                                         │
│  dataset · target_column · business_problem              │
└────────────────────────┬─────────────────────────────────┘
                         │  AgentInput (Pydantic validation)
                         ▼
┌──────────────────────────────────────────────────────────┐
│  MIRAAgent  (agent/mira_agent.py)                         │
│  Loads system prompt v0.5.0 + mira-recommend skill        │
│                                                           │
│  Phase 1 → python3 scripts/EDA.py                        │
│    LLM infers priority_metric from business_problem       │
│    → data_card.json                                       │
│                          CoT gate                         │
│  Phase 2 → python3 scripts/Modeltrain.py                 │
│    Ranks 5 models by inferred priority metric             │
│    → model_selection.json                                 │
│                          CoT gate                         │
│  Phase 3 → mira-recommend skill                          │
│    Computes reasoning confidence_score + flags[]          │
│    Computes routing_zone                                  │
│    → recommendation.json                                  │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  THREE-ZONE HITL GATE  (agent/main.py)                    │
│                                                           │
│  escalation_rules.py checks 8 hard rules                  │
│                                                           │
│  Zone 1 (confidence ≥ 0.85, no flags)                    │
│    → AUTO-PROCEED                                         │
│                                                           │
│  Zone 2 (confidence 0.70-0.84 or soft flags)             │
│    → CLI prompt: model, AUC, confidence, flags            │
│    → Human: yes/no + override category + rationale        │
│                                                           │
│  Zone 3 (confidence < 0.70 or hard escalation rule)      │
│    → CLI prompt: full detail + escalation rule list       │
│    → Human: yes/no + override category + rationale        │
│                                                           │
│  Override log → logs/overrides/{run_id}_override.json    │
└────────────────────────┬─────────────────────────────────┘
                         │  (if approved)
                         ▼
┌──────────────────────────────────────────────────────────┐
│  EVAL RUNNER  (evals/eval_runner.py)                      │
│  7 layers → processed/{run_id}_eval_report.json          │
└──────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
mis6349-shibani-11-p1/
├── agent/
│   ├── main.py                  # CLI entry point + three-zone HITL gate
│   ├── mira_agent.py            # MIRAAgent class — drives the full pipeline
│   └── escalation_rules.py      # 8 hard escalation rules (Session 6)
├── scripts/
│   ├── EDA.py                   # Phase 1: data cleaning + LLM metric inference
│   └── Modeltrain.py            # Phase 2: 5-model CV + stress tests
├── skills/
│   └── mira-recommend/
│       └── SKILL.md             # AgentSkills skill — Phase 3 recommendation
├── evals/
│   ├── eval_runner.py           # Orchestrates 7-layer eval
│   ├── behavior_evals.py        # Phase-level structure checks
│   ├── quality_evals.py         # Cross-phase performance checks
│   ├── system_evals.py          # Completion + timing checks
│   ├── unit_tests.py            # 18 deterministic schema tests
│   ├── hitl_gate.py             # 7-factor risk scoring
│   ├── production_checklist.py  # 7-item deployment readiness
│   ├── judge_agent.py           # Optional LLM judge
│   ├── golden_dataset.json      # 28 test cases across 4 scenario zones
│   └── golden_dataset_runner.py # Runs all 28 cases against eval system
├── prompts/
│   └── mira_agent_v0_5_0.md    # Active system prompt
├── schemas/
│   └── input_schema.py          # Pydantic AgentInput validation
├── docs/
│   ├── architecture.md          # This file
│   ├── approval_gate.md         # Session 6 HITL gate design
│   ├── premortem.md             # Predicted failure modes
│   ├── postmortem.md            # Actual vs predicted failures
│   └── evaluation_report.md    # 4-dimension scorecard
├── logs/
│   ├── runs/                    # {run_id}_run.json per run
│   └── overrides/               # {run_id}_override.json per HITL decision
└── processed/                   # {run_id}_*.json outputs
```

# ML Model Recommendation Agent - Shibani Kumar

## MIS6349 Project 1 · DOING Mode

## What This Agent Does

Analyzes a dataset and associated machine learning model evaluation metrics to generate a structured recommendation identifying the most suitable machine learning model for deployment.

# Architecture Answers

### 1. What exactly does this agent do?

Analyzes dataset characteristics and model evaluation metrics to recommend the best machine learning model for deployment and generate a structured recommendation report.



### 2. Input schema

The input schema is defined in:
schemas/input_schema.py

See:\
[Input Schema](schemas/input_schema.py)

The schema expects:

-   Dataset file path (CSV or Excel)
-   Model evaluation metrics including:
    -   accuracy
    -   precision
    -   recall
    -   F1 score
    -   ROC-AUC

Example input format:

``` json
{
  "dataset_path": "data/loan_default_dataset.xlsx",
  "model_evaluations": [
    {
      "model_name": "logistic_regression",
      "accuracy": 0.85,
      "precision": 0.90,
      "recall": 0.60,
      "f1_score": 0.72,
      "roc_auc": 0.81
    }
  ]
}
```

### 3. Output schema

The output schema is defined in:

schemas/output_schema.py

See:\
[Output Schema](schemas/output_schema.py)

The agent generates a structured recommendation report containing:

-   recommended model
-   justification for selection
-   dataset considerations
-   tradeoffs
-   human review flag

Example output:

``` json
{
  "recommended_model": "xgboost",
  "selection_reason": "Dataset shows moderate class imbalance and XGBoost achieves the highest ROC-AUC and F1 score.",
  "tradeoffs": [
    "Higher model complexity",
    "Lower precision than Logistic Regression"
  ],
  "dataset_considerations": [
    "Class imbalance detected",
    "Recall is important for detecting default cases"
  ],
  "requires_human_review": false
}
```

### 4. Tools used and why

| Tool | Justification |
|------|---------------|
| `FileReadTool` | Reads the dataset file and the model evaluation metrics input file. |
| `IPythonRunCellTool` | Executes Python code to analyze the dataset and compute summary statistics such as class distribution and missing values. |
| `FileWriteTool` | Writes the generated recommendation report to an output file. |


### 5. Success criteria

| Metric | Target | Measurement Method |
|------|------|----------------|
| Accuracy | ≥ 80% correct recommendations across 10 varied inputs | Manual review of run logs |
| Latency p50 | < 10 seconds | scripts/analyze_logs.py |
| Hard failure rate | < 5% | scripts/analyze_logs.py |
| Silent failure rate | < 10% | Output schema validator |

# Production Bar Status

| Bar | Status | Notes |
|-----|--------|-------|
| Versioned prompts | ❌ | Initial prompt version to be added |
| Error handling | ❌ | Retry and validation logic to be implemented |
| Observability | ❌ | Run logs will be stored in `logs/runs/` |
| Scope enforcement | ❌ | `max_iterations` to be configured |
| Output validation | ❌ | Pydantic schema validation planned |
| Test coverage | ❌ | Pytest tests to be added |

# Current Prompt Version

`v0.1.0` --- Initial prompt structure defining reasoning steps for model recommendation.

See `prompts/README.md` for full prompt version history.


# How to Run

1. Clone the repository

```bash
git clone https://github.com/shibani-11/mis6349-shibani-11-p1.git
cd mis6349-shibani-11-p1
```
2.  Install dependencies

pip install -r requirements.txt

3.  Place dataset and model metrics inside the `data/` directory.

4.  Run the agent

python agent/main.py

5.  The generated recommendation report will appear in:

outputs/recommendation_report.json

# How to Run Tests

pytest tests/ -v

# Architecture Document

## ML Model Recommendation Agent - Shibani Kumar

**Last updated:** March 13th, 2026

------------------------------------------------------------------------

# The 5 Questions

## Question 1: What exactly does this agent do?

Analyzes a dataset and associated machine learning model evaluation metrics to generate a structured recommendation report identifying the most suitable model for deployment.

------------------------------------------------------------------------

## Question 2: What goes in, what comes out?

### Input

The agent receives two inputs:

1.  A dataset file (CSV or Excel)
2.  A JSON file containing model evaluation metrics

Example input structure:

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
    },
    {
      "model_name": "random_forest",
      "accuracy": 0.88,
      "precision": 0.84,
      "recall": 0.81,
      "f1_score": 0.82,
      "roc_auc": 0.87
    },
    {
      "model_name": "xgboost",
      "accuracy": 0.89,
      "precision": 0.80,
      "recall": 0.90,
      "f1_score": 0.85,
      "roc_auc": 0.92
    }
  ]
}
```

Before reasoning, the agent uses Python to analyze the dataset and
generate summary statistics such as:

-   number of rows\
-   number of features\
-   class distribution\
-   missing values

Example dataset summary:

``` json
{
  "num_rows": 50000,
  "num_features": 20,
  "class_distribution": {
    "default": 0.25,
    "non_default": 0.75
  },
  "missing_values": {
    "age": 0,
    "income": 3
  }
}
```

These statistics are then used along with model evaluation metrics to
guide the recommendation.

------------------------------------------------------------------------

### Output

The agent produces a structured JSON recommendation report.

Example output:

``` json
{
  "dataset": "loan_default_dataset",
  "recommended_model": "xgboost",
  "selection_reason": "The dataset shows moderate class imbalance. XGBoost achieves the highest ROC-AUC and F1 score, which are important metrics when recall is important for detecting positive cases.",
  "tradeoffs": [
    "Lower precision than Logistic Regression",
    "Higher model complexity"
  ],
  "dataset_considerations": [
    "Dataset is moderately imbalanced",
    "Recall is important for detecting default cases"
  ],
  "requires_human_review": false
}
```

Output constraints:

-   Must be valid JSON\
-   Recommended model must exist in the input model list\
-   Metrics must not be hallucinated\
-   If required inputs are missing, `requires_human_review` must be set
    to true

------------------------------------------------------------------------

## Question 3: What tools does it need and why?

  --------------------------------------------------------------------------
  Tool                 Justification
  -------------------- -----------------------------------------------------
  FileReadTool         Reads the dataset file and model evaluation metrics
                       input file.

  IPythonRunCellTool   Executes Python code to analyze the dataset and
                       compute summary statistics such as class distribution
                       and missing values.

  FileWriteTool        Writes the generated recommendation report to an
                       output file.
  --------------------------------------------------------------------------

### Tools explicitly excluded

BrowserTool\
Not required because the agent operates entirely on local data.

External APIs\
No external services are required for this task.

Database connections\
All inputs are provided as files in the repository.

Keeping the toolset minimal reduces complexity and potential failure
points.

------------------------------------------------------------------------

## Question 4: How will you know it's working correctly?

  --------------------------------------------------------------------------
  Metric          Target            Measurement Method
  --------------- ----------------- ----------------------------------------
  Accuracy        ≥ 80% correct     Manual review of run logs
                  recommendations   
                  across 10 varied  
                  inputs            

  Latency p50     \< 10 seconds     scripts/analyze_logs.py

  Hard failure    \< 5%             scripts/analyze_logs.py
  rate                              

  Silent failure  \< 10%            Output schema validator
  rate                              
  --------------------------------------------------------------------------

Definitions:

Hard failure\
The agent crashes, throws an exception, or produces no output.

Silent failure\
The agent produces output but recommends the wrong model.

Each run is logged with prompt version, input hash, latency, and output
validation results.

------------------------------------------------------------------------

## Question 5: What are the three most likely ways it fails?

See `docs/premortem.md` for the full pre-mortem table.

Top 3 failure modes:

1.  Dataset file cannot be read\
    Detection: Python tool raises a file loading error.\
    Response: Agent logs the failure and stops execution.

2.  Missing or invalid model evaluation metrics\
    Detection: Input schema validation detects missing required fields.\
    Response: Agent flags the case for human review.

3.  Incorrect recommendation due to metric tradeoffs\
    Detection: Compare recommendation against deterministic ranking
    logic.\
    Response: Agent must justify the decision or flag the case for human
    review.

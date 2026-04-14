# agent/agents/ml_tester_agent.py
from agent.agents.base_agent import BaseAgent


class MLTesterAgent(BaseAgent):

    def __init__(self, run_id: str, output_path: str, **kwargs):
        super().__init__(
            phase="model_testing",
            persona_name="ml_test_engineer",
            role_label="ML Test Engineer",
            run_id=run_id,
            output_path=output_path,
            **kwargs
        )

    def build_message(
        self,
        dataset_path: str,
        target_column: str,
        business_problem: str,
        extra_context: dict,
        prior_context: str = ""
    ) -> str:
        priority_metric = extra_context.get("priority_metric", "roc_auc")

        return f"""
You are now executing Phase: MODEL TESTING

Dataset: {dataset_path}
Target Column: {target_column}
Business Problem: {business_problem}
Output Directory: {self.output_path}/
Run ID: {self.run_id}
Priority Metric: {priority_metric}

⚠️ CRITICAL RULES:
- NEVER use file_editor to read the dataset — 10MB limit
- ALWAYS write Python scripts and run via terminal
- Use FULL dataset — NEVER sample
- Write script to: scripts/model_test_{self.run_id}.py

Prior Phase Context:
{prior_context[:2000] if prior_context else "None"}

YOUR TASK:
1. Load model building results from:
   {self.output_path}/{self.run_id}_model_building.json
2. Load exploration results from:
   {self.output_path}/{self.run_id}_data_exploration.json
3. Retrain all models with same preprocessing
4. For each model test:
   a. Confusion matrix (TP, FP, TN, FN)
   b. Overfitting: train vs val gap > 10% = flag it
   c. Leakage: ROC-AUC > 0.99 = investigate
   d. Stability: cross-val std > 0.05 = flag it
   e. Feature importance: do top features make business sense?
5. Rank models by: {priority_metric}
6. Identify top 2 passing models
7. Save to:
   {self.output_path}/{self.run_id}_model_testing.json

OUTPUT JSON must have EXACTLY these keys:
{{
  "models_tested": [list of model names],
  "test_results": [
    {{
      "model_name": str,
      "val_accuracy": float,
      "val_roc_auc": float,
      "val_f1": float,
      "val_recall": float,
      "val_precision": float,
      "confusion_matrix": {{"TP": int, "FP": int, "TN": int, "FN": int}},
      "train_val_gap": float,
      "overfitting_detected": bool,
      "leakage_suspected": bool,
      "stability_ok": bool,
      "features_business_logical": bool,
      "passed_testing": bool,
      "fail_reasons": [list of strings]
    }}
  ],
  "top_models": [top 2 model names],
  "flagged_models": [list of model names],
  "primary_metric": "{priority_metric}",
  "confidence_score": float between 0.0 and 1.0,
  "genai_narrative": "plain English summary for technical lead"
}}
"""
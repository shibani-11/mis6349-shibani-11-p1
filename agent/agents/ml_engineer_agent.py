# agent/agents/ml_engineer_agent.py
from agent.agents.base_agent import BaseAgent


class MLEngineerAgent(BaseAgent):

    def __init__(self, run_id: str, output_path: str, **kwargs):
        super().__init__(
            phase="model_building",
            persona_name="ml_engineer",
            role_label="ML Engineer",
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
        prior_context: str = "",
        retry: bool = False
    ) -> str:
        retry_note = ""
        if retry:
            retry_note = """
⚠️ RETRY ATTEMPT:
Previous model building failed or underperformed.
Try different approaches:
- Use different model families
- Try simpler preprocessing
- Adjust class imbalance strategy
- Use GradientBoosting if XGBoost failed
"""
        priority_metric = extra_context.get("priority_metric", "roc_auc")

        return f"""
You are now executing Phase: MODEL BUILDING
{retry_note}

Dataset: {dataset_path}
Target Column: {target_column}
Business Problem: {business_problem}
Output Directory: {self.output_path}/
Run ID: {self.run_id}
Priority Metric: {priority_metric}

⚠️ CRITICAL RULES:
- NEVER use file_editor to read the dataset — 10MB limit
- ALWAYS write Python scripts and run via terminal
- Use the FULL dataset — NEVER sample
- Write script to: scripts/model_build_{self.run_id}.py

Prior Phase Context:
{prior_context[:2000] if prior_context else "None"}

YOUR TASK:
1. Load exploration results from:
   {self.output_path}/{self.run_id}_data_exploration.json
2. Load and preprocess FULL dataset:
   - Drop ID columns and single-value columns
   - OrdinalEncoder for categorical columns
   - StandardScaler for numeric columns
   - Handle class imbalance: class_weight='balanced'
   - Stratified 80/20 train/val split (random_state=42)
3. Select and train up to 5 models autonomously
4. Always include Logistic Regression as baseline
5. Use 5-fold cross-validation for each model
6. Save results to:
   {self.output_path}/{self.run_id}_model_building.json

OUTPUT JSON must have EXACTLY these keys:
{{
  "models_considered": [list of names],
  "models_evaluated": [
    {{
      "model_name": str,
      "model_family": str,
      "accuracy": float,
      "precision": float,
      "recall": float,
      "f1_score": float,
      "roc_auc": float,
      "training_time_seconds": float,
      "cross_val_score_mean": float,
      "cross_val_score_std": float,
      "overfitting_detected": bool,
      "feature_count_used": int
    }}
  ],
  "models_skipped": {{model_name: reason}},
  "primary_metric": "{priority_metric}",
  "best_model_by_metric": {{metric: model_name}},
  "feature_importance": {{feature: score}} or null,
  "preprocessing_steps": [list of steps],
  "confidence_score": float between 0.0 and 1.0,
  "genai_narrative": "plain English summary"
}}
"""
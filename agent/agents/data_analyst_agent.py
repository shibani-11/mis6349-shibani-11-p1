# agent/agents/data_analyst_agent.py
from agent.agents.base_agent import BaseAgent


class DataAnalystAgent(BaseAgent):

    def __init__(self, run_id: str, output_path: str, **kwargs):
        super().__init__(
            phase="data_exploration",
            persona_name="data_analyst",
            role_label="Data Analyst",
            run_id=run_id,
            output_path=output_path,
            **kwargs
        )

    def build_message(
        self,
        dataset_path: str,
        target_column: str,
        business_problem: str,
        extra_context: dict
    ) -> str:
        return f"""
You are now executing Phase: DATA EXPLORATION

Dataset: {dataset_path}
Target Column: {target_column}
Business Problem: {business_problem}
Output Directory: {self.output_path}/
Run ID: {self.run_id}
Extra Context: {extra_context}

⚠️ CRITICAL RULES:
- NEVER use file_editor to read the dataset — 10MB limit
- ALWAYS write a Python script and run via terminal
- Use the FULL dataset — NEVER sample or truncate
- Write script to: scripts/explore_{self.run_id}.py

YOUR TASK:
1. Write and run a Python script that:
   - Reads the FULL dataset using pandas
   - Profiles every column: dtype, nulls, unique values, samples
   - Analyzes target column ({target_column}) distribution
   - Detects class imbalance (flag if minority < 20%)
   - Identifies data quality issues
   - Computes numeric statistics (mean, std, min, max)
2. Save output to:
   {self.output_path}/{self.run_id}_data_exploration.json

OUTPUT JSON must have EXACTLY these keys:
{{
  "row_count": int,
  "column_count": int,
  "inferred_task_type": "classification" or "regression",
  "overall_missing_pct": float,
  "duplicate_row_count": int,
  "target_distribution": {{"0": int, "1": int}},
  "class_imbalance_detected": bool,
  "imbalance_ratio": float,
  "numeric_columns": [list of names],
  "categorical_columns": [list of names],
  "id_columns": [list of names],
  "text_columns": [list of names],
  "columns": [list of column profiles],
  "numeric_stats": {{col_name: {{mean, std, min, max}}}},
  "quality_issues": [list of strings],
  "confidence_score": float between 0.0 and 1.0,
  "genai_narrative": "2-3 sentence plain English summary"
}}
"""
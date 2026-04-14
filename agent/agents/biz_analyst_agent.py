# agent/agents/biz_analyst_agent.py
from agent.agents.base_agent import BaseAgent


class BizAnalystAgent(BaseAgent):

    def __init__(self, run_id: str, output_path: str, **kwargs):
        super().__init__(
            phase="recommendation",
            persona_name="business_analyst",
            role_label="Business Analyst",
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
You are now executing Phase: RECOMMENDATION

Dataset: {dataset_path}
Target Column: {target_column}
Business Problem: {business_problem}
Output Directory: {self.output_path}/
Run ID: {self.run_id}

Prior Phase Context:
{prior_context[:3000] if prior_context else "None"}

YOUR TASK:
1. Load ALL prior phase results:
   - {self.output_path}/{self.run_id}_data_exploration.json
   - {self.output_path}/{self.run_id}_model_building.json
   - {self.output_path}/{self.run_id}_model_testing.json
2. Select best model that PASSED testing
3. Translate ALL metrics into business language
4. Write boardroom-ready recommendation
5. Save to:
   {self.output_path}/{self.run_id}_recommendation.json
6. Also save full combined report to:
   {self.output_path}/{self.run_id}_report.json

OUTPUT JSON must have EXACTLY these keys:
{{
  "recommended_model": str,
  "selection_reason": str (business language, no jargon),
  "primary_metric_value": float,
  "tradeoffs": [list of 3 honest tradeoffs],
  "alternative_model": str,
  "next_steps": [list of 3 concrete actions],
  "deployment_considerations": [list],
  "risks": [list],
  "confidence_score": float between 0.0 and 1.0,
  "requires_human_review": bool,
  "human_review_reason": str or null,
  "executive_summary": str
}}

RULES FOR executive_summary:
- Maximum 1 paragraph
- Zero technical jargon
- No ROC-AUC, F1, precision, recall as raw terms
- Translate: "catches X% of likely defaulters"
- End with clear YES or NO on proceeding to deployment
"""
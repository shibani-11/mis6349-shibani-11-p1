# evals/judge_agent.py
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from openhands.sdk import LLM, Agent, Conversation
from agent.tools import get_tools

load_dotenv()

_JARGON = [
    "ROC-AUC", "F1 score", "F1-score", "precision score",
    "recall score", "AUC", "AUROC", "confusion matrix",
]

_REQUIRED_KEYS = [
    "recommended_model", "selection_reason", "primary_metric_value",
    "tradeoffs", "alternative_model", "next_steps",
    "deployment_considerations", "risks", "confidence_score",
    "requires_human_review", "human_review_reason", "executive_summary",
]

_SYSTEM_PROMPT = (
    "You are MIRA Judge, an expert ML systems evaluator. "
    "Critically assess AI-generated model recommendations across five dimensions: "
    "technical accuracy, business communication, completeness, consistency, and actionability. "
    "Be strict and specific. Flag every deficiency. Respond with valid JSON only."
)


class JudgeAgent:
    """LLM-as-a-Judge: evaluates MIRA Phase 4 recommendation quality.

    Runs via OpenHands SDK and writes a structured judge report JSON.
    Scoring: 0-10 per dimension; overall >= 7.0 => passed.
    Verdict: APPROVED | NEEDS_REVISION | REJECTED
    """

    def __init__(self, run_id: str, output_path: str):
        self.run_id = run_id
        self.output_path = Path(output_path)
        self.output_file = self.output_path / f"{run_id}_judge_report.json"
        self.duration = 0.0
        self.success = False

        self.llm = LLM(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o"),
            api_key=os.getenv("LLM_API_KEY"),
        )

    def _load(self, phase: str) -> dict:
        f = self.output_path / f"{self.run_id}_{phase}.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def _build_message(
        self,
        exploration: dict,
        building: dict,
        testing: dict,
        recommendation: dict,
    ) -> str:
        models = building.get("models_evaluated", [])
        test_results = testing.get("test_results", [])
        best = max(models, key=lambda m: m.get("roc_auc", 0), default={})
        scores_map = {
            m.get("model_name", "?"): round(m.get("roc_auc", 0), 4)
            for m in models
        }

        return f"""
Evaluate the MIRA Phase 4 recommendation below against verified pipeline ground truth.

=== GROUND TRUTH (verified) ===
Dataset rows        : {exploration.get('row_count', 'N/A')}
Class imbalance     : {exploration.get('class_imbalance_detected', 'N/A')}
Models (ROC-AUC)    : {json.dumps(scores_map)}
Objectively best    : {best.get('model_name', 'N/A')} @ {best.get('roc_auc', 0):.4f}
Top models (testing): {testing.get('top_models', [])}
Flagged models      : {testing.get('flagged_models', [])}
Leakage detected    : {any(r.get('leakage_suspected', False) for r in test_results)}
Overfitting detected: {any(r.get('overfitting_detected', False) for r in test_results)}

=== RECOMMENDATION TO EVALUATE ===
{json.dumps(recommendation, indent=2)[:3500]}

=== SCORING RUBRIC (0-10 each) ===
accuracy         — Did the agent recommend the objectively best model that passed testing?
business_language — Is executive_summary truly free of jargon ({', '.join(_JARGON[:4])} etc.)?
completeness     — Are all {len(_REQUIRED_KEYS)} required keys present, non-null, and substantive?
consistency      — Do stated tradeoffs/risks accurately reflect the actual metric data?
actionability    — Are next_steps specific enough to execute (not vague platitudes)?

=== OUTPUT INSTRUCTIONS ===
Write the following JSON exactly to: {self.output_file}

{{
  "run_id": "{self.run_id}",
  "dimensions": {{
    "accuracy":          {{"score": <int 0-10>, "reasoning": "<1-2 sentences>"}},
    "business_language": {{"score": <int 0-10>, "reasoning": "<1-2 sentences>"}},
    "completeness":      {{"score": <int 0-10>, "reasoning": "<1-2 sentences>"}},
    "consistency":       {{"score": <int 0-10>, "reasoning": "<1-2 sentences>"}},
    "actionability":     {{"score": <int 0-10>, "reasoning": "<1-2 sentences>"}}
  }},
  "overall_score": <float, mean of 5 dimension scores>,
  "passed": <bool, true if overall_score >= 7.0>,
  "critical_issues": [<issue strings for any dimension scoring < 5>],
  "strengths": [<dimension names scoring >= 8>],
  "verdict": "<APPROVED|NEEDS_REVISION|REJECTED>"
}}

Use the file editor tool to write this JSON to the path above. Do nothing else.
"""

    def run(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  AGENT    : LLM Judge")
        print(f"  TASK     : Evaluate Phase 4 Recommendation")
        print(f"  RUN ID   : {self.run_id}")
        print(f"{'='*60}")

        exploration = self._load("data_exploration")
        building = self._load("model_building")
        testing = self._load("model_testing")
        recommendation = self._load("recommendation")

        if not recommendation:
            print("  ✗ No recommendation output found — skipping judge")
            return {
                "run_id": self.run_id,
                "error": "no_recommendation",
                "passed": False,
                "verdict": "SKIPPED",
            }

        agent = Agent(
            llm=self.llm,
            tools=get_tools(),
            system_prompt=_SYSTEM_PROMPT,
            max_iterations=15,
        )

        conversation = Conversation(agent=agent, workspace=os.getcwd())

        start = time.time()
        conversation.send_message(
            self._build_message(exploration, building, testing, recommendation)
        )
        conversation.run()
        self.duration = round(time.time() - start, 2)

        if self.output_file.exists():
            self.success = True
            result = json.loads(self.output_file.read_text())
            print(f"\n  ✓ Judge completed in {self.duration}s")
            print(f"  Verdict : {result.get('verdict', 'N/A')}  "
                  f"Score: {result.get('overall_score', 'N/A')}/10")
            return result

        self.success = False
        print("  ✗ Judge failed — no output file written")
        return {
            "run_id": self.run_id,
            "error": "no_output",
            "passed": False,
            "verdict": "ERROR",
        }

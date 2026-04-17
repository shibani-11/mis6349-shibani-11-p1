# agent/orchestrator.py
# v0.2.0 — Agentic orchestrator. Replaces the 4-phase subprocess pipeline
# with a single unified agent run. The LLM drives all phases internally.

import json
import time
from pathlib import Path
from datetime import datetime

from agent.agents import RecommendationAgent
from schemas.input_schema import AgentInput


class MIRAOrchestrator:
    """
    MIRA Orchestrator v0.2.0

    Boots the unified RecommendationAgent with the dataset and business
    problem, then steps back. The agent decides what to explore, which
    models to train, and when it has enough evidence to recommend.

    Output files written by the agent:
      {run_id}_data_card.json       — data profiling + analyst findings
      {run_id}_model_selection.json — training results + test verdict
      {run_id}_recommendation.json  — final deployment recommendation
    """

    def __init__(self, agent_input: AgentInput):
        self.input = agent_input
        self.run_id = agent_input.run_id
        self.output_path = agent_input.output_path
        self.extra_context = agent_input.extra_context or {}
        self.decisions = []
        self.total_start = time.time()

        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path("logs/runs").mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  🚀 MIRA — Model Intelligence & Recommendation Agent")
        print(f"  Agentic System v0.2.0")
        print(f"{'='*60}")
        print(f"  Run ID   : {self.run_id}")
        print(f"  Dataset  : {self.input.dataset_path}")
        print(f"  Target   : {self.input.target_column}")
        print(f"  Problem  : {self.input.business_problem[:55]}...")
        print(f"{'='*60}\n")

    def _log_decision(self, agent: str, decision: str, reason: str):
        entry = {
            "agent": agent,
            "decision": decision,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.decisions.append(entry)
        print(f"\n  🧠 {agent}: {decision}")
        print(f"     {reason}\n")

    def run(self) -> dict:
        """
        Boot the unified agent and let it drive all phases.
        Returns a dict with keys: data_card, model_selection, recommendation.
        """
        priority_metric = self.extra_context.get("priority_metric", "roc_auc")
        domain = self.extra_context.get("domain", "general")

        agent = RecommendationAgent(
            run_id=self.run_id,
            output_path=self.output_path,
            max_iterations=self.input.max_iterations,
        )

        message = agent.build_message(
            dataset_path=self.input.dataset_path,
            target_column=self.input.target_column,
            business_problem=self.input.business_problem,
            priority_metric=priority_metric,
            domain=domain,
        )

        print("📋 ORCHESTRATOR: Starting unified agent run\n")
        print("   The agent will self-direct through:")
        print("   Phase 1 → Data Analyst  (data_card.json)")
        print("   Phase 2 → ML Engineer   (model_selection.json)")
        print("   Phase 3 → ML Test Engineer (appends to model_selection.json)")
        print("   Phase 4 → Data Scientist (recommendation.json)\n")

        outputs = agent.run(message, self.input.business_problem)
        metrics = agent.get_metrics()

        total_duration = round(time.time() - self.total_start, 2)

        recommendation = outputs.get("recommendation", {})
        if recommendation:
            self._log_decision(
                "MIRA Agent",
                "COMPLETE ✅",
                f"Recommended: {recommendation.get('recommended_model', 'N/A')}, "
                f"Confidence: {recommendation.get('confidence_score', 'N/A')}, "
                f"Deploy: {'YES' if not recommendation.get('requires_human_review') else 'REVIEW NEEDED'}"
            )
        else:
            self._log_decision(
                "MIRA Agent",
                "INCOMPLETE ✗",
                "Agent did not produce a recommendation — check logs"
            )

        self._write_log(outputs, metrics, total_duration)

        print(f"\n{'='*60}")
        print(f"  ✅ MIRA Complete in {total_duration}s")
        print(f"  📁 Reports: {self.output_path}")
        print(f"{'='*60}\n")

        return outputs

    def _write_log(self, outputs: dict, metrics: dict, total_duration: float):
        recommendation = outputs.get("recommendation", {})
        data_card = outputs.get("data_card", {})
        model_sel = outputs.get("model_selection", {})

        log = {
            "agent": "MIRA Orchestrator v0.2.0",
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_path": self.input.dataset_path,
            "target_column": self.input.target_column,
            "business_problem": self.input.business_problem,
            "total_duration_seconds": total_duration,
            "outputs_produced": {
                "data_card": bool(data_card),
                "model_selection": bool(model_sel),
                "recommendation": bool(recommendation),
            },
            "recommended_model": recommendation.get("recommended_model"),
            "confidence_score": recommendation.get("confidence_score"),
            "requires_human_review": recommendation.get("requires_human_review"),
            "test_verdict": model_sel.get("test_verdict"),
            "agent_metrics": metrics,
            "orchestrator_decisions": self.decisions,
        }

        log_path = Path("logs/runs") / f"{self.run_id}_orchestrator.json"
        log_path.write_text(json.dumps(log, indent=2))
        print(f"  📋 Log: {log_path}")

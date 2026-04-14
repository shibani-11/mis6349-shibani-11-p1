# agent/orchestrator.py
import json
import time
from pathlib import Path
from datetime import datetime

from agent.agents import (
    DataAnalystAgent,
    MLEngineerAgent,
    MLTesterAgent,
    BizAnalystAgent,
)
from schemas.input_schema import AgentInput


class MIRAOrchestrator:
    """
    MIRA Orchestrator — manages sequential agent delegation.

    Makes autonomous decisions at each phase:
    - Should we proceed or retry?
    - Is the output sufficient?
    - What context should the next agent receive?
    """

    MIN_CONFIDENCE = 0.6
    MAX_RETRIES = 2
    MIN_ROC_AUC = 0.60
    MIN_ROWS = 10000

    def __init__(self, agent_input: AgentInput):
        self.input = agent_input
        self.run_id = agent_input.run_id
        self.output_path = agent_input.output_path
        self.extra_context = agent_input.extra_context or {}
        self.agent_metrics = []
        self.decisions = []
        self.total_start = time.time()

        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path("logs/runs").mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  🚀 Starting MIRA — Model Intelligence & Recommendation Agent")
        print(f"  Multi-Agent System")
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
        print(f"\n  🧠 Orchestrator → {agent}: {decision}")
        print(f"     Reason: {reason}\n")

    def _load_output(self, phase: str) -> dict:
        f = Path(self.output_path) / f"{self.run_id}_{phase}.json"
        if f.exists():
            return json.loads(f.read_text())
        return {}

    def _get_prior_context(self, phases: list) -> str:
        parts = []
        for phase in phases:
            output = self._load_output(phase)
            if output:
                narrative = output.get("genai_narrative", "")
                if narrative:
                    parts.append(f"[{phase.upper()}]: {narrative}")
        return "\n\n".join(parts)

    def run_data_exploration(self) -> dict:
        agent = DataAnalystAgent(
            run_id=self.run_id,
            output_path=self.output_path,
            max_iterations=self.input.max_iterations,
        )
        message = agent.build_message(
            dataset_path=self.input.dataset_path,
            target_column=self.input.target_column,
            business_problem=self.input.business_problem,
            extra_context=self.extra_context,
        )

        output = {}
        for attempt in range(1, self.MAX_RETRIES + 1):
            if attempt > 1:
                print(f"\n  🔄 Retry {attempt} — Data Analyst")

            output = agent.run(message, self.input.business_problem)
            self.agent_metrics.append(agent.get_metrics())

            if not output:
                self._log_decision("Data Analyst", "RETRY",
                                   "No output produced")
                continue

            row_count = output.get("row_count", 0)
            confidence = output.get("confidence_score", 0)

            if row_count < self.MIN_ROWS:
                self._log_decision("Data Analyst", "RETRY",
                                   f"Only {row_count} rows — needs full dataset")
                continue

            if confidence < self.MIN_CONFIDENCE:
                self._log_decision("Data Analyst", "RETRY",
                                   f"Confidence {confidence} too low")
                continue

            self._log_decision("Data Analyst", "PROCEED ✅",
                               f"{row_count} rows analyzed, confidence {confidence}")
            break

        return output

    def run_model_building(self, exploration_output: dict) -> dict:
        prior_context = self._get_prior_context(["data_exploration"])
        agent = MLEngineerAgent(
            run_id=self.run_id,
            output_path=self.output_path,
            max_iterations=self.input.max_iterations,
        )

        output = {}
        retry = False

        for attempt in range(1, self.MAX_RETRIES + 1):
            if attempt > 1:
                print(f"\n  🔄 Retry {attempt} — ML Engineer")
                retry = True

            message = agent.build_message(
                dataset_path=self.input.dataset_path,
                target_column=self.input.target_column,
                business_problem=self.input.business_problem,
                extra_context=self.extra_context,
                prior_context=prior_context,
                retry=retry,
            )

            output = agent.run(message, self.input.business_problem)
            self.agent_metrics.append(agent.get_metrics())

            if not output:
                self._log_decision("ML Engineer", "RETRY",
                                   "No output produced")
                continue

            models = output.get("models_evaluated", [])
            if not models:
                self._log_decision("ML Engineer", "RETRY",
                                   "No models trained")
                continue

            priority = self.extra_context.get("priority_metric", "roc_auc")
            best_score = max(
                (m.get(priority, 0) for m in models), default=0
            )

            if best_score < self.MIN_ROC_AUC:
                self._log_decision(
                    "ML Engineer", "RETRY",
                    f"Best {priority} ({best_score:.3f}) below "
                    f"threshold {self.MIN_ROC_AUC}"
                )
                continue

            self._log_decision(
                "ML Engineer", "PROCEED ✅",
                f"{len(models)} models trained, "
                f"best {priority}: {best_score:.3f}"
            )
            break

        return output

    def run_model_testing(self, building_output: dict) -> dict:
        prior_context = self._get_prior_context([
            "data_exploration", "model_building"
        ])
        agent = MLTesterAgent(
            run_id=self.run_id,
            output_path=self.output_path,
            max_iterations=self.input.max_iterations,
        )
        message = agent.build_message(
            dataset_path=self.input.dataset_path,
            target_column=self.input.target_column,
            business_problem=self.input.business_problem,
            extra_context=self.extra_context,
            prior_context=prior_context,
        )

        output = agent.run(message, self.input.business_problem)
        self.agent_metrics.append(agent.get_metrics())

        if not output:
            self._log_decision("ML Test Engineer", "WARNING ⚠️",
                               "No output — proceeding with caution")
        else:
            top = output.get("top_models", [])
            flagged = output.get("flagged_models", [])
            self._log_decision(
                "ML Test Engineer", "PROCEED ✅",
                f"Top models: {top}, Flagged: {flagged}"
            )

        return output

    def run_recommendation(self, testing_output: dict) -> dict:
        prior_context = self._get_prior_context([
            "data_exploration", "model_building", "model_testing"
        ])
        agent = BizAnalystAgent(
            run_id=self.run_id,
            output_path=self.output_path,
            max_iterations=self.input.max_iterations,
        )
        message = agent.build_message(
            dataset_path=self.input.dataset_path,
            target_column=self.input.target_column,
            business_problem=self.input.business_problem,
            extra_context=self.extra_context,
            prior_context=prior_context,
        )

        output = agent.run(message, self.input.business_problem)
        self.agent_metrics.append(agent.get_metrics())

        if output:
            self._log_decision(
                "Business Analyst", "COMPLETE ✅",
                f"Recommended: {output.get('recommended_model', 'N/A')}, "
                f"Confidence: {output.get('confidence_score', 'N/A')}"
            )

        return output

    def run(self) -> dict:
        results = {}

        print("📋 ORCHESTRATOR: Phase 1 — Data Exploration")
        exploration = self.run_data_exploration()
        results["data_exploration"] = exploration
        if not exploration:
            print("  ✗ Data Exploration failed — stopping")
            self._write_log(results)
            return results

        print("📋 ORCHESTRATOR: Phase 2 — Model Building")
        building = self.run_model_building(exploration)
        results["model_building"] = building
        if not building:
            print("  ✗ Model Building failed — stopping")
            self._write_log(results)
            return results

        print("📋 ORCHESTRATOR: Phase 3 — Model Testing")
        testing = self.run_model_testing(building)
        results["model_testing"] = testing

        print("📋 ORCHESTRATOR: Phase 4 — Recommendation")
        recommendation = self.run_recommendation(testing)
        results["recommendation"] = recommendation

        total_duration = round(time.time() - self.total_start, 2)
        self._write_log(results, total_duration=total_duration)

        print(f"\n{'='*60}")
        print(f"  ✅ MIRA Complete in {total_duration}s")
        print(f"  📁 Reports: {self.output_path}")
        print(f"{'='*60}\n")

        return results

    def _write_log(self, results: dict, total_duration: float = 0):
        log = {
            "agent": "MIRA Orchestrator",
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_path": self.input.dataset_path,
            "target_column": self.input.target_column,
            "business_problem": self.input.business_problem,
            "phases_completed": [k for k, v in results.items() if v],
            "phases_failed": [k for k, v in results.items() if not v],
            "total_duration_seconds": total_duration,
            "agent_metrics": self.agent_metrics,
            "orchestrator_decisions": self.decisions,
        }
        log_path = Path("logs/runs") / f"{self.run_id}_orchestrator.json"
        log_path.write_text(json.dumps(log, indent=2))
        print(f"  📋 Orchestrator log: {log_path}")
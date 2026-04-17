# agent/orchestrator.py
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

from agent.agents import RecommendationAgent, NarrativeAgent
from schemas.input_schema import AgentInput


class MIRAOrchestrator:
    """
    MIRA Orchestrator — manages sequential agent delegation.

    Phases 1-3 use template scripts directly (fast, reliable).
    Phase 4 uses LLM agent for business reasoning.
    """

    MIN_CONFIDENCE = 0.6
    MAX_RETRIES = 2
    MIN_ROC_AUC = 0.60
    MIN_ROWS = 1000

    def __init__(self, agent_input: AgentInput):
        self.input = agent_input
        self.run_id = agent_input.run_id
        self.output_path = agent_input.output_path
        self.extra_context = agent_input.extra_context or {}
        self.agent_metrics = []
        self.decisions = []
        self.total_start = time.time()

        self._narrative = NarrativeAgent()

        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path("logs/runs").mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  🚀 Starting MIRA — Model Intelligence & Recommendation Agent")
        print(f"  Multi-Agent System v2.0")
        print(f"{'='*60}")
        print(f"  Run ID   : {self.run_id}")
        print(f"  Dataset  : {self.input.dataset_path}")
        print(f"  Target   : {self.input.target_column}")
        print(f"  Problem  : {self.input.business_problem[:55]}...")
        print(f"  Phases   : {self.input.analysis_phases}")
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

    def _track_metrics(
        self, phase: str, role: str,
        success: bool, confidence: float,
        output_file: str, errors: list = None
    ):
        self.agent_metrics.append({
            "phase": phase,
            "role": role,
            "success": success,
            "duration_seconds": 0,
            "confidence": confidence,
            "output_file": output_file,
            "errors": errors or []
        })

    # ── Phase 1: Data Exploration ──────────────────────────────

    def run_data_exploration(self) -> dict:
        """Run data exploration using template script."""
        output_file = (
            f"{self.output_path}{self.run_id}_data_exploration.json"
        )

        print(f"\n{'='*60}")
        print(f"  AGENT    : Data Analyst")
        print(f"  PHASE    : DATA EXPLORATION")
        print(f"  TEMPLATE : scripts/data_profiler.py")
        print(f"{'='*60}")

        start = time.time()
        result = subprocess.run(
            [
                "python3", "scripts/data_profiler.py",
                self.input.dataset_path,
                self.input.target_column,
                output_file
            ],
            capture_output=False
        )
        duration = round(time.time() - start, 2)

        if result.returncode == 0 and Path(output_file).exists():
            output = json.loads(Path(output_file).read_text())
            row_count  = output.get("row_count", 0)
            confidence = output.get("confidence_score", 0.9)

            print(f"\n  ✓ Data Exploration complete in {duration}s")
            print(f"  📄 Output: {output_file}")

            print(f"  🗣  NarrativeAgent: generating Data Analyst narrative...")
            output["genai_narrative"] = self._narrative.generate(
                "data_exploration", output,
                self.input.business_problem, "data_analyst"
            )
            Path(output_file).write_text(json.dumps(output, indent=2))

            self._log_decision(
                "Data Analyst", "PROCEED ✅",
                f"{row_count:,} rows analyzed, "
                f"confidence: {confidence}"
            )
            self._track_metrics(
                "data_exploration", "Data Analyst",
                True, confidence, output_file
            )
            return output
        else:
            print(f"\n  ✗ Data Exploration failed")
            self._log_decision(
                "Data Analyst", "FAILED ✗",
                "Template script returned non-zero exit code"
            )
            self._track_metrics(
                "data_exploration", "Data Analyst",
                False, 0, output_file,
                ["Template script failed"]
            )
            return {}

    # ── Phase 2: Model Building ────────────────────────────────

    def run_model_building(self, exploration_output: dict) -> dict:
        """Run model building using template script."""
        output_file = (
            f"{self.output_path}{self.run_id}_model_building.json"
        )
        priority = self.extra_context.get("priority_metric", "roc_auc")

        print(f"\n{'='*60}")
        print(f"  AGENT    : ML Engineer")
        print(f"  PHASE    : MODEL BUILDING")
        print(f"  TEMPLATE : scripts/model_trainer.py")
        print(f"{'='*60}")

        start = time.time()

        for attempt in range(1, self.MAX_RETRIES + 1):
            if attempt > 1:
                print(f"\n  🔄 Retry {attempt}/{self.MAX_RETRIES}")

            result = subprocess.run(
                [
                    "python3", "scripts/model_trainer.py",
                    self.input.dataset_path,
                    self.input.target_column,
                    output_file,
                    priority
                ],
                capture_output=False
            )

            duration = round(time.time() - start, 2)

            if result.returncode != 0:
                self._log_decision(
                    "ML Engineer", "RETRY",
                    f"Template failed on attempt {attempt}"
                )
                continue

            if not Path(output_file).exists():
                self._log_decision(
                    "ML Engineer", "RETRY",
                    "Output file not created"
                )
                continue

            output = json.loads(Path(output_file).read_text())
            models = output.get("models_evaluated", [])

            if not models:
                self._log_decision(
                    "ML Engineer", "RETRY",
                    "No models were trained"
                )
                continue

            best_score = max(
                (m.get(priority, 0) for m in models), default=0
            )

            if best_score < self.MIN_ROC_AUC:
                self._log_decision(
                    "ML Engineer", "WARNING ⚠️",
                    f"Best {priority} ({best_score:.3f}) below "
                    f"threshold {self.MIN_ROC_AUC} — proceeding anyway"
                )

            print(f"\n  ✓ Model Building complete in {duration}s")
            print(f"  📄 Output: {output_file}")

            print(f"  🗣  NarrativeAgent: generating Data Analyst narrative...")
            output["genai_narrative"] = self._narrative.generate(
                "model_building", output,
                self.input.business_problem, "data_analyst"
            )
            Path(output_file).write_text(json.dumps(output, indent=2))

            self._log_decision(
                "ML Engineer", "PROCEED ✅",
                f"{len(models)} models trained, "
                f"best {priority}: {best_score:.3f}"
            )
            self._track_metrics(
                "model_building", "ML Engineer",
                True, output.get("confidence_score", 0.85),
                output_file
            )
            return output

        # All retries failed
        print(f"\n  ✗ Model Building failed after {self.MAX_RETRIES} attempts")
        self._track_metrics(
            "model_building", "ML Engineer",
            False, 0, output_file,
            [f"Failed after {self.MAX_RETRIES} attempts"]
        )
        return {}

    # ── Phase 3: Model Testing ─────────────────────────────────

    def run_model_testing(self, building_output: dict) -> dict:
        """Run model testing using template script."""
        building_file = (
            f"{self.output_path}{self.run_id}_model_building.json"
        )
        output_file = (
            f"{self.output_path}{self.run_id}_model_testing.json"
        )
        priority = self.extra_context.get("priority_metric", "roc_auc")

        print(f"\n{'='*60}")
        print(f"  AGENT    : ML Test Engineer")
        print(f"  PHASE    : MODEL TESTING")
        print(f"  TEMPLATE : scripts/model_evaluator.py")
        print(f"{'='*60}")

        start = time.time()
        result = subprocess.run(
            [
                "python3", "scripts/model_evaluator.py",
                self.input.dataset_path,
                self.input.target_column,
                building_file,
                output_file,
                priority
            ],
            capture_output=False
        )
        duration = round(time.time() - start, 2)

        if result.returncode == 0 and Path(output_file).exists():
            output = json.loads(Path(output_file).read_text())
            top     = output.get("top_models", [])
            flagged = output.get("flagged_models", [])

            print(f"\n  ✓ Model Testing complete in {duration}s")
            print(f"  📄 Output: {output_file}")

            print(f"  🗣  NarrativeAgent: generating ML Engineer narrative...")
            output["genai_narrative"] = self._narrative.generate(
                "model_testing", output,
                self.input.business_problem, "ml_test_engineer"
            )
            Path(output_file).write_text(json.dumps(output, indent=2))

            self._log_decision(
                "ML Test Engineer", "PROCEED ✅",
                f"Top models: {top}, Flagged: {flagged}"
            )
            self._track_metrics(
                "model_testing", "ML Test Engineer",
                True, output.get("confidence_score", 0.85),
                output_file
            )
            return output
        else:
            print(f"\n  ✗ Model Testing failed")
            self._log_decision(
                "ML Test Engineer", "FAILED ✗",
                "Template script failed"
            )
            self._track_metrics(
                "model_testing", "ML Test Engineer",
                False, 0, output_file,
                ["Template script failed"]
            )
            return {}

    # ── Phase 4: Recommendation ────────────────────────────────

    def run_recommendation(self, testing_output: dict) -> dict:
        """Run recommendation using LLM agent."""
        prior_context = self._get_prior_context([
            "data_exploration",
            "model_building",
            "model_testing"
        ])

        print(f"\n{'='*60}")
        print(f"  AGENT    : Data Scientist")
        print(f"  PHASE    : RECOMMENDATION")
        print(f"  MODE     : LLM Agent")
        print(f"{'='*60}")

        agent = RecommendationAgent(
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

        start = time.time()
        output = agent.run(message, self.input.business_problem)
        duration = round(time.time() - start, 2)

        self.agent_metrics.append(agent.get_metrics())

        if output:
            print(f"\n  ✓ Recommendation complete in {duration}s")
            self._log_decision(
                "Data Scientist", "COMPLETE ✅",
                f"Recommended: {output.get('recommended_model', 'N/A')}, "
                f"Confidence: {output.get('confidence_score', 'N/A')}"
            )
        else:
            print(f"\n  ✗ Recommendation failed")
            self._log_decision(
                "Data Scientist", "FAILED ✗",
                "Agent produced no output"
            )

        return output

    # ── Main Run ───────────────────────────────────────────────

    def run(self) -> dict:
        """Run only the phases specified in analysis_phases."""
        results = {}
        phases  = self.input.analysis_phases

        print(f"📋 ORCHESTRATOR: Running phases: {phases}\n")

        if "data_exploration" in phases:
            print("📋 ORCHESTRATOR: Starting Phase 1 — Data Exploration")
            exploration = self.run_data_exploration()
            results["data_exploration"] = exploration
            if not exploration:
                print("  ✗ Data Exploration failed — stopping")
                self._write_log(results)
                return results

        if "model_building" in phases:
            print("📋 ORCHESTRATOR: Starting Phase 2 — Model Building")
            exploration = self._load_output("data_exploration")
            building = self.run_model_building(exploration)
            results["model_building"] = building
            if not building:
                print("  ✗ Model Building failed — stopping")
                self._write_log(results)
                return results

        if "model_testing" in phases:
            print("📋 ORCHESTRATOR: Starting Phase 3 — Model Testing")
            building = self._load_output("model_building")
            testing = self.run_model_testing(building)
            results["model_testing"] = testing
            if not testing:
                print("  ⚠️  Model Testing failed — continuing to recommendation")

        if "recommendation" in phases:
            print("📋 ORCHESTRATOR: Starting Phase 4 — Recommendation")
            testing = self._load_output("model_testing")
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
        print(f"  📋 Log: {log_path}")
# evals/eval_runner.py
import json
from pathlib import Path
from datetime import datetime

from evals.behavior_evals import (
    eval_data_exploration,
    eval_model_building,
    eval_model_testing,
    eval_recommendation,
)
from evals.quality_evals import eval_output_quality
from evals.system_evals import eval_system
from evals.hitl_gate import evaluate_hitl_risk
from evals.production_checklist import run_production_checklist
from evals.unit_tests import run_unit_tests


class EvalRunner:
    """
    Runs all MIRA evaluation layers after pipeline completion:

    Layer 1  — Behavior evals   (4 phases: structure & key-field checks)
    Layer 2  — Quality eval     (cross-phase performance checks)
    Layer 3  — System eval      (orchestrator completion & timing)
    Layer 4  — Unit tests       (8 deterministic Phase 4 output checks)
    Layer 5  — HITL gate        (7 risk factors → human review decision)
    Layer 6  — Production       (7-item production readiness checklist)
    Layer 7  — LLM Judge        (optional, runs JudgeAgent via OpenHands)

    Overall score = mean(behavior_avg, quality, system, unit_pct,
                         checklist_pct)
    LLM Judge score is reported separately and does not affect overall.
    """

    def __init__(
        self,
        run_id: str,
        output_path: str,
        priority_metric: str = "roc_auc",
        run_judge: bool = False,
    ):
        self.run_id = run_id
        self.output_path = Path(output_path)
        self.priority_metric = priority_metric
        self.run_judge = run_judge

    def _load(self, phase: str) -> dict:
        f = self.output_path / f"{self.run_id}_{phase}.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def _load_orch_log(self) -> dict:
        f = Path("logs/runs") / f"{self.run_id}_orchestrator.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def run(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  Running MIRA Evals — Run ID: {self.run_id}")
        print(f"{'='*60}")

        # MIRA v0.3.0 writes: data_card, model_selection (Phase 2+3 combined), recommendation
        data_card      = self._load("data_card")
        model_selection = self._load("model_selection")  # Phase 3 fields appended here
        recommendation = self._load("recommendation")
        orch = self._load_orch_log()

        # --- Layer 1: Behavior -------------------------------------------
        print("  [1/6] Behavior evals...")
        behavior = {
            "data_card": eval_data_exploration(data_card),
            "model_selection": eval_model_building(model_selection),
            "model_testing": eval_model_testing(model_selection),
            "recommendation": eval_recommendation(recommendation),
        }

        # --- Layer 2: Quality --------------------------------------------
        print("  [2/6] Quality eval...")
        quality = eval_output_quality(
            data_card, model_selection, model_selection, recommendation,
            priority_metric=self.priority_metric,
        )

        # --- Layer 3: System ---------------------------------------------
        print("  [3/6] System eval...")
        agent_metrics = orch.get("agent_metrics", {})
        if isinstance(agent_metrics, dict):
            agent_metrics = [agent_metrics]
        outputs = orch.get("outputs_produced", {})
        phases_completed = [k for k, v in outputs.items() if v]
        system = eval_system(
            agent_metrics=agent_metrics,
            orchestrator_decisions=orch.get("orchestrator_decisions", []),
            total_duration=orch.get("total_duration_seconds", 0),
            phases_completed=phases_completed,
            errors=orch.get("phases_failed", []),
        )

        # --- Layer 4: Unit tests -----------------------------------------
        print("  [4/6] Unit tests...")
        unit = run_unit_tests()

        # --- Layer 5: HITL gate ------------------------------------------
        print("  [5/6] HITL risk gate...")
        hitl = evaluate_hitl_risk(
            data_card, model_selection, model_selection, recommendation,
            priority_metric=self.priority_metric,
        )

        # --- Layer 6: Production checklist -------------------------------
        print("  [6/6] Production checklist...")
        checklist = run_production_checklist(
            data_card, model_selection, model_selection, recommendation,
            priority_metric=self.priority_metric,
        )

        # --- Layer 7: LLM Judge (optional) -------------------------------
        judge = None
        if self.run_judge and recommendation:
            print("  [7/7] LLM Judge (may take a few minutes)...")
            from evals.judge_agent import JudgeAgent
            judge_agent = JudgeAgent(
                run_id=self.run_id,
                output_path=str(self.output_path),
            )
            judge = judge_agent.run()
        elif self.run_judge and not recommendation:
            print("  [7/7] LLM Judge skipped — no recommendation output")

        # --- Overall score -----------------------------------------------
        b_scores = [v["pct"] for v in behavior.values()]
        avg_b = sum(b_scores) / len(b_scores) if b_scores else 0

        component_scores = [avg_b, quality["pct"], system["pct"],
                            unit["pct"], checklist["pct"]]
        overall = round(sum(component_scores) / len(component_scores), 1)

        # --- Build report ------------------------------------------------
        report = {
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_score": overall,
            "overall_passed": overall >= 70,
            "behavior_evals": behavior,
            "quality_eval": quality,
            "system_eval": system,
            "unit_tests": unit,
            "hitl_gate": hitl,
            "production_checklist": checklist,
            "judge_eval": judge,
            "summary": {
                "behavior_avg_pct": round(avg_b, 1),
                "quality_pct": quality["pct"],
                "system_pct": system["pct"],
                "unit_tests_pct": unit["pct"],
                "checklist_pct": checklist["pct"],
                "judge_score": (judge.get("overall_score") if judge else None),
                "all_behavior_passed": all(
                    v["passed"] for v in behavior.values()
                ),
                "quality_passed": quality["passed"],
                "system_passed": system["passed"],
                "unit_tests_passed": unit["overall_passed"],
                "hitl_triggered": hitl["hitl_triggered"],
                "production_ready": checklist["production_ready"],
                "judge_verdict": (judge.get("verdict") if judge else "NOT_RUN"),
            },
        }

        out = self.output_path / f"{self.run_id}_eval_report.json"
        out.write_text(json.dumps(report, indent=2))
        self._print_results(report)
        return report

    def _print_results(self, report: dict):
        s = report["summary"]
        print(f"\n{'='*60}")
        print(f"  MIRA EVAL RESULTS")
        print(f"{'='*60}")
        status = "✅ PASSED" if report["overall_passed"] else "❌ FAILED"
        print(f"  Overall Score    : {report['overall_score']}%  {status}")
        print(f"{'='*60}")

        def row(label, pct, passed):
            icon = "✅" if passed else "⚠️ "
            print(f"  {icon}  {label:<22} {pct}%")

        row("Behavior Evals", s["behavior_avg_pct"], s["all_behavior_passed"])
        for phase, r in report["behavior_evals"].items():
            icon = "✅" if r["passed"] else "❌"
            print(f"       {icon} {phase:<25} {r['pct']}%")

        row("Quality Eval", s["quality_pct"], s["quality_passed"])
        row("System Eval", s["system_pct"], s["system_passed"])
        row("Unit Tests", s["unit_tests_pct"], s["unit_tests_passed"])
        row("Prod Checklist", s["checklist_pct"], s["production_ready"])

        hitl_icon = "⚠️ " if s["hitl_triggered"] else "✅"
        hitl_label = "TRIGGERED" if s["hitl_triggered"] else "not triggered"
        print(f"  {hitl_icon}  {'HITL Gate':<22} {hitl_label}")

        if s["judge_verdict"] and s["judge_verdict"] != "NOT_RUN":
            j_score = s.get("judge_score", "?")
            j_verdict = s["judge_verdict"]
            j_icon = "✅" if j_verdict == "APPROVED" else "⚠️ "
            print(f"  {j_icon}  {'LLM Judge':<22} {j_verdict}  ({j_score}/10)")

        print(f"{'='*60}")
        print(f"  Production Ready : "
              f"{'✅ YES' if s['production_ready'] else '❌ NO'}")
        print(f"{'='*60}")
        print(f"  Report: processed/{report['run_id']}_eval_report.json")
        print(f"{'='*60}\n")

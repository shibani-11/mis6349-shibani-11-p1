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


class EvalRunner:
    """Runs all 3 eval types after MIRA completes."""

    def __init__(
        self,
        run_id: str,
        output_path: str,
        priority_metric: str = "roc_auc"
    ):
        self.run_id = run_id
        self.output_path = Path(output_path)
        self.priority_metric = priority_metric

    def _load(self, phase: str) -> dict:
        f = self.output_path / f"{self.run_id}_{phase}.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def _load_orch_log(self) -> dict:
        f = Path("logs/runs") / f"{self.run_id}_orchestrator.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def run(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  🧪 Running MIRA Evals — Run ID: {self.run_id}")
        print(f"{'='*60}")

        exploration = self._load("data_exploration")
        building = self._load("model_building")
        testing = self._load("model_testing")
        recommendation = self._load("recommendation")
        orch = self._load_orch_log()

        print("  Running behavior evals...")
        behavior = {
            "data_exploration": eval_data_exploration(exploration),
            "model_building": eval_model_building(building),
            "model_testing": eval_model_testing(testing),
            "recommendation": eval_recommendation(recommendation),
        }

        print("  Running quality evals...")
        quality = eval_output_quality(
            exploration, building, testing, recommendation,
            priority_metric=self.priority_metric
        )

        print("  Running system evals...")
        system = eval_system(
            agent_metrics=orch.get("agent_metrics", []),
            orchestrator_decisions=orch.get(
                "orchestrator_decisions", []
            ),
            total_duration=orch.get("total_duration_seconds", 0),
            phases_completed=orch.get("phases_completed", []),
            errors=orch.get("phases_failed", [])
        )

        b_scores = [v["pct"] for v in behavior.values()]
        avg_b = sum(b_scores) / len(b_scores) if b_scores else 0
        overall = round(
            (avg_b + quality["pct"] + system["pct"]) / 3, 1
        )

        report = {
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_score": overall,
            "overall_passed": overall >= 70,
            "behavior_evals": behavior,
            "quality_eval": quality,
            "system_eval": system,
            "summary": {
                "behavior_avg_pct": round(avg_b, 1),
                "quality_pct": quality["pct"],
                "system_pct": system["pct"],
                "all_behavior_passed": all(
                    v["passed"] for v in behavior.values()
                ),
                "quality_passed": quality["passed"],
                "system_passed": system["passed"],
            }
        }

        out = self.output_path / f"{self.run_id}_eval_report.json"
        out.write_text(json.dumps(report, indent=2))
        self._print_results(report)
        return report

    def _print_results(self, report: dict):
        print(f"\n{'='*60}")
        print(f"  📊 MIRA EVAL RESULTS")
        print(f"{'='*60}")
        status = "✅ PASSED" if report["overall_passed"] else "❌ FAILED"
        print(f"  Overall Score  : {report['overall_score']}%  {status}")
        print(f"{'='*60}")

        b = report["summary"]
        print(f"  Behavior Evals : {b['behavior_avg_pct']}%  "
              f"{'✅' if b['all_behavior_passed'] else '⚠️'}")
        for phase, r in report["behavior_evals"].items():
            icon = "✅" if r["passed"] else "❌"
            print(f"    {icon} {phase:<25} {r['pct']}%")

        print(f"  Quality Eval   : {b['quality_pct']}%  "
              f"{'✅' if b['quality_passed'] else '⚠️'}")
        print(f"  System Eval    : {b['system_pct']}%  "
              f"{'✅' if b['system_passed'] else '⚠️'}")
        print(f"{'='*60}")
        print(f"  📄 Report: "
              f"processed/{report['run_id']}_eval_report.json")
        print(f"{'='*60}\n")
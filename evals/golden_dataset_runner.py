# evals/golden_dataset_runner.py
"""
System test runner using the 10-case golden dataset.
Tests the evaluation logic (behavior_evals, hitl_gate, production_checklist)
against known-good and known-bad recommendation fixtures.

Usage:
    python -m evals.golden_dataset_runner
    python -m evals.golden_dataset_runner --case case_01
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from evals.behavior_evals import eval_recommendation
from evals.hitl_gate import evaluate_hitl_risk
from evals.production_checklist import run_production_checklist


_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def _load_cases() -> list:
    return json.loads(_DATASET_PATH.read_text())["cases"]


def run_case(case: dict) -> dict:
    inputs = case["inputs"]
    rec = case["candidate_recommendation"]
    expected = case["expected_eval"]

    exploration = inputs["data_exploration"]
    building = inputs["model_building"]
    testing = inputs["model_testing"]

    behavior = eval_recommendation(rec)
    hitl = evaluate_hitl_risk(exploration, building, testing, rec)
    checklist = run_production_checklist(exploration, building, testing, rec)

    actual = {
        "behavior_passes": behavior["passed"],
        "hitl_triggered": hitl["hitl_triggered"],
        "production_ready": checklist["production_ready"],
    }

    assertions = []
    for key, expected_val in expected.items():
        if key not in actual:
            continue
        actual_val = actual[key]
        passed = actual_val == expected_val
        assertions.append({
            "assertion": key,
            "expected": expected_val,
            "actual": actual_val,
            "passed": passed,
        })

    all_passed = all(a["passed"] for a in assertions)

    return {
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "description": case["description"],
        "tags": case["tags"],
        "assertions": assertions,
        "passed": all_passed,
        "behavior_detail": behavior,
        "hitl_detail": hitl,
        "checklist_detail": checklist,
    }


def run_all(filter_case: str = None) -> dict:
    cases = _load_cases()
    if filter_case:
        cases = [c for c in cases if c["case_id"] == filter_case]

    results = [run_case(c) for c in cases]

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pct = round(passed / total * 100, 1) if total else 0

    print(f"\n{'='*65}")
    print(f"  MIRA Golden Dataset — System Test Results")
    print(f"{'='*65}")
    print(f"  Cases run   : {total}")
    print(f"  Passed      : {passed}")
    print(f"  Failed      : {total - passed}")
    print(f"  Pass rate   : {pct}%")
    print(f"{'='*65}")

    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {icon}  {r['case_id']}  {r['scenario']}")
        if not r["passed"]:
            for a in r["assertions"]:
                if not a["passed"]:
                    print(f"       ✗ {a['assertion']}: "
                          f"expected={a['expected']} got={a['actual']}")

    print(f"{'='*65}\n")

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate_pct": pct,
        "overall_passed": pct >= 80,
        "results": results,
    }

    out_path = Path("processed") / "golden_dataset_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"  Report saved to: {out_path}\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None,
                        help="Run a single case by case_id, e.g. case_01")
    args = parser.parse_args()
    run_all(filter_case=args.case)

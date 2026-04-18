# evals/quality_evals.py


def eval_output_quality(
    exploration: dict,
    building: dict,
    testing: dict,
    recommendation: dict,
    priority_metric: str = "roc_auc"
) -> dict:
    checks = {}
    score = 0
    total = 0

    def check(name, condition, weight=1, detail=""):
        nonlocal score, total
        total += weight
        passed = bool(condition)
        checks[name] = {
            "passed": passed,
            "weight": weight,
            "detail": detail
        }
        if passed:
            score += weight

    # building and testing both point to model_selection (Phase 3 appended)
    models = building.get("models_trained", [])
    best_score = max(
        (m.get("cv_roc_auc_mean", 0) for m in models), default=0
    )

    check("data_fully_analyzed",
          exploration.get("rows", 0) >= 1000,
          weight=2,
          detail=f"Rows: {exploration.get('rows', 0)}")
    check("imbalance_detected",
          "class_imbalance_detected" in exploration, weight=2)
    check("minimum_performance",
          best_score >= 0.65,
          weight=3,
          detail=f"Best cv_roc_auc_mean: {best_score:.3f}")
    check("multiple_models",
          len(models) >= 2,
          weight=2,
          detail=f"Models: {len(models)}")
    check("models_passed_testing",
          testing.get("test_verdict") == "PASS",
          weight=3,
          detail=f"test_verdict: {testing.get('test_verdict')}")
    check("recommendation_made",
          bool(recommendation.get("recommended_model")), weight=3)
    check("high_confidence",
          recommendation.get("confidence_score", 0) >= 0.7,
          weight=2,
          detail=f"Confidence: {recommendation.get('confidence_score', 0)}")
    check("actionable_steps",
          len(recommendation.get("next_steps", [])) >= 3, weight=2)
    check("business_summary",
          len(recommendation.get("executive_summary", "")) > 100,
          weight=2)

    pct = round(score / total * 100, 1) if total > 0 else 0
    return {
        "eval_type": "quality",
        "checks": checks,
        "score": score,
        "total": total,
        "pct": pct,
        "passed": pct >= 70,
        "best_model_score": best_score,
        "priority_metric": priority_metric
    }
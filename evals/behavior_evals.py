# evals/behavior_evals.py


def eval_data_exploration(output: dict) -> dict:
    checks = {}
    score = 0
    total = 0

    def check(name, condition, weight=1):
        nonlocal score, total
        total += weight
        passed = bool(condition)
        checks[name] = {"passed": passed, "weight": weight}
        if passed:
            score += weight

    check("used_full_dataset",
          output.get("rows", 0) >= 1000, weight=3)
    check("computed_class_distribution",
          bool(output.get("class_distribution")), weight=2)
    check("detected_class_imbalance",
          "class_imbalance_detected" in output, weight=2)
    check("computed_minority_ratio",
          "minority_class_ratio" in output, weight=2)
    check("profiled_features",
          output.get("features", 0) > 0, weight=1)
    check("computed_missing_values",
          "missing_value_summary" in output, weight=1)
    check("found_quality_issues",
          "data_quality_issues" in output, weight=1)
    check("has_genai_narrative",
          len(output.get("genai_narrative", "")) > 20, weight=1)
    check("has_recommended_approach",
          len(output.get("recommended_approach", "")) > 10, weight=1)

    pct = round(score / total * 100, 1) if total > 0 else 0
    return {
        "phase": "data_exploration",
        "checks": checks,
        "score": score,
        "total": total,
        "pct": pct,
        "passed": pct >= 70
    }


def eval_model_building(output: dict) -> dict:
    checks = {}
    score = 0
    total = 0

    def check(name, condition, weight=1):
        nonlocal score, total
        total += weight
        passed = bool(condition)
        checks[name] = {"passed": passed, "weight": weight}
        if passed:
            score += weight

    models = output.get("models_trained", [])

    check("trained_multiple_models", len(models) >= 2, weight=3)
    check("included_baseline",
          any("logistic" in m.get("name", "").lower()
              for m in models), weight=2)
    check("used_cross_validation",
          all("cv_roc_auc_mean" in m for m in models)
          if models else False, weight=2)
    check("recorded_overfitting_gap",
          all("overfitting_gap" in m for m in models)
          if models else False, weight=2)
    check("selected_model_named",
          bool(output.get("selected_model")), weight=2)
    check("applied_preprocessing",
          bool(output.get("preprocessing_applied")), weight=1)
    check("handled_imbalance",
          output.get("class_imbalance_handled", False), weight=2)
    check("has_rejected_models",
          len(output.get("rejected_models", [])) >= 1, weight=1)
    check("has_genai_narrative",
          len(output.get("genai_narrative", "")) > 20, weight=1)

    pct = round(score / total * 100, 1) if total > 0 else 0
    return {
        "phase": "model_building",
        "checks": checks,
        "score": score,
        "total": total,
        "pct": pct,
        "passed": pct >= 70
    }


def eval_model_testing(output: dict) -> dict:
    checks = {}
    score = 0
    total = 0

    def check(name, condition, weight=1):
        nonlocal score, total
        total += weight
        passed = bool(condition)
        checks[name] = {"passed": passed, "weight": weight}
        if passed:
            score += weight

    # Phase 3 fields are appended to model_selection by the ML Test Engineer
    check("checked_overfitting",
          "overfitting_detected" in output, weight=2)
    check("recorded_overfitting_gap",
          "overfitting_gap" in output, weight=2)
    check("checked_leakage",
          "leakage_detected" in output, weight=2)
    check("checked_stability",
          "stability_flag" in output, weight=2)
    check("has_feature_importance",
          bool(output.get("feature_importance")), weight=2)
    check("has_test_verdict",
          output.get("test_verdict") in ["PASS", "FAIL"], weight=3)
    check("has_test_findings",
          "test_findings" in output, weight=1)
    check("selected_model_named",
          bool(output.get("selected_model")), weight=2)

    pct = round(score / total * 100, 1) if total > 0 else 0
    return {
        "phase": "model_testing",
        "checks": checks,
        "score": score,
        "total": total,
        "pct": pct,
        "passed": pct >= 70
    }


def eval_recommendation(output: dict) -> dict:
    checks = {}
    score = 0
    total = 0

    def check(name, condition, weight=1):
        nonlocal score, total
        total += weight
        passed = bool(condition)
        checks[name] = {"passed": passed, "weight": weight}
        if passed:
            score += weight

    summary = output.get("executive_summary", "")
    jargon = ["ROC-AUC", "F1 score", "precision score", "recall score"]

    check("has_recommended_model",
          bool(output.get("recommended_model")), weight=3)
    check("has_selection_reason",
          len(output.get("selection_reason", "")) > 30, weight=2)
    check("has_next_steps",
          len(output.get("next_steps", [])) >= 3, weight=2)
    check("has_tradeoffs",
          len(output.get("tradeoffs", [])) >= 2, weight=1)
    check("has_alternative_model",
          bool(output.get("alternative_model")), weight=1)
    check("has_confidence_score",
          0 <= output.get("confidence_score", -1) <= 1, weight=1)
    check("has_human_review_flag",
          "requires_human_review" in output, weight=1)
    check("has_executive_summary",
          len(summary) > 50, weight=2)
    check("no_jargon_in_summary",
          not any(j in summary for j in jargon), weight=2)
    check("has_deployment_verdict",
          "YES" in summary.upper() or "NO" in summary.upper(),
          weight=2)

    pct = round(score / total * 100, 1) if total > 0 else 0
    return {
        "phase": "recommendation",
        "checks": checks,
        "score": score,
        "total": total,
        "pct": pct,
        "passed": pct >= 70
    }
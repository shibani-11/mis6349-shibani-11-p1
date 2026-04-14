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
          output.get("row_count", 0) > 10000, weight=3)
    check("identified_task_type",
          output.get("inferred_task_type") in
          ["classification", "regression"], weight=2)
    check("detected_class_imbalance",
          "class_imbalance_detected" in output, weight=2)
    check("computed_target_distribution",
          bool(output.get("target_distribution")), weight=2)
    check("identified_numeric_columns",
          bool(output.get("numeric_columns")), weight=1)
    check("computed_missing_pct",
          "overall_missing_pct" in output, weight=1)
    check("found_quality_issues",
          "quality_issues" in output, weight=1)
    check("has_genai_narrative",
          len(output.get("genai_narrative", "")) > 20, weight=1)
    check("has_confidence_score",
          0 <= output.get("confidence_score", -1) <= 1, weight=1)

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

    models = output.get("models_evaluated", [])

    check("trained_multiple_models", len(models) >= 2, weight=3)
    check("included_baseline",
          any("logistic" in m.get("model_name", "").lower()
              for m in models), weight=2)
    check("used_cross_validation",
          all("cross_val_score_mean" in m for m in models)
          if models else False, weight=2)
    check("recorded_training_time",
          all("training_time_seconds" in m for m in models)
          if models else False, weight=1)
    check("checked_overfitting",
          all("overfitting_detected" in m for m in models)
          if models else False, weight=2)
    check("applied_preprocessing",
          bool(output.get("preprocessing_steps")), weight=1)
    check("handled_imbalance",
          any("balanced" in str(s).lower()
              for s in output.get("preprocessing_steps", [])),
          weight=2)
    check("has_primary_metric",
          bool(output.get("primary_metric")), weight=1)
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

    results = output.get("test_results", [])

    check("tested_models", len(results) >= 2, weight=2)
    check("checked_overfitting",
          all("overfitting_detected" in r for r in results)
          if results else False, weight=2)
    check("checked_leakage",
          all("leakage_suspected" in r for r in results)
          if results else False, weight=2)
    check("checked_stability",
          all("stability_ok" in r for r in results)
          if results else False, weight=2)
    check("generated_confusion_matrix",
          all("confusion_matrix" in r for r in results)
          if results else False, weight=1)
    check("identified_top_models",
          len(output.get("top_models", [])) >= 1, weight=2)
    check("flagged_failures", "flagged_models" in output, weight=1)
    check("has_genai_narrative",
          len(output.get("genai_narrative", "")) > 20, weight=1)

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
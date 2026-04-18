# evals/production_checklist.py
"""
7-item production readiness checklist for MIRA.

Items:
  1. Model meets minimum performance threshold (>=0.65 on priority metric)
  2. No data leakage detected in any model
  3. Recommended model passed overfitting check
  4. All 12 required output fields present and non-empty
  5. Executive summary is free of technical jargon
  6. Deployment verdict is explicitly stated (YES or NO)
  7. Human review flag is accurately set for risk level

Items 1-3 are CRITICAL — a single failure blocks production readiness.
Items 4-7 are STANDARD — all must pass, but failure is not immediately
blocking if the human review flag is set correctly.

production_ready = all critical items passed AND >= 6 of 7 total passed.
"""

_REQUIRED_KEYS = [
    "recommended_model", "selection_reason", "primary_metric_value",
    "all_models_summary", "model_comparison_narrative", "business_impact",
    "tradeoffs", "alternative_model", "next_steps",
    "deployment_considerations", "risks", "test_verdict_summary",
    "feature_drivers", "confidence_score",
    "requires_human_review", "human_review_reason", "executive_summary",
]

_JARGON_TERMS = [
    "ROC-AUC", "F1 score", "F1-score", "precision score",
    "recall score", "AUC", "AUROC",
]


def run_production_checklist(
    exploration: dict,
    building: dict,
    testing: dict,
    recommendation: dict,
    priority_metric: str = "roc_auc",
) -> dict:
    """
    Returns a checklist report. Key fields:
      checklist        : list of 7 item dicts
      passed_count     : int
      total_items      : int (always 7)
      pct              : float
      critical_passed  : bool  (items 1-3 all passed)
      production_ready : bool
      verdict          : str
    """
    items = []

    def item(name: str, passed: bool, detail: str = "", critical: bool = False):
        items.append({
            "item": name,
            "passed": passed,
            "detail": detail,
            "critical": critical,
        })

    # building and testing both point to model_selection (Phase 3 appended)
    models = building.get("models_trained", [])

    # 1. Model performance threshold (CRITICAL) --------------------------
    best_score = max(
        (m.get("cv_roc_auc_mean", 0) for m in models), default=0
    )
    item(
        "model_meets_performance_threshold",
        best_score >= 0.65,
        f"Best cv_roc_auc_mean: {best_score:.3f} (minimum: 0.65)",
        critical=True,
    )

    # 2. No data leakage (CRITICAL) --------------------------------------
    leakage = testing.get("leakage_detected", False)
    item(
        "no_data_leakage_detected",
        not leakage,
        "All models passed leakage check" if not leakage
        else "Leakage detected — deployment blocked",
        critical=True,
    )

    # 3. Recommended model passes overfitting check (CRITICAL) -----------
    rec_model = recommendation.get("recommended_model", "")
    overfit = testing.get("overfitting_detected", False)
    item(
        "recommended_model_no_overfitting",
        not overfit,
        (f"{rec_model} passed overfitting check"
         if not overfit else f"{rec_model} shows overfitting (gap: {testing.get('overfitting_gap', '?')})"),
        critical=True,
    )

    # 4. All 12 required fields present ----------------------------------
    missing_keys = [k for k in _REQUIRED_KEYS if k not in recommendation]
    empty_keys = [
        k for k in _REQUIRED_KEYS
        if k in recommendation and (
            recommendation[k] is None
            or (isinstance(recommendation[k], str) and len(recommendation[k]) < 3)
            or (isinstance(recommendation[k], list) and len(recommendation[k]) == 0)
        )
    ]
    all_present = len(missing_keys) == 0 and len(empty_keys) == 0
    detail_parts = []
    if missing_keys:
        detail_parts.append(f"missing: {missing_keys}")
    if empty_keys:
        detail_parts.append(f"empty: {empty_keys}")
    item(
        "all_required_fields_present",
        all_present,
        "; ".join(detail_parts) if detail_parts else "All 12 fields present and populated",
    )

    # 5. No jargon in executive summary ----------------------------------
    summary = recommendation.get("executive_summary", "")
    found_jargon = [j for j in _JARGON_TERMS if j in summary]
    item(
        "executive_summary_jargon_free",
        len(found_jargon) == 0,
        f"Jargon found: {found_jargon}" if found_jargon
        else "No technical jargon detected in executive summary",
    )

    # 6. Deployment verdict explicit (YES or NO) -------------------------
    verdict_present = "YES" in summary.upper() or "NO" in summary.upper()
    item(
        "deployment_verdict_explicit",
        verdict_present,
        "YES/NO verdict found" if verdict_present
        else "No deployment verdict (YES/NO) in executive summary",
    )

    # 7. Human review flag accurately set --------------------------------
    hitl_flag = recommendation.get("requires_human_review")
    critical_risk_present = leakage or best_score < 0.60
    flag_is_bool = isinstance(hitl_flag, bool)
    flag_accurate = flag_is_bool and (
        not critical_risk_present or hitl_flag is True
    )
    item(
        "human_review_flag_accurate",
        flag_accurate,
        (f"requires_human_review={hitl_flag}, "
         f"critical_risk_present={critical_risk_present}"),
    )

    # --- Aggregate -------------------------------------------------------
    passed_count = sum(1 for i in items if i["passed"])
    total_items = len(items)
    pct = round(passed_count / total_items * 100, 1) if total_items else 0
    critical_passed = all(i["passed"] for i in items if i["critical"])
    production_ready = critical_passed and passed_count >= 6

    return {
        "eval_type": "production_checklist",
        "checklist": items,
        "passed_count": passed_count,
        "total_items": total_items,
        "pct": pct,
        "critical_items_passed": critical_passed,
        "production_ready": production_ready,
        "verdict": (
            "READY FOR PRODUCTION"
            if production_ready else
            "NOT READY — resolve checklist failures before deployment"
        ),
    }

# evals/hitl_gate.py
"""
Human-in-the-Loop (HITL) risk gate for MIRA.

Evaluates seven risk factors across the pipeline outputs and determines
whether a human must review before a deployment decision is made.

Risk levels and weights:
  CRITICAL  (5) — data leakage detected
  HIGH      (4) — all models below minimum performance threshold
  HIGH      (3) — ambiguous model selection (top-2 within 0.02)
  HIGH      (3) — severe class imbalance (>90%) with no resampling
  MEDIUM    (2) — recommended model shows overfitting
  MEDIUM    (2) — recommendation confidence < 0.60
  LOW       (1) — selection reason too thin (<50 chars)

HITL is triggered when total_risk_score >= 5.
"""


def evaluate_hitl_risk(
    exploration: dict,
    building: dict,
    testing: dict,
    recommendation: dict,
    priority_metric: str = "roc_auc",
) -> dict:
    """
    Returns a risk assessment dict. Key fields:
      hitl_triggered   : bool
      total_risk_score : int
      risks_identified : list of risk objects
      highest_severity : CRITICAL | HIGH | MEDIUM | LOW | NONE
      recommendation   : human-readable action string
    """
    risks = []
    total_risk = 0

    def add_risk(name, severity, weight, detail):
        nonlocal total_risk
        total_risk += weight
        risks.append({
            "risk": name,
            "severity": severity,
            "weight": weight,
            "detail": detail,
        })

    # building and testing both point to model_selection (Phase 3 appended)
    models = building.get("models_trained", [])

    # --- CRITICAL: data leakage -----------------------------------------
    if testing.get("leakage_detected", False):
        add_risk(
            "data_leakage_detected",
            "CRITICAL",
            5,
            f"Leakage detected — investigate feature pipeline before deployment",
        )

    # --- HIGH: all models below performance floor -----------------------
    best_score = max(
        (m.get("cv_roc_auc_mean", 0) for m in models), default=0
    )
    if best_score < 0.60:
        add_risk(
            "poor_model_performance",
            "HIGH",
            4,
            f"Best cv_roc_auc_mean: {best_score:.3f} — below 0.60 floor",
        )

    # --- HIGH: ambiguous selection (top-2 within 0.02) ------------------
    scores = sorted(
        [m.get("cv_roc_auc_mean", 0) for m in models], reverse=True
    )
    if len(scores) >= 2 and (scores[0] - scores[1]) < 0.02:
        add_risk(
            "ambiguous_model_selection",
            "HIGH",
            3,
            f"Top-2 models differ by {scores[0] - scores[1]:.4f} on cv_roc_auc_mean",
        )

    # --- HIGH: severe class imbalance without resampling ----------------
    minority_ratio = exploration.get("minority_class_ratio", 0.5)
    minority_pct = minority_ratio * 100
    resampled = building.get("class_imbalance_handled", False)
    if minority_pct < 10 and not resampled:
        add_risk(
            "severe_imbalance_unhandled",
            "HIGH",
            3,
            f"Minority class {minority_pct:.1f}% with no resampling applied",
        )

    # --- MEDIUM: recommended model overfits -----------------------------
    rec_model_name = recommendation.get("recommended_model", "")
    if rec_model_name and testing.get("overfitting_detected", False):
        add_risk(
            "recommended_model_overfits",
            "MEDIUM",
            2,
            f"{rec_model_name} shows overfitting (gap: {testing.get('overfitting_gap', '?')})",
        )

    # --- MEDIUM: low recommendation confidence --------------------------
    confidence = recommendation.get("confidence_score", 1.0)
    if isinstance(confidence, (int, float)) and confidence < 0.60:
        add_risk(
            "low_confidence_recommendation",
            "MEDIUM",
            2,
            f"Confidence score {confidence:.2f} < 0.60 threshold",
        )

    # --- LOW: thin selection justification ------------------------------
    reason = recommendation.get("selection_reason", "")
    if len(reason) < 50:
        add_risk(
            "thin_business_justification",
            "LOW",
            1,
            f"selection_reason has only {len(reason)} chars — lacks substance",
        )

    # --- Aggregate -------------------------------------------------------
    hitl_triggered = total_risk >= 5

    severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    highest = "NONE"
    for sev in severity_order:
        if any(r["severity"] == sev for r in risks):
            highest = sev
            break

    primary_reason = risks[0]["detail"] if risks else "No significant risks"

    return {
        "hitl_triggered": hitl_triggered,
        "total_risk_score": total_risk,
        "risk_threshold": 5,
        "risks_identified": risks,
        "risk_count": len(risks),
        "highest_severity": highest,
        "primary_reason": primary_reason,
        "recommendation": (
            "PAUSE — Human review required before deployment decision"
            if hitl_triggered else
            "PROCEED — Risk level acceptable for automated recommendation"
        ),
    }

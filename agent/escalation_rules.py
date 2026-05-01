# agent/escalation_rules.py
"""
Hard Escalation Rules for MIRA — Session 6 Architecture.

Hard escalation rules bypass the confidence threshold entirely.
A case that hits any rule goes to Zone 3 (Priority Human Review)
regardless of how high the confidence score is.

Rule taxonomy (Session 6):
  Category 1 — Data Quality    : agent's information is incomplete or unreliable
  Category 2 — Model Quality   : model cannot be trusted for production deployment
  Category 3 — Semantic Ambiguity : domain logic is genuinely unclear
  Category 4 — Adversarial Signal : signs of manipulation or unusual input
"""

from pathlib import Path


def evaluate_escalation_rules(data_card: dict, model_selection: dict) -> dict:
    """
    Evaluate all hard escalation rules against the pipeline outputs.

    Returns:
        {
            "rules_triggered": [list of triggered rule dicts],
            "hard_escalation": bool,   # True if ANY rule fired
            "highest_category": str,   # DATA_QUALITY | MODEL_QUALITY | AMBIGUITY | ADVERSARIAL | NONE
            "summary": str             # human-readable summary
        }
    """
    triggered = []

    models = model_selection.get("models_trained", [])
    winner_name = model_selection.get("selected_model", "")
    winner = next((m for m in models if m.get("name") == winner_name), {})
    runner_up_name = model_selection.get("runner_up_model", "")
    runner_up = next((m for m in models if m.get("name") == runner_up_name), {})

    winner_auc = winner.get("cv_roc_auc_mean", 0)
    runner_up_auc = runner_up.get("cv_roc_auc_mean", 0)
    auc_gap = round(winner_auc - runner_up_auc, 4)

    rows = data_card.get("rows", 9999)
    minority_ratio = data_card.get("minority_class_ratio", 0.5)
    missing_summary = data_card.get("missing_value_summary", {})
    leakage_detected = model_selection.get("leakage_detected", False)
    overfitting_detected = model_selection.get("overfitting_detected", False)
    overfitting_gap = model_selection.get("overfitting_gap", 0.0)
    stability_flag = model_selection.get("stability_flag", False)
    test_verdict = model_selection.get("test_verdict", "PASS")
    imbalance_handled = model_selection.get("class_imbalance_handled", True)
    priority_metric = data_card.get("priority_metric", "roc_auc")

    # ── CATEGORY 1: Data Quality ─────────────────────────────────────────────

    if leakage_detected:
        triggered.append({
            "rule_name": "DATA_LEAKAGE_DETECTED",
            "category": "DATA_QUALITY",
            "severity": "CRITICAL",
            "trigger": "leakage_detected = True in model_selection.json",
            "detail": (
                f"Model CV AUC > 0.99 — suspected data leakage. "
                f"The feature pipeline must be investigated before any deployment."
            ),
            "routing": "zone_3",
            "override_authority": "Human reviewer must confirm pipeline investigation is complete",
        })

    if rows < 1000:
        triggered.append({
            "rule_name": "INSUFFICIENT_TRAINING_DATA",
            "category": "DATA_QUALITY",
            "severity": "HIGH",
            "trigger": f"rows={rows} after cleaning — below 1,000 minimum",
            "detail": (
                f"Only {rows:,} rows remained after EDA cleaning. "
                f"5-fold CV on fewer than 1,000 samples produces unreliable AUC estimates."
            ),
            "routing": "zone_3",
            "override_authority": "Human reviewer explicitly accepts low-N risk",
        })

    if missing_summary:
        total_original = rows
        max_imputed_pct = 0.0
        worst_col = None
        for col, count in missing_summary.items():
            if isinstance(count, (int, float)) and total_original > 0:
                pct = count / total_original
                if pct > max_imputed_pct:
                    max_imputed_pct = pct
                    worst_col = col
        if max_imputed_pct > 0.30:
            triggered.append({
                "rule_name": "HEAVY_IMPUTATION",
                "category": "DATA_QUALITY",
                "severity": "MEDIUM",
                "trigger": f"Column '{worst_col}' had {max_imputed_pct:.0%} values imputed",
                "detail": (
                    f"'{worst_col}' had {max_imputed_pct:.0%} null values filled with median/mode. "
                    f"Heavy imputation may introduce bias the model trains on but won't see in production."
                ),
                "routing": "zone_2",
                "override_authority": "Human confirms imputation strategy is appropriate for this domain",
            })

    # ── CATEGORY 2: Model Quality ─────────────────────────────────────────────

    if overfitting_detected and abs(overfitting_gap) > 0.20:
        triggered.append({
            "rule_name": "SEVERE_OVERFITTING",
            "category": "MODEL_QUALITY",
            "severity": "HIGH",
            "trigger": f"overfitting_gap={overfitting_gap:.4f} — exceeds 0.20 severe threshold",
            "detail": (
                f"{winner_name} has a train-to-val gap of {overfitting_gap:.4f}. "
                f"This model has memorised the training data and will degrade significantly on new data."
            ),
            "routing": "zone_3",
            "override_authority": "Human confirms model is still acceptable with noted generalisation risk",
        })
    elif overfitting_detected:
        triggered.append({
            "rule_name": "OVERFITTING_DETECTED",
            "category": "MODEL_QUALITY",
            "severity": "MEDIUM",
            "trigger": f"overfitting_gap={overfitting_gap:.4f} — exceeds 0.10 threshold",
            "detail": (
                f"{winner_name} train-to-val gap is {overfitting_gap:.4f} (threshold: 0.10). "
                f"Model may not generalise well to unseen data."
            ),
            "routing": "zone_2",
            "override_authority": "Human confirms generalisation risk is acceptable for this use case",
        })

    if test_verdict == "FAIL":
        triggered.append({
            "rule_name": "TEST_VERDICT_FAIL",
            "category": "MODEL_QUALITY",
            "severity": "HIGH",
            "trigger": "test_verdict = FAIL in model_selection.json",
            "detail": (
                "Phase 3 stress tests failed — either overfitting or leakage was confirmed. "
                "Deployment is not recommended without human investigation."
            ),
            "routing": "zone_3",
            "override_authority": "Human reviews stress test findings and accepts identified risks",
        })

    if winner_auc < 0.65:
        triggered.append({
            "rule_name": "BELOW_PERFORMANCE_FLOOR",
            "category": "MODEL_QUALITY",
            "severity": "HIGH",
            "trigger": f"winner cv_roc_auc_mean={winner_auc:.4f} — below 0.65 floor",
            "detail": (
                f"Best model AUC of {winner_auc:.4f} is below the minimum deployment threshold of 0.65. "
                f"All models may lack sufficient discriminative power for this business problem."
            ),
            "routing": "zone_3",
            "override_authority": "Human explicitly accepts below-floor performance or rejects deployment",
        })

    # ── CATEGORY 3: Semantic Ambiguity ────────────────────────────────────────

    if auc_gap < 0.02 and len(models) >= 2:
        triggered.append({
            "rule_name": "AMBIGUOUS_MODEL_SELECTION",
            "category": "SEMANTIC_AMBIGUITY",
            "severity": "MEDIUM",
            "trigger": f"Top-2 AUC gap={auc_gap:.4f} — within 0.02 statistical noise",
            "detail": (
                f"{winner_name} ({winner_auc:.4f}) vs {runner_up_name} ({runner_up_auc:.4f}): "
                f"gap of {auc_gap:.4f} is within noise range. Selection is statistically arbitrary. "
                f"Human should decide based on operational criteria: cost, interpretability, team familiarity."
            ),
            "routing": "zone_2",
            "override_authority": "Human selects model based on non-performance operational criteria",
        })

    if priority_metric in ("recall", "f1_score") and winner.get("cv_recall_mean", 1.0) < 0.60:
        winner_recall = winner.get("cv_recall_mean", 0)
        triggered.append({
            "rule_name": "METRIC_OBJECTIVE_MISMATCH",
            "category": "SEMANTIC_AMBIGUITY",
            "severity": "MEDIUM",
            "trigger": (
                f"Priority metric={priority_metric} but winner recall={winner_recall:.4f} < 0.60"
            ),
            "detail": (
                f"EDA inferred '{priority_metric}' as the priority metric from the business problem, "
                f"but {winner_name}'s recall is only {winner_recall:.4f}. "
                f"The model ranks well on AUC but may not serve the stated business objective."
            ),
            "routing": "zone_2",
            "override_authority": "Human confirms whether AUC or recall is the correct business priority",
        })

    if minority_ratio < 0.10 and not imbalance_handled:
        triggered.append({
            "rule_name": "SEVERE_IMBALANCE_UNHANDLED",
            "category": "SEMANTIC_AMBIGUITY",
            "severity": "HIGH",
            "trigger": f"minority_ratio={minority_ratio:.3f} < 0.10 and class_imbalance_handled=False",
            "detail": (
                f"Minority class is only {minority_ratio*100:.1f}% of the dataset "
                f"but no class weighting or resampling was applied. "
                f"The model likely predicts the majority class to minimise loss — "
                f"recall on the minority class may be near zero."
            ),
            "routing": "zone_3",
            "override_authority": "Human confirms why imbalance handling was not applied",
        })

    # ── CATEGORY 4: Adversarial Signal ────────────────────────────────────────

    feature_importance = model_selection.get("feature_importance", {})
    if feature_importance:
        total_imp = sum(feature_importance.values())
        if total_imp > 0:
            top_imp = max(feature_importance.values()) / total_imp
            top_feat = max(feature_importance, key=feature_importance.get)
            if top_imp > 0.60:
                triggered.append({
                    "rule_name": "FEATURE_IMPORTANCE_CONCENTRATION",
                    "category": "ADVERSARIAL_SIGNAL",
                    "severity": "MEDIUM",
                    "trigger": f"'{top_feat}' accounts for {top_imp:.0%} of total feature importance",
                    "detail": (
                        f"'{top_feat}' dominates with {top_imp:.0%} of importance. "
                        f"This level of concentration may indicate latent leakage — "
                        f"the EDA single-feature AUC check only catches features above 0.98. "
                        f"Verify this feature is not derived from the target."
                    ),
                    "routing": "zone_2",
                    "override_authority": "Human confirms feature is a legitimate predictor measured before the target event",
                })

    # ── Aggregate ─────────────────────────────────────────────────────────────

    hard_escalation = any(
        r["routing"] == "zone_3" for r in triggered
    )

    category_order = ["DATA_QUALITY", "MODEL_QUALITY", "SEMANTIC_AMBIGUITY", "ADVERSARIAL_SIGNAL"]
    highest_category = "NONE"
    for cat in category_order:
        if any(r["category"] == cat for r in triggered):
            highest_category = cat
            break

    if not triggered:
        summary = "No hard escalation rules triggered — proceed to confidence threshold check."
    else:
        names = [r["rule_name"] for r in triggered]
        summary = f"{len(triggered)} rule(s) triggered: {', '.join(names)}"

    return {
        "rules_triggered": triggered,
        "rules_count": len(triggered),
        "hard_escalation": hard_escalation,
        "highest_category": highest_category,
        "summary": summary,
    }

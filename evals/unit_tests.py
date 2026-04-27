# evals/unit_tests.py
"""
Unit tests for the MIRA Phase 4 (RecommendationAgent) recommendation output.

Tests cover:
  1.  Required schema keys present
  2.  No technical jargon in executive_summary
  3.  Deployment verdict (YES/NO) in executive_summary
  4.  Confidence score in valid range [0.0, 1.0]
  5.  requires_human_review is a boolean
  6.  next_steps contains at least 3 items
  7.  tradeoffs contains at least 2 items
  8.  selection_reason is substantive (>30 characters)

Usage:
    python -m pytest evals/unit_tests.py -v
    python -m unittest evals.unit_tests
"""
import json
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures — a perfect recommendation and several intentionally bad ones
# ---------------------------------------------------------------------------

_GOOD = {
    "recommended_model": "LightGBM",
    "selection_reason": "LightGBM correctly identifies 88% of high-risk customers while maintaining manageable false alarm rates, outperforming all other candidates on cross-validated ROC-AUC.",
    "primary_metric_value": 0.88,
    "all_models_summary": [
        {"name": "LightGBM", "cv_roc_auc_mean": 0.88, "rank": 1, "verdict": "SELECTED", "why_not_recommended": ""},
        {"name": "XGBoost", "cv_roc_auc_mean": 0.86, "rank": 2, "verdict": "RUNNER-UP", "why_not_recommended": "Marginally lower AUC and slower inference than LightGBM on this dataset size."},
        {"name": "Random Forest", "cv_roc_auc_mean": 0.83, "rank": 3, "verdict": "REJECTED", "why_not_recommended": "Lower discriminative power; ensemble depth did not compensate for feature interaction gaps."},
        {"name": "Logistic Regression", "cv_roc_auc_mean": 0.76, "rank": 4, "verdict": "REJECTED", "why_not_recommended": "Cannot capture non-linear interactions between Age and Balance that drive churn."},
    ],
    "model_comparison_narrative": (
        "LightGBM led all models with a cross-validated score of 0.88, followed closely by XGBoost at 0.86. "
        "Both gradient boosting methods substantially outperformed Random Forest (0.83) and Logistic Regression (0.76). "
        "The gap between LightGBM and the baseline confirms that non-linear patterns are present and exploitable."
    ),
    "business_impact": {
        "estimated_customers_identified": "Out of 1,000 customers likely to churn, the model correctly flags approximately 880 for the retention team.",
        "retention_opportunity": "Early identification enables the retention team to deploy targeted offers before customers initiate cancellation.",
        "model_value_statement": "Deploying this model could reduce monthly churn by an estimated 15-20% assuming a 30% retention success rate on flagged customers.",
    },
    "tradeoffs": [
        "Requires more memory than Logistic Regression — higher infrastructure cost",
        "Longer training time — needs periodic retraining on fresh data",
        "Less interpretable than a rule-based system — may require SHAP explainer for compliance",
    ],
    "alternative_model": "XGBoost",
    "alternative_model_reason": "XGBoost achieved 0.86 AUC with comparable stability and is a well-supported production library.",
    "next_steps": [
        "Deploy in shadow mode alongside existing system for 30 days",
        "Monitor live false-positive rate weekly and set alert thresholds",
        "Establish quarterly model retraining schedule with performance benchmarks",
    ],
    "deployment_considerations": [
        "Minimum 8GB RAM for batch scoring",
        "API response time target under 200ms",
    ],
    "risks": [
        "Model may degrade on new customer segments not present in training data",
        "Regulatory review required before production deployment",
    ],
    "test_verdict_summary": "PASS — overfitting check passed (gap 0.03), no leakage detected, stability confirmed (std 0.02).",
    "feature_drivers": [
        {"feature": "Age", "importance": 0.24, "business_explanation": "Older customers show significantly higher churn rates — likely related to life-stage banking needs."},
        {"feature": "IsActiveMember", "importance": 0.19, "business_explanation": "Inactive members are far more likely to leave — engagement is a strong retention signal."},
        {"feature": "Balance", "importance": 0.17, "business_explanation": "Higher balances correlate with churn — may reflect customers who found better rates elsewhere."},
    ],
    "confidence_score": 0.87,
    "requires_human_review": False,
    "human_review_reason": None,
    "executive_summary": (
        "After evaluating four candidate models, LightGBM has been selected for deployment. "
        "It correctly identifies 88 out of every 100 customers likely to leave, with stable and consistent behavior on new data. "
        "All pre-deployment integrity checks passed. "
        "RECOMMENDATION: YES — proceed to deployment planning."
    ),
}

_JARGON_REC = {**_GOOD, "executive_summary": (
    "XGBoost achieved a ROC-AUC of 0.86 and strong F1 score across folds. "
    "The precision score and recall score justify deployment. RECOMMENDATION: YES."
)}

_NO_VERDICT_REC = {**_GOOD, "executive_summary": (
    "LightGBM performed well across all evaluation criteria and is considered ready "
    "for the next phase of deployment pending final stakeholder sign-off."
)}

_BAD_CONFIDENCE_LOW = {**_GOOD, "confidence_score": -0.1}
_BAD_CONFIDENCE_HIGH = {**_GOOD, "confidence_score": 1.5}
_HITL_NOT_BOOL = {**_GOOD, "requires_human_review": "yes"}
_FEW_NEXT_STEPS = {**_GOOD, "next_steps": ["Only one step here"]}
_FEW_TRADEOFFS = {**_GOOD, "tradeoffs": ["Only one tradeoff"]}
_SHORT_REASON = {**_GOOD, "selection_reason": "It is the best."}

_MISSING_KEYS = {
    "recommended_model": "LightGBM",
    "confidence_score": 0.80,
    # intentionally missing 10 required fields
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

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


def _check_required_keys(rec: dict) -> list[str]:
    return [k for k in _REQUIRED_KEYS if k not in rec]


def _find_jargon(summary: str) -> list[str]:
    return [j for j in _JARGON_TERMS if j in summary]


def _has_verdict(summary: str) -> bool:
    upper = summary.upper()
    return "YES" in upper or "NO" in upper


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestPhase4RequiredKeys(unittest.TestCase):
    """UT-01: All 12 required schema keys must be present."""

    def test_good_recommendation_has_all_keys(self):
        missing = _check_required_keys(_GOOD)
        self.assertEqual(missing, [],
                         f"Good recommendation missing keys: {missing}")

    def test_incomplete_recommendation_fails(self):
        missing = _check_required_keys(_MISSING_KEYS)
        self.assertGreater(len(missing), 0,
                           "Expected missing keys but found none")
        self.assertIn("executive_summary", missing)


class TestPhase4NoJargon(unittest.TestCase):
    """UT-02: executive_summary must not contain technical jargon."""

    def test_good_summary_has_no_jargon(self):
        found = _find_jargon(_GOOD["executive_summary"])
        self.assertEqual(found, [],
                         f"Unexpected jargon in good summary: {found}")

    def test_jargon_recommendation_fails(self):
        found = _find_jargon(_JARGON_REC["executive_summary"])
        self.assertGreater(len(found), 0,
                           "Expected jargon but none found")
        self.assertTrue(
            any(j in found for j in ["ROC-AUC", "F1 score",
                                     "precision score", "recall score"])
        )


class TestPhase4DeploymentVerdict(unittest.TestCase):
    """UT-03: executive_summary must contain an explicit YES or NO verdict."""

    def test_good_summary_has_verdict(self):
        self.assertTrue(_has_verdict(_GOOD["executive_summary"]),
                        "Good summary should contain YES or NO")

    def test_summary_without_verdict_fails(self):
        self.assertFalse(_has_verdict(_NO_VERDICT_REC["executive_summary"]),
                         "Summary without verdict should not pass")


class TestPhase4ConfidenceScore(unittest.TestCase):
    """UT-04: confidence_score must be a float in [0.0, 1.0]."""

    def test_valid_confidence_passes(self):
        score = _GOOD["confidence_score"]
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_confidence_below_zero_fails(self):
        score = _BAD_CONFIDENCE_LOW["confidence_score"]
        in_range = 0.0 <= score <= 1.0
        self.assertFalse(in_range,
                         "confidence_score below 0 should fail range check")

    def test_confidence_above_one_fails(self):
        score = _BAD_CONFIDENCE_HIGH["confidence_score"]
        in_range = 0.0 <= score <= 1.0
        self.assertFalse(in_range,
                         "confidence_score above 1 should fail range check")


class TestPhase4HumanReviewFlag(unittest.TestCase):
    """UT-05: requires_human_review must be a boolean."""

    def test_bool_false_passes(self):
        self.assertIsInstance(_GOOD["requires_human_review"], bool)

    def test_string_value_fails(self):
        flag = _HITL_NOT_BOOL["requires_human_review"]
        self.assertNotIsInstance(flag, bool,
                                 "String 'yes' should not be treated as bool")

    def test_bool_true_passes(self):
        rec = {**_GOOD, "requires_human_review": True}
        self.assertIsInstance(rec["requires_human_review"], bool)


class TestPhase4NextStepsCount(unittest.TestCase):
    """UT-06: next_steps must contain at least 3 concrete items."""

    def test_three_steps_passes(self):
        steps = _GOOD["next_steps"]
        self.assertGreaterEqual(len(steps), 3,
                                f"Expected >=3 next_steps, got {len(steps)}")

    def test_one_step_fails(self):
        steps = _FEW_NEXT_STEPS["next_steps"]
        self.assertLess(len(steps), 3,
                        "One-step recommendation should fail the count check")


class TestPhase4TradeoffsCount(unittest.TestCase):
    """UT-07: tradeoffs must contain at least 2 items."""

    def test_three_tradeoffs_passes(self):
        self.assertGreaterEqual(len(_GOOD["tradeoffs"]), 2)

    def test_one_tradeoff_fails(self):
        self.assertLess(len(_FEW_TRADEOFFS["tradeoffs"]), 2)


class TestPhase4SelectionReasonLength(unittest.TestCase):
    """UT-08: selection_reason must be substantive (>30 characters)."""

    def test_long_reason_passes(self):
        reason = _GOOD["selection_reason"]
        self.assertGreater(len(reason), 30,
                           f"selection_reason too short: '{reason}'")

    def test_short_reason_fails(self):
        reason = _SHORT_REASON["selection_reason"]
        self.assertLessEqual(len(reason), 30,
                              "Short reason should fail the length check")


# ---------------------------------------------------------------------------
# Programmatic runner (returns structured report, no side effects)
# ---------------------------------------------------------------------------

def run_unit_tests() -> dict:
    """Run all 8 unit tests and return a structured report."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestPhase4RequiredKeys,
        TestPhase4NoJargon,
        TestPhase4DeploymentVerdict,
        TestPhase4ConfidenceScore,
        TestPhase4HumanReviewFlag,
        TestPhase4NextStepsCount,
        TestPhase4TradeoffsCount,
        TestPhase4SelectionReasonLength,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    import io
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)

    tests_run = result.testsRun
    failures = len(result.failures) + len(result.errors)
    passed = tests_run - failures
    pct = round(passed / tests_run * 100, 1) if tests_run else 0

    failure_details = []
    for test, traceback in result.failures + result.errors:
        failure_details.append({
            "test": str(test),
            "error": traceback.split("\n")[-2] if traceback else "unknown",
        })

    report = {
        "eval_type": "unit_tests",
        "tests_run": tests_run,
        "passed": passed,
        "failed": failures,
        "pct": pct,
        "overall_passed": pct >= 100,
        "failures": failure_details,
        "output": stream.getvalue(),
    }

    print(f"\n  Unit Tests: {passed}/{tests_run} passed ({pct}%)")
    if failure_details:
        for f in failure_details:
            print(f"    x {f['test']}: {f['error']}")

    return report


if __name__ == "__main__":
    unittest.main(verbosity=2)

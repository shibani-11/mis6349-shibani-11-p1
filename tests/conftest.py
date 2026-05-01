# tests/conftest.py
# Shared pytest fixtures for all MIRA test suites.

import pytest
from unittest.mock import MagicMock
from schemas.input_schema import AgentInput


SAMPLE_RECOMMENDATION = {
    "recommended_model": "LightGBM",
    "selection_reason": "LightGBM achieved the highest cross-validated ROC-AUC (0.88) with low variance across folds, outperforming all candidates on the inferred priority metric.",
    "primary_metric_value": 0.88,
    "all_models_summary": [
        {"name": "LightGBM", "cv_roc_auc_mean": 0.88, "rank": 1, "verdict": "SELECTED", "why_not_recommended": ""},
        {"name": "XGBoost", "cv_roc_auc_mean": 0.86, "rank": 2, "verdict": "RUNNER-UP", "why_not_recommended": "Marginally lower AUC."},
    ],
    "model_comparison_narrative": "LightGBM led all models with a cross-validated score of 0.88, followed by XGBoost at 0.86.",
    "business_impact": {
        "estimated_customers_identified": "880 of 1,000 at-risk customers correctly flagged.",
        "retention_opportunity": "Early flagging enables targeted retention offers.",
        "model_value_statement": "Estimated 15-20% reduction in monthly churn.",
    },
    "tradeoffs": [
        "Higher memory footprint than Logistic Regression",
        "Requires periodic retraining on fresh data",
    ],
    "alternative_model": "XGBoost",
    "alternative_model_reason": "XGBoost achieved 0.86 AUC with comparable stability.",
    "next_steps": [
        "Deploy in shadow mode for 30 days",
        "Monitor live false-positive rate weekly",
        "Establish quarterly retraining schedule",
    ],
    "deployment_considerations": ["Minimum 8GB RAM for batch scoring"],
    "risks": ["Model degrades if customer behaviour shifts significantly"],
    "test_verdict_summary": "PASS — no overfitting detected (gap=0.03), no data leakage, CV std=0.012.",
    "feature_drivers": [
        {"feature": "Age", "importance": 0.24, "plain_english": "Older customers churn less frequently."},
        {"feature": "Balance", "importance": 0.19, "plain_english": "Customers with zero balance churn at higher rates."},
    ],
    "confidence_score": 0.87,
    "routing_zone": "zone_1",
    "flags": [],
    "requires_human_review": False,
    "human_review_reason": None,
    "executive_summary": "LightGBM is the recommended model for deployment. It achieves 0.88 ROC-AUC with strong stability. DEPLOY: YES",
}


@pytest.fixture
def valid_agent_input():
    return AgentInput(
        dataset_path="data/raw/Churn_Modelling.csv",
        target_column="Exited",
        business_problem="Identify customers likely to churn so retention can intervene early.",
    )


@pytest.fixture
def valid_recommendation():
    return SAMPLE_RECOMMENDATION.copy()


@pytest.fixture
def minimal_invalid_recommendation():
    return {"recommended_model": "XGBoost"}


@pytest.fixture
def mock_agent_success():
    """Mock agent_fn that returns a valid recommendation on first call."""
    return MagicMock(return_value=SAMPLE_RECOMMENDATION.copy())


@pytest.fixture
def mock_agent_fail_once():
    """Mock agent_fn that raises APIRateLimitError on the first call, succeeds on the second."""
    from agent.runner import APIRateLimitError
    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise APIRateLimitError("rate limit hit")
        return SAMPLE_RECOMMENDATION.copy()

    mock = MagicMock(side_effect=side_effect)
    mock.call_count_tracker = call_count
    return mock


@pytest.fixture
def mock_agent_always_fail():
    """Mock agent_fn that always raises APIRateLimitError."""
    from agent.runner import APIRateLimitError
    return MagicMock(side_effect=APIRateLimitError("persistent rate limit"))

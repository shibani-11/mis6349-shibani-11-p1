"""
Test retry logic - Mock API failure → retry → success.
"""
import pytest
from unittest.mock import patch, MagicMock
from agent.runner import run_with_retry
from agent.logger import setup_logger


def test_retry_on_failure():
    """Test that agent retries on API failure."""
    call_count = 0
    
    def mock_failing_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("API temporarily unavailable")
        # Third call succeeds
        return {
            "dataset": "test",
            "recommended_model": "xgboost",
            "selection_reason": "Best model",
            "tradeoffs": [],
            "dataset_considerations": [],
            "requires_human_review": False
        }
    
    input_data = {
        "dataset_path": "data/test.csv",
        "model_evaluations": [
            {"model_name": "xgboost", "accuracy": 0.9, "precision": 0.9,
             "recall": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
        ]
    }
    
    logger = setup_logger("test")
    
    with patch('agent.runner.run_with_validation', side_effect=[
        Exception("API failure"),
        Exception("API failure"),
        {
            "dataset": "test",
            "recommended_model": "xgboost",
            "selection_reason": "Best model",
            "tradeoffs": [],
            "dataset_considerations": [],
            "requires_human_review": False
        }
    ]):
        output = run_with_retry(input_data, logger, max_retries=3)
    
    assert output is not None
    assert output["recommended_model"] == "xgboost"


def test_max_retries_exhausted():
    """Test that agent uses fallback after max retries."""
    input_data = {
        "dataset_path": "data/test.csv",
        "model_evaluations": [
            {"model_name": "model_a", "accuracy": 0.9, "precision": 0.9,
             "recall": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
        ]
    }
    
    logger = setup_logger("test")
    
    # Always fail
    with patch('agent.runner.run_with_validation', side_effect=Exception("API always fails")):
        output = run_with_retry(input_data, logger, max_retries=3)
    
    # Should return fallback with human review flag
    assert output is not None
    assert output["requires_human_review"] is True
    assert "error" in output


def test_retry_with_exponential_backoff():
    """Test that retry uses exponential backoff."""
    import time
    
    call_times = []
    
    def mock_call(*args, **kwargs):
        call_times.append(time.time())
        if len(call_times) < 3:
            raise Exception("API failure")
        return {
            "dataset": "test",
            "recommended_model": "xgboost",
            "selection_reason": "Best model",
            "tradeoffs": [],
            "dataset_considerations": [],
            "requires_human_review": False
        }
    
    input_data = {
        "dataset_path": "data/test.csv",
        "model_evaluations": [
            {"model_name": "xgboost", "accuracy": 0.9, "precision": 0.9,
             "recall": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
        ]
    }
    
    logger = setup_logger("test")
    
    with patch('agent.runner.run_with_validation', side_effect=mock_call):
        output = run_with_retry(input_data, logger, max_retries=3)
    
    # Check that backoff is increasing
    if len(call_times) >= 3:
        backoff1 = call_times[1] - call_times[0]
        backoff2 = call_times[2] - call_times[1]
        assert backoff2 >= backoff1  # Exponential backoff


def test_no_retry_on_success():
    """Test that agent doesn't retry on success."""
    input_data = {
        "dataset_path": "data/test.csv",
        "model_evaluations": [
            {"model_name": "xgboost", "accuracy": 0.9, "precision": 0.9,
             "recall": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
        ]
    }
    
    logger = setup_logger("test")
    
    with patch('agent.runner.run_with_validation', return_value={
        "dataset": "test",
        "recommended_model": "xgboost",
        "selection_reason": "Best model",
        "tradeoffs": [],
        "dataset_considerations": [],
        "requires_human_review": False
    }) as mock_validate:
        output = run_with_retry(input_data, logger, max_retries=3)
    
    # Should only be called once (no retry needed)
    assert mock_validate.call_count == 1
    assert output["recommended_model"] == "xgboost"

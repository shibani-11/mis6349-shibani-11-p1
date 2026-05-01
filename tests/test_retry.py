# tests/test_retry.py
# Tests for the three runner.py error-handling patterns:
#   run_with_retry, run_with_validation, run_with_fallback

import pytest
from unittest.mock import MagicMock, patch
from agent.runner import (
    run_with_retry,
    run_with_validation,
    run_with_fallback,
    APIRateLimitError,
    APITimeoutError,
    OutputValidationError,
)


def test_run_with_retry_succeeds_first_attempt(mock_agent_success):
    result = run_with_retry(mock_agent_success, {"task": "test"})
    assert result is not None
    mock_agent_success.assert_called_once()


def test_run_with_retry_succeeds_after_one_failure(mock_agent_fail_once):
    result = run_with_retry(mock_agent_fail_once, {"task": "test"}, max_retries=3)
    assert result is not None
    assert mock_agent_fail_once.call_count == 2


def test_run_with_retry_raises_after_max_retries(mock_agent_always_fail):
    with pytest.raises(APIRateLimitError):
        run_with_retry(mock_agent_always_fail, {"task": "test"}, max_retries=3)
    assert mock_agent_always_fail.call_count == 3


def test_run_with_validation_passes_valid_output(mock_agent_success, valid_recommendation):
    mock_agent_success.return_value = valid_recommendation
    with patch("agent.runner.run_with_retry", return_value=valid_recommendation):
        result = run_with_validation(mock_agent_success, {"task": "test"})
    assert result is not None
    assert result["recommended_model"] == "LightGBM"


def test_run_with_validation_raises_on_persistent_bad_output():
    bad_output = {"recommended_model": "XGBoost"}
    agent_fn = MagicMock(return_value=bad_output)
    with patch("agent.runner.run_with_retry", return_value=bad_output):
        with pytest.raises(OutputValidationError):
            run_with_validation(agent_fn, {"task": "test"})


def test_run_with_fallback_routes_to_human_review_on_exception():
    failing_primary = MagicMock(side_effect=APIRateLimitError("limit"))
    failing_fallback = MagicMock(side_effect=APIRateLimitError("limit"))

    with patch("agent.runner.run_with_validation", side_effect=Exception("both failed")):
        result = run_with_fallback(failing_primary, failing_fallback, {"task": "test"})

    assert result["status"] == "NEEDS_REVIEW"
    assert "ESCALATED" in result["flags"]
    assert result["confidence_score"] == 0.0


def test_run_with_retry_handles_timeout_error():
    agent_fn = MagicMock(side_effect=APITimeoutError("timeout"))
    with pytest.raises(APITimeoutError):
        run_with_retry(agent_fn, {"task": "test"}, max_retries=2)
    assert agent_fn.call_count == 2

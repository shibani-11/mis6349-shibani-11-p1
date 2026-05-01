# tests/test_happy_path.py
# Happy path tests: valid input → correct output structure and field values.

import pytest
from agent.validator import validate_output
from schemas.input_schema import AgentInput


def test_valid_recommendation_passes_validation(valid_recommendation):
    result, errors = validate_output(valid_recommendation)
    assert errors == [], f"Unexpected errors: {errors}"
    assert result is not None


def test_all_required_keys_present(valid_recommendation):
    from agent.validator import REQUIRED_RECOMMENDATION_KEYS
    result, errors = validate_output(valid_recommendation)
    assert errors == []
    for key in REQUIRED_RECOMMENDATION_KEYS:
        assert key in result, f"Missing key in validated output: {key}"


def test_confidence_score_in_valid_range(valid_recommendation):
    result, errors = validate_output(valid_recommendation)
    assert errors == []
    assert 0.0 <= result["confidence_score"] <= 1.0


def test_routing_zone_valid_value(valid_recommendation):
    result, errors = validate_output(valid_recommendation)
    assert errors == []
    assert result["routing_zone"] in ("zone_1", "zone_2", "zone_3")


def test_flags_is_list(valid_recommendation):
    result, errors = validate_output(valid_recommendation)
    assert errors == []
    assert isinstance(result["flags"], list)


def test_executive_summary_has_verdict(valid_recommendation):
    result, errors = validate_output(valid_recommendation)
    assert errors == []
    summary_upper = result["executive_summary"].upper()
    assert "YES" in summary_upper or "NO" in summary_upper


def test_requires_human_review_is_bool(valid_recommendation):
    result, errors = validate_output(valid_recommendation)
    assert errors == []
    assert isinstance(result["requires_human_review"], bool)


def test_agent_input_auto_generates_run_id(valid_agent_input):
    assert valid_agent_input.run_id.startswith("run_")
    assert len(valid_agent_input.run_id) > 4


def test_agent_input_defaults_are_set(valid_agent_input):
    assert valid_agent_input.task_type == "auto"
    assert valid_agent_input.max_models == 5
    assert valid_agent_input.max_iterations == 40

# tests/test_edge_cases.py
# Edge case and boundary tests: malformed/empty/invalid inputs are handled without crash.

import pytest
from pydantic import ValidationError
from agent.validator import validate_output
from schemas.input_schema import AgentInput


def test_validate_output_rejects_non_dict():
    result, errors = validate_output("not a dict")
    assert result is None
    assert any("dict" in e for e in errors)


def test_validate_output_rejects_missing_keys(minimal_invalid_recommendation):
    result, errors = validate_output(minimal_invalid_recommendation)
    assert result is None
    assert any("Missing required keys" in e for e in errors)


def test_validate_output_rejects_confidence_out_of_range(valid_recommendation):
    rec = valid_recommendation.copy()
    rec["confidence_score"] = 1.5
    result, errors = validate_output(rec)
    assert result is None
    assert any("confidence_score" in e for e in errors)


def test_validate_output_rejects_invalid_routing_zone(valid_recommendation):
    rec = valid_recommendation.copy()
    rec["routing_zone"] = "zone_9"
    result, errors = validate_output(rec)
    assert result is None
    assert any("routing_zone" in e for e in errors)


def test_validate_output_rejects_flags_as_string(valid_recommendation):
    rec = valid_recommendation.copy()
    rec["flags"] = "DATA_LEAKAGE_DETECTED"
    result, errors = validate_output(rec)
    assert result is None
    assert any("flags" in e for e in errors)


def test_validate_output_rejects_requires_human_review_as_string(valid_recommendation):
    rec = valid_recommendation.copy()
    rec["requires_human_review"] = "yes"
    result, errors = validate_output(rec)
    assert result is None
    assert any("requires_human_review" in e for e in errors)


def test_validate_output_rejects_executive_summary_without_verdict(valid_recommendation):
    rec = valid_recommendation.copy()
    rec["executive_summary"] = "This model performed well on the dataset."
    result, errors = validate_output(rec)
    assert result is None
    assert any("executive_summary" in e for e in errors)


def test_agent_input_rejects_unsupported_file_format():
    with pytest.raises(ValidationError):
        AgentInput(
            dataset_path="data/test.pdf",
            target_column="label",
            business_problem="Test",
        )


def test_agent_input_rejects_blank_target_column():
    with pytest.raises(ValidationError):
        AgentInput(
            dataset_path="data/test.csv",
            target_column="   ",
            business_problem="Test",
        )


def test_agent_input_rejects_blank_business_problem():
    with pytest.raises(ValidationError):
        AgentInput(
            dataset_path="data/test.csv",
            target_column="label",
            business_problem="",
        )


def test_validate_output_rejects_empty_dict():
    result, errors = validate_output({})
    assert result is None
    assert errors

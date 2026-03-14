"""
Test happy path - Valid input produces expected output format.
"""
import pytest
from agent.runner import run_with_validation
from agent.logger import setup_logger


def test_valid_input_produces_valid_output(sample_input):
    """Test that valid input produces a valid output format."""
    logger = setup_logger("test")
    output = run_with_validation(sample_input, logger)
    
    assert output is not None
    assert "recommended_model" in output
    assert "selection_reason" in output
    assert "tradeoffs" in output
    assert "dataset_considerations" in output
    assert "requires_human_review" in output


def test_output_is_json_serializable(sample_input):
    """Test that output can be serialized to JSON."""
    import json
    logger = setup_logger("test")
    output = run_with_validation(sample_input, logger)
    
    # Should not raise
    json_str = json.dumps(output)
    parsed = json.loads(json_str)
    assert parsed == output


def test_recommended_model_is_string(sample_input):
    """Test that recommended_model is a string."""
    logger = setup_logger("test")
    output = run_with_validation(sample_input, logger)
    
    assert isinstance(output["recommended_model"], str)


def test_tradeoffs_is_list(sample_input):
    """Test that tradeoffs is a list."""
    logger = setup_logger("test")
    output = run_with_validation(sample_input, logger)
    
    assert isinstance(output["tradeoffs"], list)


def test_requires_human_review_is_boolean(sample_input):
    """Test that requires_human_review is a boolean."""
    logger = setup_logger("test")
    output = run_with_validation(sample_input, logger)
    
    assert isinstance(output["requires_human_review"], bool)

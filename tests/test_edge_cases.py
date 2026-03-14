"""
Test edge cases - Malformed/empty/boundary inputs handled without crash.
"""
import pytest
from agent.runner import run_with_retry, run_with_validation
from agent.validator import validate_input, validate_output
from agent.logger import setup_logger


def test_empty_model_evaluations():
    """Test handling of empty model evaluations list."""
    input_data = {
        "dataset_path": "data/test.csv",
        "model_evaluations": []
    }
    
    logger = setup_logger("test")
    output = run_with_validation(input_data, logger)
    
    # Should still produce output (with human review flag)
    assert output is not None


def test_missing_required_field():
    """Test handling of missing required input fields."""
    input_data = {
        "dataset_path": "data/test.csv"
        # Missing model_evaluations
    }
    
    is_valid, error = validate_input(input_data)
    assert not is_valid
    assert "model_evaluations" in error


def test_missing_model_fields():
    """Test handling of model evaluation with missing fields."""
    input_data = {
        "dataset_path": "data/test.csv",
        "model_evaluations": [
            {"model_name": "model_a"}  # Missing metrics
        ]
    }
    
    is_valid, error = validate_input(input_data)
    assert not is_valid


def test_invalid_metric_type():
    """Test handling of invalid metric types."""
    input_data = {
        "dataset_path": "data/test.csv",
        "model_evaluations": [
            {"model_name": "model_a", "accuracy": "invalid", "precision": 0.9,
             "recall": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
        ]
    }
    
    is_valid, error = validate_input(input_data)
    assert not is_valid


def test_single_model():
    """Test handling of single model in evaluations."""
    input_data = {
        "dataset_path": "data/test.csv",
        "model_evaluations": [
            {"model_name": "model_a", "accuracy": 0.9, "precision": 0.9,
             "recall": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
        ]
    }
    
    logger = setup_logger("test")
    output = run_with_validation(input_data, logger)
    
    assert output is not None
    assert output["recommended_model"] == "model_a"


def test_many_models():
    """Test handling of many models in evaluations."""
    input_data = {
        "dataset_path": "data/test.csv",
        "model_evaluations": [
            {"model_name": f"model_{i}", "accuracy": 0.8 + i*0.01, "precision": 0.8 + i*0.01,
             "recall": 0.8 + i*0.01, "f1_score": 0.8 + i*0.01, "roc_auc": 0.8 + i*0.01}
            for i in range(20)
        ]
    }
    
    logger = setup_logger("test")
    output = run_with_validation(input_data, logger)
    
    assert output is not None
    assert len(output["tradeoffs"]) >= 0


def test_extreme_metric_values():
    """Test handling of extreme metric values."""
    input_data = {
        "dataset_path": "data/test.csv",
        "model_evaluations": [
            {"model_name": "model_a", "accuracy": 1.0, "precision": 1.0,
             "recall": 1.0, "f1_score": 1.0, "roc_auc": 1.0}
        ]
    }
    
    logger = setup_logger("test")
    output = run_with_validation(input_data, logger)
    
    assert output is not None
    assert output["recommended_model"] == "model_a"


def test_zero_metric_values():
    """Test handling of zero metric values."""
    input_data = {
        "dataset_path": "data/test.csv",
        "model_evaluations": [
            {"model_name": "model_a", "accuracy": 0.0, "precision": 0.0,
             "recall": 0.0, "f1_score": 0.0, "roc_auc": 0.0}
        ]
    }
    
    logger = setup_logger("test")
    output = run_with_validation(input_data, logger)
    
    assert output is not None
    assert output["requires_human_review"] is True

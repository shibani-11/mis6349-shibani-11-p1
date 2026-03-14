"""
pytest fixtures for agent tests.
"""
import pytest
import json
from pathlib import Path
from agent.main import main
from agent.runner import run_with_retry, run_with_validation
from agent.validator import validate_output, validate_input


@pytest.fixture
def sample_input():
    """Sample valid input for testing."""
    return {
        "dataset_path": "data/test_dataset.csv",
        "model_evaluations": [
            {
                "model_name": "logistic_regression",
                "accuracy": 0.85,
                "precision": 0.90,
                "recall": 0.60,
                "f1_score": 0.72,
                "roc_auc": 0.81
            },
            {
                "model_name": "random_forest",
                "accuracy": 0.88,
                "precision": 0.84,
                "recall": 0.81,
                "f1_score": 0.82,
                "roc_auc": 0.87
            },
            {
                "model_name": "xgboost",
                "accuracy": 0.89,
                "precision": 0.80,
                "recall": 0.90,
                "f1_score": 0.85,
                "roc_auc": 0.92
            }
        ]
    }


@pytest.fixture
def mock_api(monkeypatch):
    """Mock API for testing."""
    class MockAPI:
        def __init__(self):
            self.call_count = 0
            self.should_fail = False
            
        def call(self, prompt):
            self.call_count += 1
            if self.should_fail:
                raise Exception("API call failed")
            return {"recommended_model": "xgboost", "selection_reason": "Test"}
    
    return MockAPI()


@pytest.fixture
def agent_instance():
    """Create an agent instance for testing."""
    from agent.runner import run_with_retry
    return run_with_retry


@pytest.fixture
def sample_inputs():
    """Multiple sample inputs for edge case testing."""
    return {
        "valid": {
            "dataset_path": "data/test.csv",
            "model_evaluations": [
                {"model_name": "model_a", "accuracy": 0.9, "precision": 0.9, 
                 "recall": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
            ]
        },
        "empty_evaluations": {
            "dataset_path": "data/test.csv",
            "model_evaluations": []
        },
        "missing_fields": {
            "dataset_path": "data/test.csv",
            "model_evaluations": [
                {"model_name": "model_a", "accuracy": 0.9}
            ]
        },
        "invalid_metrics": {
            "dataset_path": "data/test.csv",
            "model_evaluations": [
                {"model_name": "model_a", "accuracy": "invalid", "precision": 0.9,
                 "recall": 0.9, "f1_score": 0.9, "roc_auc": 0.9}
            ]
        }
    }

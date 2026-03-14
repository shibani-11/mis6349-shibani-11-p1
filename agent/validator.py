"""
Output schema validation using pydantic.
"""
from typing import Tuple, Any, Dict


def validate_output(output: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate the agent output against the expected schema.
    
    Args:
        output: The agent's output dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = [
        "dataset",
        "recommended_model", 
        "selection_reason",
        "tradeoffs",
        "dataset_considerations",
        "requires_human_review"
    ]
    
    # Check required fields
    for field in required_fields:
        if field not in output:
            return False, f"Missing required field: {field}"
    
    # Validate types
    if not isinstance(output["recommended_model"], str):
        return False, "recommended_model must be a string"
    
    if not isinstance(output["selection_reason"], str):
        return False, "selection_reason must be a string"
    
    if not isinstance(output["tradeoffs"], list):
        return False, "tradeoffs must be a list"
    
    if not isinstance(output["dataset_considerations"], list):
        return False, "dataset_considerations must be a list"
    
    if not isinstance(output["requires_human_review"], bool):
        return False, "requires_human_review must be a boolean"
    
    return True, ""


def validate_input(input_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate the input data against the expected schema.
    
    Args:
        input_data: The input data dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if "model_evaluations" not in input_data:
        return False, "Missing required field: model_evaluations"
    
    if not isinstance(input_data["model_evaluations"], list):
        return False, "model_evaluations must be a list"
    
    if len(input_data["model_evaluations"]) == 0:
        return False, "model_evaluations cannot be empty"
    
    required_eval_fields = ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    
    for eval_model in input_data["model_evaluations"]:
        for field in required_eval_fields:
            if field not in eval_model:
                return False, f"Missing required field in model evaluation: {field}"
    
    return True, ""

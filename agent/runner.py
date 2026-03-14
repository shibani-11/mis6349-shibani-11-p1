"""
Agent runner with retry, validation, and fallback mechanisms.
"""
import time
from typing import Dict, Any, Optional
from agent.validator import validate_output
from agent.logger import get_logger


def run_with_retry(input_data: Dict[str, Any], logger, max_retries: int = 3) -> Dict[str, Any]:
    """
    Run the agent with retry logic.
    
    Args:
        input_data: The input data containing dataset_path and model_evaluations
        logger: Logger instance
        max_retries: Maximum number of retry attempts
        
    Returns:
        The agent's recommendation output
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = run_with_validation(input_data, logger)
            if result is not None:
                return result
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
        
        if attempt < max_retries - 1:
            time.sleep(1 * (attempt + 1))  # Exponential backoff
    
    # All retries failed, return error result
    return run_with_fallback(input_data, logger, str(last_error))


def run_with_validation(input_data: Dict[str, Any], logger) -> Optional[Dict[str, Any]]:
    """
    Run the agent with output validation.
    
    Args:
        input_data: The input data
        logger: Logger instance
        
    Returns:
        Validated output or None if validation fails
    """
    # TODO: Implement actual agent logic here
    # This is a placeholder that returns a mock response
    output = {
        "dataset": "unknown",
        "recommended_model": "unknown",
        "selection_reason": "Agent not yet implemented",
        "tradeoffs": [],
        "dataset_considerations": [],
        "requires_human_review": True
    }
    
    # Validate output
    is_valid, error = validate_output(output)
    if not is_valid:
        logger.error(f"Output validation failed: {error}")
        return None
    
    return output


def run_with_fallback(input_data: Dict[str, Any], logger, error: str) -> Dict[str, Any]:
    """
    Run the agent with fallback mechanism when all else fails.
    
    Args:
        input_data: The input data
        logger: Logger instance
        error: The error message from failed attempts
        
    Returns:
        Fallback output indicating human review is needed
    """
    logger.error(f"All attempts failed, using fallback: {error}")
    
    return {
        "dataset": input_data.get("dataset_path", "unknown"),
        "recommended_model": "unknown",
        "selection_reason": f"Agent failed to produce valid recommendation: {error}",
        "tradeoffs": [],
        "dataset_considerations": [],
        "requires_human_review": True,
        "error": error
    }

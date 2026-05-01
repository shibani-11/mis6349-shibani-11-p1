# agent/runner.py
# Three error-handling patterns required by the Production Bar rubric.
# All agent calls in main.py should route through these functions.

import time
import json
import hashlib
from agent.validator import validate_output
from agent.logger import RunLogger

logger = RunLogger()


class APIRateLimitError(Exception):
    pass


class APITimeoutError(Exception):
    pass


class OutputValidationError(Exception):
    pass


def run_with_retry(agent_fn, task: dict, max_retries: int = 3):
    """
    Pattern 1 — Exponential backoff retry.
    Handles transient API failures (rate limits, timeouts).
    Re-raises on final attempt — never swallows exceptions.
    """
    for attempt in range(max_retries):
        try:
            return agent_fn(task)
        except (APIRateLimitError, APITimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt   # 1s, 2s, 4s
            time.sleep(wait)


def run_with_validation(agent_fn, task: dict, max_corrections: int = 1):
    """
    Pattern 2 — Output schema validation with one self-correction attempt.
    If validation fails twice, the error surfaces — never silently passes bad output.
    """
    output = run_with_retry(agent_fn, task)
    validated, errors = validate_output(output)

    if errors and max_corrections > 0:
        correction_task = _build_correction_prompt(task, output, errors)
        output = run_with_retry(agent_fn, correction_task)
        validated, errors = validate_output(output)

    if errors:
        raise OutputValidationError(f"Validation failed after correction: {errors}")

    return validated


def run_with_fallback(primary_fn, fallback_fn, task: dict,
                      confidence_threshold: float = 0.70):
    """
    Pattern 3 — Graceful degradation.
    Routes to fallback if confidence is low. Routes to human review if fallback also fails.
    Never returns None — always has a safe handoff.
    """
    try:
        result = run_with_validation(primary_fn, task)
        if result.confidence_score < confidence_threshold:
            result = run_with_validation(fallback_fn, task)
    except Exception as e:
        return _route_to_human_review(task, error=str(e))

    if result is None:
        return _route_to_human_review(task, error="Both agents returned None")

    return result


def _build_correction_prompt(original_task: dict, bad_output: dict,
                              errors: list) -> dict:
    correction = original_task.copy()
    correction["_correction_context"] = {
        "previous_output": bad_output,
        "validation_errors": errors,
        "instruction": (
            "Your previous output failed schema validation. "
            "Correct the following errors and try again: "
            + "; ".join(errors)
        ),
    }
    return correction


def _route_to_human_review(task: dict, error: str = None) -> dict:
    logger.log_escalation(task, error)
    return {
        "status": "NEEDS_REVIEW",
        "result": "Agent could not complete this task. Routed to human review.",
        "error": error,
        "confidence_score": 0.0,
        "flags": ["ESCALATED"],
        "reasoning_steps": [],
    }

# agent/validator.py
# Output schema validation for MIRA recommendation output.
# Called by runner.py after every agent response before downstream use.

from pydantic import ValidationError
from typing import Any


REQUIRED_RECOMMENDATION_KEYS = [
    "recommended_model", "selection_reason", "primary_metric_value",
    "all_models_summary", "model_comparison_narrative", "business_impact",
    "tradeoffs", "alternative_model", "alternative_model_reason", "next_steps",
    "deployment_considerations", "risks", "test_verdict_summary",
    "feature_drivers", "confidence_score", "routing_zone", "flags",
    "requires_human_review", "human_review_reason", "executive_summary",
]


def validate_output(raw_output: Any) -> tuple:
    """
    Returns (validated_output, errors).
    If validation passes: (dict, [])
    If validation fails: (None, [error messages])

    Checks that raw_output is a dict containing all required recommendation keys
    and that key field types are correct.
    """
    if not isinstance(raw_output, dict):
        return None, [f"Output must be a dict, got {type(raw_output).__name__}"]

    errors = []

    missing = [k for k in REQUIRED_RECOMMENDATION_KEYS if k not in raw_output]
    if missing:
        errors.append(f"Missing required keys: {missing}")

    confidence = raw_output.get("confidence_score")
    if confidence is not None:
        if not isinstance(confidence, (int, float)):
            errors.append(f"confidence_score must be float, got {type(confidence).__name__}")
        elif not (0.0 <= float(confidence) <= 1.0):
            errors.append(f"confidence_score {confidence} is outside [0.0, 1.0]")

    flags = raw_output.get("flags")
    if flags is not None and not isinstance(flags, list):
        errors.append(f"flags must be a list, got {type(flags).__name__}")

    routing_zone = raw_output.get("routing_zone")
    if routing_zone is not None and routing_zone not in ("zone_1", "zone_2", "zone_3"):
        errors.append(f"routing_zone '{routing_zone}' must be zone_1, zone_2, or zone_3")

    requires_review = raw_output.get("requires_human_review")
    if requires_review is not None and not isinstance(requires_review, bool):
        errors.append(f"requires_human_review must be bool, got {type(requires_review).__name__}")

    executive_summary = raw_output.get("executive_summary", "")
    if isinstance(executive_summary, str):
        upper = executive_summary.upper()
        if "YES" not in upper and "NO" not in upper:
            errors.append("executive_summary must end with YES or NO deployment verdict")

    if errors:
        return None, errors

    return raw_output, []

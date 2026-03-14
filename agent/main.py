"""
Main entry point for the ML Model Recommendation Agent.
Runs the agent end-to-end: reads input, analyzes data, generates recommendation.
"""
import json
from pathlib import Path
from agent.runner import run_with_retry
from agent.logger import setup_logger


def main():
    """Run the agent with the given input file."""
    logger = setup_logger()
    
    # Default input path - can be made configurable
    input_path = Path("input.json")
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        print(json.dumps({
            "error": "Input file not found",
            "requires_human_review": True
        }))
        return
    
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    
    result = run_with_retry(input_data, logger)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""
Runs agent on eval set, outputs scorecard.
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.runner import run_with_retry
from agent.logger import setup_logger


def load_eval_dataset(eval_file: str) -> list:
    """Load evaluation dataset from file."""
    with open(eval_file, 'r') as f:
        return json.load(f)


def run_eval(eval_file: str, output_file: str = None) -> dict:
    """
    Run agent on evaluation set.
    
    Args:
        eval_file: Path to evaluation dataset JSON file
        output_file: Optional path to save results
        
    Returns:
        Dictionary containing evaluation results
    """
    logger = setup_logger("eval")
    
    logger.info(f"Loading eval dataset from {eval_file}")
    eval_cases = load_eval_dataset(eval_file)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(eval_cases),
        "successful": 0,
        "failed": 0,
        "requires_human_review": 0,
        "cases": []
    }
    
    for i, case in enumerate(eval_cases):
        logger.info(f"Running eval case {i+1}/{len(eval_cases)}")
        
        try:
            output = run_with_retry(case, logger)
            
            case_result = {
                "case_id": case.get("id", i),
                "success": output.get("requires_human_review", True) is False,
                "requires_human_review": output.get("requires_human_review", True),
                "recommended_model": output.get("recommended_model"),
                "output": output
            }
            
            if output.get("requires_human_review"):
                results["requires_human_review"] += 1
                results["failed"] += 1
            else:
                results["successful"] += 1
                
        except Exception as e:
            logger.error(f"Case {i+1} failed with exception: {str(e)}")
            case_result = {
                "case_id": case.get("id", i),
                "success": False,
                "error": str(e)
            }
            results["failed"] += 1
        
        results["cases"].append(case_result)
    
    # Calculate metrics
    results["success_rate"] = results["successful"] / results["total_cases"] if results["total_cases"] > 0 else 0
    results["human_review_rate"] = results["requires_human_review"] / results["total_cases"] if results["total_cases"] > 0 else 0
    
    logger.info(f"Eval complete: {results['successful']}/{results['total_cases']} successful")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return results


def print_scorecard(results: dict):
    """Print evaluation scorecard."""
    print("\n" + "="*50)
    print("EVALUATION SCORECARD")
    print("="*50)
    print(f"Total cases:      {results['total_cases']}")
    print(f"Successful:       {results['successful']}")
    print(f"Failed:           {results['failed']}")
    print(f"Human review:    {results['requires_human_review']}")
    print("-"*50)
    print(f"Success rate:     {results['success_rate']*100:.1f}%")
    print(f"Human review rate: {results['human_review_rate']*100:.1f}%")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run agent on evaluation set")
    parser.add_argument("--eval-file", required=True, help="Path to evaluation dataset")
    parser.add_argument("--output", help="Path to save results JSON")
    args = parser.parse_args()
    
    results = run_eval(args.eval_file, args.output)
    print_scorecard(results)


if __name__ == "__main__":
    main()

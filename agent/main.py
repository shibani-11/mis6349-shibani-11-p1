import json
from pathlib import Path

from agent.runner import run_with_retry
from agent.logger import log_run
from schemas.input_schema import AgentInput


INPUT_PATH = Path("data/processed/loan_model_results.json")


def select_best_model(data):

    models = data["models"]

    models_sorted = sorted(
        models,
        key=lambda x: x["metrics"]["f1"],
        reverse=True
    )

    best = models_sorted[0]
    runner_up = models_sorted[1]

    return {
        "recommended_model": best["name"],
        "runner_up_model": runner_up["name"],
        "problem_type": data["problem_type"],
        "decision_summary": f"{best['name']} achieved the highest F1 score.",
        "primary_reason": "Best predictive performance.",
        "tradeoffs": ["Higher complexity"],
        "risk_flags": [],
        "metrics_considered": best["metrics"],
        "deployment_readiness": {
            "status": "ready",
            "notes": "Suitable for deployment"
        }
    }


def run_agent(input_data):

    return select_best_model(input_data)


def main():

    with open(INPUT_PATH) as f:
        data = json.load(f)

    AgentInput(**data)

    output = run_with_retry(run_agent, data)

    log_run(data, output)

    print("Recommended Model:", output["recommended_model"])


if __name__ == "__main__":
    main()

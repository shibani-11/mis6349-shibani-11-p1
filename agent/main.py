# agent/main.py
# Entry point for the ML Engineer Agent

from agent.agent import MLEngineerAgent
from schemas.input_schema import AgentInput


def main():
    agent_input = AgentInput(
        # ── Change these 3 lines for any new dataset ──
        dataset_path="data/loan_default_dataset.xlsx",
        target_column="default",
        business_problem=(
            "A bank wants to identify which loan applicants are likely "
            "to default so they can reduce credit risk and make better "
            "lending decisions."
        ),
        # ─────────────────────────────────────────────
        task_type="auto",
        max_models=5,
        max_iterations=40,
        extra_context={
            "domain": "finance",
            "priority_metric": "roc_auc"
        }
    )

    agent = MLEngineerAgent(agent_input)
    agent.run()


if __name__ == "__main__":
    main()

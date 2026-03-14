import json
from pathlib import Path

from openhands.sdk import Agent, Conversation, LLM

from agent.runner import run_with_retry
from agent.validator import validate_output
from agent.logger import log_run

from schemas.input_schema import AgentInput


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data/processed/loan_model_results.json"
PROMPT_FILE = ROOT / "prompts/v1.0.0_system.txt"


def load_prompt():

    with open(PROMPT_FILE) as f:
        return f.read()


def build_agent():

    # Uses your configured LLM subscription automatically
    llm = LLM()

    agent = Agent(
        llm=llm,
        system_prompt=load_prompt()
    )

    return agent


def run_agent(agent, input_data):

    conversation = Conversation(agent=agent)

    user_prompt = f"""
Analyze the following ML model evaluation results and recommend the best model for deployment.

Input data:

{json.dumps(input_data, indent=2)}

Return JSON only.
"""

    conversation.send_message(user_prompt)

    response = conversation.run()

    return json.loads(response)


def main():

    with open(INPUT_FILE) as f:
        data = json.load(f)

    # validate input schema
    AgentInput(**data)

    agent = build_agent()

    output = run_with_retry(
        lambda d: run_agent(agent, d),
        data
    )

    validate_output(output)

    log_run(data, output)

    print("\nRecommended Model:", output["recommended_model"])


if __name__ == "__main__":
    main()

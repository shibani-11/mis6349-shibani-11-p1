import json
from pathlib import Path
from datetime import datetime

from openhands.sdk import Agent, Conversation, LLM

from agent.validator import validate_output
from agent.logger import log_run


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"

INPUT_FILE = DATA_DIR / "model_results.json"


def load_prompt(version: str = "v1.0.0"):
    prompt_path = PROMPTS_DIR / f"{version}_system.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    return prompt_path.read_text()


def load_input():
    if not INPUT_FILE.exists():
        raise FileNotFoundError("model_results.json not found in data/")

    with open(INPUT_FILE) as f:
        return json.load(f)


def build_user_prompt(model_results: dict):

    return f"""
You are an ML Model Recommendation Agent.

Analyze the following model evaluation metrics and determine the best model.

Model Results:
{json.dumps(model_results, indent=2)}

Return output strictly as JSON with the following fields:
- best_model
- reasoning
- metrics_summary
"""


def run_agent(prompt_version="v1.0.0"):

    system_prompt = load_prompt(prompt_version)
    model_results = load_input()

    user_prompt = build_user_prompt(model_results)

    llm = LLM()

    agent = Agent(
        llm=llm,
        system_prompt=system_prompt
    )

    conversation = Conversation(agent=agent)

    conversation.send_message(user_prompt)

    response = conversation.run()

    return response


def main():

    start_time = datetime.utcnow()

    try:

        output = run_agent()

        validated_output = validate_output(output)

        log_run(
            success=True,
            output=validated_output,
            latency=str(datetime.utcnow() - start_time)
        )

        print("\nAgent Recommendation:\n")
        print(validated_output)

    except Exception as e:

        log_run(
            success=False,
            output=str(e),
            latency=str(datetime.utcnow() - start_time)
        )

        print("Agent failed:", e)


if __name__ == "__main__":
    main()

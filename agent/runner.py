import time
from agent.validator import validate_output


def run_with_retry(agent_function, input_data, retries=3):

    for attempt in range(retries):

        try:

            output = agent_function(input_data)

            if validate_output(output):
                return output

        except Exception as e:

            print("Attempt failed:", e)

        time.sleep(2 ** attempt)

    raise Exception("Agent failed after retries")

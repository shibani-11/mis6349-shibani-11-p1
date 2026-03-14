from schemas.output_schema import AgentOutput


def validate_output(data):

    try:
        AgentOutput(**data)
        return True

    except Exception as e:
        print("Schema validation failed:", e)
        return False

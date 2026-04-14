from schemas.output_schema import AnalysisReport


def validate_output(data):
    try:
        AnalysisReport(**data)
        return True
    except Exception as e:
        print("Schema validation failed:", e)
        return False
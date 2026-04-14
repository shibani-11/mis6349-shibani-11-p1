import json
from pathlib import Path
from datetime import datetime


LOG_DIR = Path("logs/runs")


def log_run(input_data, output_data):

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    log_file = LOG_DIR / f"run_{timestamp}.json"

    log = {
        "timestamp": timestamp,
        "input": input_data,
        "output": output_data
    }

    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)

    print("Run logged:", log_file)

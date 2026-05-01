# agent/logger.py
# Structured run logger — every agent run writes a JSON log entry.
# RunLogger is used by runner.py and mira_agent.py.

import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone


class RunLogger:
    def __init__(self, log_dir: str = "logs/runs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        run_id: str,
        input_data: dict,
        output: dict,
        prompt_version: str,
        latency_ms: int,
        tool_calls: list = None,
        error: str = None,
        retry_count: int = 0,
    ) -> str:
        entry = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_version": prompt_version,
            "input_hash": hashlib.sha256(
                json.dumps(input_data, sort_keys=True).encode()
            ).hexdigest()[:12],
            "output": output,
            "latency_ms": latency_ms,
            "tool_calls": tool_calls or [],
            "error": error,
            "retry_count": retry_count,
        }
        path = self.log_dir / f"{run_id}_run.json"
        path.write_text(json.dumps(entry, indent=2))
        return run_id

    def log_escalation(self, task: dict, error: str = None):
        entry = {
            "type": "ESCALATION",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_hash": hashlib.sha256(
                json.dumps(task, sort_keys=True).encode()
            ).hexdigest()[:12],
            "error": error,
        }
        path = self.log_dir / f"escalation_{int(time.time())}.json"
        path.write_text(json.dumps(entry, indent=2))

#!/usr/bin/env python3
"""
scripts/analyze_logs.py
Parse logs/runs/ for latency, failure rate, and cost statistics.
Run: python scripts/analyze_logs.py
"""
import json
import sys
import argparse
from pathlib import Path


LOG_DIR = Path("logs/runs")


def load_run_logs(log_dir: Path) -> list:
    logs = []
    for path in sorted(log_dir.glob("*_run.json")):
        try:
            data = json.loads(path.read_text())
            data["_source_file"] = path.name
            logs.append(data)
        except (json.JSONDecodeError, IOError):
            print(f"  WARNING: Could not parse {path.name}", file=sys.stderr)
    return logs


def percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def print_latency_stats(logs: list):
    durations = [r["duration_seconds"] for r in logs if "duration_seconds" in r]
    if not durations:
        print("  No latency data found.")
        return
    print(f"\n-- Latency (seconds) ------------------------------")
    print(f"  Runs with timing data : {len(durations)}")
    print(f"  p50                   : {percentile(durations, 50):.1f}s")
    print(f"  p90                   : {percentile(durations, 90):.1f}s")
    print(f"  p95                   : {percentile(durations, 95):.1f}s")
    print(f"  Min                   : {min(durations):.1f}s")
    print(f"  Max                   : {max(durations):.1f}s")
    print(f"  Mean                  : {sum(durations)/len(durations):.1f}s")

    slowest = max(logs, key=lambda r: r.get("duration_seconds", 0))
    print(f"  Slowest run           : {slowest.get('run_id', '?')} ({slowest.get('duration_seconds', 0):.1f}s)")


def print_failure_stats(logs: list):
    total = len(logs)
    if total == 0:
        print("  No run logs found.")
        return

    hard_failures  = [r for r in logs if not r.get("success", True)]
    successes      = [r for r in logs if r.get("success", False)]
    no_recommend   = [r for r in logs if r.get("success", False) and not r.get("recommended_model")]

    print(f"\n-- Failure Rate -----------------------------------")
    print(f"  Total runs            : {total}")
    print(f"  Successes             : {len(successes)}  ({len(successes)/total*100:.1f}%)")
    print(f"  Hard failures         : {len(hard_failures)}  ({len(hard_failures)/total*100:.1f}%)")
    print(f"  No recommendation     : {len(no_recommend)}  ({len(no_recommend)/total*100:.1f}%)")


def print_cost_stats(logs: list):
    cost_logs = [r for r in logs if "cost_tracking" in r]
    if not cost_logs:
        print(f"\n-- Cost Tracking ----------------------------------")
        print("  No cost_tracking data found (requires MIRA v0.5.0+).")
        return

    costs   = [r["cost_tracking"].get("eda_cost_usd", 0.0) for r in cost_logs]
    tokens  = [r["cost_tracking"].get("eda_total_tokens", 0) for r in cost_logs]

    print(f"\n-- Cost (EDA metric inference call) --------------─")
    print(f"  Runs with cost data   : {len(cost_logs)}")
    print(f"  Avg EDA tokens/run    : {sum(tokens)/len(tokens):.0f}")
    print(f"  Avg EDA cost/run      : ${sum(costs)/len(costs):.6f}")
    print(f"  Total EDA cost        : ${sum(costs):.4f}")


def print_model_stats(logs: list):
    models = [r.get("recommended_model") for r in logs if r.get("recommended_model")]
    if not models:
        return
    counts = {}
    for m in models:
        counts[m] = counts.get(m, 0) + 1
    print(f"\n-- Most Recommended Models ------------------------")
    for model, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {model:<30} {count} runs  ({count/len(models)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze MIRA run logs")
    parser.add_argument("--log-dir", default=str(LOG_DIR), help="Path to logs/runs/")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        sys.exit(1)

    logs = load_run_logs(log_dir)
    if not logs:
        print(f"No *_run.json files found in {log_dir}")
        sys.exit(0)

    print(f"\n{'='*52}")
    print(f"  MIRA Run Log Analysis  ({len(logs)} runs in {log_dir})")
    print(f"{'='*52}")

    print_latency_stats(logs)
    print_failure_stats(logs)
    print_cost_stats(logs)
    print_model_stats(logs)
    print()


if __name__ == "__main__":
    main()

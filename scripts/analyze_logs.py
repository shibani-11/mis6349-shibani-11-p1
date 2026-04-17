# scripts/analyze_logs.py
# Analyze MIRA run logs and produce metrics report

import json
import os
from pathlib import Path
from datetime import datetime


def analyze_logs(logs_dir: str = "logs/runs") -> dict:
    """Parse all orchestrator logs and compute metrics."""

    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"No logs found at {logs_dir}")
        return {}

    # Load all orchestrator logs
    orch_logs = []
    for f in logs_path.glob("*_orchestrator.json"):
        with open(f) as fp:
            orch_logs.append(json.load(fp))

    if not orch_logs:
        print("No orchestrator logs found.")
        return {}

    print(f"\n{'='*55}")
    print(f"  MIRA Run Log Analysis")
    print(f"  Logs found: {len(orch_logs)}")
    print(f"{'='*55}\n")

    # ── Metrics ───────────────────────────────────────────────
    total_runs       = len(orch_logs)
    successful_runs  = 0
    failed_runs      = 0
    total_durations  = []
    phase_durations  = {
        "data_exploration": [],
        "model_building": [],
        "model_testing": [],
        "recommendation": []
    }
    decisions_made   = []
    retry_counts     = []
    datasets_used    = []

    for log in orch_logs:
        phases_completed = log.get("phases_completed", [])
        phases_failed    = log.get("phases_failed", [])
        duration         = log.get("total_duration_seconds", 0)
        decisions        = log.get("orchestrator_decisions", [])
        agent_metrics    = log.get("agent_metrics", [])

        # Success = all 4 phases completed
        if len(phases_completed) == 4:
            successful_runs += 1
        else:
            failed_runs += 1

        total_durations.append(duration)
        decisions_made.append(len(decisions))
        datasets_used.append(log.get("dataset_path", "unknown"))

        # Phase durations from agent metrics
        for metric in agent_metrics:
            phase = metric.get("phase", "")
            dur   = metric.get("duration_seconds", 0)
            if phase in phase_durations:
                phase_durations[phase].append(dur)

        # Count retries
        retries = sum(
            1 for d in decisions
            if d.get("decision", "").startswith("RETRY")
        )
        retry_counts.append(retries)

    # ── Compute stats ─────────────────────────────────────────
    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0

    def p50(lst):
        if not lst:
            return 0
        s = sorted(lst)
        return round(s[len(s) // 2], 1)

    success_rate = round(successful_runs / total_runs * 100, 1)
    failure_rate = round(failed_runs / total_runs * 100, 1)

    report = {
        "summary": {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate_pct": success_rate,
            "failure_rate_pct": failure_rate,
        },
        "latency": {
            "avg_total_duration_seconds": avg(total_durations),
            "p50_total_duration_seconds": p50(total_durations),
            "phase_avg_seconds": {
                phase: avg(durs)
                for phase, durs in phase_durations.items()
            },
            "phase_p50_seconds": {
                phase: p50(durs)
                for phase, durs in phase_durations.items()
            }
        },
        "orchestrator": {
            "avg_decisions_per_run": avg(decisions_made),
            "avg_retries_per_run": avg(retry_counts),
            "total_retries": sum(retry_counts),
        },
        "datasets": list(set(datasets_used))
    }

    # ── Print report ──────────────────────────────────────────
    s = report["summary"]
    l = report["latency"]
    o = report["orchestrator"]

    print(f"  📊 Run Statistics")
    print(f"  {'─'*45}")
    print(f"  Total Runs      : {s['total_runs']}")
    print(f"  Success Rate    : {s['success_rate_pct']}%")
    print(f"  Failure Rate    : {s['failure_rate_pct']}%")

    print(f"\n  ⏱️  Latency")
    print(f"  {'─'*45}")
    print(f"  Avg Total       : {l['avg_total_duration_seconds']}s")
    print(f"  P50 Total       : {l['p50_total_duration_seconds']}s")
    print(f"\n  Phase Averages:")
    for phase, dur in l['phase_avg_seconds'].items():
        print(f"    {phase:<25} {dur}s")

    print(f"\n  🧠 Orchestrator")
    print(f"  {'─'*45}")
    print(f"  Avg Decisions   : {o['avg_decisions_per_run']}")
    print(f"  Avg Retries     : {o['avg_retries_per_run']}")
    print(f"  Total Retries   : {o['total_retries']}")

    print(f"\n  📁 Datasets Used:")
    for ds in report["datasets"]:
        print(f"    - {ds}")

    print(f"\n{'='*55}\n")

    # Save report
    out = Path("logs") / "analysis_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✅ Report saved to: {out}\n")

    return report


if __name__ == "__main__":
    analyze_logs()

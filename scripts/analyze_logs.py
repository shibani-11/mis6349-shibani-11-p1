"""
Parses logs/runs/ for latency, failure rate stats.
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import statistics


def load_run_logs(logs_dir: str = "logs/runs") -> List[Dict]:
    """
    Load all run log files from the logs directory.
    
    Args:
        logs_dir: Directory containing run logs
        
    Returns:
        List of run log dictionaries
    """
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"Logs directory not found: {logs_dir}")
        return []
    
    run_logs = []
    for log_file in logs_path.glob("*.json"):
        try:
            with open(log_file, 'r') as f:
                run_logs.append(json.load(f))
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    return run_logs


def analyze_latency(run_logs: List[Dict]) -> Dict:
    """
    Analyze latency statistics.
    
    Args:
        run_logs: List of run log dictionaries
        
    Returns:
        Dictionary of latency statistics
    """
    latencies = [log.get("latency_seconds", 0) for log in run_logs if "latency_seconds" in log]
    
    if not latencies:
        return {"count": 0}
    
    return {
        "count": len(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "p50": statistics.quantiles(latencies, n=100)[49] if len(latencies) >= 50 else statistics.median(latencies),
        "p90": statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else statistics.median(latencies),
        "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else statistics.median(latencies),
    }


def analyze_failure_rate(run_logs: List[Dict]) -> Dict:
    """
    Analyze failure rate statistics.
    
    Args:
        run_logs: List of run log dictionaries
        
    Returns:
        Dictionary of failure statistics
    """
    total = len(run_logs)
    if total == 0:
        return {"total": 0}
    
    hard_failures = sum(1 for log in run_logs if not log.get("success", True) and "error" in log)
    silent_failures = sum(1 for log in run_logs if not log.get("success", True) and "error" not in log)
    successful = sum(1 for log in run_logs if log.get("success", False))
    human_review = sum(1 for log in run_logs if log.get("output", {}).get("requires_human_review", False))
    
    return {
        "total": total,
        "successful": successful,
        "hard_failures": hard_failures,
        "silent_failures": silent_failures,
        "human_review": human_review,
        "hard_failure_rate": hard_failures / total,
        "silent_failure_rate": silent_failures / total,
        "success_rate": successful / total,
    }


def analyze_prompt_versions(run_logs: List[Dict]) -> Dict:
    """
    Analyze performance by prompt version.
    
    Args:
        run_logs: List of run log dictionaries
        
    Returns:
        Dictionary mapping prompt versions to their stats
    """
    version_stats = {}
    
    for log in run_logs:
        version = log.get("prompt_version", "unknown")
        if version not in version_stats:
            version_stats[version] = {
                "count": 0,
                "successful": 0,
                "latencies": []
            }
        
        version_stats[version]["count"] += 1
        
        if log.get("success", False):
            version_stats[version]["successful"] += 1
        
        if "latency_seconds" in log:
            version_stats[version]["latencies"].append(log["latency_seconds"])
    
    # Calculate rates
    for version, stats in version_stats.items():
        stats["success_rate"] = stats["successful"] / stats["count"] if stats["count"] > 0 else 0
        if stats["latencies"]:
            stats["avg_latency"] = statistics.mean(stats["latencies"])
        else:
            stats["avg_latency"] = 0
    
    return version_stats


def print_report(stats: Dict):
    """Print analysis report."""
    print("\n" + "="*60)
    print("LOG ANALYSIS REPORT")
    print("="*60)
    
    # Latency
    if "count" in stats["latency"] and stats["latency"]["count"] > 0:
        print("\n📊 LATENCY STATISTICS")
        print("-"*40)
        print(f"  Total runs:     {stats['latency']['count']}")
        print(f"  Mean:           {stats['latency']['mean']:.2f}s")
        print(f"  Median (p50):   {stats['latency']['median']:.2f}s")
        print(f"  p90:            {stats['latency']['p90']:.2f}s")
        print(f"  p95:            {stats['latency']['p95']:.2f}s")
        print(f"  Min:            {stats['latency']['min']:.2f}s")
        print(f"  Max:            {stats['latency']['max']:.2f}s")
    
    # Failures
    print("\n❌ FAILURE STATISTICS")
    print("-"*40)
    print(f"  Total runs:          {stats['failures']['total']}")
    print(f"  Successful:          {stats['failures']['successful']}")
    print(f"  Hard failures:       {stats['failures']['hard_failures']}")
    print(f"  Silent failures:     {stats['failures']['silent_failures']}")
    print(f"  Human review:        {stats['failures']['human_review']}")
    print("-"*40)
    print(f"  Success rate:        {stats['failures']['success_rate']*100:.1f}%")
    print(f"  Hard failure rate:   {stats['failures']['hard_failure_rate']*100:.1f}%")
    print(f"  Silent failure rate:  {stats['failures']['silent_failure_rate']*100:.1f}%")
    
    # By version
    if stats["versions"]:
        print("\n📝 BY PROMPT VERSION")
        print("-"*40)
        for version, vstats in stats["versions"].items():
            print(f"  {version}:")
            print(f"    Runs: {vstats['count']}, Success: {vstats['success_rate']*100:.1f}%, Avg latency: {vstats['avg_latency']:.2f}s")
    
    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze agent run logs")
    parser.add_argument("--logs-dir", default="logs/runs", help="Directory containing run logs")
    parser.add_argument("--output", help="Path to save analysis JSON")
    args = parser.parse_args()
    
    print(f"Loading logs from {args.logs_dir}...")
    run_logs = load_run_logs(args.logs_dir)
    
    if not run_logs:
        print("No run logs found.")
        return
    
    stats = {
        "timestamp": datetime.now().isoformat(),
        "latency": analyze_latency(run_logs),
        "failures": analyze_failure_rate(run_logs),
        "versions": analyze_prompt_versions(run_logs)
    }
    
    print_report(stats)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Analysis saved to {args.output}")


if __name__ == "__main__":
    main()

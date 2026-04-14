# evals/system_evals.py


def eval_system(
    agent_metrics: list,
    orchestrator_decisions: list,
    total_duration: float,
    phases_completed: list,
    errors: list
) -> dict:
    checks = {}
    score = 0
    total = 0

    def check(name, condition, weight=1, detail=""):
        nonlocal score, total
        total += weight
        passed = bool(condition)
        checks[name] = {
            "passed": passed,
            "weight": weight,
            "detail": detail
        }
        if passed:
            score += weight

    all_phases = [
        "data_exploration", "model_building",
        "model_testing", "recommendation"
    ]

    check("all_phases_completed",
          all(p in phases_completed for p in all_phases),
          weight=3,
          detail=f"Completed: {phases_completed}")
    check("no_critical_errors",
          len(errors) == 0,
          weight=3,
          detail=f"Errors: {errors}")
    check("completed_in_time",
          total_duration < 3600,
          weight=1,
          detail=f"Duration: {total_duration:.0f}s")
    check("all_agents_succeeded",
          all(m.get("success") for m in agent_metrics),
          weight=3,
          detail=f"Successes: {[m.get('success') for m in agent_metrics]}")
    check("orchestrator_made_decisions",
          len(orchestrator_decisions) > 0,
          weight=2,
          detail=f"Decisions: {len(orchestrator_decisions)}")
    check("all_agents_tracked",
          len(agent_metrics) >= 4,
          weight=1,
          detail=f"Agents: {len(agent_metrics)}")

    avg_confidence = (
        sum(m.get("confidence", 0) for m in agent_metrics) /
        len(agent_metrics) if agent_metrics else 0
    )
    check("high_avg_confidence",
          avg_confidence >= 0.6,
          weight=2,
          detail=f"Avg confidence: {avg_confidence:.2f}")

    pct = round(score / total * 100, 1) if total > 0 else 0
    return {
        "eval_type": "system",
        "checks": checks,
        "score": score,
        "total": total,
        "pct": pct,
        "passed": pct >= 70,
        "total_duration_seconds": total_duration,
        "phases_completed": phases_completed,
        "avg_agent_confidence": round(avg_confidence, 2)
    }
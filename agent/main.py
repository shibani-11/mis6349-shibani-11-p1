# agent/main.py
import sys
import json
import time
import uuid
from pathlib import Path
from agent.mira_agent import MIRAAgent
from agent.escalation_rules import evaluate_escalation_rules
from evals.eval_runner import EvalRunner
from schemas.input_schema import AgentInput

CONFIG_FILE = Path(".current_run_config")
RUN_ID_FILE = Path(".current_run_id")
OVERRIDE_LOG_DIR = Path("logs/overrides")

OVERRIDE_CATEGORIES = {
    "1": ("PERFORMANCE_ACCEPTABLE",  "Human judged AUC acceptable despite flag for this specific use case"),
    "2": ("BUSINESS_PRIORITY",       "Human chose differently based on cost, interpretability, or team familiarity"),
    "3": ("DATA_QUALITY_RESOLVED",   "Human confirms data issue is understood and doesn't affect deployment"),
    "4": ("DOMAIN_KNOWLEDGE",        "Human applied domain expertise the agent could not infer from the dataset"),
    "5": ("METRIC_MISMATCH",         "Human overrides the inferred metric — business actually prioritises a different one"),
    "6": ("RISK_ACCEPTED",           "Human acknowledges the risk (overfitting, stability) and proceeds explicitly"),
    "7": ("AGENT_ERROR",             "Agent reasoning was factually wrong — flagged for prompt review"),
}


# ── Config helpers ────────────────────────────────────────────────────────────

def save_config(config: dict):
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    print(f"  Config saved to: {CONFIG_FILE}")


def load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return None


def list_datasets() -> list:
    data_dir = Path("data/raw")
    if not data_dir.exists():
        return []
    datasets = []
    for ext in ["*.csv", "*.xlsx", "*.xls", "*.parquet"]:
        datasets.extend(data_dir.rglob(ext))
    return sorted(datasets)


# ── User prompt ───────────────────────────────────────────────────────────────

def prompt_user() -> dict:
    print(f"\n{'='*60}")
    print(f"  MIRA — Model Intelligence & Recommendation Agent")
    print(f"  Agentic System v0.5.0")
    print(f"{'='*60}\n")

    datasets = list_datasets()
    if not datasets:
        print("  No datasets found in data/raw/")
        print("  Add your dataset and rerun.\n")
        sys.exit(1)

    if len(datasets) == 1:
        dataset = str(datasets[0])
        mb = Path(dataset).stat().st_size / (1024 * 1024)
        print(f"  Dataset detected: {dataset} ({mb:.1f} MB)\n")
    else:
        print("  Available datasets:\n")
        for i, ds in enumerate(datasets, 1):
            mb = ds.stat().st_size / (1024 * 1024)
            print(f"     [{i}] {ds}  ({mb:.1f} MB)")
        print()
        while True:
            c = input(f"  Pick dataset [1-{len(datasets)}]: ").strip()
            if c.isdigit() and 1 <= int(c) <= len(datasets):
                dataset = str(datasets[int(c) - 1])
                break
            print(f"     Enter 1-{len(datasets)}")

    print("  Reading columns...\n")
    try:
        import pandas as pd
        df = (
            pd.read_csv(dataset, nrows=0)
            if dataset.endswith(".csv")
            else pd.read_excel(dataset, nrows=0)
        )
        cols = list(df.columns)
        for i, c in enumerate(cols, 1):
            print(f"     [{i:>2}] {c}")
        print()
    except Exception as e:
        print(f"  Could not read columns: {e}")
        cols = []

    while True:
        t = input(f"  Target column [1-{len(cols)}] or name: ").strip()
        if t.isdigit() and 1 <= int(t) <= len(cols):
            target = cols[int(t) - 1]
            break
        elif t in cols:
            target = t
            break
        print("     Not found. Try again.")
    print(f"\n  Target: {target}\n")

    print("  Describe the business problem:\n")
    while True:
        problem = input("  Problem: ").strip()
        if len(problem) >= 20:
            break
        print("     Please be more descriptive.")

    print(f"\n{'='*60}")
    print(f"  MIRA Configuration")
    print(f"{'='*60}")
    print(f"  Dataset  : {dataset}")
    print(f"  Target   : {target}")
    print(f"  Problem  : {problem[:55]}...")
    print(f"  Metric   : inferred from business problem by MIRA")
    print(f"{'='*60}")

    confirm = input("\n  Start MIRA? (yes/no): ").strip().lower()
    if confirm not in ["yes", "y"]:
        print("\n  Cancelled.\n")
        sys.exit(0)

    return {
        "dataset_path": dataset,
        "target_column": target,
        "business_problem": problem,
    }


# ── Utilities ─────────────────────────────────────────────────────────────────

def clear_results(output_path: str):
    folder = Path(output_path)
    folder.mkdir(exist_ok=True)
    cleared = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix in (".json", ".csv")
    ]
    for f in cleared:
        f.unlink()
    if cleared:
        print(f"\n  Cleared {len(cleared)} old result files")


def build_agent_input(config: dict, run_id: str) -> AgentInput:
    return AgentInput(
        dataset_path=config["dataset_path"],
        target_column=config["target_column"],
        business_problem=config["business_problem"],
        task_type="auto",
        max_models=5,
        max_iterations=40,
        analysis_phases=[],
        run_id=run_id,
        extra_context={},
    )


def run_evals(run_id: str, output_path: str):
    print("\n  Running evals...")
    try:
        data_card_path = Path(output_path) / f"{run_id}_data_card.json"
        priority_metric = "roc_auc"
        if data_card_path.exists():
            data_card = json.loads(data_card_path.read_text())
            priority_metric = data_card.get("priority_metric", "roc_auc")

        EvalRunner(
            run_id=run_id,
            output_path=output_path,
            priority_metric=priority_metric,
        ).run()
    except Exception as e:
        print(f"  Evals failed: {e}")


# ── Override logging ──────────────────────────────────────────────────────────

def write_override_log(
    run_id: str,
    routing_zone: str,
    escalation_rules: list,
    recommendation: dict,
    human_decision: str,
    override_category: str,
    human_rationale: str,
    review_duration_seconds: float,
):
    OVERRIDE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = {
        "run_id": run_id,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "routing_zone": routing_zone,
        "escalation_rules_triggered": [r["rule_name"] for r in escalation_rules],
        "agent_recommendation": {
            "recommended_model":  recommendation.get("recommended_model"),
            "primary_metric_value": recommendation.get("primary_metric_value"),
            "confidence_score":   recommendation.get("confidence_score"),
            "flags":              recommendation.get("flags", []),
            "requires_human_review": recommendation.get("requires_human_review"),
            "human_review_reason":   recommendation.get("human_review_reason"),
        },
        "human_decision": human_decision,
        "override_category": override_category,
        "human_rationale": human_rationale,
        "review_duration_seconds": round(review_duration_seconds, 1),
        "agent_version": "MIRA v0.5.0",
        "prompt_version": "mira_agent_v0_5_0.md",
    }
    log_path = OVERRIDE_LOG_DIR / f"{run_id}_override.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"  Override log: {log_path}")


# ── HITL Approval Gate (three-zone) ──────────────────────────────────────────

def hitl_approval_gate(run_id: str, output_path: str) -> bool:
    """
    Three-zone HITL approval gate (Session 6 architecture).

    Zone 1 (Auto-Approve Eligible): confidence >= 0.85, no flags, no escalation rules
    Zone 2 (Standard Human Review): confidence 0.70-0.84 or soft warnings
    Zone 3 (Priority Human Review): confidence < 0.70 or any hard escalation rule

    Returns True to proceed to evals, False if human rejected the recommendation.
    """
    rec_path   = Path(output_path) / f"{run_id}_recommendation.json"
    dc_path    = Path(output_path) / f"{run_id}_data_card.json"
    ms_path    = Path(output_path) / f"{run_id}_model_selection.json"

    if not rec_path.exists():
        return True  # no recommendation produced — let evals report the failure

    rec          = json.loads(rec_path.read_text())
    data_card    = json.loads(dc_path.read_text())   if dc_path.exists()  else {}
    model_sel    = json.loads(ms_path.read_text())   if ms_path.exists()  else {}

    confidence   = rec.get("confidence_score", 1.0)
    flags        = rec.get("flags", [])
    routing_zone = rec.get("routing_zone", "zone_2")

    # Run hard escalation rules
    escalation   = evaluate_escalation_rules(data_card, model_sel)
    rules_fired  = escalation["rules_triggered"]

    # Override routing_zone if escalation rules say zone_3
    if escalation["hard_escalation"]:
        routing_zone = "zone_3"
    elif not flags and confidence >= 0.85:
        routing_zone = "zone_1"
    else:
        routing_zone = "zone_2"

    # ── Zone 1: Auto-Approve Eligible ─────────────────────────────────────
    if routing_zone == "zone_1":
        print(f"\n{'='*60}")
        print(f"  HITL GATE — Zone 1: Auto-Approve Eligible")
        print(f"{'='*60}")
        print(f"  Model      : {rec.get('recommended_model')}")
        print(f"  Confidence : {confidence}  (threshold: 0.85)")
        print(f"  Flags      : none")
        print(f"  Escalation : none")
        print(f"  Decision   : AUTO-PROCEED to evals")
        print(f"{'='*60}\n")
        return True

    # ── Zone 2: Standard Human Review ─────────────────────────────────────
    if routing_zone == "zone_2":
        print(f"\n{'='*60}")
        print(f"  HITL GATE — Zone 2: Standard Human Review Required")
        print(f"{'='*60}")
        print(f"  Model      : {rec.get('recommended_model')}")
        print(f"  AUC        : {rec.get('primary_metric_value')}")
        print(f"  Confidence : {confidence}")
        print(f"  Flags      : {', '.join(flags) if flags else 'none'}")
        if rules_fired:
            print(f"  Rules      : {', '.join(r['rule_name'] for r in rules_fired)}")
        print(f"\n  Review reason  : {rec.get('human_review_reason', 'confidence in standard review range')}")
        print(f"\n  Executive summary:")
        print(f"  {rec.get('executive_summary', '')[:300]}")
        print(f"\n  Full recommendation: {rec_path}")
        print(f"{'='*60}")

    # ── Zone 3: Priority Human Review ─────────────────────────────────────
    else:
        print(f"\n{'='*60}")
        print(f"  HITL GATE — Zone 3: Priority Human Review Required")
        print(f"{'='*60}")
        print(f"  Model      : {rec.get('recommended_model')}")
        print(f"  AUC        : {rec.get('primary_metric_value')}")
        print(f"  Confidence : {confidence}  *** BELOW SAFE THRESHOLD ***")
        print(f"  Flags      : {', '.join(flags) if flags else 'none'}")
        print(f"\n  Hard escalation rules triggered:")
        for rule in rules_fired:
            print(f"    [{rule['severity']}] {rule['rule_name']}")
            print(f"      {rule['detail']}")
        print(f"\n  Risk reason    : {rec.get('human_review_reason', 'N/A')}")
        print(f"  Test verdict   : {rec.get('test_verdict_summary', 'N/A')[:120]}")
        print(f"\n  Top feature drivers:")
        for fd in rec.get("feature_drivers", [])[:3]:
            print(f"    {fd.get('feature')}: {fd.get('importance', 0):.4f}")
        print(f"\n  Executive summary:")
        print(f"  {rec.get('executive_summary', '')[:300]}")
        print(f"\n  Full recommendation: {rec_path}")
        print(f"{'='*60}")

    # ── Human decision prompt ──────────────────────────────────────────────
    review_start = time.time()

    while True:
        decision = input("\n  Approve this recommendation? (yes/no): ").strip().lower()
        if decision in ("yes", "y", "no", "n"):
            break
        print("     Enter yes or no.")

    review_duration = time.time() - review_start
    human_decision = "APPROVED" if decision in ("yes", "y") else "REJECTED"

    # ── Override category (required for any human decision in Zone 2/3) ───
    print(f"\n  Select override category (required for audit log):")
    for k, (name, desc) in OVERRIDE_CATEGORIES.items():
        print(f"    [{k}] {name}")
        print(f"        {desc}")

    while True:
        cat_choice = input("\n  Category [1-7]: ").strip()
        if cat_choice in OVERRIDE_CATEGORIES:
            override_category, _ = OVERRIDE_CATEGORIES[cat_choice]
            break
        print("     Enter 1-7.")

    rationale = input("\n  Brief rationale (1-2 sentences): ").strip()

    # ── Write override log ────────────────────────────────────────────────
    write_override_log(
        run_id=run_id,
        routing_zone=routing_zone,
        escalation_rules=rules_fired,
        recommendation=rec,
        human_decision=human_decision,
        override_category=override_category,
        human_rationale=rationale,
        review_duration_seconds=review_duration,
    )

    if human_decision == "APPROVED":
        print(f"\n  Approved ({override_category}). Proceeding to eval scoring.\n")
        return True
    else:
        print(f"\n  Rejected ({override_category}). Run marked as human-reviewed and declined.\n")
        rec["hitl_decision"] = "REJECTED"
        rec["hitl_note"] = rationale
        rec_path.write_text(json.dumps(rec, indent=2))
        return False


# ── Main entry point ──────────────────────────────────────────────────────────

def main():
    """
    MIRA — Model Intelligence & Recommendation Agent v0.5.0

    Usage:
      python3 -m agent.main        <- full agentic run
      python3 -m agent.main evals  <- re-run evals on last completed run
    """
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    output_path = "processed/"

    if mode == "evals":
        config = load_config()
        if not config:
            print("\n  No run config found. Run MIRA first.\n")
            sys.exit(1)
        run_evals(config["run_id"], output_path)
        return

    if mode not in ("run", "all"):
        print(f"\n  Unknown mode: '{mode}'")
        print(f"  Valid options: run, evals\n")
        sys.exit(1)

    config = prompt_user()
    clear_results(output_path)

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    config["run_id"] = run_id
    save_config(config)

    print(f"\n  New run started: {run_id}\n")

    agent_input = build_agent_input(config, run_id)

    MIRAAgent(agent_input).run()

    # Three-zone HITL approval gate — must pass before evals run
    proceed = hitl_approval_gate(run_id, output_path)

    if proceed:
        run_evals(run_id, output_path)

    print(f"\n  MIRA analysis complete!")
    print(f"   data_card:       {output_path}{run_id}_data_card.json")
    print(f"   model_selection: {output_path}{run_id}_model_selection.json")
    print(f"   recommendation:  {output_path}{run_id}_recommendation.json")
    if proceed:
        print(f"   eval report:     {output_path}{run_id}_eval_report.json")
    else:
        print(f"   eval report:     skipped — recommendation rejected at HITL gate")
    print(f"   override log:    logs/overrides/{run_id}_override.json\n")

    CONFIG_FILE.unlink(missing_ok=True)
    RUN_ID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

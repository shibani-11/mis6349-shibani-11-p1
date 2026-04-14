# agent/main.py
import sys
import json
import uuid
from pathlib import Path
from agent.orchestrator import MIRAOrchestrator
from evals.eval_runner import EvalRunner
from schemas.input_schema import AgentInput

CONFIG_FILE = ".current_run_config"


def save_config(config: dict):
    Path(CONFIG_FILE).write_text(json.dumps(config, indent=2))


def load_config():
    f = Path(CONFIG_FILE)
    return json.loads(f.read_text()) if f.exists() else None


def list_datasets() -> list:
    data_dir = Path("data/raw")
    datasets = []
    for ext in ["*.csv", "*.xlsx", "*.xls", "*.parquet"]:
        datasets.extend(data_dir.rglob(ext))
    return sorted(datasets)


def prompt_user():
    print(f"\n{'='*60}")
    print(f"  🚀 MIRA — Model Intelligence & Recommendation Agent")
    print(f"  Multi-Agent System with Evals")
    print(f"{'='*60}\n")

    datasets = list_datasets()
    if not datasets:
        print("  ⚠️  No datasets in data/raw/ — add one and rerun\n")
        sys.exit(1)

    if len(datasets) == 1:
        dataset = str(datasets[0])
        mb = Path(dataset).stat().st_size / (1024 * 1024)
        print(f"  📁 Dataset: {dataset} ({mb:.1f} MB)\n")
    else:
        print("  📁 Available datasets:\n")
        for i, ds in enumerate(datasets, 1):
            mb = ds.stat().st_size / (1024 * 1024)
            print(f"     [{i}] {ds}  ({mb:.1f} MB)")
        print()
        while True:
            c = input(f"  Pick [1-{len(datasets)}]: ").strip()
            if c.isdigit() and 1 <= int(c) <= len(datasets):
                dataset = str(datasets[int(c) - 1])
                break
            print(f"     ⚠️  Enter 1-{len(datasets)}")

    # Show columns
    print("  🔍 Reading columns...\n")
    try:
        import pandas as pd
        df = (pd.read_csv(dataset, nrows=0)
              if dataset.endswith(".csv")
              else pd.read_excel(dataset, nrows=0))
        cols = list(df.columns)
        for i, c in enumerate(cols, 1):
            print(f"     [{i:>2}] {c}")
        print()
    except Exception as e:
        print(f"  ⚠️  Could not read columns: {e}")
        cols = []

    # Target column
    while True:
        t = input(
            f"  🎯 Target column [1-{len(cols)}] or name: "
        ).strip()
        if t.isdigit() and 1 <= int(t) <= len(cols):
            target = cols[int(t) - 1]
            break
        elif t in cols:
            target = t
            break
        print("     ⚠️  Not found.")
    print(f"\n  ✓ Target: {target}\n")

    # Business problem
    print("  💼 Describe the business problem:\n")
    while True:
        problem = input("  Problem: ").strip()
        if len(problem) >= 20:
            break
        print("     ⚠️  Please be more descriptive.")

    # Optional
    print(f"\n  ⚙️  Optional (Enter for defaults)\n")
    domain = input(
        "  🏢 Domain [default: general]: "
    ).strip() or "general"
    metric = input(
        "  📊 Metric (roc_auc/f1_score/recall) [default: roc_auc]: "
    ).strip() or "roc_auc"
    if metric not in ["roc_auc", "f1_score", "accuracy", "recall"]:
        metric = "roc_auc"

    # Confirm
    print(f"\n{'='*60}")
    print(f"  ✅ Configuration")
    print(f"{'='*60}")
    print(f"  Dataset  : {dataset}")
    print(f"  Target   : {target}")
    print(f"  Problem  : {problem[:55]}...")
    print(f"  Domain   : {domain}")
    print(f"  Metric   : {metric}")
    print(f"{'='*60}")

    confirm = input("\n  Start MIRA? (yes/no): ").strip().lower()
    if confirm not in ["yes", "y"]:
        print("\n  ❌ Cancelled.\n")
        sys.exit(0)

    return {
        "dataset_path": dataset,
        "target_column": target,
        "business_problem": problem,
        "domain": domain,
        "priority_metric": metric
    }


def main():
    """
    MIRA Multi-Agent System

    Usage:
      python -m agent.main         ← full run + evals
      python -m agent.main evals   ← evals only on last run
    """
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    output_path = "processed/"

    # Evals only
    if mode == "evals":
        config = load_config()
        if not config:
            print("\n⚠️  No run found. Run MIRA first.\n")
            sys.exit(1)
        EvalRunner(
            run_id=config["run_id"],
            output_path=output_path,
            priority_metric=config.get("priority_metric", "roc_auc")
        ).run()
        return

    # Full run
    config = prompt_user()

    # Clear old results
    folder = Path(output_path)
    folder.mkdir(exist_ok=True)
    cleared = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix == ".json"
    ]
    for f in cleared:
        f.unlink()
    if cleared:
        print(f"\n🗑️  Cleared {len(cleared)} old result files")

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    save_config({**config, "run_id": run_id})
    print(f"\n🆔 Run ID: {run_id}\n")

    agent_input = AgentInput(
        dataset_path=config["dataset_path"],
        target_column=config["target_column"],
        business_problem=config["business_problem"],
        task_type="auto",
        max_models=5,
        max_iterations=40,
        analysis_phases=[
            "data_exploration",
            "model_building",
            "model_testing",
            "recommendation"
        ],
        run_id=run_id,
        extra_context={
            "domain": config.get("domain", "general"),
            "priority_metric": config.get("priority_metric", "roc_auc")
        }
    )

    # Run orchestrator
    MIRAOrchestrator(agent_input).run()

    # Run evals automatically
    print("\n🧪 Running evals...")
    EvalRunner(
        run_id=run_id,
        output_path=output_path,
        priority_metric=config.get("priority_metric", "roc_auc")
    ).run()


if __name__ == "__main__":
    main()
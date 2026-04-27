# agent/main.py
import sys
import json
import uuid
from pathlib import Path
from agent.mira_agent import MIRAAgent
from evals.eval_runner import EvalRunner
from schemas.input_schema import AgentInput

CONFIG_FILE = Path(".current_run_config")
RUN_ID_FILE = Path(".current_run_id")


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


def prompt_user() -> dict:
    print(f"\n{'='*60}")
    print(f"  MIRA — Model Intelligence & Recommendation Agent")
    print(f"  Agentic System v0.3.0")
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
        # Read priority_metric from data_card.json written by EDA phase
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


def main():
    """
    MIRA — Model Intelligence & Recommendation Agent v0.3.0

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

    run_evals(run_id, output_path)

    print(f"\n  MIRA analysis complete!")
    print(f"   data_card:       {output_path}{run_id}_data_card.json")
    print(f"   model_selection: {output_path}{run_id}_model_selection.json")
    print(f"   recommendation:  {output_path}{run_id}_recommendation.json")
    print(f"   eval report:     {output_path}{run_id}_eval_report.json\n")

    CONFIG_FILE.unlink(missing_ok=True)
    RUN_ID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

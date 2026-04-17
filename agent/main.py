# agent/main.py
import sys
import json
import uuid
from pathlib import Path
from agent.orchestrator import MIRAOrchestrator
from evals.eval_runner import EvalRunner
from schemas.input_schema import AgentInput

ALL_PHASES = [
    "data_exploration",
    "model_building",
    "model_testing",
    "recommendation"
]

CONFIG_FILE = Path(".current_run_config")
RUN_ID_FILE = Path(".current_run_id")


def save_config(config: dict):
    """Save run config to file."""
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    print(f"  💾 Config saved to: {CONFIG_FILE}")


def load_config() -> dict:
    """Load saved run config."""
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return None


def list_datasets() -> list:
    """Find all datasets in data/raw/."""
    data_dir = Path("data/raw")
    if not data_dir.exists():
        return []
    datasets = []
    for ext in ["*.csv", "*.xlsx", "*.xls", "*.parquet"]:
        datasets.extend(data_dir.rglob(ext))
    return sorted(datasets)


def prompt_user() -> dict:
    """Ask user for dataset, target, and business problem."""
    print(f"\n{'='*60}")
    print(f"  🚀 MIRA — Model Intelligence & Recommendation Agent")
    print(f"  Multi-Agent System v2.0")
    print(f"{'='*60}\n")

    # Auto-detect dataset
    datasets = list_datasets()
    if not datasets:
        print("  ⚠️  No datasets found in data/raw/")
        print("  Add your dataset and rerun.\n")
        sys.exit(1)

    if len(datasets) == 1:
        dataset = str(datasets[0])
        mb = Path(dataset).stat().st_size / (1024 * 1024)
        print(f"  📁 Dataset detected: {dataset} ({mb:.1f} MB)\n")
    else:
        print("  📁 Available datasets:\n")
        for i, ds in enumerate(datasets, 1):
            mb = ds.stat().st_size / (1024 * 1024)
            print(f"     [{i}] {ds}  ({mb:.1f} MB)")
        print()
        while True:
            c = input(f"  Pick dataset [1-{len(datasets)}]: ").strip()
            if c.isdigit() and 1 <= int(c) <= len(datasets):
                dataset = str(datasets[int(c) - 1])
                break
            print(f"     ⚠️  Enter 1-{len(datasets)}")

    # Show columns
    print("  🔍 Reading columns...\n")
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
        print("     ⚠️  Not found. Try again.")
    print(f"\n  ✓ Target: {target}\n")

    # Business problem
    print("  💼 Describe the business problem:\n")
    while True:
        problem = input("  Problem: ").strip()
        if len(problem) >= 20:
            break
        print("     ⚠️  Please be more descriptive.")

    # Optional settings
    print(f"\n  ⚙️  Optional (press Enter for defaults)\n")
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
    print(f"  ✅ MIRA Configuration")
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


def clear_results(output_path: str):
    """Clear previous run results."""
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


def build_agent_input(config: dict, phases: list, run_id: str) -> AgentInput:
    """Build AgentInput from config."""
    return AgentInput(
        dataset_path=config["dataset_path"],
        target_column=config["target_column"],
        business_problem=config["business_problem"],
        task_type="auto",
        max_models=5,
        max_iterations=40,
        analysis_phases=phases,
        run_id=run_id,
        extra_context={
            "domain": config.get("domain", "general"),
            "priority_metric": config.get("priority_metric", "roc_auc")
        }
    )


def run_evals(run_id: str, output_path: str, priority_metric: str):
    """Run evals on completed phase outputs."""
    print("\n🧪 Running evals...")
    try:
        EvalRunner(
            run_id=run_id,
            output_path=output_path,
            priority_metric=priority_metric
        ).run()
    except Exception as e:
        print(f"  ⚠️  Evals failed: {e}")


def main():
    """
    MIRA — Model Intelligence & Recommendation Agent

    Usage:
      python3 -m agent.main                  ← full run (all phases)
      python3 -m agent.main data_exploration ← Phase 1 only (fresh start)
      python3 -m agent.main model_building   ← Phase 2 (continues run)
      python3 -m agent.main model_testing    ← Phase 3 (continues run)
      python3 -m agent.main recommendation   ← Phase 4 (continues run)
      python3 -m agent.main evals            ← run evals only
    """
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    output_path = "processed/"

    # ── Evals only mode ────────────────────────────────────────
    if mode == "evals":
        config = load_config()
        if not config:
            print("\n⚠️  No run config found. Run MIRA first.\n")
            sys.exit(1)
        run_evals(
            config["run_id"],
            output_path,
            config.get("priority_metric", "roc_auc")
        )
        return

    # ── Validate phase argument ────────────────────────────────
    if mode not in ALL_PHASES and mode != "all":
        print(f"\n❌ Unknown mode: '{mode}'")
        print(f"   Valid options: {ALL_PHASES + ['all', 'evals']}\n")
        sys.exit(1)

    # ── Determine if this is a fresh start ────────────────────
    is_first_phase = (
        mode == "all" or
        mode == ALL_PHASES[0]  # "data_exploration"
    )

    if is_first_phase:
        # ── Fresh run: prompt user, clear old results ──────────
        config = prompt_user()
        clear_results(output_path)

        run_id = f"run_{uuid.uuid4().hex[:8]}"
        config["run_id"] = run_id
        save_config(config)

        print(f"\n🆔 New run started: {run_id}\n")
        phases = ALL_PHASES if mode == "all" else [ALL_PHASES[0]]

    else:
        # ── Continuing run: load saved config ─────────────────
        config = load_config()

        if not config:
            print(f"\n⚠️  No active run found.")
            print(f"   Start a new run first:")
            print(f"   python3 -m agent.main data_exploration\n")
            sys.exit(1)

        run_id = config.get("run_id")
        if not run_id:
            print(f"\n⚠️  Run ID missing from config.")
            print(f"   Start fresh:")
            print(f"   python3 -m agent.main data_exploration\n")
            sys.exit(1)

        phases = [mode]

        print(f"\n{'='*60}")
        print(f"  🚀 MIRA — Continuing Run")
        print(f"{'='*60}")
        print(f"  Run ID   : {run_id}")
        print(f"  Phase    : {mode.upper().replace('_', ' ')}")
        print(f"  Dataset  : {config['dataset_path']}")
        print(f"  Target   : {config['target_column']}")
        print(f"  Problem  : {config['business_problem'][:55]}...")
        print(f"{'='*60}\n")

    # ── Build agent input ──────────────────────────────────────
    agent_input = build_agent_input(config, phases, run_id)

    # ── Run orchestrator ───────────────────────────────────────
    orchestrator = MIRAOrchestrator(agent_input)
    orchestrator.run()

    # ── Run evals after each phase ─────────────────────────────
    run_evals(
        run_id,
        output_path,
        config.get("priority_metric", "roc_auc")
    )

    # ── Tell user what to run next ─────────────────────────────
    if mode != "all":
        current_idx = ALL_PHASES.index(mode)
        if current_idx < len(ALL_PHASES) - 1:
            next_phase = ALL_PHASES[current_idx + 1]
            print(f"\n▶️  Run next phase:")
            print(f"   python3 -m agent.main {next_phase}\n")
        else:
            print(f"\n🎉 MIRA analysis complete!")
            print(f"   Full report: {output_path}{run_id}_report.json")
            print(f"   Eval report: {output_path}{run_id}_eval_report.json\n")
            # Clean up config files
            CONFIG_FILE.unlink(missing_ok=True)
            RUN_ID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
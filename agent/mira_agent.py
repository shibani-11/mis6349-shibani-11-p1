# agent/mira_agent.py
# Single entry point for all MIRA agent logic:
#   Phase 1 — EDA.py
#   Phase 2 — Modeltrain.py
#   Phase 3 — mira-recommend skill (recommendation.json)

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.sdk.context.skills.skill import Skill
from openhands.sdk.context.agent_context import AgentContext
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from schemas.input_schema import AgentInput

load_dotenv()

PROMPT_VERSION = "mira_agent_v0_5_0.md"
SKILL_PATH = Path("skills/mira-recommend/SKILL.md")

REQUIRED_KEYS = {
    "data_card": [
        "rows", "features", "class_distribution", "class_imbalance_detected",
        "minority_class_ratio", "missing_value_summary", "high_correlation_features",
        "data_quality_issues", "recommended_approach", "genai_narrative",
    ],
    "model_selection": [
        "models_trained", "excluded_models", "selected_model", "runner_up_model",
        "selection_reasoning", "runner_up_reasoning", "rejected_models",
        "class_imbalance_handled", "imbalance_strategy", "preprocessing_applied", "genai_narrative",
        "overfitting_detected", "overfitting_gap", "leakage_detected",
        "stability_flag", "test_verdict", "test_findings", "feature_importance",
    ],
    "recommendation": [
        "recommended_model", "selection_reason", "primary_metric_value",
        "all_models_summary", "model_comparison_narrative", "business_impact",
        "tradeoffs", "alternative_model", "alternative_model_reason", "next_steps",
        "deployment_considerations", "risks", "test_verdict_summary",
        "feature_drivers", "confidence_score", "requires_human_review",
        "human_review_reason", "executive_summary",
    ],
}


class MIRAAgent:
    """
    MIRA — Model Intelligence & Recommendation Agent v0.5.0

    Drives the full OpenHands agent loop across three phases:
      Phase 1 → EDA.py             → data_card.json
      Phase 2 → Modeltrain.py      → model_selection.json
      Phase 3 → mira-recommend     → recommendation.json
    """

    def __init__(self, agent_input: AgentInput):
        self.input = agent_input
        self.run_id = agent_input.run_id
        self.output_path = Path(agent_input.output_path)
        self.extra_context = agent_input.extra_context or {}
        self.decisions = []
        self.duration = 0.0
        self.success = False

        self.output_path.mkdir(parents=True, exist_ok=True)
        Path("logs/runs").mkdir(parents=True, exist_ok=True)

        self.llm = LLM(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o"),
            api_key=os.getenv("LLM_API_KEY"),
        )

        print(f"\n{'='*60}")
        print(f"  MIRA — Model Intelligence & Recommendation Agent")
        print(f"  Agentic System v0.5.0")
        print(f"{'='*60}")
        print(f"  Run ID   : {self.run_id}")
        print(f"  Dataset  : {self.input.dataset_path}")
        print(f"  Target   : {self.input.target_column}")
        print(f"  Problem  : {self.input.business_problem[:55]}...")
        print(f"{'='*60}\n")

    # ── Private helpers ────────────────────────────────────────────

    def _load_system_prompt(self) -> str:
        path = Path("prompts") / PROMPT_VERSION
        if not path.exists():
            raise FileNotFoundError(f"System prompt not found: {path}")
        return path.read_text()

    def _load_skill(self) -> Skill | None:
        if not SKILL_PATH.exists():
            print(f"  WARNING: Skill not found at {SKILL_PATH}")
            return None
        try:
            skill = Skill.load(SKILL_PATH, strict=False)
            print(f"  Skill loaded: {skill.name}")
            return skill
        except Exception as e:
            print(f"  WARNING: Failed to load skill — {e}")
            return None

    def _tools(self) -> list[Tool]:
        return [
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ]

    def _path(self, suffix: str) -> Path:
        return self.output_path / f"{self.run_id}_{suffix}.json"

    def _read_output(self, suffix: str) -> dict:
        path = self._path(suffix)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                print(f"  WARNING: Could not parse {path}")
        return {}

    def _check_schema(self, suffix: str) -> list[str]:
        data = self._read_output(suffix)
        if not data:
            return []
        return [k for k in REQUIRED_KEYS.get(suffix, []) if k not in data]

    def _schema_error_message(self, suffix: str, missing: list[str]) -> str:
        return (
            f"SCHEMA VIOLATION in {self._path(suffix)}.\n"
            f"Missing required keys: {missing}\n"
            f"Rewrite the file with ALL required keys from the system prompt schema.\n"
            f"Do not invent your own schema. Fill in real values. Then continue."
        )

    def _build_message(self) -> str:
        data_card      = self._path("data_card")
        cleaned        = self.output_path / f"{self.run_id}_cleaned.csv"
        model_sel      = self._path("model_selection")
        recommendation = self._path("recommendation")

        business_problem_arg = self.input.business_problem.replace('"', '\\"')

        return f"""
## Run Context

- **Run ID:** {self.run_id}
- **Dataset:** {self.input.dataset_path}
- **Target column:** {self.input.target_column}
- **Business problem:** {self.input.business_problem}
- **Priority metric:** inferred by EDA from the business problem

---

## Exact Commands — Run in Order

Do not write any Python. Do not skip any step. Run each command exactly as shown.

**Phase 1 — EDA:**
```
python3 scripts/EDA.py --dataset {self.input.dataset_path} --target {self.input.target_column} --output {data_card} --cleaned-output {cleaned} --business-problem "{business_problem_arg}"
```

**Phase 2 — Model Training + Stress Tests:**
```
python3 scripts/Modeltrain.py --cleaned-data {cleaned} --data-card {data_card} --target {self.input.target_column} --output {model_sel}
```

After Phase 1: verify `SCHEMA OK` is printed.
After Phase 2: verify `SCHEMA OK` is printed.

**Phase 3 — Deployment Recommendation:**

After Phase 2 is complete, generate the recommendation report by following the mira-recommend skill instructions in your available skills. Use:
- TerminalTool to read `{data_card}` and `{model_sel}`
- FileEditorTool to write `{recommendation}`

After writing the file, print `RECOMMENDATION OK`.
Call TaskTracker once all three phases are complete.
"""

    def _recommendation_push(self) -> str:
        data_card   = self._path("data_card")
        model_sel   = self._path("model_selection")
        recommend   = self._path("recommendation")
        return (
            f"Phases 1 and 2 are complete. Now generate the deployment recommendation report (Phase 3).\n\n"
            f"Follow the mira-recommend skill instructions listed in your available skills.\n\n"
            f"Step 1 — Read the output files using TerminalTool:\n"
            f"  cat {data_card}\n"
            f"  cat {model_sel}\n\n"
            f"Step 2 — Compute the required fields (confidence_score, requires_human_review, "
            f"all_models_summary, test_verdict_summary, deploy_word) exactly as specified in the skill.\n\n"
            f"Step 3 — Write the recommendation JSON to:\n"
            f"  {recommend}\n"
            f"Use FileEditorTool to write the file. The JSON must include ALL required keys from the skill schema.\n\n"
            f"Step 4 — Print: RECOMMENDATION OK\n\n"
            f"Then call TaskTracker to finish."
        )

    def _log_decision(self, decision: str, reason: str):
        entry = {"decision": decision, "reason": reason, "timestamp": datetime.utcnow().isoformat()}
        self.decisions.append(entry)
        print(f"\n  [MIRA] {decision}")
        print(f"     {reason}\n")

    # ── Main run ───────────────────────────────────────────────────

    def run(self) -> dict:
        skill = self._load_skill()
        agent_context = AgentContext(skills=[skill]) if skill else None

        agent = Agent(
            llm=self.llm,
            tools=self._tools(),
            system_prompt=self._load_system_prompt(),
            max_iterations=self.input.max_iterations,
            agent_context=agent_context,
        )

        conversation = Conversation(agent=agent, workspace=os.getcwd())

        print("  Phases: EDA → Modeltrain → mira-recommend\n")

        start = time.time()
        conversation.send_message(self._build_message())
        conversation.run()

        for _ in range(6):
            schema_errors = {
                s: self._check_schema(s)
                for s in ("data_card", "model_selection")
                if self._read_output(s) and self._check_schema(s)
            }
            phases_done = (
                self._read_output("data_card") and
                self._read_output("model_selection") and
                not schema_errors
            )
            recommendation_done = bool(self._read_output("recommendation"))

            if phases_done and recommendation_done:
                break

            completed = [s for s in ("data_card", "model_selection") if self._read_output(s)]

            if schema_errors:
                error_msgs = "\n\n".join(self._schema_error_message(s, m) for s, m in schema_errors.items())
                print(f"\n  Schema violations in {list(schema_errors.keys())} — pushing fix...")
                conversation.send_message(f"STOP. Schema validation failed.\n\n{error_msgs}")
            elif "model_selection" not in completed:
                print(f"\n  Paused after Phase 1 — pushing Phase 2...")
                conversation.send_message(
                    "Phase 1 is done. Run Phase 2 — Modeltrain.py. Then invoke mira-recommend. Then TaskTracker."
                )
            elif phases_done and not recommendation_done:
                print(f"\n  Phases 1 & 2 done — pushing mira-recommend skill...")
                conversation.send_message(self._recommendation_push())
            else:
                break

            conversation.run()

        self.duration = round(time.time() - start, 2)
        return self._collect_outputs()

    def _collect_outputs(self) -> dict:
        data_card  = self._read_output("data_card")
        model_sel  = self._read_output("model_selection")
        recommend  = self._read_output("recommendation")

        schema_failures = {
            s: self._check_schema(s)
            for s in ("data_card", "model_selection")
            if self._read_output(s) and self._check_schema(s)
        }

        if data_card and model_sel and recommend and not schema_failures:
            self.success = True
            print(f"\n  All three phases complete in {self.duration}s")
        elif schema_failures:
            self.success = False
            print(f"\n  Schema errors after {self.duration}s: {list(schema_failures.keys())}")
        elif data_card and model_sel and not recommend:
            self.success = False
            print(f"\n  Phases 1 & 2 done but recommendation missing after {self.duration}s")
        else:
            self.success = False
            print(f"\n  Incomplete after {self.duration}s")

        self._log_decision(
            "COMPLETE" if self.success else "INCOMPLETE",
            (
                f"Recommended: {recommend.get('recommended_model', 'N/A')}, "
                f"Confidence: {recommend.get('confidence_score', 'N/A')}"
            ) if recommend else "No recommendation produced — check logs"
        )

        self._write_log(data_card, model_sel, recommend)
        return {"data_card": data_card, "model_selection": model_sel, "recommendation": recommend}

    def _write_log(self, data_card: dict, model_sel: dict, recommend: dict):
        log = {
            "agent": "MIRA v0.5.0",
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_path": self.input.dataset_path,
            "target_column": self.input.target_column,
            "business_problem": self.input.business_problem,
            "duration_seconds": self.duration,
            "success": self.success,
            "outputs_produced": {
                "data_card":      bool(data_card),
                "model_selection": bool(model_sel),
                "recommendation":  bool(recommend),
            },
            "recommended_model":     recommend.get("recommended_model") if recommend else None,
            "confidence_score":      recommend.get("confidence_score") if recommend else None,
            "requires_human_review": recommend.get("requires_human_review") if recommend else None,
            "test_verdict":          model_sel.get("test_verdict") if model_sel else None,
            "decisions": self.decisions,
        }
        log_path = Path("logs/runs") / f"{self.run_id}_run.json"
        log_path.write_text(json.dumps(log, indent=2))
        print(f"  Log: {log_path}")

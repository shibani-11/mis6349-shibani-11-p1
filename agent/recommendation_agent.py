# agent/recommendation_agent.py
# v0.3.0 — Unified agentic agent. Drives all four phases in a single
# conversation loop via role-injected messages. Replaces the 4-phase
# subprocess orchestrator with one autonomous agent run.

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from openhands.sdk import LLM, Agent, Conversation
from agent.tools import get_tools

load_dotenv()

PROMPT_VERSION = "mira_agent_v0_3_0.md"


class RecommendationAgent:
    """
    Unified MIRA agent. Given a dataset and business problem, the LLM
    autonomously explores data, trains models, evaluates quality, and
    produces a deployment recommendation — in one conversation loop.

    Personas are injected via the initial message (Option A), so the
    LLM transitions roles based on what it observes, not a hardcoded
    phase sequence.
    """

    def __init__(self, run_id: str, output_path: str, max_iterations: int = 40):
        self.run_id = run_id
        self.output_path = Path(output_path)
        self.max_iterations = max_iterations
        self.duration = 0.0
        self.success = False

        self.llm = LLM(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o"),
            api_key=os.getenv("LLM_API_KEY"),
        )

    def _load_system_prompt(self, business_problem: str) -> str:
        path = Path("prompts") / PROMPT_VERSION
        if not path.exists():
            raise FileNotFoundError(
                f"System prompt not found: {path}. "
                f"Expected prompts/{PROMPT_VERSION}"
            )
        prompt = path.read_text()
        return (
            prompt
            .replace("{output_path}", str(self.output_path) + "/")
            .replace("{run_id}", self.run_id)
        )

    def build_message(
        self,
        dataset_path: str,
        target_column: str,
        business_problem: str,
        priority_metric: str = "roc_auc",
        domain: str = "general",
    ) -> str:
        data_card_path     = self.output_path / f"{self.run_id}_data_card.json"
        model_sel_path     = self.output_path / f"{self.run_id}_model_selection.json"
        recommendation_path = self.output_path / f"{self.run_id}_recommendation.json"

        return f"""
## Run Context

- **Run ID:** {self.run_id}
- **Dataset:** {dataset_path}
- **Target column:** {target_column}
- **Business problem:** {business_problem}
- **Domain:** {domain}
- **Priority metric:** {priority_metric}
- **Output directory:** {self.output_path}/

---

## Your Output Files

You must produce exactly these three files before marking the run complete:

1. `{data_card_path}` — written after Phase 1 (Data Analyst)
2. `{model_sel_path}` — written after Phases 2 & 3 (ML Engineer + ML Test Engineer)
3. `{recommendation_path}` — written after Phase 4 (Data Scientist)

---

## Start Here — Phase 1: Data Analyst

Begin by adopting the **Data Analyst** persona from your system instructions.

Explore `{dataset_path}` thoroughly. Profile every column, compute real class
balance for `{target_column}`, identify correlations, and flag data quality issues.

Write all exploration code to a `.py` file before running it. When your analysis
is complete, write your findings to `{data_card_path}` using the schema in your
system instructions.

Then — based only on what the data tells you — decide which models to train and
transition to Phase 2.
"""

    def run(self, message: str, business_problem: str) -> dict:
        system_prompt = self._load_system_prompt(business_problem)

        agent = Agent(
            llm=self.llm,
            tools=get_tools(),
            system_prompt=system_prompt,
            max_iterations=self.max_iterations,
        )

        conversation = Conversation(
            agent=agent,
            workspace=os.getcwd()
        )

        start = time.time()
        conversation.send_message(message)
        conversation.run()

        # If recommendation not yet written, push the agent to continue
        # (handles cases where the agent paused between phases)
        for push in range(3):
            if self._read_output("recommendation"):
                break
            completed = [
                s for s in ("data_card", "model_selection")
                if self._read_output(s)
            ]
            if not completed:
                break
            print(f"\n  ↪ Agent paused after {completed[-1]} — pushing to continue...")
            conversation.send_message(
                f"Continue. You have written: {completed}. "
                f"Do not stop — complete the remaining phases and write "
                f"recommendation.json, then use TaskTracker to finish."
            )
            conversation.run()

        self.duration = round(time.time() - start, 2)
        return self._collect_outputs()

    def _collect_outputs(self) -> dict:
        """Read all three output files written by the agent."""
        data_card      = self._read_output("data_card")
        model_sel      = self._read_output("model_selection")
        recommendation = self._read_output("recommendation")

        if recommendation:
            self.success = True
            print(f"\n  ✓ MIRA agent completed in {self.duration}s")
            print(f"  📄 data_card:      {self.output_path}/{self.run_id}_data_card.json")
            print(f"  📄 model_selection:{self.output_path}/{self.run_id}_model_selection.json")
            print(f"  📄 recommendation: {self.output_path}/{self.run_id}_recommendation.json")
        else:
            self.success = False
            print(f"\n  ✗ Agent did not produce a recommendation after {self.duration}s")

        return {
            "data_card": data_card,
            "model_selection": model_sel,
            "recommendation": recommendation,
        }

    def _read_output(self, suffix: str) -> dict:
        path = self.output_path / f"{self.run_id}_{suffix}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                print(f"  ⚠️  Could not parse {path}")
        return {}

    def get_metrics(self) -> dict:
        recommendation = self._read_output("recommendation")
        return {
            "phase": "unified_agent",
            "role": "MIRA Agent (v0.2.0)",
            "success": self.success,
            "duration_seconds": self.duration,
            "confidence": recommendation.get("confidence_score", 0.0),
            "output_files": {
                "data_card":      str(self.output_path / f"{self.run_id}_data_card.json"),
                "model_selection": str(self.output_path / f"{self.run_id}_model_selection.json"),
                "recommendation": str(self.output_path / f"{self.run_id}_recommendation.json"),
            },
        }

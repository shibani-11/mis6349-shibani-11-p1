# agent/agents/base_agent.py
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from openhands.sdk import LLM, Agent, Conversation
from agent.tools import get_tools

load_dotenv()


class BaseAgent:
    """Base class for all MIRA specialist agents."""

    def __init__(
        self,
        phase: str,
        persona_name: str,
        role_label: str,
        run_id: str,
        output_path: str,
        max_iterations: int = 40,
    ):
        self.phase = phase
        self.persona_name = persona_name
        self.role_label = role_label
        self.run_id = run_id
        self.output_path = Path(output_path)
        self.max_iterations = max_iterations
        self.output_file = self.output_path / f"{run_id}_{phase}.json"
        self.duration = 0.0
        self.success = False
        self.confidence = 0.0
        self.errors = []

        self.llm = LLM(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o"),
            api_key=os.getenv("LLM_API_KEY"),
        )

    def load_persona(self, business_problem: str) -> str:
        prompts_dir = Path("prompts")
        matches = sorted(
            prompts_dir.glob(f"{self.persona_name}_v*.md")
        )
        if not matches:
            raise FileNotFoundError(
                f"No prompt for persona '{self.persona_name}'"
            )
        prompt = matches[-1].read_text()
        return prompt.replace("{business_problem}", business_problem)

    def run(self, message: str, business_problem: str) -> dict:
        print(f"\n{'='*60}")
        print(f"  AGENT    : {self.role_label}")
        print(f"  PHASE    : {self.phase.upper().replace('_', ' ')}")
        print(f"  RUN ID   : {self.run_id}")
        print(f"{'='*60}")

        system_prompt = self.load_persona(business_problem)

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
        self.duration = round(time.time() - start, 2)

        if self.output_file.exists():
            self.success = True
            output = json.loads(self.output_file.read_text())
            self.confidence = output.get("confidence_score", 0.8)
            print(f"\n  ✓ {self.role_label} completed in {self.duration}s")
            print(f"  📄 Output: {self.output_file}")
            return output
        else:
            self.success = False
            print(f"\n  ✗ {self.role_label} failed — no output file")
            return {}

    def get_metrics(self) -> dict:
        return {
            "phase": self.phase,
            "role": self.role_label,
            "success": self.success,
            "duration_seconds": self.duration,
            "confidence": self.confidence,
            "output_file": str(self.output_file),
            "errors": self.errors,
        }
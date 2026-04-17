# agent/agents/narrative_agent.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openhands.sdk import LLM

load_dotenv()

_PERSONA_MAP = {
    "data_analyst":     "data_analyst_v0_1_0.md",
    "ml_engineer":      "ml_engineer_v0_1_0.md",
    "ml_test_engineer": "ml_test_engineer_v0_1_0.md",
    "data_scientist":   "data_scientist_v0_1_0.md",
}

_PHASE_PROMPTS = {
    "data_exploration": (
        "Summarize this data exploration in 2-3 sentences from your persona's perspective. "
        "Highlight the most important business-relevant finding (e.g., class imbalance, "
        "missing data, key column distributions). No technical jargon — speak to a business audience."
    ),
    "model_building": (
        "Summarize what was learned during model building in 2-3 sentences. "
        "Focus on which models were trained and how they compare on the priority metric. "
        "Translate model performance into business terms — avoid raw metric names."
    ),
    "model_testing": (
        "Summarize the model testing results in 2-3 sentences. "
        "Highlight which models passed quality checks and any risks flagged (overfitting, leakage). "
        "Frame findings in terms of business readiness, not statistical tests."
    ),
}

_COMPACT_KEYS = {
    "data_exploration": [
        "row_count", "column_count", "target_column",
        "class_distribution", "missing_value_summary",
        "class_imbalance_detected", "minority_class_ratio",
    ],
    "model_building": [
        "models_evaluated", "class_imbalance_detected", "minority_class_ratio",
    ],
    "model_testing": [
        "test_results", "top_models", "flagged_models", "primary_metric",
    ],
}


class NarrativeAgent:
    """
    Generates a persona-voiced narrative for a pipeline phase using a
    single OpenHands LLM completion call (no tools, no conversation loop).
    """

    def __init__(self):
        self.llm = LLM(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o"),
            api_key=os.getenv("LLM_API_KEY"),
        )

    def _load_persona(self, persona_name: str, business_problem: str) -> str:
        filename = _PERSONA_MAP.get(persona_name)
        if not filename:
            return f"You are a {persona_name} analyzing an ML pipeline."
        path = Path("prompts") / filename
        if not path.exists():
            return f"You are a {persona_name} analyzing an ML pipeline."
        return path.read_text().replace("{business_problem}", business_problem)

    def generate(
        self,
        phase: str,
        phase_output: dict,
        business_problem: str,
        persona_name: str,
    ) -> str:
        system_prompt = self._load_persona(persona_name, business_problem)
        task = _PHASE_PROMPTS.get(
            phase,
            "Summarize the key findings from this phase in 2-3 sentences for a business audience."
        )

        keys = _COMPACT_KEYS.get(phase, list(phase_output.keys())[:10])
        compact = {k: phase_output[k] for k in keys if k in phase_output}

        user_message = (
            f"Business problem: {business_problem}\n\n"
            f"Phase output summary:\n{json.dumps(compact, indent=2)}\n\n"
            f"{task}"
        )

        try:
            response = self.llm.completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=250,
                temperature=0.5,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [NarrativeAgent] LLM call failed for {phase}: {e}")
            return phase_output.get("genai_narrative", "")

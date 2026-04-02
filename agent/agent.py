# agent/agent.py
# ML Engineer Agent — 5-phase business problem solver
# Built on OpenHands SDK


import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path


from dotenv import load_dotenv
from openhands.sdk import LLM, Agent, Conversation


from agent.tools import get_tools
from schemas.input_schema import AgentInput
from schemas.output_schema import AnalysisReport


load_dotenv()




def load_persona(persona_name: str, business_problem: str) -> str:
    """Load a persona prompt file and inject the business problem."""
    prompts_dir = Path("prompts")
    
    # Find the latest version of the persona file
    matches = sorted(prompts_dir.glob(f"{persona_name}_v*.md"))
    if not matches:
        raise FileNotFoundError(
            f"No prompt file found for persona '{persona_name}' in prompts/"
        )
    
    prompt_path = matches[-1]  # use latest version
    prompt = prompt_path.read_text()
    return prompt.replace("{business_problem}", business_problem)




def build_phase_message(phase: str, agent_input: AgentInput, prior_context: str = "") -> str:
    """
    Build the instruction message sent to the agent for each phase.
    Includes dataset info, business problem, and any prior phase results.
    """
    base = f"""
You are now executing Phase: {phase.upper()}


Dataset: {agent_input.dataset_path}
Target Column: {agent_input.target_column}
Task Type: {agent_input.task_type}
Business Problem: {agent_input.business_problem}
Output Directory: {agent_input.output_path}
Run ID: {agent_input.run_id}
"""


    if agent_input.extra_context:
        base += f"\nAdditional Context: {json.dumps(agent_input.extra_context)}"


    if agent_input.id_columns:
        base += f"\nKnown ID Columns (exclude from features): {agent_input.id_columns}"


    if agent_input.date_columns:
        base += f"\nKnown Date Columns: {agent_input.date_columns}"


    if prior_context:
        base += f"\n\nPrior Phase Results Summary:\n{prior_context}"


    phase_instructions = {
        "descriptive": """
Your task:
1. Read the dataset from the path above using pandas
2. Profile every column: dtype, null count, unique values, sample values
3. Identify numeric, categorical, datetime, text, and ID columns
4. Describe the target column distribution
5. Infer the task type if set to 'auto'
6. Write your findings as a JSON matching the DescriptiveAnalysis schema
7. Save it to: {output_path}{run_id}_descriptive.json
8. End with a genai_narrative: 1 paragraph in plain English
""",
        "diagnostic": """
Your task:
1. Load the dataset and the descriptive results from prior phase
2. Check for class imbalance in the target column
3. Compute correlation matrix and flag pairs with |corr| > 0.85
4. Detect outliers using IQR method
5. Identify skewed numeric columns (|skew| > 1)
6. Flag any columns that risk target leakage
7. Recommend preprocessing and feature engineering steps
8. Write findings as JSON matching DiagnosticAnalysis schema
9. Save to: {output_path}{run_id}_diagnostic.json
10. End with a genai_narrative prioritizing issues by severity
""",
        "predictive": """
Your task:
1. Load the dataset and apply preprocessing from diagnostic phase
2. Based on dataset characteristics, autonomously select up to {max_models} appropriate models
3. For each model:
   a. Train using 5-fold cross-validation
   b. Compute all relevant metrics for the task type
   c. Record training time
   d. Check for overfitting (train vs val gap)
4. Rank models by primary metric
5. Compute feature importance for tree-based models
6. Write results as JSON matching PredictiveAnalysis schema
7. Save to: {output_path}{run_id}_predictive.json
8. End with genai_narrative explaining results in business terms
""",
        "prescriptive": """
Your task:
1. Load predictive results from prior phase
2. Select the best model considering:
   - Business problem context
   - Primary metric performance
   - Overfitting status
   - Model interpretability needs
3. Justify the selection in business terms
4. Name an alternative model
5. List 3+ concrete next steps
6. Suggest hyperparameter tuning directions
7. Write results as JSON matching PrescriptiveAnalysis schema
8. Save to: {output_path}{run_id}_prescriptive.json
9. End with genai_narrative written for a manager
""",
        "genai_synthesis": """
Your task:
1. Load results from all 4 prior phases
2. Write an executive briefing that:
   - States the recommendation in one sentence
   - Summarizes the 3 most important findings
   - Identifies top 2 business risks
   - States confidence level (high/medium/low) and why
   - Gives a clear YES/NO on proceeding to deployment
3. Write results as JSON matching GenAISynthesis schema
4. Save to: {output_path}{run_id}_genai_synthesis.json
5. Also save the complete AnalysisReport to:
   {output_path}{run_id}_report.json
"""
    }


    instruction = phase_instructions.get(phase, "")
    instruction = instruction.replace("{output_path}", agent_input.output_path)
    instruction = instruction.replace("{run_id}", agent_input.run_id)
    instruction = instruction.replace("{max_models}", str(agent_input.max_models))


    return base + instruction




class MLEngineerAgent:
    """
    Multi-persona ML agent that solves business problems
    by running 5 sequential analysis phases, each powered
    by a different expert persona.
    """


    PHASE_PERSONA_MAP = {
        "descriptive":     "data_analyst",
        "diagnostic":      "data_engineer",
        "predictive":      "ml_engineer",
        "prescriptive":    "business_analyst",
        "genai_synthesis": "executive_advisor",
    }


    def __init__(self, agent_input: AgentInput):
        self.input = agent_input
        self.run_id = agent_input.run_id
        self.results = {}
        self.phase_durations = {}
        self.errors = []


        # Ensure output directory exists
        Path(agent_input.output_path).mkdir(parents=True, exist_ok=True)
        Path("logs/runs").mkdir(parents=True, exist_ok=True)


        # Set up LLM — using OpenAI GPT-4o via LiteLLM
        self.llm = LLM(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o"),
            api_key=os.getenv("LLM_API_KEY"),
        )


    def _run_phase(self, phase: str, prior_context: str = "") -> str:
        """Run a single phase with its corresponding persona."""
        persona_name = self.PHASE_PERSONA_MAP[phase]
        print(f"\n{'='*60}")
        print(f"  PHASE: {phase.upper()}")
        print(f"  PERSONA: {persona_name.replace('_', ' ').title()}")
        print(f"{'='*60}")


        # Load persona prompt
        system_prompt = load_persona(persona_name, self.input.business_problem)


        # Build agent with this persona's system prompt
        agent = Agent(
            llm=self.llm,
            tools=get_tools(),
            system_prompt=system_prompt,
            max_iterations=self.input.max_iterations,
        )


        # Build phase-specific instruction
        message = build_phase_message(phase, self.input, prior_context)


        # Run the conversation
        workspace = os.getcwd()
        conversation = Conversation(agent=agent, workspace=workspace)


        start = time.time()
        conversation.send_message(message)
        conversation.run()
        duration = time.time() - start


        self.phase_durations[phase] = round(duration, 2)
        print(f"\n  ✓ Phase '{phase}' completed in {duration:.1f}s")


        # Load and return the phase output for context chaining
        output_file = Path(self.input.output_path) / f"{self.run_id}_{phase}.json"
        if output_file.exists():
            return output_file.read_text()
        return ""


    def run(self) -> dict:
        """
        Run all 5 phases sequentially.
        Each phase receives the prior phase's output as context.
        """
        print(f"\n🚀 Starting ML Engineer Agent")
        print(f"   Run ID     : {self.run_id}")
        print(f"   Dataset    : {self.input.dataset_path}")
        print(f"   Target     : {self.input.target_column}")
        print(f"   Problem    : {self.input.business_problem}")


        total_start = time.time()
        prior_context = ""
        phases_completed = []
        phases_skipped = []


        for phase in self.input.analysis_phases:
            try:
                result = self._run_phase(phase, prior_context)
                prior_context = result  # chain output to next phase
                phases_completed.append(phase)
                self.results[phase] = result
            except Exception as e:
                error_msg = f"Phase '{phase}' failed: {str(e)}"
                print(f"\n  ✗ {error_msg}")
                self.errors.append(error_msg)
                phases_skipped.append(phase)


        total_duration = round(time.time() - total_start, 2)


        # Write run log
        self._write_log(phases_completed, phases_skipped, total_duration)


        print(f"\n✅ Agent complete in {total_duration}s")
        print(f"   Report: {self.input.output_path}{self.run_id}_report.json")


        return self.results


    def _write_log(self, phases_completed: list, phases_skipped: list, total_duration: float):
        """Write a structured run log to logs/runs/."""
        log = {
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_path": self.input.dataset_path,
            "target_column": self.input.target_column,
            "business_problem": self.input.business_problem,
            "phases_completed": phases_completed,
            "phases_skipped": phases_skipped,
            "phase_durations_seconds": self.phase_durations,
            "total_duration_seconds": total_duration,
            "errors": self.errors,
            "llm_model": os.getenv("LLM_MODEL", "openai/gpt-4o"),
        }
        log_path = Path("logs/runs") / f"{self.run_id}.json"
        log_path.write_text(json.dumps(log, indent=2))
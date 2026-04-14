# agent/agent.py
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openhands.sdk import LLM, Agent, Conversation

from agent.tools import get_tools
from schemas.input_schema import AgentInput

load_dotenv()


def load_persona(persona_name: str, business_problem: str) -> str:
    prompts_dir = Path("prompts")
    matches = sorted(prompts_dir.glob(f"{persona_name}_v*.md"))
    if not matches:
        raise FileNotFoundError(
            f"No prompt file found for persona '{persona_name}' in prompts/"
        )
    prompt = matches[-1].read_text()
    return prompt.replace("{business_problem}", business_problem)


def build_phase_message(phase: str, agent_input: AgentInput, prior_context: str = "") -> str:
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

        "data_exploration": """
ROLE: Data Analyst

⚠️  CRITICAL INSTRUCTIONS — READ BEFORE DOING ANYTHING:
- NEVER use file_editor to read the dataset — it has a 10MB size limit
- ALWAYS use the terminal tool to run pandas commands
- ALWAYS write Python scripts to a .py file and run them via terminal
- The FULL dataset is at: {dataset_path}
- You MUST use ALL rows — never sample or truncate

CORRECT way to read data:
  Write a script to explore.py then run: python3 explore.py

WRONG way (never do this):
  file_editor view {dataset_path}

Your task:
1. Read the FULL dataset using pandas via terminal
2. Profile every column:
   - dtype, null count, null percentage
   - unique value count
   - sample values (first 3 non-null)
   - column type: numeric, categorical, datetime, text, or ID
3. Analyze the target column ({target_column}) distribution
   - Count of each class
   - Class imbalance ratio
4. Detect class imbalance
   - Flag if minority class < 20% of total
5. Identify data quality issues:
   - Missing values per column
   - Duplicate rows
   - Columns with only 1 unique value (useless features)
6. Compute statistics for ALL numeric columns:
   - mean, std, min, max, 25th, 50th, 75th percentile
7. Save ALL findings as JSON to:
   {output_path}{run_id}_data_exploration.json

   The JSON MUST contain these exact keys:
   {{
     "row_count": int,
     "column_count": int,
     "inferred_task_type": "classification" or "regression",
     "overall_missing_pct": float,
     "duplicate_row_count": int,
     "target_distribution": {{"0": int, "1": int}},
     "class_imbalance_detected": bool,
     "imbalance_ratio": float,
     "numeric_columns": [list of column names],
     "categorical_columns": [list of column names],
     "id_columns": [list of column names],
     "text_columns": [list of column names],
     "columns": [list of column profiles],
     "numeric_stats": {{column_name: {{mean, std, min, max}}}},
     "quality_issues": [list of issues found],
     "genai_narrative": "2-3 sentence plain English summary"
   }}

8. End with genai_narrative:
   - What does this dataset tell us about the business problem?
   - Written for a non-technical business stakeholder
   - Maximum 3 sentences
""",

        "model_building": """
ROLE: ML Engineer

⚠️  CRITICAL INSTRUCTIONS — READ BEFORE DOING ANYTHING:
- NEVER use file_editor to read the dataset — it has a 10MB size limit
- ALWAYS use terminal tool to run pandas and sklearn via Python scripts
- Write all code to .py files and run them via terminal
- The FULL dataset is at: {dataset_path}
- You MUST use ALL rows — never sample or truncate
- Load prior phase results from: {output_path}{run_id}_data_exploration.json

Your task:
1. Load the FULL dataset using pandas via terminal
2. Load data exploration results from prior phase
3. Apply preprocessing:
   - Drop ID columns identified in exploration phase
   - Handle missing values (impute or drop)
   - Encode categorical columns (LabelEncoder or OneHotEncoder)
   - Scale numeric columns (StandardScaler)
   - Handle class imbalance using class_weight='balanced'
4. Split data: 80% train, 20% validation (random_state=42)
5. Autonomously select up to {max_models} models based on:
   - Dataset size and task type from exploration phase
   - Class imbalance detected in prior phase
   - Always include: Logistic Regression as baseline
   - Add tree-based: Random Forest, XGBoost, LightGBM
   - Add one more based on your judgment
6. Train each model using 5-fold cross-validation
7. For each model record:
   - model_name, model_family
   - accuracy, precision, recall, f1_score, roc_auc
   - training_time_seconds
   - overfitting_detected (train score > val score by >10%)
   - cross_val_score_mean, cross_val_score_std
   - feature_count_used
8. Save results as JSON to:
   {output_path}{run_id}_model_building.json

   The JSON MUST contain these exact keys:
   {{
     "models_considered": [list],
     "models_evaluated": [list of model metric dicts],
     "models_skipped": {{model: reason}},
     "primary_metric": "roc_auc",
     "best_model_by_metric": {{metric: model_name}},
     "feature_importance": {{feature: score}} or null,
     "preprocessing_steps": [list of steps applied],
     "genai_narrative": "plain English summary"
   }}

9. End with genai_narrative:
   - Which models were built and why
   - What do the results suggest?
   - Written for a technical lead
""",

        "model_testing": """
ROLE: ML Test Engineer

⚠️  CRITICAL INSTRUCTIONS — READ BEFORE DOING ANYTHING:
- NEVER use file_editor to read the dataset — it has a 10MB size limit
- ALWAYS use terminal tool to run Python scripts
- Write all code to .py files and run them via terminal
- The FULL dataset is at: {dataset_path}
- Load model building results from: {output_path}{run_id}_model_building.json
- Load exploration results from: {output_path}{run_id}_data_exploration.json

Your task:
1. Load the FULL dataset and retrain all models from prior phase
   using the same preprocessing and hyperparameters
2. For each model run rigorous testing:
   a. Evaluate on held-out validation set (20%)
   b. Generate confusion matrix (TP, FP, TN, FN)
   c. Check overfitting:
      - train_score vs val_score
      - Flag if gap > 10%
   d. Check data leakage:
      - Flag if ROC-AUC > 0.99 on first run
   e. Test stability:
      - Cross-val std > 0.05 = unstable, flag it
   f. Check feature importance business logic:
      - Are top features financially meaningful?
      - Flag if random/ID columns are top features
3. Flag models that FAIL any of:
   - Overfitting detected
   - Possible data leakage
   - Unstable cross-validation
   - Top features make no business sense
4. Rank all models by: {priority_metric}
5. Identify top 2 passing models
6. Save test report as JSON to:
   {output_path}{run_id}_model_testing.json

   The JSON MUST contain these exact keys:
   {{
     "models_tested": [list of model names],
     "test_results": [{{
       "model_name": str,
       "val_accuracy": float,
       "val_roc_auc": float,
       "val_f1": float,
       "confusion_matrix": {{TP, FP, TN, FN}},
       "train_val_gap": float,
       "overfitting_detected": bool,
       "leakage_suspected": bool,
       "stability_ok": bool,
       "features_business_logical": bool,
       "passed_testing": bool,
       "fail_reasons": [list]
     }}],
     "top_models": [top 2 model names],
     "flagged_models": [list of failed model names],
     "primary_metric": "{priority_metric}",
     "genai_narrative": "plain English summary"
   }}

7. End with genai_narrative:
   - Which models passed testing and why
   - Any major risks found
   - Written for a technical lead
""",

        "recommendation": """
ROLE: Business Analyst

⚠️  CRITICAL INSTRUCTIONS — READ BEFORE DOING ANYTHING:
- Load ALL prior phase results:
  - {output_path}{run_id}_data_exploration.json
  - {output_path}{run_id}_model_building.json
  - {output_path}{run_id}_model_testing.json
- Base recommendation ONLY on models that passed testing

Your task:
1. Review all prior phase results thoroughly
2. Select the single best model considering:
   - Passed all testing checks
   - Best {priority_metric} score
   - Business interpretability
   - Deployment feasibility
   - Risk to the business if wrong predictions are made
3. Translate performance into business terms:
   - e.g. "ROC-AUC of 0.85 means the model correctly
     ranks 85% of defaulters above non-defaulters"
   - e.g. "Recall of 0.78 means we catch 78% of actual
     defaulters before they default"
4. Write full recommendation covering:
   - recommended_model and why in business terms
   - selection_reason (business language, no jargon)
   - primary_metric_value
   - tradeoffs (top 3, honest)
   - alternative_model (runner-up)
   - next_steps (3 concrete actions)
   - deployment_considerations
   - risks and mitigation
   - confidence_score (0.0-1.0)
   - requires_human_review (true if confidence < 0.75)
   - human_review_reason (if applicable)
5. Save recommendation as JSON to:
   {output_path}{run_id}_recommendation.json

   The JSON MUST contain these exact keys:
   {{
     "recommended_model": str,
     "selection_reason": str,
     "primary_metric_value": float,
     "tradeoffs": [list],
     "alternative_model": str,
     "next_steps": [list of 3],
     "deployment_considerations": [list],
     "risks": [list],
     "confidence_score": float,
     "requires_human_review": bool,
     "human_review_reason": str or null,
     "executive_summary": str
   }}

6. End with executive_summary:
   - Maximum 1 paragraph
   - Boardroom-ready, zero technical jargon
   - Translate ALL metrics into business outcomes
   - End with a clear YES or NO on deploying this model
"""
    }

    instruction = phase_instructions.get(phase, "")
    instruction = instruction.replace("{dataset_path}", agent_input.dataset_path)
    instruction = instruction.replace("{target_column}", agent_input.target_column)
    instruction = instruction.replace("{output_path}", agent_input.output_path)
    instruction = instruction.replace("{run_id}", agent_input.run_id)
    instruction = instruction.replace("{max_models}", str(agent_input.max_models))
    instruction = instruction.replace(
        "{priority_metric}",
        agent_input.extra_context.get("priority_metric", "roc_auc")
        if agent_input.extra_context else "roc_auc"
    )

    return base + instruction


class MIRA:
    """
    MIRA — Model Intelligence & Recommendation Agent
    Multi-role ML agent that solves business problems
    through 4 industry-standard phases, each powered
    by a different professional persona.
    """

    PHASE_PERSONA_MAP = {
        "data_exploration": "data_analyst",
        "model_building":   "ml_engineer",
        "model_testing":    "ml_test_engineer",
        "recommendation":   "business_analyst",
    }

    PHASE_LABELS = {
        "data_exploration": "Data Analyst",
        "model_building":   "ML Engineer",
        "model_testing":    "ML Test Engineer",
        "recommendation":   "Business Analyst",
    }

    def __init__(self, agent_input: AgentInput):
        self.input = agent_input
        self.run_id = agent_input.run_id
        self.results = {}
        self.phase_durations = {}
        self.errors = []

        Path(agent_input.output_path).mkdir(parents=True, exist_ok=True)
        Path("logs/runs").mkdir(parents=True, exist_ok=True)

        self.llm = LLM(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o"),
            api_key=os.getenv("LLM_API_KEY"),
        )

    def _run_phase(self, phase: str, prior_context: str = "") -> str:
        persona_name = self.PHASE_PERSONA_MAP[phase]
        role_label = self.PHASE_LABELS[phase]

        print(f"\n{'='*60}")
        print(f"  PHASE    : {phase.upper().replace('_', ' ')}")
        print(f"  ROLE     : {role_label}")
        print(f"  RUN ID   : {self.run_id}")
        print(f"{'='*60}")

        system_prompt = load_persona(persona_name, self.input.business_problem)

        agent = Agent(
            llm=self.llm,
            tools=get_tools(),
            system_prompt=system_prompt,
            max_iterations=self.input.max_iterations,
        )

        message = build_phase_message(phase, self.input, prior_context)
        workspace = os.getcwd()
        conversation = Conversation(agent=agent, workspace=workspace)

        start = time.time()
        conversation.send_message(message)
        conversation.run()
        duration = time.time() - start

        self.phase_durations[phase] = round(duration, 2)
        print(f"\n  ✓ '{phase.replace('_', ' ')}' completed in {duration:.1f}s")

        output_file = Path(self.input.output_path) / f"{self.run_id}_{phase}.json"
        if output_file.exists():
            content = output_file.read_text()
            print(f"  📄 Output saved: {output_file}")
            return content

        print(f"  ⚠️  No output file found for phase '{phase}'")
        return ""

    def run(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  🚀 Starting MIRA — Model Intelligence & Recommendation Agent")
        print(f"{'='*60}")
        print(f"  Run ID   : {self.run_id}")
        print(f"  Dataset  : {self.input.dataset_path}")
        print(f"  Target   : {self.input.target_column}")
        print(f"  Problem  : {self.input.business_problem[:55]}...")
        print(f"  Phases   : {' → '.join(self.input.analysis_phases)}")
        print(f"{'='*60}")

        total_start = time.time()
        prior_context = ""
        phases_completed = []
        phases_skipped = []

        for phase in self.input.analysis_phases:
            try:
                result = self._run_phase(phase, prior_context)
                prior_context = result
                phases_completed.append(phase)
                self.results[phase] = result
            except Exception as e:
                error_msg = f"Phase '{phase}' failed: {str(e)}"
                print(f"\n  ✗ {error_msg}")
                self.errors.append(error_msg)
                phases_skipped.append(phase)

        total_duration = round(time.time() - total_start, 2)
        self._write_log(phases_completed, phases_skipped, total_duration)

        print(f"\n{'='*60}")
        print(f"  ✅ MIRA complete in {total_duration}s")
        print(f"  📁 Reports: {self.input.output_path}")
        if phases_skipped:
            print(f"  ✗ Skipped : {', '.join(phases_skipped)}")
        print(f"{'='*60}\n")

        return self.results

    def _write_log(self, phases_completed, phases_skipped, total_duration):
        log = {
            "agent": "MIRA — Model Intelligence & Recommendation Agent",
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
        print(f"  📋 Log: {log_path}")
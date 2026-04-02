from pydantic import BaseModel, field_validator, model_validator
from typing import List, Dict, Optional
import uuid


class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


class ModelResult(BaseModel):
    name: str
    metrics: ModelMetrics
    latency_ms: float
    training_time_sec: float
    inference_cost: str
    interpretability: str
    robustness_notes: str
    deployment_notes: str


# Supported file extensions for datasets
VALID_EXTENSIONS = [".csv", ".xlsx", ".xls", ".parquet", ".json"]


class AgentInput(BaseModel):
    # Required fields
    dataset_path: str
    target_column: str
    business_problem: str
    
    # Optional fields with defaults
    task_type: str = "auto"
    max_models: int = 5
    max_iterations: int = 40
    output_path: str = "processed/"
    
    # Optional context
    extra_context: Optional[Dict] = None
    id_columns: Optional[List[str]] = None
    date_columns: Optional[List[str]] = None
    
    # Analysis phases to run (default all 5)
    analysis_phases: Optional[List[str]] = None
    
    # Auto-generated fields
    run_id: Optional[str] = None
    
    @field_validator("run_id", mode="before")
    @classmethod
    def generate_run_id(cls, v):
        if v is None or v == "":
            return f"run_{uuid.uuid4().hex[:8]}"
        return v
    
    @model_validator(mode="after")
    def set_run_id_if_none(self):
        if self.run_id is None:
            self.run_id = f"run_{uuid.uuid4().hex[:8]}"
        return self
    
    @field_validator("dataset_path")
    @classmethod
    def validate_file_format(cls, v):
        # Check that the file has a valid extension
        has_valid_ext = any(v.lower().endswith(ext) for ext in VALID_EXTENSIONS)
        if not has_valid_ext:
            raise ValueError(
                f"Invalid file format. Supported: {', '.join(VALID_EXTENSIONS)}"
            )
        return v
    
    @field_validator("target_column")
    @classmethod
    def validate_target_column(cls, v):
        # Ensure target column is not empty or just whitespace
        if not v or not v.strip():
            raise ValueError("target_column cannot be empty")
        return v.strip()
    
    @field_validator("analysis_phases", mode="before")
    @classmethod
    def set_default_phases(cls, v):
        if v is None:
            return ["descriptive", "diagnostic", "predictive", "prescriptive", "genai_synthesis"]
        return v

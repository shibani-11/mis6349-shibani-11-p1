from __future__ import annotations
import os
import uuid
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, validator


class AgentInput(BaseModel):

    # ── Required ──────────────────────────────────────────────────
    dataset_path: str = Field(
        ...,
        description="Path to dataset. Supports .csv, .xlsx, .xls, .tsv, .parquet, .json",
        example="data/raw/train.csv"
    )

    target_column: str = Field(
        ...,
        description="Column the agent should predict.",
        example="Loan Status"
    )

    business_problem: str = Field(
        ...,
        description="Plain English description of the business problem to solve.",
        example="Identify which loan applicants are likely to default."
    )

    # ── Optional: Task ─────────────────────────────────────────────
    task_type: Literal["auto", "classification", "regression", "clustering", "time_series"] = Field(
        default="auto",
        description="ML task type. 'auto' lets the agent infer from target column."
    )

    analysis_phases: List[
        Literal["data_exploration", "model_building", "model_testing", "recommendation"]
    ] = Field(
        default=["data_exploration", "model_building", "model_testing", "recommendation"],
        description="Which phases to run."
    )

    max_models: int = Field(default=5, ge=1, le=10)
    max_iterations: int = Field(default=40, ge=5, le=100)

    # ── Optional: Dataset Hints ────────────────────────────────────
    date_columns: Optional[List[str]] = Field(default=None)
    id_columns: Optional[List[str]] = Field(default=None)
    text_columns: Optional[List[str]] = Field(default=None)

    # ── Optional: Run Metadata ─────────────────────────────────────
    run_id: str = Field(
        default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}"
    )

    output_path: str = Field(default="processed/")

    extra_context: Optional[Dict[str, str]] = Field(
        default=None,
        example={"domain": "finance", "priority_metric": "roc_auc"}
    )

    # ── Validators ─────────────────────────────────────────────────
    @validator("dataset_path")
    def validate_file_format(cls, v):
        allowed = {".csv", ".xlsx", ".xls", ".tsv", ".parquet", ".json"}
        ext = os.path.splitext(v)[-1].lower()
        if ext not in allowed:
            raise ValueError(f"Unsupported format '{ext}'. Allowed: {allowed}")
        return v

    @validator("target_column")
    def validate_target_not_empty(cls, v):
        if not v.strip():
            raise ValueError("target_column cannot be blank.")
        return v.strip()

    @validator("business_problem")
    def validate_business_problem_not_empty(cls, v):
        if not v.strip():
            raise ValueError("business_problem cannot be blank.")
        return v.strip()

    @validator("output_path")
    def validate_output_path(cls, v):
        return v.rstrip("/") + "/"

    class Config:
        use_enum_values = True
        validate_assignment = True
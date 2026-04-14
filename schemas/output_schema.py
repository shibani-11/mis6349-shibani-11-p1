from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    cardinality: Literal["low", "medium", "high", "identifier"]
    sample_values: List[Any]
    is_target: bool = False


class DescriptiveAnalysis(BaseModel):
    row_count: int
    column_count: int
    target_column: str
    inferred_task_type: Literal["classification", "regression", "clustering", "time_series"]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    text_columns: List[str]
    id_columns: List[str]
    column_profiles: List[ColumnProfile]
    overall_missing_pct: float
    duplicate_row_count: int
    memory_usage_mb: float
    genai_narrative: str


class CorrelationPair(BaseModel):
    col_a: str
    col_b: str
    correlation: float
    concern: Literal["multicollinearity", "target_leakage", "informative", "none"]


class DiagnosticAnalysis(BaseModel):
    class_imbalance_detected: bool
    imbalance_ratio: Optional[float] = None
    imbalance_handling_strategy: Optional[str] = None
    high_correlation_pairs: List[CorrelationPair]
    potential_target_leakage_columns: List[str]
    outlier_columns: List[str]
    outlier_method_used: str
    skewed_columns: List[str]
    recommended_preprocessing: List[str]
    recommended_feature_engineering: List[str]
    data_quality_score: float
    genai_narrative: str


class ModelMetrics(BaseModel):
    model_name: str
    model_family: Literal[
        "linear", "tree_based", "ensemble", "boosting",
        "neural_network", "svm", "clustering", "other"
    ]
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    log_loss: Optional[float] = None
    matthews_corrcoef: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    mape: Optional[float] = None
    silhouette_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    training_time_seconds: float
    cross_val_score_mean: Optional[float] = None
    cross_val_score_std: Optional[float] = None
    overfitting_detected: bool = False
    feature_count_used: int


class PredictiveAnalysis(BaseModel):
    models_considered: List[str]
    models_evaluated: List[ModelMetrics]
    models_skipped: Dict[str, str]
    primary_metric: str
    best_model_by_metric: Dict[str, str]
    feature_importance: Optional[Dict[str, float]] = None
    genai_narrative: str


class PrescriptiveAnalysis(BaseModel):
    recommended_model: str
    selection_reason: str
    primary_metric_value: float
    tradeoffs: List[str]
    alternative_model: Optional[str] = None
    next_steps: List[str]
    hyperparameter_suggestions: Dict[str, Any]
    deployment_considerations: List[str]
    confidence_score: float
    genai_narrative: str


class GenAISynthesis(BaseModel):
    executive_summary: str
    key_findings: List[str]
    risks: List[str]
    dataset_specific_insights: List[str]
    requires_human_review: bool
    human_review_reason: Optional[str] = None
    confidence_level: Literal["high", "medium", "low"]


class RunMetadata(BaseModel):
    run_id: str
    persona_used: str
    llm_model: str
    total_iterations_used: int
    total_tool_calls: int
    phases_completed: List[str]
    phases_skipped: List[str]
    total_duration_seconds: float
    phase_durations: Dict[str, float]
    errors_encountered: List[str]
    openhands_sdk_version: str


class AnalysisReport(BaseModel):
    run_id: str
    dataset_path: str
    target_column: str
    task_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    descriptive: DescriptiveAnalysis
    diagnostic: DiagnosticAnalysis
    predictive: PredictiveAnalysis
    prescriptive: PrescriptiveAnalysis
    genai_synthesis: GenAISynthesis
    metadata: RunMetadata
    recommended_model: str
    primary_metric: str
    primary_metric_value: float
    requires_human_review: bool

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
        use_enum_values = True
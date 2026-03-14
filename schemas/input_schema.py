from pydantic import BaseModel
from typing import List, Dict


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


class AgentInput(BaseModel):
    dataset_name: str
    problem_type: str
    business_context: str
    models: List[ModelResult]

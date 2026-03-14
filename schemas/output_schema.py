from pydantic import BaseModel
from typing import List, Dict


class DeploymentReadiness(BaseModel):
    status: str
    notes: str


class AgentOutput(BaseModel):
    recommended_model: str
    runner_up_model: str
    problem_type: str
    decision_summary: str
    primary_reason: str
    tradeoffs: List[str]
    risk_flags: List[str]
    metrics_considered: Dict
    deployment_readiness: DeploymentReadiness

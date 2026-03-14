"""
Pydantic model for expected input schema.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ModelEvaluation(BaseModel):
    """Model evaluation metrics."""
    model_name: str = Field(..., description="Name of the model")
    accuracy: float = Field(..., description="Model accuracy score")
    precision: float = Field(..., description="Model precision score")
    recall: float = Field(..., description="Model recall score")
    f1_score: float = Field(..., description="Model F1 score")
    roc_auc: float = Field(..., description="Model ROC-AUC score")


class InputSchema(BaseModel):
    """Expected input schema for the agent."""
    dataset_path: Optional[str] = Field(None, description="Path to the dataset file")
    model_evaluations: List[ModelEvaluation] = Field(..., description="List of model evaluations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_path": "data/loan_default_dataset.xlsx",
                "model_evaluations": [
                    {
                        "model_name": "logistic_regression",
                        "accuracy": 0.85,
                        "precision": 0.90,
                        "recall": 0.60,
                        "f1_score": 0.72,
                        "roc_auc": 0.81
                    }
                ]
            }
        }

"""
Pydantic model for expected output schema.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class OutputSchema(BaseModel):
    """Expected output schema for the agent recommendation."""
    dataset: str = Field(..., description="Name of the dataset")
    recommended_model: str = Field(..., description="Name of the recommended model")
    selection_reason: str = Field(..., description="Explanation for why this model was recommended")
    tradeoffs: List[str] = Field(default_factory=list, description="Trade-offs to consider for the recommended model")
    dataset_considerations: List[str] = Field(default_factory=list, description="Dataset-specific factors that influenced the recommendation")
    requires_human_review: bool = Field(..., description="Whether human review is needed")
    error: Optional[str] = Field(None, description="Error message if something went wrong")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset": "loan_default_dataset",
                "recommended_model": "xgboost",
                "selection_reason": "The dataset shows moderate class imbalance. XGBoost achieves the highest ROC-AUC and F1 score.",
                "tradeoffs": [
                    "Lower precision than Logistic Regression",
                    "Higher model complexity"
                ],
                "dataset_considerations": [
                    "Dataset is moderately imbalanced",
                    "Recall is important for detecting default cases"
                ],
                "requires_human_review": False
            }
        }

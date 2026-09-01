from pydantic import BaseModel, Field
from typing import List

class ItemServingAdjustment(BaseModel):
    label: str = Field(..., description="Food class label")
    serving_count: float = Field(..., ge=0.1, le=10.0, description="Multiplier for portion size (e.g. 1.5 portions)")

class RecalculateNutritionRequest(BaseModel):
    adjustments: List[ItemServingAdjustment]

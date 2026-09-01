from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class BoundingBox(BaseModel):
    x_min: float = Field(..., description="Top-left X coordinate")
    y_min: float = Field(..., description="Top-left Y coordinate")
    x_max: float = Field(..., description="Bottom-right X coordinate")
    y_max: float = Field(..., description="Bottom-right Y coordinate")

class DetectedFoodItem(BaseModel):
    label: str = Field(..., description="Detected food category name")
    confidence: float = Field(..., description="Prediction confidence score between 0.0 and 1.0")
    class_id: int = Field(..., description="Model class integer ID")
    bbox: BoundingBox = Field(..., description="Bounding box pixel coordinates")
    confidence_level: str = Field(..., description="Qualitative confidence rating: high, medium, low")

class FoodDetectionResponse(BaseModel):
    total_detections: int
    detections: List[DetectedFoodItem]
    annotated_image_url: Optional[str] = None
    has_low_confidence: bool = False
    processing_time_ms: float

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from app.schemas.detection import DetectedFoodItem

class MacroRange(BaseModel):
    min_val: float
    max_val: float
    avg_val: float
    unit: str = "g"

class CalorieRange(BaseModel):
    min_kcal: float
    max_kcal: float
    avg_kcal: float
    unit: str = "kcal"

class FoodItemNutrition(BaseModel):
    label: str
    display_name: str
    category: str
    serving_name: str
    serving_grams: float
    serving_count: float = 1.0
    calories: CalorieRange
    protein: MacroRange
    carbs: MacroRange
    fat: MacroRange
    fiber: Optional[MacroRange] = None
    is_vegetarian: bool
    is_vegan: bool
    is_gluten_free: bool
    confidence: float
    confidence_level: str

class MealNutritionSummary(BaseModel):
    total_calories: CalorieRange
    total_protein: MacroRange
    total_carbs: MacroRange
    total_fat: MacroRange
    detected_items: List[FoodItemNutrition]
    disclaimer: str = "Estimates are derived from visual detection and standard portion assumptions. Not intended as medical or laboratory measurements."
    has_uncertain_items: bool = False
    annotated_image_url: Optional[str] = None
    processing_time_ms: float

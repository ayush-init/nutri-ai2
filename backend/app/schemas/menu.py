from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.schemas.nutrition import CalorieRange, MacroRange

class MenuItem(BaseModel):
    name: str = Field(..., description="Cleaned menu item / dish name")
    estimated_calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    is_vegetarian: bool = True
    category: str = "main_course"
    score: float = 0.0
    recommendation_reason: Optional[str] = None

class MenuRecommendationResponse(BaseModel):
    preference_applied: str = Field(..., description="E.g. balanced, high_protein, low_calorie, vegetarian")
    top_recommendations: List[MenuItem]
    all_items: List[MenuItem]
    explanation: str
    total_items_analyzed: int
    processing_time_ms: float

class MenuTextRequest(BaseModel):
    menu_text: str = Field(..., description="Pasted text of the restaurant or canteen menu")
    preference: str = Field("balanced", description="User preference: balanced, high_protein, low_calorie, vegetarian")

class FoodComparisonRequest(BaseModel):
    food_names: List[str] = Field(..., min_items=2, description="List of 2 or more foods to compare")
    preference: str = Field("balanced", description="Preference goal: balanced, high_protein, low_calorie, vegetarian")

class FoodComparisonItem(BaseModel):
    name: str
    calories: float
    protein: float
    carbs: float
    fat: float
    is_vegetarian: bool
    score: float
    pros: List[str]
    cons: List[str]

class FoodComparisonResponse(BaseModel):
    winner: str
    preference_applied: str
    summary_reason: str
    comparison_table: List[FoodComparisonItem]
    processing_time_ms: float

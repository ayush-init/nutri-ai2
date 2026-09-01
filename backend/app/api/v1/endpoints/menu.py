from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.models.analysis import AnalysisHistory
from app.services.ocr.menu_service import menu_service
from app.schemas.menu import (
    MenuTextRequest,
    MenuRecommendationResponse,
    FoodComparisonRequest,
    FoodComparisonResponse
)

router = APIRouter()

@router.post("/recommend", response_model=MenuRecommendationResponse, summary="Analyze & Rank Menu Items by Dietary Preference")
def recommend_menu(request: MenuTextRequest, db: Session = Depends(get_db)):
    """
    Extracts individual dishes from menu text, determines macro parameters,
    and applies preference-based ranking (balanced, high_protein, low_calorie, vegetarian).
    """
    dish_names = menu_service.extract_items_from_text(request.menu_text)
    if not dish_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract any recognizable menu dishes from the provided text."
        )

    response = menu_service.rank_menu(dish_names, preference=request.preference)

    # Persist in Neon DB history
    history_entry = AnalysisHistory(
        analysis_type="menu_analysis",
        image_filename=None,
        summary_title=f"Menu Analysis ({request.preference})",
        total_calories=response.top_recommendations[0].estimated_calories if response.top_recommendations else None,
        total_protein=response.top_recommendations[0].protein_g if response.top_recommendations else None,
        total_carbs=response.top_recommendations[0].carbs_g if response.top_recommendations else None,
        total_fat=response.top_recommendations[0].fat_g if response.top_recommendations else None,
        payload=response.model_dump()
    )
    db.add(history_entry)
    db.commit()

    return response

@router.post("/compare", response_model=FoodComparisonResponse, summary="Compare Multiple Foods (What's the Better Choice?)")
def compare_foods(request: FoodComparisonRequest, db: Session = Depends(get_db)):
    """
    Compares 2 or more foods side-by-side with transparent macro trade-offs,
    pros/cons, preference scoring, and an explained winning choice.
    """
    response = menu_service.compare_foods(request.food_names, preference=request.preference)
    
    # Persist in Neon DB history
    history_entry = AnalysisHistory(
        analysis_type="food_comparison",
        image_filename=None,
        summary_title=f"Comparison: {' vs '.join(request.food_names)}",
        total_calories=None,
        total_protein=None,
        total_carbs=None,
        total_fat=None,
        payload=response.model_dump()
    )
    db.add(history_entry)
    db.commit()

    return response

from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from app.core.database import get_db
from app.models.food import FoodItem
from app.services.vision.image_service import image_service
from app.services.vision.yolo_service import yolo_service
from app.services.nutrition.nutrition_service import nutrition_service
from app.schemas.detection import FoodDetectionResponse
from app.schemas.nutrition import MealNutritionSummary

router = APIRouter()

@router.get("/items", summary="List Supported Food Classes & Base Nutrition")
def list_food_items(db: Session = Depends(get_db)):
    """
    Returns all 15 supported food classes with base nutritional values per 100g
    and default serving sizes stored in the Neon database.
    """
    items = db.query(FoodItem).order_by(FoodItem.label).all()
    return {
        "total": len(items),
        "items": [
            {
                "id": item.id,
                "label": item.label,
                "display_name": item.display_name,
                "category": item.category,
                "calories_per_100g": item.calories_per_100g,
                "protein_per_100g": item.protein_per_100g,
                "carbs_per_100g": item.carbs_per_100g,
                "fat_per_100g": item.fat_per_100g,
                "serving_name": item.default_serving_name,
                "serving_grams": item.default_serving_grams,
                "is_vegetarian": item.is_vegetarian,
                "is_vegan": item.is_vegan,
                "is_gluten_free": item.is_gluten_free
            }
            for item in items
        ]
    }

@router.post("/detect", response_model=FoodDetectionResponse, summary="Detect Foods with YOLO")
async def detect_foods(
    file: UploadFile = File(..., description="Food photo for detection"),
    conf_threshold: float = Query(0.25, ge=0.05, le=0.95, description="Confidence score threshold")
):
    """
    Detects visible food classes in an uploaded photo using fine-tuned YOLO.
    Returns bounding boxes, confidence ratings, and an annotated image link.
    """
    contents, ext = await image_service.validate_and_read(file)
    img = image_service.decode_image(contents)
    processed_img = image_service.resize_preserving_aspect_ratio(img, max_dim=1280)

    detections, annotated_img, latency_ms = yolo_service.detect(processed_img, conf_threshold=conf_threshold)
    annotated_url = yolo_service.save_annotated(annotated_img)

    has_low_conf = any(d.confidence < 0.40 for d in detections)

    return FoodDetectionResponse(
        total_detections=len(detections),
        detections=detections,
        annotated_image_url=annotated_url,
        has_low_confidence=has_low_conf,
        processing_time_ms=latency_ms
    )

@router.post("/analyze", response_model=MealNutritionSummary, summary="Analyze Meal (YOLO Detection + Nutrition Engine)")
async def analyze_food_meal(
    file: UploadFile = File(..., description="Food photo for full nutritional analysis"),
    conf_threshold: float = Query(0.25, ge=0.05, le=0.95, description="Confidence score threshold"),
    db: Session = Depends(get_db)
):
    """
    End-to-End Food Analysis Pipeline:
    1. Validates and preprocesses image via OpenCV.
    2. Runs YOLO object detection to identify food items and bounding boxes.
    3. Queries Neon PostgreSQL for nutritional parameters.
    4. Computes calorie and macronutrient estimation ranges with portion multipliers.
    """
    contents, ext = await image_service.validate_and_read(file)
    img = image_service.decode_image(contents)
    processed_img = image_service.resize_preserving_aspect_ratio(img, max_dim=1280)

    detections, annotated_img, latency_ms = yolo_service.detect(processed_img, conf_threshold=conf_threshold)
    annotated_url = yolo_service.save_annotated(annotated_img)

    summary = nutrition_service.calculate_meal_nutrition(
        db=db,
        detections=detections,
        annotated_url=annotated_url,
        latency_ms=latency_ms
    )

    return summary

@router.get("/annotated-image/{filename}", summary="Stream Annotated Image")
def get_annotated_image(filename: str):
    filepath = os.path.join(yolo_service.annotated_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotated image not found")
    return FileResponse(filepath, media_type="image/jpeg")

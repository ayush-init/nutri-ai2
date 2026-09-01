from sqlalchemy.orm import Session
from app.models.food import FoodItem
from app.schemas.nutrition import FoodItemNutrition, MealNutritionSummary, CalorieRange, MacroRange
from app.schemas.detection import DetectedFoodItem
from typing import List, Dict, Any

# Uncertainty buffer (?15% for photo-based portion variance)
VARIANCE_FACTOR = 0.15

class NutritionEngineService:
    """
    Calculates estimated nutritional ranges for detected food classes
    using nutritional values from the Neon PostgreSQL database.
    """

    @staticmethod
    def get_or_fallback_food(db: Session, label: str) -> Dict[str, Any]:
        item = db.query(FoodItem).filter(FoodItem.label == label).first()
        if item:
            return {
                "label": item.label,
                "display_name": item.display_name,
                "category": item.category,
                "cal_100g": item.calories_per_100g,
                "prot_100g": item.protein_per_100g,
                "carbs_100g": item.carbs_per_100g,
                "fat_100g": item.fat_per_100g,
                "fiber_100g": item.fiber_per_100g,
                "serving_name": item.default_serving_name,
                "serving_grams": item.default_serving_grams,
                "is_veg": item.is_vegetarian,
                "is_vegan": item.is_vegan,
                "is_gf": item.is_gluten_free
            }
        
        # Fallback default values
        return {
            "label": label,
            "display_name": label.replace("_", " ").title(),
            "category": "general_food",
            "cal_100g": 120.0,
            "prot_100g": 4.0,
            "carbs_100g": 18.0,
            "fat_100g": 3.0,
            "fiber_100g": 2.0,
            "serving_name": "1 standard portion",
            "serving_grams": 100.0,
            "is_veg": True,
            "is_vegan": False,
            "is_gf": True
        }

    @classmethod
    def calculate_meal_nutrition(
        cls,
        db: Session,
        detections: List[DetectedFoodItem],
        annotated_url: str = None,
        latency_ms: float = 0.0
    ) -> MealNutritionSummary:
        
        item_nutritions: List[FoodItemNutrition] = []
        tot_min_cal = 0.0
        tot_max_cal = 0.0
        tot_avg_cal = 0.0

        tot_min_prot = 0.0
        tot_max_prot = 0.0
        tot_avg_prot = 0.0

        tot_min_carb = 0.0
        tot_max_carb = 0.0
        tot_avg_carb = 0.0

        tot_min_fat = 0.0
        tot_max_fat = 0.0
        tot_avg_fat = 0.0

        has_uncertain = False

        # Group by label to avoid duplicate row spam for multiple detections of same food
        label_groups: Dict[str, List[DetectedFoodItem]] = {}
        for det in detections:
            label_groups.setdefault(det.label, []).append(det)

        for label, group in label_groups.items():
            best_det = max(group, key=lambda d: d.confidence)
            count = float(len(group))
            food_meta = cls.get_or_fallback_food(db, label)

            multiplier = (food_meta["serving_grams"] / 100.0) * count

            avg_cal = food_meta["cal_100g"] * multiplier
            min_cal = round(avg_cal * (1.0 - VARIANCE_FACTOR), 1)
            max_cal = round(avg_cal * (1.0 + VARIANCE_FACTOR), 1)
            avg_cal = round(avg_cal, 1)

            avg_prot = food_meta["prot_100g"] * multiplier
            min_prot = round(avg_prot * (1.0 - VARIANCE_FACTOR), 1)
            max_prot = round(avg_prot * (1.0 + VARIANCE_FACTOR), 1)
            avg_prot = round(avg_prot, 1)

            avg_carb = food_meta["carbs_100g"] * multiplier
            min_carb = round(avg_carb * (1.0 - VARIANCE_FACTOR), 1)
            max_carb = round(avg_carb * (1.0 + VARIANCE_FACTOR), 1)
            avg_carb = round(avg_carb, 1)

            avg_fat = food_meta["fat_100g"] * multiplier
            min_fat = round(avg_fat * (1.0 - VARIANCE_FACTOR), 1)
            max_fat = round(avg_fat * (1.0 + VARIANCE_FACTOR), 1)
            avg_fat = round(avg_fat, 1)

            tot_min_cal += min_cal
            tot_max_cal += max_cal
            tot_avg_cal += avg_cal

            tot_min_prot += min_prot
            tot_max_prot += max_prot
            tot_avg_prot += avg_prot

            tot_min_carb += min_carb
            tot_max_carb += max_carb
            tot_avg_carb += avg_carb

            tot_min_fat += min_fat
            tot_max_fat += max_fat
            tot_avg_fat += avg_fat

            if best_det.confidence < 0.40:
                has_uncertain = True

            item_nutritions.append(
                FoodItemNutrition(
                    label=label,
                    display_name=food_meta["display_name"],
                    category=food_meta["category"],
                    serving_name=food_meta["serving_name"],
                    serving_grams=food_meta["serving_grams"],
                    serving_count=count,
                    calories=CalorieRange(min_kcal=min_cal, max_kcal=max_cal, avg_kcal=avg_cal),
                    protein=MacroRange(min_val=min_prot, max_val=max_prot, avg_val=avg_prot),
                    carbs=MacroRange(min_val=min_carb, max_val=max_carb, avg_val=avg_carb),
                    fat=MacroRange(min_val=min_fat, max_val=max_fat, avg_val=avg_fat),
                    is_vegetarian=food_meta["is_veg"],
                    is_vegan=food_meta["is_vegan"],
                    is_gluten_free=food_meta["is_gf"],
                    confidence=best_det.confidence,
                    confidence_level=best_det.confidence_level
                )
            )

        return MealNutritionSummary(
            total_calories=CalorieRange(min_kcal=round(tot_min_cal, 1), max_kcal=round(tot_max_cal, 1), avg_kcal=round(tot_avg_cal, 1)),
            total_protein=MacroRange(min_val=round(tot_min_prot, 1), max_val=round(tot_max_prot, 1), avg_val=round(tot_avg_prot, 1)),
            total_carbs=MacroRange(min_val=round(tot_min_carb, 1), max_val=round(tot_max_carb, 1), avg_val=round(tot_avg_carb, 1)),
            total_fat=MacroRange(min_val=round(tot_min_fat, 1), max_val=round(tot_max_fat, 1), avg_val=round(tot_avg_fat, 1)),
            detected_items=item_nutritions,
            has_uncertain_items=has_uncertain,
            annotated_image_url=annotated_url,
            processing_time_ms=latency_ms
        )

nutrition_service = NutritionEngineService()

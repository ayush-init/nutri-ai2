from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.models.food import FoodItem
from app.schemas.nutrition import FoodItemNutrition, MealNutritionSummary, CalorieRange, MacroRange
from app.schemas.detection import DetectedFoodItem
from typing import List, Dict, Any

# Uncertainty buffer (±15% for photo-based portion variance)
VARIANCE_FACTOR = 0.15

class NutritionEngineService:
    """
    Calculates estimated nutritional ranges for detected food classes
    using nutritional values from the Neon PostgreSQL database.
    """

    @staticmethod
    def get_or_fallback_food(db: Session, label: str) -> Dict[str, Any]:
        clean_label = label.strip().lower()
        variants = [
            clean_label,
            clean_label.replace(" ", "_"),
            clean_label.replace("_", " "),
            clean_label.split("_")[-1],  # e.g. 'rice' from 'steamed_rice'
            clean_label.split("_")[0]   # e.g. 'chicken' from 'chicken_curry'
        ]

        item = None
        for v in variants:
            item = db.query(FoodItem).filter(
                or_(
                    FoodItem.label == v,
                    FoodItem.label == v.replace(" ", "_"),
                    FoodItem.label.ilike(f"%{v}%"),
                    FoodItem.display_name.ilike(f"%{v}%")
                )
            ).first()
            if item:
                break

        if item:
            return {
                "label": item.label,
                "display_name": item.display_name,
                "category": item.category,
                "cal_100g": item.calories_per_100g,
                "prot_100g": item.protein_per_100g,
                "carbs_100g": item.carbs_per_100g,
                "fat_100g": item.fat_per_100g,
                "fiber_100g": item.fiber_per_100g or 0.0,
                "serving_name": item.default_serving_name or "1 standard portion",
                "serving_grams": item.default_serving_grams or 100.0,
                "is_veg": item.is_vegetarian if item.is_vegetarian is not None else True,
                "is_vegan": item.is_vegan if item.is_vegan is not None else False,
                "is_gf": item.is_gluten_free if item.is_gluten_free is not None else True
            }
        
        # Fallback default values
        return {
            "label": label,
            "display_name": label.replace("_", " ").title(),
            "category": "general_food",
            "cal_100g": 135.0,
            "prot_100g": 5.0,
            "carbs_100g": 20.0,
            "fat_100g": 4.0,
            "fiber_100g": 2.0,
            "serving_name": "1 standard portion",
            "serving_grams": 120.0,
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

        for det in detections:
            if det.confidence < 0.40:
                has_uncertain = True

            finfo = cls.get_or_fallback_food(db, det.label)

            # Heuristic portion sizing based on bounding box relative area
            box_area = (det.bbox.x_max - det.bbox.x_min) * (det.bbox.y_max - det.bbox.y_min)
            if box_area > 300000:
                portion_mult = 1.3
            elif box_area < 40000:
                portion_mult = 0.6
            else:
                portion_mult = 1.0

            est_grams = round(finfo["serving_grams"] * portion_mult, 1)
            scale = est_grams / 100.0

            # Base macro calculations for this detected item
            avg_cal = round(finfo["cal_100g"] * scale, 1)
            min_cal = round(avg_cal * (1.0 - VARIANCE_FACTOR), 1)
            max_cal = round(avg_cal * (1.0 + VARIANCE_FACTOR), 1)

            avg_p = round(finfo["prot_100g"] * scale, 1)
            min_p = round(avg_p * (1.0 - VARIANCE_FACTOR), 1)
            max_p = round(avg_p * (1.0 + VARIANCE_FACTOR), 1)

            avg_c = round(finfo["carbs_100g"] * scale, 1)
            min_c = round(avg_c * (1.0 - VARIANCE_FACTOR), 1)
            max_c = round(avg_c * (1.0 + VARIANCE_FACTOR), 1)

            avg_f = round(finfo["fat_100g"] * scale, 1)
            min_f = round(avg_f * (1.0 - VARIANCE_FACTOR), 1)
            max_f = round(avg_f * (1.0 + VARIANCE_FACTOR), 1)

            avg_fib = round(finfo["fiber_100g"] * scale, 1)
            min_fib = round(avg_fib * (1.0 - VARIANCE_FACTOR), 1)
            max_fib = round(avg_fib * (1.0 + VARIANCE_FACTOR), 1)

            # Accumulate totals
            tot_min_cal += min_cal
            tot_max_cal += max_cal
            tot_avg_cal += avg_cal

            tot_min_prot += min_p
            tot_max_prot += max_p
            tot_avg_prot += avg_p

            tot_min_carb += min_c
            tot_max_carb += max_c
            tot_avg_carb += avg_c

            tot_min_fat += min_f
            tot_max_fat += max_f
            tot_avg_fat += avg_f

            item_nutritions.append(FoodItemNutrition(
                label=det.label,
                display_name=finfo["display_name"],
                category=finfo["category"],
                serving_name=finfo["serving_name"],
                serving_grams=est_grams,
                serving_count=portion_mult,
                calories=CalorieRange(min_kcal=min_cal, max_kcal=max_cal, avg_kcal=avg_cal),
                protein=MacroRange(min_val=min_p, max_val=max_p, avg_val=avg_p),
                carbs=MacroRange(min_val=min_c, max_val=max_c, avg_val=avg_c),
                fat=MacroRange(min_val=min_f, max_val=max_f, avg_val=avg_f),
                fiber=MacroRange(min_val=min_fib, max_val=max_fib, avg_val=avg_fib),
                is_vegetarian=finfo["is_veg"],
                is_vegan=finfo["is_vegan"],
                is_gluten_free=finfo["is_gf"],
                confidence=det.confidence,
                confidence_level=det.confidence_level
            ))

        return MealNutritionSummary(
            total_calories=CalorieRange(
                min_kcal=round(tot_min_cal, 1),
                max_kcal=round(tot_max_cal, 1),
                avg_kcal=round(tot_avg_cal, 1)
            ),
            total_protein=MacroRange(
                min_val=round(tot_min_prot, 1),
                max_val=round(tot_max_prot, 1),
                avg_val=round(tot_avg_prot, 1)
            ),
            total_carbs=MacroRange(
                min_val=round(tot_min_carb, 1),
                max_val=round(tot_max_carb, 1),
                avg_val=round(tot_avg_carb, 1)
            ),
            total_fat=MacroRange(
                min_val=round(tot_min_fat, 1),
                max_val=round(tot_max_fat, 1),
                avg_val=round(tot_avg_fat, 1)
            ),
            detected_items=item_nutritions,
            disclaimer="Estimates are derived from visual multi-dish detection and standard portion assumptions. Not intended as medical or laboratory measurements.",
            has_uncertain_items=has_uncertain,
            annotated_image_url=annotated_url,
            processing_time_ms=latency_ms
        )

nutrition_service = NutritionEngineService()

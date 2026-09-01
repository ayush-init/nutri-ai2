import re
import time
from typing import List, Dict, Any, Tuple
from app.schemas.menu import MenuItem, MenuRecommendationResponse, FoodComparisonItem, FoodComparisonResponse

# Known culinary dish dictionary for accurate menu intelligence
DISH_KNOWLEDGE_BASE = {
    "paneer butter masala": {"cal": 380.0, "prot": 14.0, "carbs": 12.0, "fat": 30.0, "veg": True, "cat": "curry"},
    "dal tadka": {"cal": 180.0, "prot": 9.0, "carbs": 26.0, "fat": 4.5, "veg": True, "cat": "lentil"},
    "dal makhani": {"cal": 280.0, "prot": 10.0, "carbs": 24.0, "fat": 16.0, "veg": True, "cat": "lentil"},
    "jeera rice": {"cal": 190.0, "prot": 3.5, "carbs": 38.0, "fat": 2.5, "veg": True, "cat": "rice"},
    "steamed rice": {"cal": 130.0, "prot": 2.7, "carbs": 28.0, "fat": 0.3, "veg": True, "cat": "rice"},
    "roti": {"cal": 90.0, "prot": 3.0, "carbs": 18.0, "fat": 0.8, "veg": True, "cat": "bread"},
    "butter naan": {"cal": 260.0, "prot": 6.0, "carbs": 38.0, "fat": 9.0, "veg": True, "cat": "bread"},
    "tandoori chicken": {"cal": 260.0, "prot": 32.0, "carbs": 3.0, "fat": 12.0, "veg": False, "cat": "meat"},
    "chicken biryani": {"cal": 450.0, "prot": 25.0, "carbs": 52.0, "fat": 15.0, "veg": False, "cat": "rice"},
    "veg biryani": {"cal": 320.0, "prot": 7.0, "carbs": 50.0, "fat": 10.0, "veg": True, "cat": "rice"},
    "grilled chicken salad": {"cal": 240.0, "prot": 30.0, "carbs": 8.0, "fat": 8.0, "veg": False, "cat": "salad"},
    "caesar salad": {"cal": 310.0, "prot": 8.0, "carbs": 12.0, "fat": 26.0, "veg": True, "cat": "salad"},
    "pizza slice": {"cal": 285.0, "prot": 12.0, "carbs": 36.0, "fat": 10.0, "veg": True, "cat": "fast_food"},
    "cheeseburger": {"cal": 480.0, "prot": 24.0, "carbs": 40.0, "fat": 24.0, "veg": False, "cat": "fast_food"},
    "rajma chawal": {"cal": 340.0, "prot": 13.0, "carbs": 58.0, "fat": 5.0, "veg": True, "cat": "combo"},
    "chole bhature": {"cal": 550.0, "prot": 12.0, "carbs": 68.0, "fat": 26.0, "veg": True, "cat": "curry"}
}

class MenuIntelligenceService:
    """
    Extracts, standardizes, evaluates, ranks, and compares dishes based on dietary preferences.
    """

    @staticmethod
    def extract_items_from_text(raw_text: str) -> List[str]:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        extracted = []
        for line in lines:
            # Strip prices (e.g. $12.99, Rs. 250, 180/-)
            cleaned = re.sub(r"(?:rs\.?|\$|inr|usd)?\s*\d+(?:\.\d+)?(?:/-)?", "", line, flags=re.IGNORECASE)
            # Remove numbering like 1. 2)
            cleaned = re.sub(r"^\d+[\.\)\-:]\s*", "", cleaned)
            cleaned = cleaned.strip(" -?*,;")
            if len(cleaned) >= 3 and not any(kw in cleaned.lower() for kw in ["menu", "appetizer", "beverages", "desserts", "main course", "today's special"]):
                extracted.append(cleaned)
        return extracted

    @staticmethod
    def match_dish_nutrition(dish_name: str) -> Dict[str, Any]:
        lower = dish_name.lower()
        for known_name, meta in DISH_KNOWLEDGE_BASE.items():
            if known_name in lower or lower in known_name:
                return meta
        # Generic heuristic if dish unknown
        return {"cal": 280.0, "prot": 8.0, "carbs": 35.0, "fat": 10.0, "veg": True, "cat": "general"}

    @classmethod
    def score_item(cls, meta: Dict[str, Any], preference: str) -> Tuple[float, str]:
        cal = meta["cal"]
        prot = meta["prot"]
        fat = meta["fat"]
        is_veg = meta["veg"]

        pref = preference.lower()
        if pref == "high_protein":
            # Maximize protein per 100 calories
            prot_density = (prot * 4.0) / max(cal, 1.0)
            score = round(prot * 2.0 + (prot_density * 50.0) - (fat * 0.5), 1)
            reason = f"High protein content ({prot}g) delivering strong protein density."
        elif pref == "low_calorie":
            score = round(1000.0 / max(cal, 50.0) + (prot * 1.5) - (fat * 2.0), 1)
            reason = f"Low calorie density ({cal} kcal) with minimal added fats."
        elif pref == "vegetarian":
            if not is_veg:
                score = -100.0
                reason = "Contains non-vegetarian ingredients."
            else:
                score = round((prot * 2.0) + 50.0 - (fat * 0.5), 1)
                reason = f"100% vegetarian option with balanced {prot}g plant protein."
        else: # "balanced"
            # Balanced protein-to-carb-to-fat ratio
            prot_ratio = (prot * 4.0) / max(cal, 1.0)
            fat_ratio = (fat * 9.0) / max(cal, 1.0)
            balance_penalty = abs(prot_ratio - 0.25) * 50.0 + abs(fat_ratio - 0.25) * 50.0
            score = round(100.0 - balance_penalty, 1)
            reason = f"Well-balanced macronutrient profile ({prot}g protein, {meta['carbs']}g carbs, {fat}g fat)."

        return score, reason

    @classmethod
    def rank_menu(cls, dish_names: List[str], preference: str = "balanced") -> MenuRecommendationResponse:
        start_time = time.time()
        scored_items: List[MenuItem] = []

        for name in dish_names:
            meta = cls.match_dish_nutrition(name)
            score, reason = cls.score_item(meta, preference)
            scored_items.append(
                MenuItem(
                    name=name,
                    estimated_calories=meta["cal"],
                    protein_g=meta["prot"],
                    carbs_g=meta["carbs"],
                    fat_g=meta["fat"],
                    is_vegetarian=meta["veg"],
                    category=meta["cat"],
                    score=score,
                    recommendation_reason=reason
                )
            )

        # Sort descending by score
        scored_items.sort(key=lambda x: x.score, reverse=True)
        top = [item for item in scored_items if item.score > 0][:3]
        latency = round((time.time() - start_time) * 1000, 2)

        explanation = f"Evaluated {len(scored_items)} items against '{preference}' goal. Recommendations prioritize optimal calorie-to-protein efficiency and dietary compliance."

        return MenuRecommendationResponse(
            preference_applied=preference,
            top_recommendations=top,
            all_items=scored_items,
            explanation=explanation,
            total_items_analyzed=len(scored_items),
            processing_time_ms=latency
        )

    @classmethod
    def compare_foods(cls, food_names: List[str], preference: str = "balanced") -> FoodComparisonResponse:
        start_time = time.time()
        items: List[FoodComparisonItem] = []

        for name in food_names:
            meta = cls.match_dish_nutrition(name)
            score, _ = cls.score_item(meta, preference)
            
            pros = []
            cons = []
            if meta["prot"] >= 15.0:
                pros.append(f"High in protein ({meta['prot']}g)")
            elif meta["prot"] < 5.0:
                cons.append(f"Low protein content ({meta['prot']}g)")

            if meta["cal"] <= 250.0:
                pros.append(f"Low calorie footprint ({meta['cal']} kcal)")
            elif meta["cal"] >= 450.0:
                cons.append(f"Calorie-dense ({meta['cal']} kcal)")

            if meta["fat"] <= 6.0:
                pros.append("Low dietary fat")
            elif meta["fat"] >= 18.0:
                cons.append(f"Elevated fat content ({meta['fat']}g)")

            if not pros:
                pros.append("Standard everyday portion")
            if not cons:
                cons.append("Moderate overall balance")

            items.append(
                FoodComparisonItem(
                    name=name,
                    calories=meta["cal"],
                    protein=meta["prot"],
                    carbs=meta["carbs"],
                    fat=meta["fat"],
                    is_vegetarian=meta["veg"],
                    score=score,
                    pros=pros,
                    cons=cons
                )
            )

        items.sort(key=lambda x: x.score, reverse=True)
        winner = items[0].name
        latency = round((time.time() - start_time) * 1000, 2)

        summary = f"'{winner}' ranks highest for '{preference}' preference because of superior macronutrient efficiency and lower caloric overhead."

        return FoodComparisonResponse(
            winner=winner,
            preference_applied=preference,
            summary_reason=summary,
            comparison_table=items,
            processing_time_ms=latency
        )

menu_service = MenuIntelligenceService()

"""
FoodLens AI - Seed All 103 Food Classes into Neon PostgreSQL
"""

import sys
import os
import yaml
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.core.database import SessionLocal
from app.models.food import FoodItem

FOOD_103_DEFAULTS = {
    "apple": {"cal": 52.0, "p": 0.3, "c": 13.8, "f": 0.2, "fib": 2.4, "serving": "1 medium apple", "g": 182, "veg": True, "vegan": True, "gf": True, "cat": "fruits"},
    "banana": {"cal": 89.0, "p": 1.1, "c": 22.8, "f": 0.3, "fib": 2.6, "serving": "1 medium banana", "g": 118, "veg": True, "vegan": True, "gf": True, "cat": "fruits"},
    "bread": {"cal": 265.0, "p": 9.0, "c": 49.0, "f": 3.2, "fib": 2.7, "serving": "1 slice", "g": 35, "veg": True, "vegan": True, "gf": False, "cat": "grains"},
    "broccoli": {"cal": 34.0, "p": 2.8, "c": 6.6, "f": 0.4, "fib": 2.6, "serving": "1 cup florets", "g": 90, "veg": True, "vegan": True, "gf": True, "cat": "vegetables"},
    "cake": {"cal": 371.0, "p": 5.0, "c": 53.0, "f": 15.0, "fib": 1.0, "serving": "1 slice", "g": 80, "veg": True, "vegan": False, "gf": False, "cat": "desserts"},
    "carrot": {"cal": 41.0, "p": 0.9, "c": 9.6, "f": 0.2, "fib": 2.8, "serving": "1 medium carrot", "g": 61, "veg": True, "vegan": True, "gf": True, "cat": "vegetables"},
    "chicken_duck": {"cal": 239.0, "p": 27.0, "c": 0.0, "f": 14.0, "fib": 0.0, "serving": "1 breast portion", "g": 120, "veg": False, "vegan": False, "gf": True, "cat": "meat"},
    "cucumber": {"cal": 15.0, "p": 0.7, "c": 3.6, "f": 0.1, "fib": 0.5, "serving": "1/2 cup sliced", "g": 52, "veg": True, "vegan": True, "gf": True, "cat": "vegetables"},
    "egg": {"cal": 155.0, "p": 13.0, "c": 1.1, "f": 11.0, "fib": 0.0, "serving": "1 large egg", "g": 50, "veg": True, "vegan": False, "gf": True, "cat": "dairy_eggs"},
    "french_fries": {"cal": 312.0, "p": 3.4, "c": 41.0, "f": 15.0, "fib": 3.8, "serving": "1 medium order", "g": 117, "veg": True, "vegan": True, "gf": True, "cat": "fast_food"},
    "ice_cream": {"cal": 207.0, "p": 3.5, "c": 24.0, "f": 11.0, "fib": 0.7, "serving": "1 scoop", "g": 72, "veg": True, "vegan": False, "gf": True, "cat": "desserts"},
    "lettuce": {"cal": 15.0, "p": 1.4, "c": 2.9, "f": 0.2, "fib": 1.3, "serving": "1 cup shredded", "g": 47, "veg": True, "vegan": True, "gf": True, "cat": "vegetables"},
    "noodles": {"cal": 138.0, "p": 4.5, "c": 25.0, "f": 2.1, "fib": 1.2, "serving": "1 bowl", "g": 160, "veg": True, "vegan": True, "gf": False, "cat": "grains"},
    "onion": {"cal": 40.0, "p": 1.1, "c": 9.3, "f": 0.1, "fib": 1.7, "serving": "1 medium onion", "g": 110, "veg": True, "vegan": True, "gf": True, "cat": "vegetables"},
    "pepper": {"cal": 31.0, "p": 1.0, "c": 6.0, "f": 0.3, "fib": 2.1, "serving": "1 medium pepper", "g": 119, "veg": True, "vegan": True, "gf": True, "cat": "vegetables"},
    "pie": {"cal": 290.0, "p": 3.0, "c": 40.0, "f": 13.0, "fib": 1.5, "serving": "1 slice", "g": 125, "veg": True, "vegan": False, "gf": False, "cat": "desserts"},
    "potato": {"cal": 87.0, "p": 1.9, "c": 20.0, "f": 0.1, "fib": 1.8, "serving": "1 medium potato", "g": 150, "veg": True, "vegan": True, "gf": True, "cat": "vegetables"},
    "rice": {"cal": 130.0, "p": 2.7, "c": 28.0, "f": 0.3, "fib": 0.4, "serving": "1 cup cooked", "g": 150, "veg": True, "vegan": True, "gf": True, "cat": "grains"},
    "steak": {"cal": 271.0, "p": 26.0, "c": 0.0, "f": 19.0, "fib": 0.0, "serving": "1 cooked steak", "g": 150, "veg": False, "vegan": False, "gf": True, "cat": "meat"},
    "strawberry": {"cal": 32.0, "p": 0.7, "c": 7.7, "f": 0.3, "fib": 2.0, "serving": "1 cup whole", "g": 144, "veg": True, "vegan": True, "gf": True, "cat": "fruits"},
    "tomato": {"cal": 18.0, "p": 0.9, "c": 3.9, "f": 0.2, "fib": 1.2, "serving": "1 medium tomato", "g": 123, "veg": True, "vegan": True, "gf": True, "cat": "vegetables"},
}

def seed():
    db = SessionLocal()
    print("Seeding Food 103 classes into Neon DB...")
    
    yaml_path = Path("datasets/food_yolo_103/data.yaml")
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            class_names = list(data.get("names", {}).values())
    else:
        class_names = list(FOOD_103_DEFAULTS.keys())

    count_added = 0
    count_updated = 0

    for name in class_names:
        label = name.strip().lower().replace(" ", "_")
        disp_name = name.strip().replace("_", " ").title()
        
        info = FOOD_103_DEFAULTS.get(label, {
            "cal": 120.0, "p": 4.0, "c": 18.0, "f": 3.0, "fib": 2.0,
            "serving": "1 standard portion", "g": 100, "veg": True, "vegan": False, "gf": True, "cat": "general_food"
        })

        existing = db.query(FoodItem).filter(FoodItem.label == label).first()
        if existing:
            count_updated += 1
        else:
            item = FoodItem(
                label=label,
                display_name=disp_name,
                category=info["cat"],
                calories_per_100g=info["cal"],
                protein_per_100g=info["p"],
                carbs_per_100g=info["c"],
                fat_per_100g=info["f"],
                fiber_per_100g=info["fib"],
                default_serving_name=info["serving"],
                default_serving_grams=info["g"],
                is_vegetarian=info["veg"],
                is_vegan=info["vegan"],
                is_gluten_free=info["gf"]
            )
            db.add(item)
            count_added += 1

    db.commit()
    db.close()
    print(f"✓ DB Seeding complete! Added: {count_added}, Existing: {count_updated}")

if __name__ == "__main__":
    seed()

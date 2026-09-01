from app.core.database import engine, Base, SessionLocal
from app.models.food import FoodItem

SEED_DATA = [
    {"label": "bread", "display_name": "Bread", "category": "grain", "cal_100g": 265.0, "prot_100g": 9.0, "carbs_100g": 49.0, "fat_100g": 3.2, "serving_name": "1 slice", "serving_grams": 35.0, "is_veg": True, "is_vegan": True, "is_gf": False},
    {"label": "tomato", "display_name": "Tomato", "category": "vegetable", "cal_100g": 18.0, "prot_100g": 0.9, "carbs_100g": 3.9, "fat_100g": 0.2, "serving_name": "1 medium tomato", "serving_grams": 120.0, "is_veg": True, "is_vegan": True, "is_gf": True},
    {"label": "carrot", "display_name": "Carrot", "category": "vegetable", "cal_100g": 41.0, "prot_100g": 0.9, "carbs_100g": 9.6, "fat_100g": 0.2, "serving_name": "1 medium carrot", "serving_grams": 60.0, "is_veg": True, "is_vegan": True, "is_gf": True},
    {"label": "chicken_duck", "display_name": "Cooked Poultry / Chicken", "category": "meat", "cal_100g": 239.0, "prot_100g": 27.0, "carbs_100g": 0.0, "fat_100g": 14.0, "serving_name": "1 piece / breast", "serving_grams": 150.0, "is_veg": False, "is_vegan": False, "is_gf": True},
    {"label": "steak", "display_name": "Beef Steak / Red Meat", "category": "meat", "cal_100g": 271.0, "prot_100g": 26.0, "carbs_100g": 0.0, "fat_100g": 19.0, "serving_name": "1 palm-size steak", "serving_grams": 200.0, "is_veg": False, "is_vegan": False, "is_gf": True},
    {"label": "potato", "display_name": "Boiled / Baked Potato", "category": "vegetable", "cal_100g": 87.0, "prot_100g": 1.9, "carbs_100g": 20.1, "fat_100g": 0.1, "serving_name": "1 medium potato", "serving_grams": 150.0, "is_veg": True, "is_vegan": True, "is_gf": True},
    {"label": "broccoli", "display_name": "Broccoli", "category": "vegetable", "cal_100g": 34.0, "prot_100g": 2.8, "carbs_100g": 6.6, "fat_100g": 0.4, "serving_name": "1 cup florets", "serving_grams": 90.0, "is_veg": True, "is_vegan": True, "is_gf": True},
    {"label": "ice_cream", "display_name": "Ice Cream", "category": "dessert", "cal_100g": 207.0, "prot_100g": 3.5, "carbs_100g": 24.0, "fat_100g": 11.0, "serving_name": "1 scoop", "serving_grams": 75.0, "is_veg": True, "is_vegan": False, "is_gf": True},
    {"label": "strawberry", "display_name": "Strawberry", "category": "fruit", "cal_100g": 32.0, "prot_100g": 0.7, "carbs_100g": 7.7, "fat_100g": 0.3, "serving_name": "1 cup berries", "serving_grams": 100.0, "is_veg": True, "is_vegan": True, "is_gf": True},
    {"label": "lettuce", "display_name": "Lettuce / Salad Greens", "category": "vegetable", "cal_100g": 15.0, "prot_100g": 1.4, "carbs_100g": 2.9, "fat_100g": 0.2, "serving_name": "1 bowl salad", "serving_grams": 60.0, "is_veg": True, "is_vegan": True, "is_gf": True},
    {"label": "onion", "display_name": "Onion", "category": "vegetable", "cal_100g": 40.0, "prot_100g": 1.1, "carbs_100g": 9.3, "fat_100g": 0.1, "serving_name": "1 medium onion", "serving_grams": 80.0, "is_veg": True, "is_vegan": True, "is_gf": True},
    {"label": "rice", "display_name": "Cooked Rice", "category": "grain", "cal_100g": 130.0, "prot_100g": 2.7, "carbs_100g": 28.2, "fat_100g": 0.3, "serving_name": "1 standard bowl", "serving_grams": 150.0, "is_veg": True, "is_vegan": True, "is_gf": True},
    {"label": "cucumber", "display_name": "Cucumber", "category": "vegetable", "cal_100g": 15.0, "prot_100g": 0.7, "carbs_100g": 3.6, "fat_100g": 0.1, "serving_name": "1 whole cucumber", "serving_grams": 120.0, "is_veg": True, "is_vegan": True, "is_gf": True},
    {"label": "pepper", "display_name": "Bell Pepper", "category": "vegetable", "cal_100g": 31.0, "prot_100g": 1.0, "carbs_100g": 6.0, "fat_100g": 0.3, "serving_name": "1 medium pepper", "serving_grams": 120.0, "is_veg": True, "is_vegan": True, "is_gf": True},
    {"label": "pie", "display_name": "Pastry / Pie", "category": "dessert", "cal_100g": 290.0, "prot_100g": 3.0, "carbs_100g": 40.0, "fat_100g": 14.0, "serving_name": "1 slice", "serving_grams": 120.0, "is_veg": True, "is_vegan": False, "is_gf": False}
]

def init_db_and_seed():
    print("Creating tables in Neon PostgreSQL...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        for data in SEED_DATA:
            existing = db.query(FoodItem).filter(FoodItem.label == data["label"]).first()
            if not existing:
                item = FoodItem(
                    label=data["label"],
                    display_name=data["display_name"],
                    category=data["category"],
                    calories_per_100g=data["cal_100g"],
                    protein_per_100g=data["prot_100g"],
                    carbs_per_100g=data["carbs_100g"],
                    fat_per_100g=data["fat_100g"],
                    default_serving_name=data["serving_name"],
                    default_serving_grams=data["serving_grams"],
                    is_vegetarian=data["is_veg"],
                    is_vegan=data["is_vegan"],
                    is_gluten_free=data["is_gf"]
                )
                db.add(item)
        db.commit()
        print("Database seeded with all 15 food classes successfully!")
    finally:
        db.close()

if __name__ == "__main__":
    init_db_and_seed()

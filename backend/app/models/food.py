from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class FoodItem(Base):
    __tablename__ = "food_items"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String(64), unique=True, index=True, nullable=False) # Maps to YOLO class name
    display_name = Column(String(128), nullable=False)
    category = Column(String(64), nullable=False) # e.g. vegetable, grain, meat, dessert, fruit
    
    # Nutritional values per 100 grams
    calories_per_100g = Column(Float, nullable=False)
    protein_per_100g = Column(Float, nullable=False)
    carbs_per_100g = Column(Float, nullable=False)
    fat_per_100g = Column(Float, nullable=False)
    fiber_per_100g = Column(Float, default=0.0)

    # Standard serving assumptions
    default_serving_name = Column(String(64), default="1 medium serving")
    default_serving_grams = Column(Float, default=100.0)
    
    # Dietary flags
    is_vegetarian = Column(Boolean, default=True)
    is_vegan = Column(Boolean, default=True)
    is_gluten_free = Column(Boolean, default=True)

    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

from sqlalchemy import Column, Integer, String, Float, Text, JSON, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(String(32), nullable=False) # 'food_photo', 'package_ocr', 'menu_analysis'
    image_filename = Column(String(255), nullable=True)
    summary_title = Column(String(255), nullable=False)
    
    # Quantitative summary
    total_calories = Column(Float, nullable=True)
    total_protein = Column(Float, nullable=True)
    total_carbs = Column(Float, nullable=True)
    total_fat = Column(Float, nullable=True)
    
    # Detailed JSON payload
    payload = Column(JSON, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ExtractedNutrient(BaseModel):
    name: str
    value: Optional[float] = None
    unit: str = "g"
    confidence: float = 1.0
    is_detected: bool = True

class PackageOCRResult(BaseModel):
    product_name: Optional[str] = "Packaged Food Product"
    serving_size: Optional[str] = None
    calories: ExtractedNutrient
    protein: ExtractedNutrient
    carbohydrates: ExtractedNutrient
    fat: ExtractedNutrient
    sugar: Optional[ExtractedNutrient] = None
    sodium: Optional[ExtractedNutrient] = None
    ingredients: List[str] = []
    allergens: List[str] = []
    raw_ocr_text: str
    review_required: bool = False
    review_reasons: List[str] = []
    source: str = "Extracted from package label via OCR"
    processing_time_ms: float

import re
import time
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from app.schemas.package import PackageOCRResult, ExtractedNutrient

logger = logging.getLogger(__name__)

class PackageOCRService:
    """
    Modular OCR Service for reading nutrition fact panels, ingredients,
    and allergen declarations from packaged food labels.
    """

    @staticmethod
    def preprocess_label_image(img: np.ndarray) -> np.ndarray:
        """
        Applies grayscale, contrast enhancement, and adaptive thresholding
        to maximize text legibility for OCR engines.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return enhanced

    @staticmethod
    def parse_nutrition_text(text: str) -> Dict[str, Any]:
        """
        Robust regex-based extraction parser for nutrition panels.
        Extracts calories, protein, carbs, fat, sodium, sugar, ingredients, and allergens.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = " ".join(lines)
        
        def extract_number(pattern: str, src: str) -> Optional[float]:
            match = re.search(pattern, src, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    return None
            return None

        cal = extract_number(r"(?:calories|energy|kcal)[:\s]+(\d+(?:\.\d+)?)", clean_text)
        prot = extract_number(r"(?:protein|proteins)[:\s]+(\d+(?:\.\d+)?)\s*g?", clean_text)
        carbs = extract_number(r"(?:total\s+carbohydrate|carbohydrates|carbs)[:\s]+(\d+(?:\.\d+)?)\s*g?", clean_text)
        fat = extract_number(r"(?:total\s+fat|fat|lipids)[:\s]+(\d+(?:\.\d+)?)\s*g?", clean_text)
        sugar = extract_number(r"(?:total\s+sugars|sugars|sugar)[:\s]+(\d+(?:\.\d+)?)\s*g?", clean_text)
        sodium = extract_number(r"(?:sodium|salt)[:\s]+(\d+(?:\.\d+)?)\s*(?:mg|g)?", clean_text)
        
        # Serving size extraction
        serving_match = re.search(r"serving\s+size[:\s]+([^,\n;]+)", clean_text, re.IGNORECASE)
        serving_size = serving_match.group(1).strip() if serving_match else "1 serving"

        # Ingredient extraction
        ingredients = []
        ing_match = re.search(r"ingredients?[:\s]+([^\.\n]+)", clean_text, re.IGNORECASE)
        if ing_match:
            raw_ing = ing_match.group(1)
            ingredients = [item.strip() for item in re.split(r"[,;•]", raw_ing) if len(item.strip()) > 1]

        # Allergen extraction
        allergens = []
        all_match = re.search(r"(?:contains|allergens?)[:\s]+([^\.\n]+)", clean_text, re.IGNORECASE)
        if all_match:
            raw_all = all_match.group(1)
            allergens = [item.strip() for item in re.split(r"[,;•]", raw_all) if len(item.strip()) > 1]

        # Determine review flags
        review_reasons = []
        if cal is None:
            review_reasons.append("Calories could not be detected on label")
        if prot is None:
            review_reasons.append("Protein value could not be confirmed")
        if carbs is None:
            review_reasons.append("Carbohydrate value could not be confirmed")
        if fat is None:
            review_reasons.append("Total fat value could not be confirmed")

        return {
            "serving_size": serving_size,
            "calories": cal,
            "protein": prot,
            "carbs": carbs,
            "fat": fat,
            "sugar": sugar,
            "sodium": sodium,
            "ingredients": ingredients,
            "allergens": allergens,
            "review_required": len(review_reasons) > 0,
            "review_reasons": review_reasons
        }

    @classmethod
    def analyze_package_text(cls, raw_text: str) -> PackageOCRResult:
        start_time = time.time()
        parsed = cls.parse_nutrition_text(raw_text)
        latency = round((time.time() - start_time) * 1000, 2)

        return PackageOCRResult(
            product_name="Scanned Packaged Food Item",
            serving_size=parsed["serving_size"],
            calories=ExtractedNutrient(
                name="Calories",
                value=parsed["calories"],
                unit="kcal",
                is_detected=parsed["calories"] is not None
            ),
            protein=ExtractedNutrient(
                name="Protein",
                value=parsed["protein"],
                unit="g",
                is_detected=parsed["protein"] is not None
            ),
            carbohydrates=ExtractedNutrient(
                name="Carbohydrates",
                value=parsed["carbs"],
                unit="g",
                is_detected=parsed["carbs"] is not None
            ),
            fat=ExtractedNutrient(
                name="Total Fat",
                value=parsed["fat"],
                unit="g",
                is_detected=parsed["fat"] is not None
            ),
            sugar=ExtractedNutrient(
                name="Sugar",
                value=parsed["sugar"],
                unit="g",
                is_detected=parsed["sugar"] is not None
            ) if parsed["sugar"] is not None else None,
            sodium=ExtractedNutrient(
                name="Sodium",
                value=parsed["sodium"],
                unit="mg",
                is_detected=parsed["sodium"] is not None
            ) if parsed["sodium"] is not None else None,
            ingredients=parsed["ingredients"],
            allergens=parsed["allergens"],
            raw_ocr_text=raw_text,
            review_required=parsed["review_required"],
            review_reasons=parsed["review_reasons"],
            processing_time_ms=latency
        )

ocr_service = PackageOCRService()

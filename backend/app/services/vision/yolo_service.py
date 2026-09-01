import cv2
import numpy as np
import os
import time
import uuid
import logging
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO, YOLOWorld
from app.core.config import settings
from app.schemas.detection import DetectedFoodItem, BoundingBox

logger = logging.getLogger(__name__)

CLASS_PALETTE = {
    "bread": (42, 114, 222),
    "tomato": (34, 34, 220),
    "carrot": (0, 140, 255),
    "chicken_duck": (80, 127, 255),
    "steak": (49, 39, 139),
    "potato": (102, 178, 204),
    "broccoli": (46, 139, 87),
    "ice_cream": (203, 192, 255),
    "strawberry": (71, 99, 255),
    "lettuce": (50, 205, 50),
    "onion": (180, 105, 255),
    "rice": (220, 245, 245),
    "cucumber": (113, 179, 60),
    "pepper": (0, 69, 255),
    "pie": (30, 105, 210)
}

MULTI_DISH_VOCABULARY = [
    "rice", "curry", "dal", "bread", "salad", "chicken",
    "broccoli", "potato", "soup", "noodles", "egg"
]

class YOLOFoodDetectionService:
    """
    Inference service for FoodLens YOLO Food Detection.
    Defaults to our fine-tuned custom YOLO model for high precision and accurate detections.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self._finetuned_model = None
        self._world_model = None
        self.annotated_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/annotated"))
        os.makedirs(self.annotated_dir, exist_ok=True)

    @property
    def finetuned_model(self) -> YOLO:
        if self._finetuned_model is None:
            if os.path.exists(self.model_path):
                logger.info(f"Loading custom fine-tuned FoodLens YOLO model from '{self.model_path}'")
                self._finetuned_model = YOLO(self.model_path)
            else:
                logger.warning(f"Model path '{self.model_path}' not found, falling back to yolo11n.pt")
                self._finetuned_model = YOLO("yolo11n.pt")
        return self._finetuned_model

    @property
    def world_model(self) -> YOLOWorld:
        if self._world_model is None:
            self._world_model = YOLOWorld("yolov8s-worldv2.pt")
            self._world_model.set_classes(MULTI_DISH_VOCABULARY)
        return self._world_model

    def detect(
        self,
        img: np.ndarray,
        conf_threshold: float = 0.25,
        engine: str = "finetuned"
    ) -> Tuple[List[DetectedFoodItem], np.ndarray, float]:
        """
        Runs inference on an OpenCV BGR image and returns detected items + annotated image.
        """
        start_time = time.time()

        if engine == "world":
            model = self.world_model
            names_lookup = {i: name for i, name in enumerate(MULTI_DISH_VOCABULARY)}
        else:
            model = self.finetuned_model
            names_lookup = model.names

        results = model.predict(source=img, conf=conf_threshold, verbose=False)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        detections: List[DetectedFoodItem] = []
        annotated_img = img.copy()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                raw_name = names_lookup.get(class_id, f"food_{class_id}")
                canonical_label = raw_name.strip().lower().replace(" ", "_")

                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]

                # Determine qualitative rating
                if confidence >= 0.70:
                    rating = "high"
                elif confidence >= 0.40:
                    rating = "medium"
                else:
                    rating = "low"

                detected_item = DetectedFoodItem(
                    label=canonical_label,
                    confidence=round(confidence, 3),
                    class_id=class_id,
                    bbox=BoundingBox(
                        x_min=round(x1, 1),
                        y_min=round(y1, 1),
                        x_max=round(x2, 1),
                        y_max=round(y2, 1)
                    ),
                    confidence_level=rating
                )
                detections.append(detected_item)

                # Draw crisp bounding box & badge
                color = CLASS_PALETTE.get(canonical_label, (46, 204, 113))
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.rectangle(annotated_img, pt1, pt2, color, 3)

                display_title = raw_name.replace("_", " ").title()
                label_text = f"{display_title} {int(confidence * 100)}%"
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Filled badge header
                badge_top = max(0, int(y1) - th - 10)
                cv2.rectangle(
                    annotated_img,
                    (int(x1), badge_top),
                    (int(x1) + tw + 12, int(y1)),
                    color,
                    -1
                )
                cv2.putText(
                    annotated_img,
                    label_text,
                    (int(x1) + 6, int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        return detections, annotated_img, latency_ms

    def save_annotated(self, annotated_img: np.ndarray) -> str:
        filename = f"annotated_{uuid.uuid4().hex[:10]}.jpg"
        filepath = os.path.join(self.annotated_dir, filename)
        cv2.imwrite(filepath, annotated_img)
        return f"/api/v1/food/annotated-image/{filename}"

    def save_annotated_image(self, annotated_img: np.ndarray) -> str:
        return self.save_annotated(annotated_img)

yolo_service = YOLOFoodDetectionService()

import cv2
import numpy as np
import os
import time
import uuid
import logging
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
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

class YOLOFoodDetectionService:
    """
    Singleton inference service for YOLO Food Detection.
    Loads model weights once and performs fast batched/single inference.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self._model = None
        self.annotated_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/annotated"))
        os.makedirs(self.annotated_dir, exist_ok=True)

    @property
    def model(self) -> YOLO:
        if self._model is None:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model path '{self.model_path}' not found, falling back to pretrained YOLO11n.")
                self._model = YOLO("yolo11n.pt")
            else:
                logger.info(f"Loading custom FoodLens YOLO model from '{self.model_path}'")
                self._model = YOLO(self.model_path)
        return self._model

    def detect(self, img: np.ndarray, conf_threshold: float = 0.25) -> Tuple[List[DetectedFoodItem], np.ndarray, float]:
        """
        Runs inference on an OpenCV BGR image and returns detected items + annotated image.
        """
        start_time = time.time()
        results = self.model.predict(source=img, conf=conf_threshold, verbose=False)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        detections: List[DetectedFoodItem] = []
        annotated_img = img.copy()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names.get(class_id, f"food_{class_id}")
                
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
                    label=class_name,
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

                # Draw bounding box & label on annotated image
                color = CLASS_PALETTE.get(class_name, (0, 255, 0))
                cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                label_tag = f"{class_name} {int(confidence * 100)}%"
                (w, h), _ = cv2.getTextSize(label_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_img, (int(x1), int(y1) - 20), (int(x1) + w + 6, int(y1)), color, -1)
                cv2.putText(annotated_img, label_tag, (int(x1) + 3, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return detections, annotated_img, latency_ms

    def save_annotated(self, annotated_img: np.ndarray) -> str:
        filename = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(self.annotated_dir, filename)
        cv2.imwrite(filepath, annotated_img)
        return f"/api/v1/food/annotated-image/{filename}"

yolo_service = YOLOFoodDetectionService()

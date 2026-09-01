from fastapi import APIRouter, UploadFile, File, Query, HTTPException, status
from fastapi.responses import FileResponse
from app.services.vision.image_service import image_service
from app.services.vision.yolo_service import yolo_service
from app.schemas.detection import FoodDetectionResponse
import os

router = APIRouter()

@router.post("/detect", response_model=FoodDetectionResponse, summary="Detect Foods with YOLO")
async def detect_foods(
    file: UploadFile = File(..., description="Food photo for detection"),
    conf_threshold: float = Query(0.25, ge=0.05, le=0.95, description="Confidence score threshold")
):
    """
    Detects visible food classes in an uploaded photo using fine-tuned YOLO.
    Returns bounding boxes, confidence ratings, and an annotated image link.
    """
    contents, ext = await image_service.validate_and_read(file)
    img = image_service.decode_image(contents)
    processed_img = image_service.resize_preserving_aspect_ratio(img, max_dim=1280)

    detections, annotated_img, latency_ms = yolo_service.detect(processed_img, conf_threshold=conf_threshold)
    annotated_url = yolo_service.save_annotated(annotated_img)

    has_low_conf = any(d.confidence < 0.40 for d in detections)

    return FoodDetectionResponse(
        total_detections=len(detections),
        detections=detections,
        annotated_image_url=annotated_url,
        has_low_confidence=has_low_conf,
        processing_time_ms=latency_ms
    )

@router.get("/annotated-image/{filename}", summary="Stream Annotated Image")
def get_annotated_image(filename: str):
    filepath = os.path.join(yolo_service.annotated_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotated image not found")
    return FileResponse(filepath, media_type="image/jpeg")

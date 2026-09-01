from fastapi import APIRouter, UploadFile, File, status
from app.services.vision.image_service import image_service
from typing import Dict, Any

router = APIRouter()

@router.post("/upload", status_code=status.HTTP_201_CREATED, summary="Upload & Preprocess Image")
async def upload_image(file: UploadFile = File(..., description="Food or package photo to upload")) -> Dict[str, Any]:
    """
    Uploads an image, validates format/size, decodes via OpenCV,
    normalizes dimensions, and stores for analysis.
    """
    contents, ext = await image_service.validate_and_read(file)
    img = image_service.decode_image(contents)
    processed_img = image_service.resize_preserving_aspect_ratio(img, max_dim=1280)
    meta = image_service.save_image(processed_img, ext=ext)

    return {
        "status": "success",
        "message": "Image successfully validated and preprocessed.",
        "data": meta
    }

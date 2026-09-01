import cv2
import numpy as np
import uuid
import os
from fastapi import UploadFile, HTTPException, status
from typing import Tuple, Dict, Any

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/uploads"))

os.makedirs(UPLOAD_DIR, exist_ok=True)

class ImageProcessingService:
    """
    Service for validating, decoding, preprocessing, and persisting uploaded food images.
    """

    @staticmethod
    async def validate_and_read(file: UploadFile) -> Tuple[bytes, str]:
        """
        Validates file extension, content type, and size.
        """
        # 1. Validate extension
        filename = file.filename or "upload.jpg"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        # 2. Validate MIME type
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid MIME type '{file.content_type}'. Must be JPEG, PNG, or WEBP."
            )

        # 3. Read file contents and check size
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )
        if len(contents) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of 10MB (Received {len(contents) / (1024*1024):.2f}MB)."
            )

        return contents, ext

    @staticmethod
    def decode_image(contents: bytes) -> np.ndarray:
        """
        Decodes raw byte buffer into OpenCV BGR NumPy matrix.
        Detects corrupt or malformed image files.
        """
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted or invalid image file. OpenCV could not decode image."
            )

        return img

    @staticmethod
    def resize_preserving_aspect_ratio(img: np.ndarray, max_dim: int = 1280) -> np.ndarray:
        """
        Resizes image if its height or width exceeds max_dim, preserving the exact aspect ratio.
        """
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    @classmethod
    def save_image(cls, img: np.ndarray, ext: str = ".jpg") -> Dict[str, Any]:
        """
        Saves the processed image to local storage and returns its metadata.
        """
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        success = cv2.imwrite(filepath, img)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to write image to disk."
            )

        h, w, c = img.shape
        return {
            "image_id": file_id,
            "filename": filename,
            "filepath": filepath,
            "width": w,
            "height": h,
            "channels": c,
            "size_bytes": os.path.getsize(filepath)
        }

image_service = ImageProcessingService()

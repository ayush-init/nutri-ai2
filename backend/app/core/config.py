from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "FoodLens AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"

    # Neon DB PostgreSQL Connection String
    DATABASE_URL: str

    # CORS Configuration for Frontend access
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]

    # Model Weights Path (v2 103-class model default, v1 fallback)
    YOLO_MODEL_PATH: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../runs/detect/foodlens_v2_103/weights/best.pt")
        if os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../runs/detect/foodlens_v2_103/weights/best.pt")))
        else os.path.join(os.path.dirname(__file__), "../../../runs/detect/foodlens_v1_gpu-3/weights/best.pt")
    )

    model_config = SettingsConfigDict(
        env_file=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env")),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

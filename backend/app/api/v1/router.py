from fastapi import APIRouter
from app.api.v1.endpoints import health, image, food

api_router = APIRouter()
api_router.include_router(health.router, tags=["System & Health"])
api_router.include_router(image.router, prefix="/image", tags=["Image Pipeline"])
api_router.include_router(food.router, prefix="/food", tags=["Food Detection & Analysis"])

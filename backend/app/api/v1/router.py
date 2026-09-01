from fastapi import APIRouter
from app.api.v1.endpoints import health, image, food, package, menu, history

api_router = APIRouter()
api_router.include_router(health.router, tags=["System & Health"])
api_router.include_router(image.router, prefix="/image", tags=["Image Pipeline"])
api_router.include_router(food.router, prefix="/food", tags=["Food Detection & Analysis"])
api_router.include_router(package.router, prefix="/package", tags=["Packaged Food OCR"])
api_router.include_router(menu.router, prefix="/menu", tags=["Menu Intelligence & Recommendations"])
api_router.include_router(history.router, prefix="/history", tags=["Database History & Persistence"])

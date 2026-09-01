from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

from app.core.config import settings
from app.api.v1.router import api_router
from app.core.middleware import request_timing_and_logging_middleware

def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="""
## ?? FoodLens - AI Food Intelligence & Menu Assistant API
        
FoodLens provides an end-to-end AI platform combining:
* ?? **Computer Vision:** Fine-tuned YOLO object detection with bounding box annotations.
* ?? **Nutrition Engine:** Deterministic macronutrient & calorie estimation with uncertainty ranges.
* ?? **Packaged Food OCR:** Non-hallucinated nutrition facts, ingredients, and allergen parsing.
* ?? **Menu Intelligence:** Natural language menu parsing, dietary ranking (balanced, high-protein, low-cal, vegetarian).
* ?? **Food Comparison:** Multi-item comparison with transparent trade-offs.
* ??? **Persistent Cloud Storage:** Neon PostgreSQL tracking history and records.
        """,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Middleware
    application.middleware("http")(request_timing_and_logging_middleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Routers
    application.include_router(api_router, prefix=settings.API_V1_STR)

    # Mount static assets if present
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../static"))
    os.makedirs(static_dir, exist_ok=True)
    application.mount("/static", StaticFiles(directory=static_dir), name="static")

    @application.get("/", response_class=HTMLResponse, tags=["Frontend Dashboard"])
    def landing_page():
        html_file = os.path.join(static_dir, "index.html")
        if os.path.exists(html_file):
            with open(html_file, "r", encoding="utf-8") as f:
                return f.read()
        return f"""
        <html>
            <head><title>{settings.PROJECT_NAME}</title></head>
            <body style='font-family:sans-serif; text-align:center; padding:50px;'>
                <h1>?? {settings.PROJECT_NAME}</h1>
                <p>FastAPI Backend is running and connected to Neon PostgreSQL.</p>
                <p><a href='/docs'>Interactive Swagger API Documentation</a></p>
            </body>
        </html>
        """

    return application

app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

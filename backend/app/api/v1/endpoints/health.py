from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.database import get_db
from app.core.config import settings
import time

router = APIRouter()

@router.get("/health", status_code=status.HTTP_200_OK, summary="Health Check and Database Connectivity")
def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint that validates:
    1. FastAPI application is responding.
    2. Connection to Neon PostgreSQL cloud database is active and measuring latency.
    """
    db_status = "unknown"
    db_latency_ms = None
    
    start_time = time.time()
    try:
        result = db.execute(text("SELECT 1;")).scalar()
        if result == 1:
            db_status = "connected"
        else:
            db_status = "unexpected_response"
        db_latency_ms = round((time.time() - start_time) * 1000, 2)
    except Exception as e:
        db_status = f"disconnected: {str(e)}"

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database": {
            "type": "Neon PostgreSQL",
            "status": db_status,
            "latency_ms": db_latency_ms
        }
    }

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import Generator
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Neon PostgreSQL connection pool configuration
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,      # Validates connection health before issuing queries
    pool_recycle=300,        # Recycle connections every 5 min to prevent Neon timeouts
    pool_size=5,             # Base number of pooled connections
    max_overflow=10          # Overflow capacity during burst traffic
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db() -> Generator:
    """
    FastAPI Dependency that provides a database session per request,
    ensuring proper closure after the response is sent.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

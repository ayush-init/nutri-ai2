from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from app.core.database import get_db
from app.models.analysis import AnalysisHistory

router = APIRouter()

@router.get("", summary="Get Past Analyses History from Neon PostgreSQL")
def get_history(
    analysis_type: Optional[str] = Query(None, description="Filter by type: food_photo, package_ocr, menu_analysis, food_comparison"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Retrieves stored analysis records from Neon PostgreSQL database.
    """
    query = db.query(AnalysisHistory)
    if analysis_type:
        query = query.filter(AnalysisHistory.analysis_type == analysis_type)
    
    records = query.order_by(AnalysisHistory.created_at.desc()).limit(limit).all()

    return {
        "total": len(records),
        "history": [
            {
                "id": r.id,
                "analysis_type": r.analysis_type,
                "summary_title": r.summary_title,
                "total_calories": r.total_calories,
                "total_protein": r.total_protein,
                "total_carbs": r.total_carbs,
                "total_fat": r.total_fat,
                "image_filename": r.image_filename,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "payload": r.payload
            }
            for r in records
        ]
    }

@router.get("/{history_id}", summary="Get Single Analysis Detail")
def get_history_detail(history_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisHistory).filter(AnalysisHistory.id == history_id).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History record not found")
    
    return {
        "id": record.id,
        "analysis_type": record.analysis_type,
        "summary_title": record.summary_title,
        "total_calories": record.total_calories,
        "total_protein": record.total_protein,
        "total_carbs": record.total_carbs,
        "total_fat": record.total_fat,
        "image_filename": record.image_filename,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "payload": record.payload
    }

@router.delete("/{history_id}", summary="Delete History Record")
def delete_history_record(history_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisHistory).filter(AnalysisHistory.id == history_id).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History record not found")
    
    db.delete(record)
    db.commit()
    return {"status": "success", "message": f"Deleted history record {history_id}"}

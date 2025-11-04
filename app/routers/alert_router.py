from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app.model.alert_schema import AlertCreate, AlertResponse
from app.repository import alert_repository
import uuid
router = APIRouter(
    prefix="/alert",
    tags=["Alert"],
)

@router.post("/", response_model=AlertResponse)
def create_alert(alert: AlertCreate, db: Session = Depends(get_db)):
    """Insert new alert record into database."""
    try:
        db.id = str(uuid.uuid4())
        new_alert = alert_repository.insert_alert(db, alert)
        return new_alert
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert alert: {e}")


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert_by_id(alert_id: str, db: Session = Depends(get_db)):
    """Fetch a single alert by its ID."""
    print("Fetching alert with ID:", alert_id)
    alert = alert_repository.get_alert_by_id(db, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.get("/store/{store_id}", response_model=List[AlertResponse])
def get_alerts_by_store(store_id: str, db: Session = Depends(get_db)):
    """Fetch all alerts for a specific store."""
    alerts = alert_repository.get_alerts_by_store(db, store_id)
    return alerts


@router.get("/", response_model=List[AlertResponse])
def get_all_alerts(limit: int = 50, db: Session = Depends(get_db)):
    """Fetch all alerts (default limit = 50)."""
    alerts = alert_repository.get_all_alerts(db, limit)
    return alerts

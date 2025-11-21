from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.db.session import get_db
from app.model.alert_schema import AlertCreate, AlertResponse, AlertUpdate
from app.repository import alert_repository
from typing import Optional
from fastapi import Request

import uuid
router = APIRouter(
    prefix="/alert",
    tags=["Alert"],
)

@router.post("/", response_model=AlertResponse)
def create_alert(alert: AlertCreate, db: Session = Depends(get_db)):
    """Insert new alert record into database."""
    try:
        alert.id = str(uuid.uuid4())
        new_alert = alert_repository.insert_alert(db, alert)
        return new_alert
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert alert: {e}")
    
@router.put("/{alert_id}", response_model=AlertResponse)
def update_alert(alert_id: str, alert: AlertUpdate, request: Request, db: Session = Depends(get_db)):
    """Update an existing alert record."""
    existing_alert = alert_repository.get_alert_by_id(db, alert_id)
    if not existing_alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    try:
        user = getattr(request.state, "user", None)
        if not user or not user.get("id"):
            raise HTTPException(status_code=401, detail="Unauthorized")
        updated_by = user["id"]
        updated_alert = alert_repository.update_alert(db, alert_id, alert, updated_by=updated_by)
        return updated_alert
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update alert: {e}")


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert_by_id(alert_id: str, db: Session = Depends(get_db)):
    """Fetch a single alert by its ID."""
    alert = alert_repository.get_alert_by_id(db, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.get("/store/{store_id}", response_model=List[AlertResponse])
def get_alerts_by_store(store_id: str, is_valid: Optional[bool] = None, db: Session = Depends(get_db)):
    """Fetch all alerts for a specific store."""
    alerts = alert_repository.get_alerts_by_store(db, is_valid, store_id)
    return alerts


@router.get("/", response_model=List[AlertResponse])
def get_all_alerts(limit: int = 50, db: Session = Depends(get_db)):
    """Fetch all alerts (default limit = 50)."""
    alerts = alert_repository.get_all_alerts(db, limit)
    return alerts

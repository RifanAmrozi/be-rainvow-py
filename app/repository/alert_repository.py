from sqlalchemy.orm import Session
from app.model.alert import Alert
from app.model.alert_schema import AlertCreate, AlertUpdate
from app.model.camera import Camera
from typing import Optional
from datetime import datetime

def insert_alert(db: Session, data):
    if isinstance(data, dict):
        payload = data
    elif isinstance(data, AlertCreate):
        payload = data.dict()
    else:
        raise TypeError("insert_alert() expects dict or AlertCreate model")

    if isinstance(payload.get("incident_start"), str):
        try:
            payload["incident_start"] = datetime.fromisoformat(
                payload["incident_start"].replace("Z", "")
            )
        except Exception:
            print("⚠️ Failed to parse incident_start timestamp")

    new_alert = Alert(**payload)

    db.add(new_alert)
    db.commit()
    db.refresh(new_alert)

    return new_alert

def get_alert_by_id(db: Session, id: str):
    result = (
        db.query(Alert, Camera.name.label("camera_name"), Camera.aisle_loc.label("aisle_loc"))
        .join(Camera, Alert.camera_id == Camera.id)
        .filter(Alert.id == id)
        .first()
    )

    if not result:
        return None

    alert, camera_name, aisle_loc = result
    alert_dict = alert.__dict__.copy()
    alert_dict["camera_name"] = camera_name
    alert_dict["aisle_loc"] = aisle_loc
    return alert_dict

def update_alert(db: Session, id: str, data: AlertUpdate, updated_by: Optional[str] = None):
    alert = db.query(Alert).filter(Alert.id == id).first()
    if alert:
        if data.title is not None:
            alert.title = data.title
        if data.is_valid is not None:
            alert.is_valid = data.is_valid
        if data.notes is not None:
            alert.notes = data.notes
        if updated_by is not None:
            alert.updated_by = updated_by
        db.commit()
        db.refresh(alert)
    return alert


def get_alerts_by_store(db: Session, is_valid: Optional[bool], store_id: str):
    query = (
        db.query(
            Alert,
            Camera.name.label("camera_name"),
            Camera.aisle_loc.label("aisle_loc"),
        )
        .join(Camera, Alert.camera_id == Camera.id)
        .filter(Alert.store_id == store_id)
        .order_by(Alert.incident_start.desc())
    )
    if is_valid is not None:
        query = query.filter(Alert.is_valid.is_(is_valid))
    else:
        query = query.filter(Alert.is_valid.is_(None))
    results = query.all()
    alerts = []
    for result in results:
        alert, camera_name, aisle_loc = result
        alert_dict = alert.__dict__.copy()
        alert_dict["camera_name"] = camera_name
        alert_dict["aisle_loc"] = aisle_loc
        alerts.append(alert_dict)
    return alerts


def get_all_alerts(db: Session, limit: int = 50):
    return db.query(Alert).order_by(Alert.timestamp.desc()).limit(limit).all()

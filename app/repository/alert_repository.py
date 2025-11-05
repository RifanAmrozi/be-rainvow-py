from sqlalchemy.orm import Session
from app.model.alert import Alert
from app.model.alert_schema import AlertCreate
from typing import Optional


def insert_alert(db: Session, data: AlertCreate):
    new_alert = Alert(
        id=data.id,
        title=data.title,
        incident_start=data.incident_start,
        is_valid=data.is_valid,
        video_url=data.video_url,
        notes=data.notes,
        store_id=data.store_id,
        camera_id=data.camera_id,
    )
    db.add(new_alert)
    db.commit()
    db.refresh(new_alert)
    return new_alert


def get_alert_by_id(db: Session, id: str):
    return db.query(Alert).filter(Alert.id == id).first()

def update_alert(db: Session, id: str, data: AlertCreate):
    alert = db.query(Alert).filter(Alert.id == id).first()
    if alert:
        alert.title = data.title
        alert.is_valid = data.is_valid
        alert.notes = data.notes
        db.commit()
        db.refresh(alert)
    return alert


def get_alerts_by_store(db: Session, is_valid: Optional[bool], store_id: str):
    query = db.query(Alert).filter(Alert.store_id == store_id)
    if is_valid is not None:
        query = query.filter(Alert.is_valid.isnot(None))
    else:
        query = query.filter(Alert.is_valid.is_(None))
    return query.all()


def get_all_alerts(db: Session, limit: int = 50):
    return db.query(Alert).order_by(Alert.timestamp.desc()).limit(limit).all()

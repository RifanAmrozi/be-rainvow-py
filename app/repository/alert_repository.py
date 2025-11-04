from sqlalchemy.orm import Session
from app.model.alert import Alert
from app.model.alert_schema import AlertCreate


def insert_alert(db: Session, data: AlertCreate):
    print("Inserting alert into database...", data)
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


def get_alerts_by_store(db: Session, store_id: str):
    return db.query(Alert).filter(Alert.store_id == store_id).all()


def get_all_alerts(db: Session, limit: int = 50):
    return db.query(Alert).order_by(Alert.timestamp.desc()).limit(limit).all()

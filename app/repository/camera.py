from sqlalchemy.orm import Session
from sqlalchemy import select
from app.model.camera import Camera

def get_all_cameras(db: Session, id: str = None, name: str = None):
    query = db.query(Camera)
    if id:
        query = query.filter(Camera.id == id)
    if name:
        query = query.filter(Camera.name.ilike(f"%{name}%"))
    return query.all()

def create_camera(db: Session, camera_data: dict):
    camera = Camera(**camera_data)
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera
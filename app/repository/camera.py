from sqlalchemy.orm import Session
from sqlalchemy import select
from app.model.camera import Camera

def get_all_cameras(db: Session):
    result = db.execute(select(Camera))
    return result.scalars().all()

def create_camera(db: Session, camera_data: dict):
    camera = Camera(**camera_data)
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera
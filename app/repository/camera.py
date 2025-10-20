from sqlalchemy.orm import Session
from sqlalchemy import select
from app.model.camera import Camera

def get_all_cameras(db: Session):
    result = db.execute(select(Camera))
    return result.scalars().all()

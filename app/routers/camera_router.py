from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.repository.camera import get_all_cameras, create_camera

router = APIRouter(prefix="/camera", tags=["Camera"])

@router.get("/")
def list_cameras(db: Session = Depends(get_db)):
    cameras = get_all_cameras(db)
    return cameras

@router.post("/")
def create_new_camera(payload: dict, db: Session = Depends(get_db)):
    camera = create_camera(db, payload)
    return camera
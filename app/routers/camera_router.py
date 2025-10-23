from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from app.db.session import get_db
from app.repository.camera import get_all_cameras, create_camera
from app.model.camera_schema import CameraCreate

router = APIRouter(prefix="/camera", tags=["Camera"])

@router.get("/")
def list_cameras(
    id: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    cameras = get_all_cameras(db, id=id, name=name)
    return cameras

@router.post("/")
def create_new_camera(payload: CameraCreate, db: Session = Depends(get_db)):
    camera = create_camera(db, payload)
    return camera
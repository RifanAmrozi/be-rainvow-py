from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from app.db.session import get_db
from app.repository.camera_repository import get_all_cameras, create_camera, get_ip_of_interface, get_aisle_locations
from app.model.camera_schema import CameraCreate

router = APIRouter(prefix="/camera", tags=["Camera"])

@router.get("/")
def list_cameras(
    id: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    store_id: Optional[str] = Query(None),
    aisle_loc: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    cameras = get_all_cameras(db, id=id, name=name, store_id=store_id, aisle_loc=aisle_loc)
    return cameras

@router.post("/")
def create_new_camera(payload: CameraCreate, db: Session = Depends(get_db)):
    camera = create_camera(db, payload)
    return camera


@router.get("/ip")
def get_client_ip():
    client_host = get_ip_of_interface("en1")
    return {"client_ip": client_host}

@router.get("/aisle-loc")
def get_unique_aisle_locs(db: Session = Depends(get_db), store_id: Optional[str] = None):
    locs = get_aisle_locations(db=db, store_id=store_id)
    return {"aisle_locations": locs}
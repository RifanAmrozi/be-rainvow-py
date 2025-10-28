from fastapi import HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from app.model.store import Store
from app.model.camera import Camera
from app.model.store_schema import StoreCreate
from app.repository.camera import get_local_ip
import re
import uuid

def create_store(db: Session, store_data: StoreCreate):
    new_store = Store(
        id=str(uuid.uuid4()),
        store_name=store_data.store_name,
        store_address=store_data.store_address
    )
    db.add(new_store)
    db.commit()
    db.refresh(new_store)
    return new_store

def get_stores(db: Session, id: Optional[str] = None):
    query = db.query(Store)
    if id:
        query = query.filter(Store.id == id)
    return query.all()

def evaluate_store(db: Session, id: Optional[str] = None, ip: Optional[str] = None):
    ip_local = get_local_ip() if ip is None else ip

    if not id:
        raise HTTPException(status_code=400, detail="Store ID is required")

    store = db.query(Store).filter(Store.id == id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")

    cameras = db.query(Camera).filter(Camera.store_id == store.id).all()

    for camera in cameras:
        # Replace only the IP portion before ":8889"
        camera.webrtc_url = re.sub(
            r"//([\d\.]+):8889",             # Match // followed by IP + port
            f"//{ip_local}:8889",            # Replace with the provided or local IP
            camera.webrtc_url
        )
        db.add(camera)

    db.commit()

    return cameras

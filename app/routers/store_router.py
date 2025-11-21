from fastapi import APIRouter, Depends, Query
from typing import Optional
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.repository.store_repository import create_store, get_stores, evaluate_store
from app.model.store_schema import StoreCreate, StoreResponse
from app.model.camera_schema import EvaluateRequest, CameraResponse
from typing import List

router = APIRouter(prefix="/store", tags=["Store"])

@router.post("/", response_model=StoreResponse)
def create_new_store(store_data: StoreCreate, db: Session = Depends(get_db)):
    return create_store(db, store_data)

@router.get("/", response_model=List[StoreResponse])
def list_stores(
    id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    return get_stores(db,id=id)


@router.post("/evaluate", response_model=List[CameraResponse])
def evaluate(data: EvaluateRequest, db: Session = Depends(get_db)):
    return evaluate_store(db, id=data.id, ip=data.ip)
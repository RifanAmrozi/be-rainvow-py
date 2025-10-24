from sqlalchemy.orm import Session
from app.model.store import Store
from app.model.store_schema import StoreCreate
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

def get_stores(db: Session):
    return db.query(Store).all()

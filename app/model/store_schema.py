from pydantic import BaseModel
from typing import Optional
from uuid import UUID

class StoreCreate(BaseModel):
    store_name: str
    store_address: Optional[str] = None

class StoreResponse(StoreCreate):
    id: UUID

    class Config:
        orm_mode = True

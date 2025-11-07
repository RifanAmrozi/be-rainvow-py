from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from uuid import UUID

class AlertCreate(BaseModel):
    id: Optional[str] = None
    title: str
    incident_start: datetime
    is_valid: Optional[bool] = None
    video_url: Optional[str] = None
    notes: Optional[str] = None
    store_id: str
    camera_id: str


class AlertResponse(AlertCreate):
    id: UUID
    store_id: UUID
    camera_id: UUID
    camera_name: Optional[str] = None
    aisle_loc: Optional[str] = None

    class Config:
        orm_mode = True

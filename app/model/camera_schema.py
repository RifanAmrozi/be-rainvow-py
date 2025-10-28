from pydantic import BaseModel
from typing import Optional
from uuid import UUID

class CameraCreate(BaseModel):
    name: str
    aisle_loc: str
    preview_img: Optional[str] = None
    rtsp_url: str
    store_id: str

class EvaluateRequest(BaseModel):
    id: str
    ip: Optional[str] = None

class CameraResponse(CameraCreate):
    id: UUID
    store_id: UUID
    webrtc_url: Optional[str] = None

    class Config:
        orm_mode = True
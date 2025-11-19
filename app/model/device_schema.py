from pydantic import BaseModel
from typing import Optional

class DeviceCreate(BaseModel):
    device_token: str
    user_id: Optional[str] = None
    store_id: Optional[str] = None

class DeviceResponse(DeviceCreate):
    id: str
    user_id: Optional[str] = None
    store_id: Optional[str] = None
    is_active: Optional[bool] = None

    class Config:
        orm_mode = True
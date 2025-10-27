from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Alert(BaseModel):
    id: str
    store_id: str
    camera_id: str
    timestamp: datetime
    suspicious_activity: bool
    alert_message: str
    image_url: Optional[str] = None
    video_url: Optional[str] = None

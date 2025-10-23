from pydantic import BaseModel
from typing import Optional

class CameraCreate(BaseModel):
    name: str
    aisle_loc: str
    status: Optional[bool] = True
    preview_img: Optional[str] = None
    rtsp_url: str
    store_id: str

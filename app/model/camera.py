from sqlalchemy import Column, Integer, String, Boolean
from app.db.session import engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Camera(Base):
    __tablename__ = "camera"
    __table_args__ = {"schema": "public"} 

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    aisle_loc = Column(String, nullable=False)
    preview_img = Column(String, nullable=True)
    rtsp_url = Column(String, nullable=False)
    store_id = Column(String, nullable=False)
    webrtc_url = Column(String, nullable=True)

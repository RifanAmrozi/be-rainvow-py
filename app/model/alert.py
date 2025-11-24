from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Alert(Base):
    __tablename__ = "alert"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    incident_start = Column(DateTime, nullable=False)
    is_valid = Column(Boolean, default=None)
    video_url = Column(String, nullable=True)
    photo_url = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    store_id = Column(String, index=True, nullable=False)
    camera_id = Column(String, index=True, nullable=False)
    updated_by = Column(String, nullable=True)

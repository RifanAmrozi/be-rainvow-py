from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey
import uuid
from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Device(Base):
    __tablename__ = "device"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_token = Column(String, unique=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    store_id = Column(UUID(as_uuid=True), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Device(id={self.id}, device_token={self.device_token}, active={self.is_active})>"

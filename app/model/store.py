import uuid
from sqlalchemy import Column, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Store(Base):
    __tablename__ = "store"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    store_name = Column(String, nullable=False)
    store_address = Column(String, nullable=False)

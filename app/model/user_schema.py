from pydantic import BaseModel, EmailStr
from uuid import UUID

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str
    store_id: UUID

class UserResponse(BaseModel):
    id: UUID
    username: str
    email: EmailStr
    role: str
    store_id: UUID

    class Config:
        orm_mode = True

class UserLogin(BaseModel):
    username: str
    password: str

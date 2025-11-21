from sqlalchemy.orm import Session
from app.model.user import User
from app.model.user_schema import UserCreate
from app.model.device_schema import DeviceCreate
from app.model.device import Device
from app.auth.utils import hash_password

def create_user(db: Session, data: UserCreate):
    hashed_pw = hash_password(data.password)
    new_user = User(
        username=data.username,
        email=data.email,
        password=hashed_pw,
        role=data.role,
        store_id=data.store_id,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

def get_user_by_name(db: Session, username: str=None, id: str = None):
    query = db.query(User)
    if id:
        query = query.filter(User.id == id)
    if username:
        query = query.filter(User.username == username)
    return query.first()

def register_device_token(db: Session, data: DeviceCreate):
    existing_device = db.query(Device).filter(Device.user_id == data.user_id).first()

    if existing_device:
        existing_device.device_token = data.device_token
        existing_device.user_id = data.user_id
        existing_device.store_id = data.store_id
        db.commit()
        db.refresh(existing_device)
        return existing_device
    else:
        new_device = Device(
            device_token=data.device_token,
            user_id=data.user_id,
            store_id=data.store_id,
            is_active=True
        )
        db.add(new_device)
        db.commit()
        db.refresh(new_device)
        return new_device

def get_devices(db: Session, user_id: str, store_id: str):
    query = db.query(Device)
    if user_id:
        query = query.filter(Device.user_id == user_id)
    if store_id:
        query = query.filter(Device.store_id == store_id)
    return query.all()
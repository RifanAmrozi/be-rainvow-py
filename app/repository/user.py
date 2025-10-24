from sqlalchemy.orm import Session
from app.model.user import User
from app.model.user_schema import UserCreate
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

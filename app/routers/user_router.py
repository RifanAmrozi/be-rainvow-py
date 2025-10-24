from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.repository.user import create_user, get_user_by_name
from app.model.user_schema import UserCreate, UserResponse, UserLogin
from app.auth.utils import verify_password, create_access_token

router = APIRouter(prefix="/user", tags=["User"])


@router.post("/register", response_model=UserResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    user_exist = get_user_by_name(db, user_data.username)
    if user_exist:
        raise HTTPException(status_code=400, detail="Username already exists")

    return create_user(db, user_data)


@router.post("/login")
def login(data: UserLogin, db: Session = Depends(get_db)):
    user = get_user_by_name(db, data.username, id=None)
    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.username})

    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username,
        "role": user.role,
        "store_id": user.store_id,
    }

@router.get("/{id}", response_model=UserResponse)
def get_user_by_id(id: str, db: Session = Depends(get_db)):
    user = get_user_by_name(db, username=None, id=id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

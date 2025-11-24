from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from jose import jwt, JWTError
from typing import Optional
from app.core.config import settings

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.user = None  # default

        auth: Optional[str] = request.headers.get("Authorization")
        token = None
        if auth and auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()

        if token:
            try:
                payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
                # adapt to the fields you return on login
                user_id = payload.get("id") or payload.get("sub")
                store_id = payload.get("store_id")
                request.state.user = {
                    "id": user_id,
                    "store_id": store_id,
                    "payload": payload
                }
            except JWTError:
                request.state.user = None

        response = await call_next(request)
        return response
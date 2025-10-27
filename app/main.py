from fastapi import FastAPI
from app.core.config import settings
from app.service.stream_worker import run_stream_worker
from app.db.session import test_connection
from app.routers.camera_router import router as camera_router
from app.routers.user_router import router as user_router
from app.routers.store_router import router as store_router
from app.routers.webhook_router import router as webhook_router
import asyncio
from app.service.alert import send_fake_alerts

app = FastAPI(title="Backend Server")

@app.on_event("startup")
def startup_event():
    print("🚀 Starting server...")
    asyncio.create_task(send_fake_alerts())
    test_connection()

app.include_router(camera_router)
app.include_router(user_router)
app.include_router(store_router)
app.include_router(webhook_router)

@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.get("/ping")
def ping():
    return {"ping": "pong"}

@app.get("/start")
def start_stream():
    # WARNING: blocking loop for now (for demo)
    run_stream_worker()
    return {"status": "started"}
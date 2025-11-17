from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from app.core.config import settings
from app.service.stream_worker import run_stream_worker
from app.db.session import test_connection
from app.routers.camera_router import router as camera_router
from app.routers.user_router import router as user_router
from app.routers.store_router import router as store_router
from app.routers.webhook_router import router as webhook_router
from app.websocket.websocket_router import router as websocket_router
from app.routers.alert_router import router as alert_router
from app.middleware.auth_middleware import AuthMiddleware
import asyncio
import os

app = FastAPI()

VIDEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "alert_clips"))

@app.get("/alert_clips/{filename}")
async def get_clip(filename: str):
    file_path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",       # enable seeking
            "Content-Disposition": "inline" # don't force download
        }
    )

@app.on_event("startup")
def startup_event():
    print("🚀 Starting server...")
    # comment this line to disable stream worker on startup
    app.state.stream_worker_task = asyncio.create_task(run_stream_worker(app, settings.STORE_ID))
    app.state.stop_stream_flag = False

    test_connection()

app.add_middleware(AuthMiddleware)
app.include_router(camera_router)
app.include_router(user_router)
app.include_router(store_router)
app.include_router(webhook_router)
app.include_router(websocket_router)
app.include_router(alert_router)

@app.get("/ping")
def ping():
    return {"ping": "pong"}

@app.get("/start")
async def start_stream():
    if app.state.stream_worker_task and not app.state.stream_worker_task.done():
        return {"status": "already running"}

    app.state.stop_stream_flag = False
    app.state.stream_worker_task = asyncio.create_task(run_stream_worker(app, settings.STORE_ID))
    return {"status": "started"}


@app.get("/stop")
async def stop_stream():
    if not app.state.stream_worker_task or app.state.stream_worker_task.done():
        return {"status": "no active stream"}

    app.state.stop_stream_flag = True
    await app.state.stream_worker_task
    app.state.stream_worker_task = None
    return {"status": "stopped"}


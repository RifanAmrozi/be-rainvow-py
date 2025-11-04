from fastapi import FastAPI
from app.core.config import settings
from app.service.stream_worker import run_stream_worker
from app.db.session import test_connection
from app.routers.camera_router import router as camera_router
from app.routers.user_router import router as user_router
from app.routers.store_router import router as store_router
from app.routers.webhook_router import router as webhook_router
from app.websocket.websocket_router import router as websocket_router
from app.routers.alert_router import router as alert_router
import asyncio

app = FastAPI(title="Backend Server")

@app.on_event("startup")
def startup_event():
    print("🚀 Starting server...")
    app.state.stream_worker_task = asyncio.create_task(run_stream_worker(app))
    app.state.stop_stream_flag = False

    test_connection()

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
    app.state.stream_worker_task = asyncio.create_task(run_stream_worker(app))
    return {"status": "started"}


@app.get("/stop")
async def stop_stream():
    if not app.state.stream_worker_task or app.state.stream_worker_task.done():
        return {"status": "no active stream"}

    app.state.stop_stream_flag = True
    await app.state.stream_worker_task
    app.state.stream_worker_task = None
    return {"status": "stopped"}


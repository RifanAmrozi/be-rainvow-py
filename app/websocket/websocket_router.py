from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.websocket.websocket_manager import WebSocketManager
from app.service.alert import send_alerts
import asyncio
import threading

router = APIRouter(prefix="/ws", tags=["WebSocket"])
manager = WebSocketManager()
websocket_task = None
stop_event = threading.Event()

@router.websocket("/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@router.get("/start")
async def start_websocket_alerts():
    global websocket_task, stop_event
    if websocket_task and not websocket_task.done():
        return {"status": "already running"}

    stop_event.clear()
    websocket_task = asyncio.create_task(send_alerts(manager, stop_event))
    return {"status": "websocket test started"}

@router.get("/stop")
async def stop_websocket_alerts():
    global websocket_task, stop_event
    stop_event.set()
    if websocket_task:
        websocket_task.cancel()
        websocket_task = None
    return {"status": "websocket test stopped"}

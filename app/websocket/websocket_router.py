from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .websocket_manager import WebSocketManager

router = APIRouter(prefix="/ws", tags=["WebSocket"])
manager = WebSocketManager()

@router.websocket("/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Wait for client ping messages (optional)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

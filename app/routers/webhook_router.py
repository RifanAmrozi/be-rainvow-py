from fastapi import APIRouter
from app.model.alert import Alert

router = APIRouter(prefix="/webhook", tags=["Webhook"])

@router.post("/alert")
async def receive_alert(alert: Alert):
    # Just log it for now
    print(f"🚨 Webhook received alert: {alert.dict()}")
    return {"status": "received", "alert_id": alert.id}

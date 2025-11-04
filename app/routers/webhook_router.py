from fastapi import APIRouter
from app.model.alert_schema import AlertResponse

router = APIRouter(prefix="/webhook", tags=["Webhook"])

@router.post("/alert")
async def receive_alert(alert: AlertResponse):
    # Just log it for now
    print(f"🚨 Webhook received alert: {alert.dict()}")
    return {"status": "received", "alert_id": alert.id}

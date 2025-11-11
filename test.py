from app.service.notification.apn_service import send_apn_notification

send_apn_notification(
    "4fd10beda6dff455be1cea5369f7d29344eaab86c3524810371ebf9bdaa15f98",
    {
        "title": "Test Push",
        "alert_message": "This is a test notification from FastAPI 🚀"
    }
)
from app.service.notification.apn_service import send_apn_notification

send_apn_notification(
    "e22f8b7999bac87cc788c5db87fc7b3c6fbfe0e1ce072b248941b7c35f62cb2c",
    {
        "title": "Test Push",
        "alert_message": "This is a test notification from FastAPI 🚀"
    }
)

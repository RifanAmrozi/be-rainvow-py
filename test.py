from app.service.notification import send_apn_notification

send_apn_notification(
    "83114abdcc263f212f2741ca090ca9885c84eb7e78a1403a821102e31a662637",
    {
        "title": "Test Push",
        "alert_message": "Notification from FastAPI",
        "media_url": "https://static.vecteezy.com/system/resources/thumbnails/050/037/747/small/mathematics-integral-symbol-icon-vector.jpg"
    }
)

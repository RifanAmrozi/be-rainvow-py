from app.service.notification import send_apn_notification

token =['1f6e52af5964a593c4da7459ae678aa67dbe77cb490ca8aefebec3045f9a5b23', 'b5f5438f88bb0d43a4aadcb49ba2d94b9066279253e0024dac90f91a3a0be0f9', '3072619cf601862aa1932700e3cfaa3886636cda4d97bb0575881d2214a141f4'
        ]
for i in token:
    send_apn_notification(
        i,
        {
            "title": "Test Push",
            "alert_message": "Notification from FastAPI",
            "media-url": "https://ppxajpmyuzqbnzgewzig.supabase.co/storage/v1/object/public/alert_clips/shoplifting_20251128_150430_crops-ALERT_crop.jpg",
            "photo-url": "https://ppxajpmyuzqbnzgewzig.supabase.co/storage/v1/object/public/alert_clips/shoplifting_20251128_150430_crops-ALERT_crop.jpg",
        }
    )
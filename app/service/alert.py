import asyncio
import random
import uuid
from datetime import datetime

async def send_alerts(manager, stop_flag):
    try:
        while not stop_flag.is_set():
            fake_alert = {
                "id": str(uuid.uuid4()),
                "store_id": f"store-{random.randint(1, 3)}",
                "camera_id": f"cam-{random.randint(1, 5)}",
                "timestamp": datetime.utcnow().isoformat(),
                "suspicious_activity": random.choice([True, False]),
                "alert_message": random.choice([
                    "Motion detected in restricted area",
                    "Camera disconnected",
                    "Unusual activity detected",
                    "System reboot detected"
                ]),
                "image_url": f"http://example.com/images/{random.randint(1, 5)}.jpg",
                "video_url": f"http://example.com/videos/{random.randint(1, 5)}.mp4"
            }

            await manager.broadcast(fake_alert)
            print(f"📤 Sent fake alert {fake_alert['id']}")

            await asyncio.sleep(15)

    except asyncio.CancelledError:
        print("🛑 Fake WebSocket alert loop stopped")
    finally:
        print("✅ WebSocket test stopped cleanly.")

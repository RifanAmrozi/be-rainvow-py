import asyncio
import aiohttp
import random
import uuid
from datetime import datetime
from app.core.config import settings

async def send_fake_alerts():
    url = f"http://{settings.WEBHOOK_URL}/webhook/alert"  # Change if using remote IP

    async with aiohttp.ClientSession() as session:
        while True:
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

            try:
                async with session.post(url, json=fake_alert) as resp:
                    print(f"✅ Sent fake alert {fake_alert['id']} (status: {resp.status})")
            except Exception as e:
                print(f"⚠️ Failed to send alert: {e}")

            await asyncio.sleep(60)

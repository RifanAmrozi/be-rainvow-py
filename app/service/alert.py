import asyncio
import aiohttp
import random
import uuid
from datetime import datetime
from app.core.config import settings
from app.websocket.websocket_router import manager

async def send_alerts():
    # url = f"http://{settings.WEBHOOK_URL}/webhook/alert"
    url='https://webhook.site/79d2daea-08b6-4692-87c4-8e955e576773'

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

            # 🔥 Send to connected WebSocket clients
            await manager.broadcast(fake_alert)

            # Also call webhook endpoint if you want (optional)
            try:
                async with session.post(url, json=fake_alert) as resp:
                    print(f"✅ Sent fake alert (HTTP {resp.status})")
            except Exception as e:
                print(f"⚠️ Failed to send webhook: {e}")

            await asyncio.sleep(60)

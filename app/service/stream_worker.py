import time
import datetime
import asyncio
import aiohttp
import uuid
import random
from app.service.rtsp import RtspReader
from app.client.api import ApiClient
from app.core.config import settings
from app.websocket.websocket_router import manager  # for WebSocket broadcasting


async def run_stream_worker(app):
    """
    Simulated stream worker that reads RTSP frames and periodically sends alert-like data
    to both WebSocket and an external webhook.
    """
    reader = RtspReader(settings.RTSP_DEFAULT_URL)
    if not reader.is_opened():
        print("❌ Cannot open RTSP stream")
        return

    api = ApiClient(settings.API_URL)
    frame_count = 0
    start = time.time()
    webhook_url = "https://webhook.site/e880d0cb-7b19-4b99-8099-c732a190acb1"

    print("🎥 Stream started...")

    async with aiohttp.ClientSession() as session:
        while not app.state.stop_stream_flag:
            ret, frame = reader.read_frame()
            if not ret:
                print("⚠️ Failed to grab frame, stopping...")
                break

            frame_count += 1

            # Every ~60 frames, simulate sending an alert
            if frame_count % 60 == 0:
                fps = frame_count / (time.time() - start)
                alert = {
                    "id": str(uuid.uuid4()),
                    "store_id": f"store-{random.randint(1, 3)}",
                    "camera_id": settings.CAMERA_ID,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
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

                # 🛰 Send to WebSocket
                await manager.broadcast(alert)

                # 🌐 Send to webhook (external endpoint) comment for now
                try:
                    async with session.post(webhook_url, json=alert) as resp:
                        print(f"✅ Sent alert to webhook (HTTP {resp.status}) — {alert['alert_message']}")
                except Exception as e:
                    print(f"⚠️ Failed to send webhook: {e}")

                # Also print to console for visibility
                print(f"📤 Alert sent | FPS={fps:.2f} | Total frames={frame_count}")

            await asyncio.sleep(1/30)  # Simulate ~30 FPS

    reader.release()
    print("🛑 Stream stopped.")

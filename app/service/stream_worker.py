import time
import datetime
from app.service.rtsp import RtspReader
from app.client.api import ApiClient
from app.core.config import settings
import asyncio

async def run_stream_worker(app):
    reader = RtspReader(settings.RTSP_DEFAULT_URL)
    if not reader.is_opened():
        print("❌ Cannot open RTSP stream")
        return

    api = ApiClient(settings.API_URL)
    frame_count = 0
    start = time.time()

    print("🎥 Stream started...")

    while not app.state.stop_stream_flag:
        ret, frame = reader.read_frame()
        if not ret:
            print("⚠️ Failed to grab frame, stopping...")
            break

        frame_count += 1
        if frame_count % 60 == 0:
            fps = frame_count / (time.time() - start)
            payload = {
                "camera_id": settings.CAMERA_ID,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "fps": fps,
                "frame_count": frame_count,
            }
            print("POST result:", payload)
        await asyncio.sleep(0.01)

    reader.release()
    print("🛑 Stream stopped.")

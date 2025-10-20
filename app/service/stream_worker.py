import time
import datetime
from app.service.rtsp import RtspReader
from app.client.api import ApiClient
from app.core.config import settings

def run_stream_worker():
    reader = RtspReader(settings.RTSP_URL)
    if not reader.is_opened():
        print("❌ Cannot open RTSP stream")
        return

    api = ApiClient(settings.API_URL)

    frame_count = 0
    start = time.time()

    while True:
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
            # status = api.post_stats(payload)
            print("POST result:",payload)

    reader.release()

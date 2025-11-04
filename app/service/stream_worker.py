import asyncio
import time
import cv2
import numpy as np
from collections import deque
from app.websocket.websocket_router import manager
from app.service.detection import ShopliftingPoseDetectorWithGrab
from app.service.detection import ThreadedRTSPCapture
from app.repository import alert_repository
from app.db.session import SessionLocal
import aiohttp


async def run_stream_worker(app):
    """
    Background worker that runs the shoplifting detection model,
    broadcasts WebSocket alerts, and posts webhook alerts.
    """
    print("🚀 Starting detection worker...")

    # === Initialize detector ===
    try:
        detector = ShopliftingPoseDetectorWithGrab(
            pose_model="yolo11m-pose.pt",
            debug_mode=True
        )
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return

    # === Connect to RTSP ===
    rtsp_url =  "rtsp://10.98.169.25/live/ch00_0"
    threaded_capture = ThreadedRTSPCapture(rtsp_url=rtsp_url, buffer_size=1, name="ShopliftingCam")

    try:
        threaded_capture.start()
        time.sleep(2)
        stats = threaded_capture.get_stats()
        if not stats['is_alive']:
            print("❌ Cannot start RTSP capture")
            return
    except Exception as e:
        print(f"❌ Failed to start capture: {e}")
        return

    print(f"🎥 RTSP connected at {rtsp_url}")
    detector.fps = 25
    total_alerts = 0
    frame_times = deque(maxlen=30)
    db = SessionLocal() 
    async with aiohttp.ClientSession() as session:
        try:
            while not app.state.stop_stream_flag:
                t_start = time.time()
                ret, frame = threaded_capture.read()

                if not ret or frame is None:
                    stats = threaded_capture.get_stats()
                    if stats['time_since_last_frame'] > 5.0:
                        print("⚠️ No frames received for 5s, stream may be down")
                    await asyncio.sleep(0.1)
                    continue

                frame = cv2.resize(frame, (1280, 720))
                processed, alerts = detector.process_frame(frame)

                # === If detection found, trigger alert ===
                if alerts:
                    print(alerts)
                    total_alerts += len(alerts)

                    for alert in alerts:
                        # ✅ Broadcast to websocket
                        await manager.broadcast(alert)
                        print("   🚨 Alert broadcasted via WebSocket")
                        # save to database
                        await alert_repository.insert_alert(db, alert)

                # Maintain FPS tracking
                frame_times.append(time.time() - t_start)
                current_fps = 1.0 / np.mean(frame_times) if frame_times else 0
                await asyncio.sleep(max(0, 1 / 25 - (time.time() - t_start)))

        except asyncio.CancelledError:
            print("🛑 Detection worker cancelled")
        finally:
            threaded_capture.stop()
            cv2.destroyAllWindows()
            print(f"✅ Detection worker stopped. Total alerts: {total_alerts}")

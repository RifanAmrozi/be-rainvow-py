import asyncio
import time
import cv2
import numpy as np
from collections import deque
from app.websocket.websocket_router import manager
from app.service.detection import ShopliftingPoseDetectorWithGrab
from app.service.detection import ThreadedRTSPCapture
from app.repository import alert_repository
from app.db.session import get_db, SessionLocal
from app.repository.camera import get_all_cameras
from app.repository.alert_repository import insert_alert
from app.core.config import settings
import aiohttp


async def run_stream_worker(app, camera_id: str):
    try:
        detector = ShopliftingPoseDetectorWithGrab(
            pose_model="yolo11m-pose.pt",
            debug_mode=True
        )
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    db = SessionLocal()
    camera = get_all_cameras(db, id=camera_id)
    if not camera:
        print(f"❌ Camera with id {camera_id} not found")
        return
    else:
        camera = camera[0]
    
    rtsp_url = camera.rtsp_url
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

    detector.fps = 25
    frame_times = deque(maxlen=30)
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
                for alert in alerts:
                    alert['camera_id'] = camera_id
                    alert['store_id'] = camera.store_id
                    alert['is_valid'] = None
                    await manager.broadcast(alert)
                    print(f"🚨 Alert detected: {alert}")
                    insert_alert(db, alert)

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

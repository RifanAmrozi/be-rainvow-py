import asyncio
import time
import cv2
import numpy as np
from app.service.notification.apn_service import send_apn_notification
from collections import deque
from app.websocket.websocket_router import manager
from app.service.detection import ShopliftingPoseDetectorWithGrab
from app.service.detection import ThreadedRTSPCapture
from app.service.storage import upload_video_to_supabase, get_video_public_url
from app.repository.user import get_devices
from app.db.session import SessionLocal
from app.repository.camera import get_all_cameras
from app.repository.alert_repository import insert_alert
import aiohttp
import os
from uuid import UUID
from pathlib import Path

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
                    # Upload video clip to Supabase
                    # video_filename = "shoplifting_track48_20251111_100057"+ ".mp4"
                    # base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    # video_path = os.path.join(base_dir, "alert_clips", video_filename)
                    # print("Exists:", video_path.exists())
                    # if not video_path.exists():
                    #     video_path='/Users/rifanamrozi/Documents/ada/be-rainvow-py/alert_clips/shoplifting_track48_20251111_100057.mp4'
                    # if video_path:
                    #     upload_res = upload_video_to_supabase(video_path)
                    #     print(f"Upload response: {upload_res}")
                    #     public_url = get_video_public_url(video_filename)
                    #     print(f"Public stream URL: {public_url}")
                    #     alert['video_url'] = public_url

                    # await manager.broadcast(alert)
                    print(f"🚨 Alert detected: {alert}")
                    devices = get_devices(db, user_id=None, store_id=camera.store_id)
                    for device in devices:
                        alertAPN = {k: str(v) if isinstance(v, UUID) else v for k, v in alert.items()}

                        send_apn_notification(device.device_token, alertAPN)
                    insert_alert(db, alert)

                # Maintain FPS tracking
                frame_times.append(time.time() - t_start)
                current_fps = 1.0 / np.mean(frame_times) if frame_times else 0
                await asyncio.sleep(max(0, 1 / 25 - (time.time() - t_start)))

        except asyncio.CancelledError:
            print("🛑 Detection worker cancelled")
        finally:
            print("🛑 Stopping detection worker...")
            threaded_capture.stop()
            cv2.destroyAllWindows()
            print(f"✅ Detection worker stopped. Total alerts:")

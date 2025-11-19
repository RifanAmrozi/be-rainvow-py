import asyncio
import time
import cv2
import numpy as np
from app.service.notification.apn_service import send_apn_notification
from collections import deque
from app.service.detection import ShopliftingPoseDetectorWithGrab, ThreadedRTSPCapture
from app.service.storage import get_public_url
from app.repository.user import get_devices
from app.db.session import SessionLocal, db_session
from app.repository.camera import get_all_cameras
from app.repository.alert_repository import insert_alert
from uuid import UUID


async def process_camera(app, camera):
    """Process a single camera stream asynchronously"""
    camera_id = camera.id
    rtsp_url = camera.rtsp_url
    
    print(f"🎥 Starting camera {camera_id}: {rtsp_url}")
    
    try:
        detector = ShopliftingPoseDetectorWithGrab(
            pose_model="yolo11m-pose.pt",
            debug_mode=True
        )
    except Exception as e:
        print(f"❌ Failed to initialize detector for camera {camera_id}: {e}")
        return
    
    threaded_capture = ThreadedRTSPCapture(rtsp_url=rtsp_url, buffer_size=1, name=f"Camera-{camera_id}")

    try:
        threaded_capture.start()
        await asyncio.sleep(2)  # Use async sleep
        stats = threaded_capture.get_stats()
        if not stats['is_alive']:
            print(f"❌ Cannot start RTSP capture for camera {camera_id}")
            return
    except Exception as e:
        print(f"❌ Failed to start capture for camera {camera_id}: {e}")
        return

    detector.fps = 25
    frame_times = deque(maxlen=30)
    
    try:
        while not app.state.stop_stream_flag:
            t_start = time.time()
            ret, frame = threaded_capture.read()

            if not ret or frame is None:
                stats = threaded_capture.get_stats()
                if stats['time_since_last_frame'] > 5.0:
                    print(f"⚠️ Camera {camera_id}: No frames received for 5s, stream may be down")
                await asyncio.sleep(0.1)
                continue

            frame = cv2.resize(frame, (1280, 720))
            processed, alerts = detector.process_frame(frame)
            
            for alert in alerts:
                alert['camera_id'] = camera_id
                alert['store_id'] = camera.store_id
                alert['is_valid'] = None
                video_filename = alert['video_url'] + ".mp4"
                alert["photo_url"] = get_public_url(alert["video_url"] + "_crops-ALERT_crop.jpg")
                alert["video_url"] = get_public_url(video_filename)
                print(f"🚨 Camera {camera_id} - Shoplifting alert detected:", alert)
                
                try:
                    with db_session() as db:
                        devices = get_devices(db, user_id=None, store_id=camera.store_id)
                        # insert_alert may call db.commit via db_session context, but ensure exceptions handled
                        insert_alert(db, alert)
                        # optional: send notifications after DB commit
                        for device in devices:
                            alertAPN = {k: str(v) if isinstance(v, UUID) else v for k, v in alert.items()}
                            send_apn_notification(device.device_token, alertAPN)
                except Exception as e:
                    # rollback already happened in db_session, just log and continue
                    print(f"❌ DB error handling alert for camera {camera_id}: {e}")

            # Maintain FPS
            await asyncio.sleep(max(0, 1 / 25 - (time.time() - t_start)))

    except asyncio.CancelledError:
        print(f"🛑 Camera {camera_id} - Detection worker cancelled")
    except Exception as e:
        print(f"❌ Camera {camera_id} - Error in detection loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"🛑 Camera {camera_id} - Stopping detection worker...")
        threaded_capture.stop()
        print(f"✅ Camera {camera_id} - Detection worker stopped")


async def run_stream_worker(app, store_id: str):
    """Run all cameras asynchronously"""
    db = SessionLocal()
    
    try:
        cameras = get_all_cameras(db, store_id=store_id)
        if not cameras:
            print(f"❌ No cameras found for store id {store_id}")
            return
        
        print(f"📹 Found {len(cameras)} camera(s) for store {store_id}")
        
        # Create tasks for all cameras
        camera_tasks = []
        for camera in cameras:
            task = asyncio.create_task(process_camera(app, camera))
            camera_tasks.append(task)
            print(f"✅ Started task for camera {camera.id}")
        
        # Wait for all camera tasks to complete (or be cancelled)
        try:
            await asyncio.gather(*camera_tasks, return_exceptions=True)
        except Exception as e:
            print(f"❌ Error in camera tasks: {e}")
    
    finally:
        db.close()
        print(f"✅ Stream worker for store {store_id} stopped")
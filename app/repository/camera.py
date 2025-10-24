from sqlalchemy.orm import Session
from app.model.camera import Camera
from app.model.camera_schema import CameraCreate
import socket
import yaml

def get_all_cameras(db: Session, id: str = None, name: str = None, store_id: str = None):
    query = db.query(Camera)
    if id:
        query = query.filter(Camera.id == id)
    if name:
        query = query.filter(Camera.name.ilike(f"%{name}%"))
    if store_id:
        query = query.filter(Camera.store_id == store_id)
    return query.all()



def get_local_ip() -> str:
    """Get local LAN IP (e.g., 192.168.x.x) instead of 127.0.0.1"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't have to be reachable; just used to get local IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def add_camera_to_mediamtx(name: str, rtsp_url: str, protocol: str = "tcp"):
    """
    Dynamically append a new path to mediamtx.yml
    """
    path = "mediamtx.yml"
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}

    # Ensure 'paths' section exists
    if "paths" not in data:
        data["paths"] = {}

    path_name = name.lower().replace(" ", "-")

    data["paths"][path_name] = {
        "source": rtsp_url,
        "sourceProtocol": protocol,
        "sourceOnDemand": True,
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ Added {path_name} to mediamtx.yml")


def create_camera(db: Session, camera_data: CameraCreate):
    # --- get current local IP ---
    ip_local = get_local_ip()

    # --- create WebRTC URL (for browser playback) ---
    webrtc_url = f"http://{ip_local}:8889/{camera_data.name.lower().replace(' ', '-')}/whep"

    # --- update mediamtx.yml dynamically ---
    add_camera_to_mediamtx(camera_data.name, camera_data.rtsp_url)

    # --- create DB record ---
    new_camera = Camera(
        name=camera_data.name,
        aisle_loc=camera_data.aisle_loc,
        status=camera_data.status,
        preview_img=camera_data.preview_img,
        rtsp_url=camera_data.rtsp_url,
        store_id=camera_data.store_id,
        webrtc_url=webrtc_url,
    )

    db.add(new_camera)
    db.commit()
    db.refresh(new_camera)

    return new_camera


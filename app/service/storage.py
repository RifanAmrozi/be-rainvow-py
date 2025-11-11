import os
from supabase import create_client, Client
from app.core.config import settings

supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

def upload_video_to_supabase(file_path: str, bucket_name: str = "alert_clips"):
    file_name = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        file_data = f.read()

    try:
        # Upload file to Supabase Storage
        res = supabase.storage.from_(bucket_name).upload(
            path=file_name,
            file=file_data,
            file_options={"content-type": "video/mp4"}  # ⚠️ no 'upsert' here
        )
        print("✅ Upload success:", res)
        return res
    except Exception as e:
        print("❌ Upload failed:", e)


def get_video_public_url(file_name: str) -> str:
    """
    Return direct streamable URL to the video in Supabase Storage.
    """
    public_url = supabase.storage.from_("alert_clips").get_public_url(file_name)
    print(f"🎥 Public stream URL: {public_url}")
    return public_url

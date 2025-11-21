import os
import requests
from app.core.config import settings
from pathlib import Path

def upload_video_to_supabase(file_path: str, bucket_name: str = "alert_clips"):
    """Upload video to Supabase Storage using requests."""
    file_name = os.path.basename(file_path)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        url = f"{settings.SUPABASE_URL}/storage/v1/object/{bucket_name}/{file_name}"
        headers = {
            "Authorization": f"Bearer {settings.SUPABASE_KEY}",
            "Content-Type": "video/mp4"
        }
        
        print(f"📤 Uploading video: {file_name} ({len(file_data)} bytes)")
        
        response = requests.post(url, headers=headers, data=file_data)
        
        if response.status_code in [200, 201]:
            print("✅ Video upload success")
            public_url = f"{settings.SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{file_name}"
            return {"path": file_name, "url": public_url}
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Video upload failed: {e}")
        return None

def upload_photo_to_supabase(file_path: str, bucket_name: str = "alert_clips", retries: int = 3):

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    # Convert path to flattened filename
    # Example: /path/to/alert_clips/shoplifting_track1_20251113_140501_crops/ALERT_crop.jpg
    # -> shoplifting_track1_20251113_140501_crops-ALERT_crop.jpg
    path_obj = Path(file_path)
    parent_folder = path_obj.parent.name  # Get immediate parent folder name
    original_filename = path_obj.name
    
    # Combine parent folder and filename with hyphen
    flattened_filename = f"{parent_folder}-{original_filename}"
    
    print(f"📝 Original path: {file_path}")
    print(f"📝 Flattened filename: {flattened_filename}")
    
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return None
    
    url = f"{settings.SUPABASE_URL}/storage/v1/object/{bucket_name}/{flattened_filename}"
    headers = {
        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
        "Content-Type": "image/jpeg"
    }
    
    for attempt in range(retries):
        try:
            print(f"📤 Uploading photo: {flattened_filename} ({len(file_data)} bytes) - attempt {attempt + 1}/{retries}")
            
            response = requests.post(url, headers=headers, data=file_data)
            
            if response.status_code in [200, 201]:
                print("✅ Photo upload success")
                public_url = f"{settings.SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{flattened_filename}"
                return {"path": flattened_filename, "url": public_url}
            elif response.status_code == 409:  # File exists, try upsert
                print("⚠️ File exists, trying upsert...")
                put_response = requests.put(url, headers=headers, data=file_data)
                if put_response.status_code == 200:
                    print("✅ Photo upserted successfully")
                    public_url = f"{settings.SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{flattened_filename}"
                    return {"path": flattened_filename, "url": public_url}
                else:
                    print(f"❌ Upsert failed: {put_response.status_code} - {put_response.text}")
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Photo upload failed (attempt {attempt + 1}/{retries}): {e}")
            
        if attempt < retries - 1:
            print(f"⏳ Retrying in 2 seconds...")
            import time
            time.sleep(2)
    
    return None

def get_video_public_url(file_name: str, bucket_name: str = "alert_clips") -> str:
    """Get public URL for video."""
    return f"{settings.SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{file_name}"


def get_public_url(file_name: str, bucket_name: str = "alert_clips") -> str:
    return f"{settings.SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{file_name}"
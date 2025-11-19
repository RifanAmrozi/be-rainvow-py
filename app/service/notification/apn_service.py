import tempfile
import os
from apns2.client import APNsClient
from apns2.credentials import TokenCredentials
from apns2.payload import Payload
from app.core.config import settings
import json
import traceback

def get_apns_client():
    key_path = settings.APN_KEY_PATH
    key_string = getattr(settings, "APN_KEY_STRING", None)

    if (not key_path or not os.path.exists(key_path)) and key_string:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".p8")
        tmp.write(key_string.encode("utf-8"))
        tmp.flush()
        tmp.close()
        key_path = tmp.name
        print(f"🧩 Temporary APN key file created: {key_path}")

    # 🧠 Sanity check
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"APN key file not found at: {key_path}")

    print(f"🔐 Using APN key: {key_path}")
    print(f"   Key ID: {settings.APN_KEY_ID}, Team ID: {settings.APN_TEAM_ID}")

    # ✅ Create token credentials safely
    credentials = TokenCredentials(
        auth_key_path=key_path,
        auth_key_id=settings.APN_KEY_ID,
        team_id=settings.APN_TEAM_ID,
    )

    return APNsClient(
        credentials=credentials,
        use_sandbox=settings.APN_USE_SANDBOX,
        use_alternative_port=False,
    )


def send_apn_notification(device_token: str, alert_data: dict):
    try:
        safe_alert_data = json.loads(
            json.dumps(alert_data, default=str)
        )

        media_url = safe_alert_data.pop("media_url", None)
        payload = Payload(
            alert={
                "title": safe_alert_data.get("title", "Shoplifting Alert"),
                "body": safe_alert_data.get("alert_message", "Suspicious activity detected."),
            },
            sound="alert.wav",
            # pass media URL in custom payload key (Notification Service Extension must know this key)
            custom={"alert_data": safe_alert_data, **({"media-url": media_url} if media_url else {})},
            mutable_content=True
        )
        print("Payload:", payload.dict())

        topic = settings.APN_BUNDLE_ID
        client = get_apns_client()
        response = client.send_notification(device_token, payload, topic)

    except Exception as e:
        print("❌ Failed to send APN:")
        traceback.print_exc()
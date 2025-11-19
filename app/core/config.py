from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os


class Settings(BaseSettings):
    # Database
    DB_HOST: str = Field("localhost")
    DB_PORT: int = Field(5432)
    DB_USER: str = Field("postgres")
    DB_PASSWORD: str = Field("postgres")
    DB_NAME: str = Field("postgres")

    # RTSP
    RTSP_DEFAULT_URL: str = Field("")
    RTSP_URL: str = "rtsp://10.63.47.25/live/ch00_0"
    API_URL: str = "http://172/20.10.6:8080/api/v1/save"
    CAMERA_ID: str = os.getenv("CAMERA_ID")
    STORE_ID: str = os.getenv("STORE_ID")

    # App
    APP_HOST: str = Field("0.0.0.0")
    APP_PORT: int = Field(8000)
    WEBHOOK_URL: str = Field("localhost:3000/webhook/alert")

    # Tuning
    STREAM_WORKERS: int = Field(2)

    # JWT / Auth
    SECRET_KEY: str = Field("defaultsecret")
    ALGORITHM: str = Field("HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30)

    # APN
    APN_KEY_PATH: str = os.getenv("APN_KEY_PATH")
    APN_KEY_ID: str = os.getenv("APN_KEY_ID")
    APN_TEAM_ID: str = os.getenv("APN_TEAM_ID")
    APN_BUNDLE_ID: str = os.getenv("APN_BUNDLE_ID")
    APN_USE_SANDBOX: bool = os.getenv("APN_USE_SANDBOX", "True").lower() == "true"

    # SUPABASE
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")
    SUPABASE_BUCKET: str = os.getenv("SUPABASE_BUCKET", "alert_clips")


    # 👇 this replaces your old `class Config`
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )


settings = Settings()

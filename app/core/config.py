from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Database (kept here for completeness)
    DB_HOST: str = Field("localhost")
    DB_PORT: int = Field(5432)
    DB_USER: str = Field("postgres")
    DB_PASSWORD: str = Field("postgres")
    DB_NAME: str = Field("postgres")

    # RTSP
    RTSP_DEFAULT_URL: str = Field("")
    RTSP_URL: str = "http://172.20.10.5:8889/de1/whep"
    API_URL: str = "http://172/20.10.6:8080/api/v1/save"
    CAMERA_ID: str = "cam-01"

    # App
    APP_HOST: str = Field("0.0.0.0")
    APP_PORT: int = Field(8000)

    # Tuning
    STREAM_WORKERS: int = Field(2)


class Config:
    env_file = ".env"
    env_file_encoding = "utf-8"


settings = Settings()
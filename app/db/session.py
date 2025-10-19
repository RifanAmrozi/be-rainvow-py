from sqlalchemy import create_engine, text
from app.core.config import settings


def get_database_url():
    """Build a full SQLAlchemy URL from granular env settings."""
    return (
        f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )


# Create engine from your granular fields
engine = create_engine(get_database_url())


def test_connection():
    """Try connecting to the database once at startup."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful!")
    except Exception as e:
        print("❌ Database connection failed!")
        print("Error:", e)

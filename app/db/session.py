from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.config import settings


def get_database_url():
    """Build a full SQLAlchemy URL from granular env settings."""
    return (
        f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        "?options=-csearch_path%3Dpublic"
    )


engine = create_engine(get_database_url())
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection():
    try:
        url = get_database_url()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT current_database(), current_user, inet_server_addr();"))
        print("✅ Database connection successful!")
    except Exception as e:
        print("❌ Database connection failed!")
        print("Error:", e)

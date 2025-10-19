from fastapi import FastAPI
from app.core.config import settings
from app.usecase.stream_worker import run_stream_worker
from app.db.session import test_connection

app = FastAPI(title="Backend Server")

@app.on_event("startup")
def startup_event():
    print("🚀 Starting server...")
    test_connection()
    
@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.get("/ping")
def ping():
    return {"ping": "pong"}

@app.get("/start")
def start_stream():
    # WARNING: blocking loop for now (for demo)
    run_stream_worker()
    return {"status": "started"}
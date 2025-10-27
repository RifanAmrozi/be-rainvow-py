```
# FastAPI RTSP Proxy — Clean Architecture Skeleton


## Run locally
1. Copy `.env.example` -> `.env` and fill values
2. python3 -m venv .venv && source .venv/bin/activate
3. pip3 install -r requirements.txt
4. uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload --reload-dir app
5. Open http://localhost:8000/api/stream?url=<your-rtsp-url>

## Run MediamMTX
./mediamtx mediamtx.yml

## Find Local IP
ipconfig getifaddr en0


Notes:
- This skeleton uses OpenCV VideoCapture. Many cameras require proper credentials and network access.
- For production consider: GStreamer/FFmpeg, process supervision, workers per camera, authentication, HTTPS, and rate limiting.
```


---


## Next steps & suggestions
- Add authentication (API key / OAuth) for the streaming endpoint.
- Add health checks and a reconnect strategy for flaky RTSP sources.
- Consider using a proper media server or WebRTC for low-latency browser playback.
- Add unit tests and CI.
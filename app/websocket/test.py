import websockets
import asyncio

async def test_ws():
    uri = "ws://localhost:3000/ws/alerts"
    async with websockets.connect(uri) as ws:
        print("✅ Connected!")
        await ws.send("hello server")
        while True:
            msg = await ws.recv()
            print("📩 Message:", msg)

asyncio.run(test_ws())

import asyncio
import websockets
import torch
import torchaudio
from utils import generate_speech, transcribe_audio, generate_response
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse("""
    <html>
        <body>
            <h1>Deva Voice Agent</h1>
            <button onclick="startRecording()">Start Talking</button>
            <script>
                async function startRecording() {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    const recorder = new MediaRecorder(stream);
                    recorder.ondataavailable = e => {
                        const ws = new WebSocket("ws://localhost:8000/ws");
                        ws.onopen = () => ws.send(e.data);
                        ws.onmessage = msg => {
                            const audio = new Audio(URL.createObjectURL(new Blob([msg.data])));
                            audio.play();
                        };
                    };
                    recorder.start();
                }
            </script>
        </body>
    </html>
    """)

async def handle_ws(websocket, path):
    async for message in websocket:
        # Save incoming audio
        with open("input.wav", "wb") as f:
            f.write(message)
        text = transcribe_audio("input.wav")
        response = generate_response(f"Respond empathetically: {text}")
        audio = generate_speech(response)
        torchaudio.save("output.wav", audio, 24000)
        with open("output.wav", "rb") as f:
            await websocket.send(f.read())

start_server = websockets.serve(handle_ws, "0.0.0.0", 8000)

asyncio.get_event_loop().run_until_complete(start_server)
uvicorn.run(app, host="0.0.0.0", port=8000)

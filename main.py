from faster_whisper import WhisperModel
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional
import asyncio
from ts_manager import TranscriptionManager, TranscriptionJob

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

"""
    constants
"""
CHUNK_DURATION_S=10
SAMPLE_RATE=16000
BYTES_PER_SAMPLE=2
FLUSH_SIZE=SAMPLE_RATE*BYTES_PER_SAMPLE*CHUNK_DURATION_S


""" 
    global manager for managing jobs right now everything is defaulting to default value  
"""
manager = TranscriptionManager()


@app.on_event("startup")
async def startup():
    await manager.start_workers()

@app.on_event("shutdown")
async def shutdown():
    manager.executor.shutdown(
        wait=True,
        cancel_futures=True
    )


@app.websocket("/ws")
async def audio_ws(websocket:WebSocket):
    await websocket.accept()
    
    """will soon be deprecated and we will use the hypothesis buffer"""
    clientbuffer = bytearray()
    
    try:
        while True:
            chunk = await websocket.receive_bytes()
            clientbuffer.extend(chunk)
            
            if len(clientbuffer) >= FLUSH_SIZE:
                audio_bytes = bytes(clientbuffer)
                clientbuffer.clear()
                
                job = TranscriptionJob(
                        audio_buffer=audio_bytes,
                        websocket=websocket
                )
                await manager.submit_job(job)

    except WebSocketDisconnect as e:
        print("Client disconnected")
        if len(clientbuffer) > SAMPLE_RATE: 
            # pipeline transcribe
            job = TranscriptionJob(
                    audio_buffer=bytes(clientbuffer),
                    websocket=websocket
            )
            await manager.submit_job(job)
from faster_whisper import WhisperModel
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional
import asyncio
from ts_manager import TranscriptionManager, TranscriptionJob, SessionTJob
import uuid
import json 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:8443",
        "https://127.0.0.1:8443",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_origin_regex=r"https://.*\.noustalk\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


"""
    @deprecated will be removing this as soon as possible 
"""

#@app.get("/")
#def home():
#    return FileResponse("static/index.html")



"""
    constants
"""
CHUNK_DURATION_S=5
SAMPLE_RATE=16000
BYTES_PER_SAMPLE=2
FLUSH_SIZE=SAMPLE_RATE*BYTES_PER_SAMPLE*CHUNK_DURATION_S


""" 
    global manager for managing jobs right now everything is defaulting to default value  
"""
manager = TranscriptionManager(device="cuda", compute_type="float16",condition_on_previous_text=False)


"""
    health check for amazon load balancer for checking the system load 
"""
@app.get('/health')
def health():
    qsize = manager.job_queue.qsize()
    if qsize > 30:
       return Response(status_code=503)
    return {
            "status": "ok",
            "queue_size": qsize,
            "queue_max": manager.job_queue.maxsize,
    }



@app.on_event("startup")
async def startup():
    await manager.start_workers()

@app.on_event("shutdown")
async def shutdown():
    manager.executor.shutdown(
        wait=True,
        cancel_futures=True
    )

@app.post("/transcriptions")
async def upload_audio(request: Request):
    pcm_bytes = await request.body()
    job_id = str(uuid.uuid4())

    manager.create_http_job(job_id)

    await manager.submit_job(SessionTJob(
        job_id = job_id,
        audio_buffer=pcm_bytes
    ))
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/transcriptions/{job_id}")
async def get_job_status(job_id: str):
    job = manager.jobs.get(job_id)
    
    if not job:
        return {"status": "not found"}
    
    return job

@app.get("/transcription/{job_id}/events")
async def events(job_id: str):

    async def stream():
        queue = manager.job_events[job_id]

        while True:
            event = await queue.get()
            yield (f"data: {json.dumps(event)}\n\n")

            if event["type"] == "completed":
                break

    return StreamingResponse(stream(), media_type="text/event-stream")


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



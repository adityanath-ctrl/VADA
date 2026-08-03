from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, UploadFile,File
from contextlib import asynccontextmanager
from dataclasses import dataclass
import numpy as np
from enginev1.prkt import BatchedTranscriptionEngine
import asyncio
import torch
from enginev1.ds_pipeline import DiarizationPipeLine
from workers.streaming_worker import StreamingWorker
from workers.long_job_worker import LongJobWorker
import uuid
import os
from cache.redis import redis
import json
import time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse


# GLOBAL CONSTANTS 
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_DURATION  =5
TRANSCRIBE_BYTES = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * CHUNK_DURATION
UPLOADS_DIR="uploads"

@dataclass
class AudioJob:
    audio: np.ndarray
    ws: WebSocket

"""
    -- Worker Task function 
"""
audio_queue: asyncio.Queue = asyncio.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading parakeet ....")

    model = BatchedTranscriptionEngine()
    diarizer = DiarizationPipeLine()
    streaming_worker = StreamingWorker(model,audio_queue)
    long_job = LongJobWorker(redis,model,diarizer)

    worker_task = asyncio.create_task(streaming_worker.run())
    long_task = asyncio.create_task(long_job.run())
    
    app.state.model = model
    app.state.diarizer = diarizer
    app.state.worker_task = worker_task
    app.state.streaming_worker = streaming_worker


    print("Model loaded....")
    yield 

    print("shutting down")
    worker_task.cancel()
    long_task.cancel()

    try:
        await worker_task
        await long_task
    except asyncio.CancelledError:
        pass

    del app.state.model
    del app.state.diarizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




app = FastAPI(title="noustalk stt version 2", lifespan = lifespan)



app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
   return FileResponse("static/index.html")


@app.get("/")
async def root():
    return {"status": "running", "service": "transcription"}


@app.post("/jobs/transcribe")
async def create_transcription_job(
    file: UploadFile = File(...)
):

    if not file.content_type.startswith("audio"):
        raise HTTPException(
            status_code=400,
            detail="Only audio files allowed"
        )

    job_id = str(uuid.uuid4())

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    file_path = (
        f"{UPLOAD_DIR}/{job_id}_{file.filename}"
    )

    with open(file_path, "wb") as f:
        while chunk := await file.read(
            1024 * 1024
        ):
            f.write(chunk)


    await redis.hset(
        f"job:{job_id}",
        mapping={
            "job_id": job_id,
            "status": "queued",
            "stage": "waiting",
            "progress": "0",
            "path": file_path,
        }
    )

    await redis.lpush(
        "transcription_jobs",
        job_id
    )


    return {
        "job_id": job_id,
        "status": "queued"
    }


@app.get("/health")
async def h():
    return {"success": True}


# status endpoint 
@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):

    job = await redis.hgetall(f"job:{job_id}")

    if not job:
        raise HTTPException(
            404,
            "job not found"
        )

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "stage": job.get("stage"),
        "progress": int(job.get("progress",0))
    }


@app.get("/jobs/{job_id}/result")
async def get_result(job_id: str):

    result = await redis.get(f"job:{job_id}:result")

    if not result:
        raise HTTPException(404,"Result not ready")

    return json.loads(result)


@app.websocket("/ws")
async def web(ws: WebSocket):
    await ws.accept()

    buffer = bytearray()

    try:
        while True:
           
            start = time.perf_counter()
            chunk = await ws.receive_bytes()
            end = time.perf_counter()
            print(f"[receive time] {end - start}")
            buffer.extend(chunk)
    

            if len(buffer) % 2 != 0:
                buffer = buffer[:-1]

            while len(buffer) >= TRANSCRIBE_BYTES:
                pcm = bytes(buffer[:TRANSCRIBE_BYTES])
                del buffer[:TRANSCRIBE_BYTES]

                audio = np.frombuffer(pcm,dtype=np.int16).astype(np.float32) / 32768.0

                if audio.std() < 0.005:
                    print("skipping silent chunks")
                    continue

                await app.state.streaming_worker.submit(
                    AudioJob(
                        audio=audio,
                        ws=ws,
                        )
                )

    except WebSocketDisconnect:
        pass

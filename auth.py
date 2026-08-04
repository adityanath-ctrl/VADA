from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, UploadFile, File, Security, Depends
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import numpy as np
from enginev1.prkt import BatchedTranscriptionEngine
import asyncio
import torch
from enginev1.ds_pipeline import DiarizationPipeLine
from workers.streaming_worker import StreamingWorker, AudioJob
from workers.long_job_worker import LongJobWorker
import uuid
import os
import secrets
from cache.redis import redis
import json
import time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse


load_dotenv()

# GLOBAL CONSTANTS
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_DURATION = 5
TRANSCRIBE_BYTES = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * CHUNK_DURATION
UPLOADS_DIR = os.environ.get("UPLOADS_DIR")

"""
    -- API Key auth
"""
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Comma-separated list of valid keys, e.g. API_KEYS="key1,key2,key3"
VALID_API_KEYS = set(
    k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()
)

if not VALID_API_KEYS:
    print("WARNING: No API_KEYS set in environment. All authenticated routes will reject every request.")


def _key_is_valid(key: str | None) -> bool:
    if not key:
        return False
    return any(secrets.compare_digest(key, valid) for valid in VALID_API_KEYS)


def verify_api_key(key: str | None = Security(api_key_header)):
    if not _key_is_valid(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


"""
    -- Worker Task function
"""
audio_queue: asyncio.Queue = asyncio.Queue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading parakeet ....")

    model = BatchedTranscriptionEngine()
    diarizer = DiarizationPipeLine()

    streaming_worker = StreamingWorker(model, audio_queue)

    long_job = LongJobWorker(redis, streaming_worker, diarizer)

    worker_task = asyncio.create_task(streaming_worker.run())
    long_task = asyncio.create_task(long_job.run())

    app.state.model = model
    app.state.diarizer = diarizer
    app.state.worker_task = worker_task
    app.state.long_task = long_task
    app.state.streaming_worker = streaming_worker

    print("Model loaded....")
    yield

    print("shutting down")
    worker_task.cancel()
    long_task.cancel()

    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    try:
        await long_task
    except asyncio.CancelledError:
        pass

    del app.state.model
    del app.state.diarizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="noustalk stt version 2", lifespan=lifespan)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/record_session")
def wav():
    return FileResponse("static/transcribe_upload.html")

@app.get("/health")
async def h():
    return {"success": True}


@app.post("/jobs/transcribe")
async def create_transcription_job(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):

    if not file.content_type.startswith("audio"):
        raise HTTPException(
            status_code=400,
            detail="Only audio files allowed"
        )

    job_id = str(uuid.uuid4())

    os.makedirs(UPLOADS_DIR, exist_ok=True)

    file_path = f"{UPLOADS_DIR}/{job_id}_{file.filename}"

    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
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

    await redis.lpush("transcription_jobs", job_id)

    return {
        "job_id": job_id,
        "status": "queued"
    }


# status endpoint
@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, api_key: str = Depends(verify_api_key)):

    job = await redis.hgetall(f"job:{job_id}")

    if not job:
        raise HTTPException(404, "job not found")

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "stage": job.get("stage"),
        "progress": int(job.get("progress", 0))
    }


@app.get("/jobs/{job_id}/result")
async def get_result(job_id: str, api_key: str = Depends(verify_api_key)):

    result = await redis.get(f"job:{job_id}:result")

    if not result:
        raise HTTPException(404, "Result not ready")

    return json.loads(result)


@app.websocket("/ws")
async def web(ws: WebSocket):
    api_key = ws.query_params.get("api_key")

    if not _key_is_valid(api_key):
        await ws.close(code=1008)
        return

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

                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

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
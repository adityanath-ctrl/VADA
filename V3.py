# A python single file websocket endpoint for testing
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from nemo.collections.asr.models import ASRModel
from omegaconf import open_dict

from contextlib import asynccontextmanager
from dataclasses import dataclass
import torch
import numpy as np
import asyncio

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_DURATION = 5
TRANSCRIBE_BYTES = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * CHUNK_DURATION

@dataclass
class AudioJob:
    audio: np.ndarray
    ws: WebSocket

audio_queue: asyncio.Queue = asyncio.Queue()
model_lock = asyncio.Lock()


def run_transcribe(model, audios,batch_size):
	with torch.inference_mode(), torch.autocast(device_type='cuda',dtype=torch.bfloat16):
		return model.transcribe(audio=audios,batch_size=batch_size)

async def transcription_worker(model):
    while True:
        batch: list[AudioJob] = []

        job = await audio_queue.get()
        batch.append(job)

        queue_size = audio_queue.qsize()
        if queue_size == 0:
            max_batch = 1
        elif queue_size < 5:
            max_batch = 3
        elif queue_size < 15:
            max_batch = 6
        else:
            max_batch = 10

        try:
            while len(batch) < max_batch:
                job = await asyncio.wait_for(audio_queue.get(), timeout=0.15)
                batch.append(job)
        except asyncio.TimeoutError:
            pass

        audios = [j.audio for j in batch]
        print(f"[Worker] Batch size: {len(batch)} | Queue left: {audio_queue.qsize()}")

        try:
            async with model_lock:
            	results = await asyncio.to_thread(
            		run_transcribe,
            		model,
            		audios,
            		len(audios)
            	)

            for job, hyp in zip(batch, results):

                try:
                    text = hyp.text if hyp.text is not None else ""
                    
                    await job.ws.send_json({
                        "type": "transcript",
                        "text": text
                    })

                except Exception:
                    pass

        except Exception as e:
            print(f"[Worker] Transcription error: {e}")

        for _ in batch:
            audio_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Parakeet...")
    model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

    if torch.cuda.is_available():
        model = model.to(device="cuda")
        print(f"Allocated : {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"Reserved  : {torch.cuda.memory_reserved()/1024**3:.2f} GB")


    model.eval()

    # Start background worker
    worker_task = asyncio.create_task(transcription_worker(model))
    app.state.model = model
    app.state.worker_task = worker_task

    print("Model loaded.")
    yield

    print("Shutting down...")
    worker_task.cancel()
    
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    del app.state.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="Parakeet transcription API", lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "running", "service": "transcription"}

@app.websocket("/ws")
async def web(ws: WebSocket):
    await ws.accept()
    buffer = bytearray()

    try:
        while True:
            chunk = await ws.receive_bytes()
            buffer.extend(chunk)

            if len(buffer) % 2 != 0:
                buffer = buffer[:-1]

            while len(buffer) >= TRANSCRIBE_BYTES:
                pcm = bytes(buffer[:TRANSCRIBE_BYTES])
                del buffer[:TRANSCRIBE_BYTES]

                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

                if audio.std() < 0.005:
                    print("Skipping silent chunk")
                    continue

                await audio_queue.put(AudioJob(audio=audio, ws=ws))

    except WebSocketDisconnect:
        pass
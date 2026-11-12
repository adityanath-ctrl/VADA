from faster_whisper import WhisperModel
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional
import asyncio
from ts_manager import TranscriptionManager,DiarizationManager, TranscriptionJob, AudioJob, DiarizationJob
import uuid
import json 
from fastapi.middleware.cors import CORSMiddleware
from metrics.system import CPU, GPU
import soundfile as sf
import tempfile
import os 
from contextlib import asynccontextmanager
from cache.redis import redis
import time
from dataclasses import dataclass
import httpx
from pydantic import BaseModel, HttpUrl
from nemo.collections.asr.models import ASRModel
import wave
import json
import os
import httpx


@dataclass
class AAudioJob:
    audio: np.ndarray
    ws: WebSocket
    chunk_start_time: float
    is_close: bool = False

@dataclass
class UtteranceJob:
    audio: np.ndarray
    ws: WebSocket
    utterance_id: int


# important constants begin


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16 = 2 bytes

CHUNK_DURATION = 8
OVERLAP_DURATION = 4
STEP_DURATION = CHUNK_DURATION - OVERLAP_DURATION  

CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION       
OVERLAP_SAMPLES = SAMPLE_RATE * OVERLAP_DURATION   
STEP_SAMPLES = SAMPLE_RATE * STEP_DURATION         

TRANSCRIBE_BYTES = (
    SAMPLE_RATE
    * CHANNELS
    * SAMPLE_WIDTH
    * CHUNK_DURATION
)

STEP_BYTES = (
    SAMPLE_RATE
    * CHANNELS
    * SAMPLE_WIDTH
    * STEP_DURATION
)
COMPLETION_QUEUE = "transcription:completion"

### ----- constant's for utterance method  ----- ###
UTT_MAX_SAMPLES = 30 * SAMPLE_RATE
UTT_MIN_SAMPLES = int(0.2 * SAMPLE_RATE)
UTT_BATCH_MAX = 8
UTT_BATCH_WAIT = 0.05
### ----- constant's for utterance method  ----- ###


# important constants end



manager = TranscriptionManager(device="cuda", compute_type="int8_float16",condition_on_previous_text=False)
dmanager = DiarizationManager()
audio_queue = asyncio.Queue()
utterance_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
model_lock = asyncio.Lock()

from contextlib import asynccontextmanager
import asyncio
import torch
from fastapi import FastAPI


def run_transcribe(model, audios,batch_size):
	with torch.inference_mode(), torch.autocast(device_type='cuda',dtype=torch.bfloat16):
		return model.transcribe(audio=audios,batch_size=batch_size, timestamps=True)

async def safe_send(ws: WebSocket, payload: dict):
    try:
        await ws.send_json(payload)
    except Exception:
        pass  # client already gone


async def complete_transcription(job_id: str, job: dict):
    try:
        words_json = await redis.get(
            f"job:{job_id}:words"
        )

        speakers_json = await redis.get(
            f"job:{job_id}:diar"
        )

        if not words_json:
            raise RuntimeError(
                f"ASR result missing for job {job_id}"
            )

        if not speakers_json:
            raise RuntimeError(
                f"Diarization result missing for job {job_id}"
            )

        words = json.loads(words_json)
        speaker_turns = json.loads(speakers_json)

        merged = merge_words_with_speakers(
            words,
            speaker_turns
        )

        transcript = build_text_with_speakers(
            merged
        )

        speakers = list({
            word["speaker"]
            for word in merged
            if word.get("speaker") is not None
        })

        callback_url = job.get("callback_url")

        if not callback_url:
            raise RuntimeError(
                f"Callback URL missing for job {job_id}"
            )

        payload = {
            "job_id": job_id,
            "status": "completed",
            "transcript": transcript,
            "words": merged,
            "speakers": speakers
        }


        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                callback_url,
                json=payload
            )
            
            if response.status_code >= 400:
                print(f"[ERROR] Callback {response.status_code}: {response.text}")

            response.raise_for_status()

        await redis.hset(
            f"job:{job_id}",
            "status",
            "completed"
        )

        audio_path = job.get("audio_path")

        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        await redis.delete(
            f"job:{job_id}:words",
            f"job:{job_id}:diar"
        )

    except Exception as e:
        print(
            f"[ERROR] Failed to complete transcription "
            f"{job_id}: {e}"
        )

        await redis.hset(
            f"job:{job_id}",
            mapping={
                "status": "failed",
                "error": str(e)
            }
        )


async def completion_worker():
    while True:
        try:
            job_id = await redis.lpop(COMPLETION_QUEUE)

            if job_id is None:
                await asyncio.sleep(0.5)
                continue

            print("[COMPLETION WORKER] popped:", job_id)

            job = await redis.hgetall(f"job:{job_id}")

            if not job:
                continue

            asr_done = job.get("asr_status") == "done"
            diar_done = job.get("diar_status") == "done"

            if not (asr_done and diar_done):
                continue

            print(
                "[INFO] ASR + diarization completed:",
                job_id
            )

            await complete_transcription(job_id, job)

        except Exception:
            import traceback
            traceback.print_exc()
            await asyncio.sleep(1)



async def transcription_worker(model):
    committed_state = {}

    while True:
        batch: list[AAudioJob] = []

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

        real_jobs = []
        for j in batch:
            if j.is_close:
                committed_state.pop(j.ws, None)
                print(f"[Worker] Cleaned up committed_state for closed ws")
            else:
                real_jobs.append(j)

        if not real_jobs:
            for _ in batch:
                audio_queue.task_done()
            continue

        audios = [j.audio for j in real_jobs]
        print(f"[Worker] Batch size: {len(real_jobs)} | Queue left: {audio_queue.qsize()}")

        try:
            async with model_lock:
                results = await asyncio.to_thread(
                    run_transcribe,
                    model,
                    audios,
                    len(audios)
                )

            for job, hyp in zip(real_jobs, results):
                try:
                    words = hyp.timestamp.get('word', []) if hyp.timestamp else []
                    committed_until = committed_state.get(job.ws, 0.0)
                    safe_boundary = job.chunk_start_time + STEP_DURATION

                    new_words = []
                    for w in words:
                        g_start = job.chunk_start_time + w["start"]
                        g_end = job.chunk_start_time + w["end"]
                        if g_start >= committed_until and g_start < safe_boundary:
                            new_words.append(w["word"])
                            committed_until = g_end

                    committed_state[job.ws] = max(committed_state.get(job.ws, 0.0), committed_until)

                    if new_words:
                        await job.ws.send_json({
                            "type": "transcript",
                            "text": " ".join(new_words)
                        })

                except Exception:
                    pass

        except Exception as e:
            print(f"[Worker] Transcription error: {e}")

        for _ in batch:
            audio_queue.task_done()


async def utterance_worker(model):
    while True:
        batch: list[UtteranceJob] = [await utterance_queue.get()]
        try:
            while len(batch) < UTT_BATCH_MAX:
                batch.append(
                    await asyncio.wait_for(utterance_queue.get(), timeout=UTT_BATCH_WAIT)
                )
        except asyncio.TimeoutError:
            pass

        try:
            async with model_lock:
                results = await asyncio.to_thread(
                    run_transcribe_text, model, [j.audio for j in batch], len(batch)
                )
            for job, hyp in zip(batch, results):
                text = (hyp.text if hasattr(hyp, "text") else str(hyp)) or ""
                await safe_send(job.ws, {
                    "type": "transcript",
                    "id": job.utterance_id,
                    "text": text.strip(),
                    "isFinal": True,
                })
        except Exception as e:
            print(f"[Utterance worker] error: {e}")
            for job in batch:
                await safe_send(job.ws, {
                    "type": "transcript",
                    "id": job.utterance_id,
                    "text": "",
                    "isFinal": True,
                    "error": True,
                })
        finally:
            for _ in batch:
                utterance_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading parakeet ....", flush=True)
    model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

    if torch.cuda.is_available():
        model = model.to(device="cuda")
        print(f"Allocated : {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"Reserved  : {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    model.eval()

    
    worker_task = asyncio.create_task(transcription_worker(model))
    app.state.model = model
    app.state.worker_task = worker_task
    completion_task = asyncio.create_task(
        completion_worker()
    )


    await manager.start_workers()

    print("All workers started....", flush=True)


    try:
        yield

    finally:
        print("Shutting down....", flush=True)

        worker_task.cancel()
        completion_task.cancel()
        utterance_task.cancel()   
        await asyncio.gather(
            worker_task, utterance_task, completion_task, return_exceptions=True
        )
        
        try:
            await worker_task
            await completion_task
        except asyncio.CancelledError:
            pass


        for w in manager.workers:
            w.cancel()

        del app.state.model
        del app.state.worker_task

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Shutdown complete", flush=True)



app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

def build_text_with_speakers(words: list) -> str:
    if not words:
        return ""

    lines           = []
    current_speaker = words[0]["speaker"]
    current_start   = words[0]["start"]
    current_words   = [words[0]["word"]]

    for w in words[1:]:
        if w["speaker"] == current_speaker:
            current_words.append(w["word"])
        else:
            lines.append(f"{current_speaker} [{current_start:.2f}s]: {' '.join(current_words)}")
            current_speaker = w["speaker"]
            current_start   = w["start"]
            current_words   = [w["word"]]

    lines.append(f"{current_speaker} [{current_start:.2f}s]: {' '.join(current_words)}")
    return "\n\n".join(lines)


def merge_words_with_speakers(words: list, speaker_turns: list) -> list:
    turns = sorted(speaker_turns, key=lambda t: t["start"])
    result = []

    for word in words:
        w_start, w_end = word["start"], word["end"]

        best_turn = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(w_end, turn["end"]) - max(w_start, turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = turn

        if best_turn is None:
            best_turn = min(
                turns,
                key=lambda t: min(abs(t["start"] - w_end), abs(w_start - t["end"])),
            )

        speaker = best_turn["speaker"] if best_turn else "UNKNOWN"
        result.append({**word, "speaker": speaker})

    return result


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:8443",
        "https://127.0.0.1:8443",
        "http://localhost:5173",
        "http://localhost:3000",
        "https://localhost:8000"
    ],
    allow_origin_regex=r"https://.*\.noustalk\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_text_with_speakers(words: list) -> str:
    if not words:
        return ""

    lines           = []
    current_speaker = words[0]["speaker"]
    current_start   = words[0]["start"]
    current_words   = [words[0]["word"]]

    for w in words[1:]:
        if w["speaker"] == current_speaker:
            current_words.append(w["word"])
        else:
            lines.append(f"{current_speaker} [{current_start:.2f}s]: {' '.join(current_words)}")
            current_speaker = w["speaker"]
            current_start   = w["start"]
            current_words   = [w["word"]]

    lines.append(f"{current_speaker} [{current_start:.2f}s]: {' '.join(current_words)}")
    return "\n".join(lines)


"""
    @deprecated will be removing this as soon as possible 
"""

@app.get("/")
def home():
   return FileResponse("static/index.html")


@app.get("/dashboard")
async def dashboard():
    return FileResponse("static/dashboard.html")

@app.get("/record_audio")
async def show_recorder():
    return FileResponse("static/transcribe_upload.html")


@app.get("/health/live")
def liveness():
    return {"status": "live"}


async def monitor():
    loop = asyncio.get_running_loop()

    while True:
        start = loop.time()
        await asyncio.sleep(1)
        delay = loop.time() - start - 1
        print(f"Loop delay: {delay:.3f}s")


"""
    health check for amazon load balancer for checking the system load 
"""
@app.get('/health')
def health():
    qsize = manager.job_queue.qsize()

    overloaded = (
        manager.active_jobs  >= manager.num_workers and qsize > 10
    )

    if overloaded:
        return Response(content="overloaded", status_code=503)

    return {
        "status": "ready",
        "active_jobs": manager.active_jobs,
        "queue_size": qsize,
        "active_connections": manager.active_connections
    }


@app.get("/health/metrics")
def metrics():
    return {
        "status": "ok",
        "active_connections": manager.active_connections,
        "active_jobs": manager.active_jobs,
        "queue_size": manager.job_queue.qsize(),
        "queue_capacity": manager.job_queue.maxsize,
        "workers": manager.num_workers,
        "workers_busy": manager.active_jobs,
        "workers_free": manager.num_workers - manager.active_jobs,
        "cpu_percent": CPU.get_percent(),
        "cpu_memory": CPU.get_mem(),

        ## GPU MEMORY ## 

        "gpu_utilization": GPU.get_utilization(),
        "gpu_mem_used": GPU.get_memory_used_mb()   
    }



@app.on_event("shutdown")
async def shutdown():
    for w in dmanager.workers + manager.workers:
        w.cancel()

    manager.executor.shutdown(
        wait=True,
        cancel_futures=True
    )
    dmanager.executor.shutdown(
        wait=True,
        cancel_futures=True
    )


##### HTTP endpoint 
@app.post("/transcribe")
async def upload_audio(request: Request):
    pcm_bytes = await request.body()

    if len(pcm_bytes) == 0:
        raise HTTPException(400, "Empty audio body")

    if len(pcm_bytes) % 2 != 0:
        raise HTTPException(
            400,
            "Invalid int16 PCM: odd number of bytes"
        )

    job_id = str(uuid.uuid4())

    fd, tmp_path = tempfile.mkstemp(
        suffix=".wav"
    )
 
    os.close(fd)
    
    audio_np = np.frombuffer(
        pcm_bytes,
        dtype=np.int16
    )

    sf.write(
        tmp_path,
        audio_np,
        samplerate=16000,
        subtype="PCM_16"
    )
    
    # create job state in redis
    await redis.hset(
        f"job:{job_id}",
        mapping={
            "status": "queued",
            "asr_status": "queued",
            "diar_status": "queued"
        }
    )

    # submit ASR job
    await manager.submit_job(
        AudioJob(
            job_id=job_id,
            audio_path=tmp_path
        )
    )

    # submit diarization job
    await dmanager.submit(
        DiarizationJob(
            job_id=job_id,
            audio_path=tmp_path
        )
    )
    print(
        "[INFO] submitted both jobs",
        job_id
    )
    return {
        "job_id": job_id,
        "status": "queued"
    }



@app.get("/transcribe/{job_id}")
async def get_result(job_id: str, timeout: float = 120.0, poll_interval: float = 1.0):
    start = asyncio.get_event_loop().time()

    while True:
        job = await redis.hgetall(f"job:{job_id}")
        if not job:
            raise HTTPException(404, "job not found")

        asr_status = job.get("asr_status")
        diar_status = job.get("diar_status")

        if asr_status == "done" and diar_status == "done":
            break 

        if asr_status == "failed" or diar_status == "failed":
            raise HTTPException(500, "job failed during processing")

        if asyncio.get_event_loop().time() - start > timeout:
            raise HTTPException(504, "job did not complete in time")

        await asyncio.sleep(poll_interval)

    words_json = await redis.get(f"job:{job_id}:words")
    speakers_json = await redis.get(f"job:{job_id}:diar")
    if not words_json or not speakers_json:
        raise HTTPException(500, "job completed but result missing")

    words = json.loads(words_json)
    speaker_turns = json.loads(speakers_json)
    merged = merge_words_with_speakers(words, speaker_turns)

    return {
        "job_id": job_id,
        "words": merged,
        "text": build_text_with_speakers(merged),
        "speakers": list({w["speaker"] for w in merged})
    }



class TranscriptionPayload(BaseModel):
    job_id: str
    file_url: str
    callback_url: str


###### new job and callback based endpoint 
@app.post("/transcriptions")
async def start_transcription(payload: TranscriptionPayload):
    job_id = payload.job_id
    fd, wav_path = tempfile.mkstemp(
        suffix=".wav"
    )
    os.close(fd)

    try:
        async with httpx.AsyncClient(
            timeout=None
        ) as client:

            async with client.stream(
                "GET",
                payload.file_url
            ) as response:

                response.raise_for_status()

                with wave.open(wav_path, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)

                    async for chunk in response.aiter_bytes(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            wav.writeframes(chunk)

        file_size = os.path.getsize(wav_path)

        if file_size <= 44:
            raise HTTPException(
                400,
                "Empty audio file"
            )

        await redis.hset(
            f"job:{job_id}",
            mapping={
                "status": "queued",
                "asr_status": "queued",
                "diar_status": "queued",
                "callback_url": payload.callback_url,
                "audio_path": wav_path,
                "callback_status": "pending"
            }
        )

        # ASR
        await manager.submit_job(
            AudioJob(
                job_id=job_id,
                audio_path=wav_path
            )
        )

        # Sortformer diarization
        await dmanager.submit(
            DiarizationJob(
                job_id=job_id,
                audio_path=wav_path
            )
        )

        print(
            "[INFO] submitted ASR + Sortformer",
            job_id,
            wav_path
        )

        return {
            "job_id": job_id,
            "status": "queued"
        }

    except Exception:
        if os.path.exists(wav_path):
            os.remove(wav_path)

        raise

##### new job and callback based endpoint ends




##### HTTP endpoint
@app.websocket("/ws")
async def web(ws: WebSocket):
    await ws.accept()

    buffer = bytearray()
    bytes_consumed = 0
    committed_until = 0.0

    try:
        while True:
            chunk = await ws.receive_bytes()
            chunk_audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
#            if chunk_audio.std() < 0.005:
#                if len(buffer) > 0:
#                    pcm = bytes(buffer)
#            
#                    if len(pcm) % 2 != 0:
#                        pcm = pcm[:-1]
#            
#                    if len(pcm) > 0:
#                        audio = (
#                            np.frombuffer(pcm, dtype=np.int16)
#                            .astype(np.float32)
#                            / 32768.0
#                        )
#            
#                        chunk_start_time = bytes_consumed / (SAMPLE_RATE * SAMPLE_WIDTH)
#            
#                        await audio_queue.put(
#                            AAudioJob(
#                                audio=audio,
#                                ws=ws,
#                                chunk_start_time=chunk_start_time,
#                            )
#                        )
#            
#                        bytes_consumed += len(pcm)
#                    buffer.clear()
#                continue       

            buffer.extend(chunk)

            if len(buffer) % 2 != 0:
                buffer = buffer[:-1]

            while len(buffer) >= TRANSCRIBE_BYTES:
                pcm = bytes(buffer[:TRANSCRIBE_BYTES])
                chunk_start_time = bytes_consumed / (SAMPLE_RATE * SAMPLE_WIDTH)
                del buffer[:STEP_BYTES]
                
                bytes_consumed += STEP_BYTES

                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

                await audio_queue.put(
                    AAudioJob(
                        audio=audio,
                        ws=ws,
                        chunk_start_time=chunk_start_time
                    )
                )

    except WebSocketDisconnect:
        pass

    finally:
        await audio_queue.put(AAudioJob(audio=None, ws=ws, chunk_start_time=0.0, is_close=True))



@app.websocket("/ws/utterance")
    """Each binary message = one complete utterance: [uint32 LE id][int16 PCM @16 kHz]."""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) < 4:
                continue

            uid = int.from_bytes(data[:4], "little")
            pcm = data[4:]
            if len(pcm) % 2:
                pcm = pcm[:-1]

            n = len(pcm) // 2
            if n < UTT_MIN_SAMPLES or n > UTT_MAX_SAMPLES:
                await safe_send(ws, {"type": "transcript", "id": uid, "text": "", "isFinal": True})
                continue

            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            try:
                utterance_queue.put_nowait(UtteranceJob(audio=audio, ws=ws, utterance_id=uid))
            except asyncio.QueueFull:
                await safe_send(ws, {
                    "type": "transcript", "id": uid, "text": "",
                    "isFinal": True, "error": True,
                })
    except WebSocketDisconnect:
        pass
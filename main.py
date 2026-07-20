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

app = FastAPI()

#app.mount("/static", StaticFiles(directory="static"), name="static")

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
    result = []
    for word in words:
        word_mid = (word["start"] + word["end"]) / 2
        speaker = "UNKNOWN"
        for turn in speaker_turns:
            if turn["start"] <= word_mid <= turn["end"]:
                speaker = turn["speaker"]
                break
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

#@app.get("/")
#def home():
#   return FileResponse("static/index.html")


@app.get("/dashboard")
async def dashboard():
    return FileResponse("static/dashboard.html")

@app.get("/record_audio")
async def show_recorder():
    return FileResponse("static/transcribe_upload.html")

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
manager = TranscriptionManager(device="cuda", compute_type="int8_float16",condition_on_previous_text=False)
dmanager = DiarizationManager(num_workers=1, manager=manager)


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


@app.on_event("startup")
async def startup():
    await manager.start_workers()
    await dmanager.start_workers()


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
        raise HTTPException(400, "Invalid int16 PCM: odd number of bytes")

    job_id = str(uuid.uuid4())

    fd, tmp_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)  # close the fd, soundfile will open it itself

    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16)
    sf.write(tmp_path, audio_np, samplerate=16000, subtype='PCM_16')

    manager.job_events[job_id] = {
        "asr": asyncio.get_running_loop().create_future(),
        "diar": asyncio.get_running_loop().create_future()
    }

    await manager.submit_job(AudioJob(
        job_id=job_id,
        audio_buffer=pcm_bytes,
        audio_path=tmp_path
    ))

    await dmanager.submit(DiarizationJob(
        job_id=job_id,
        audio_path = tmp_path
    ))
    
    print("[INFO]: submitted the both jobs in here")
    return {"job_id": job_id, "status": "queued"}



@app.get("/transcribe/{job_id}")
async def get_result(job_id: str):

    try:
        events = manager.job_events[job_id]
        
        if events is None:
            raise HTTPException(404, "job not found")

        print("upper events", events)
        words, speaker_turns = await asyncio.gather(
            events["asr"],
            events["diar"]
        )


        print("we are hitting this endpoint in here", words)
        merged = merge_words_with_speakers(words, speaker_turns)

        return {
            "job_id": job_id,
            "words": merged,
            "text": build_text_with_speakers(merged),
            "speakers": list({w["speaker"] for w in merged})
        }

    except Exception as e:
        raise HTTPException(500, str(e))

    finally:
        events = manager.job_events.pop(job_id)
        if events:
            asr_done = events["asr"].done()
            diar_done = events["diar"].done()

            if asr_done and diar_done:
                manager.job_events.pop(
                    job_id, 
                    None
                )

        print("[INFO] manager is shutting down", manager.job_events)


##### HTTP endpoint




@app.websocket("/ws")
async def audio_ws(websocket:WebSocket):
    await websocket.accept()
    manager.active_i()

    """will soon be deprecated and we will use the hypothesis buffer"""
    clientbuffer = bytearray()
    ws_lock = asyncio.Lock()

    try:
        while True:
            chunk = await websocket.receive_bytes()
            clientbuffer.extend(chunk)
            
            if len(clientbuffer) >= FLUSH_SIZE:
                audio_bytes = bytes(clientbuffer)
                clientbuffer.clear()
                
                job = TranscriptionJob(
                        audio_buffer=audio_bytes,
                        websocket=websocket,
                        ws_lock = ws_lock
                )
                await manager.submit_job(job)

    except WebSocketDisconnect as e:
        print("Client disconnected")

    finally:
        manager.active_d()



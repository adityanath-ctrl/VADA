import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, List
import ts_pipleline
import ds_pipeline
from cache.redis import redis
import cache.keys as cache_keys
import json

RESULT_TTL=300

@dataclass
class TranscriptionJob:
    audio_buffer: bytes
    websocket: Any


@dataclass
class AudioJob:
    audio_buffer: bytes
    job_id: str
    audio_path: str


@dataclass
class DiarizationJob:
    job_id: str
    audio_path: str


class DiarizationManager:
    def __init__(
        self,
        model: str = "nvidia/diar_streaming_sortformer_4spk-v2",
        num_workers: int = 1,
        manager: Optional["TranscriptionManager"] = None,
    ):
        self.model = ds_pipeline.DiarizationPipeLine(variant=model)
        self.queue = asyncio.Queue(maxsize=20)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.workers = []
        self.num_workers = num_workers
        self.manager = manager

    async def start_workers(self):
        for i in range(self.num_workers):
            self.workers.append(asyncio.create_task(self.worker(i)))

    async def worker(self, worker_id: int):
        while True:
            job = await self.queue.get()
            loop = asyncio.get_running_loop()

            try:
                result = await loop.run_in_executor(
                    self.executor,
                    self.model.diarize,
                    job.audio_path,
                )

                await redis.set(
                    cache_keys.diar_result(job.job_id),
                    json.dumps(result),
                    ex=RESULT_TTL
                )
            
            except Exception as e:
                future = self.manager.job_events.get(job.job_id, {}).get("diar")
                if future and not future.done():
                    future.set_exception(e)
                print(f"Diarization Worker error: {e}")
            finally:
                self.queue.task_done()


    async def submit(self, job: DiarizationJob):
        await self.queue.put(job)

class TranscriptionManager:
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cuda",
        max_batch_size: int = 24,
        max_wait_ms: int = 40,   
        queue_size: int = 100,
    ):
        self.pipeline = ts_pipleline.TranscriptionPipeline(
            model_name=model_name,
            device=device,
        )

        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0

        self.job_queue = asyncio.Queue(maxsize=queue_size)
        self.executor = ThreadPoolExecutor(max_workers=1)  # only 1 thread needed

        self.active_connections = 0
        self.active_jobs = 0
        self.job_events = {}
        self._worker_task = None

    def active_i(self):
        self.active_connections += 1

    def active_d(self):
        self.active_connections -= 1

    async def submit_job(self, job):
        await self.job_queue.put(job)

    async def start_workers(self):
        self._worker_task = asyncio.create_task(self._batch_worker())

    async def _batch_worker(self):
        while True:
            batch_jobs: List[Any] = []
            start_time = time.monotonic()

            # ---------- Collect batch ----------
            while len(batch_jobs) < self.max_batch_size:
                timeout = max(0.0, self.max_wait - (time.monotonic() - start_time))

                try:
                    job = await asyncio.wait_for(
                        self.job_queue.get(),
                        timeout=timeout
                    )
                    batch_jobs.append(job)
                except asyncio.TimeoutError:
                    break

            if not batch_jobs:
                continue

            self.active_jobs += len(batch_jobs)

            ws_jobs = []
            audio_jobs = []

            for job in batch_jobs:
                if isinstance(job, AudioJob):
                    audio_jobs.append(job)
                else:
                    ws_jobs.append(job)

            if ws_jobs:
                try:
                    audios = [j.audio_buffer for j in ws_jobs]
                    texts = await asyncio.get_running_loop().run_in_executor(
                        self.executor,
                        self.pipeline.transcribe_batch,
                        audios
                    )

                    for job, text in zip(ws_jobs, texts):
                        if text:
                            try:
                                await job.websocket.send_json({
                                    "type": "transcript",
                                    "text": text
                                })
                            except Exception:
                                pass

                except Exception as e:
                    print(f"WebSocket batch error: {e}")

            # ---------- Process HTTP AudioJobs + Redis cache ----------
            for job in audio_jobs:
                try:
                    words = await asyncio.get_running_loop().run_in_executor(
                        self.executor,
                        self.pipeline.transcribe_session,
                        job.audio_buffer
                    )
                    words = words or []

                    await redis.set(
                        cache_keys.asr_result(job.job_id),
                        json.dumps(words),
                        ex=RESULT_TTL,
                    )

                except Exception as e:
                    future = self.job_events.get(job.job_id, {}).get("asr")
                    if future and not future.done():
                        future.set_exception(e)
                    print(f"AudioJob error: {e}")

            for _ in batch_jobs:
                self.job_queue.task_done()
            self.active_jobs -= len(batch_jobs)
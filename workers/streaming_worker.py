import asyncio
from dataclasses import dataclass
from typing import Optional
import numpy as np
from fastapi import WebSocket
from enginev1.prkt import BatchedTranscriptionEngine
import time


@dataclass
class AudioJob:
    audio: np.ndarray
    ws: Optional[WebSocket] = None
    future: Optional[asyncio.Future] = None   # set by LongJobWorker instead of ws
    path:  str | None = None


class StreamingWorker:

    def __init__(self, model: BatchedTranscriptionEngine, queue: "asyncio.Queue[AudioJob]"):
        self.model = model
        self.queue = queue
        self.lock = asyncio.Lock()   

    async def run(self):
        print("Streaming Worker Started")

        while True:

            batch = await self._collect_batch()

            print(
                f"[Worker] Batch size: {len(batch)} | Queue left: {self.queue.qsize()}"
            )

            start = time.perf_counter()

            try:
                file_jobs = [
                    job for job in batch
                    if job.path is not None
                ]

                for job in file_jobs:

                    async with self.lock:
                        result = await asyncio.to_thread(
                            self.model.transcribe_file,
                            job.path
                        )

                    if job.future is not None:
                        job.future.set_result(result)

                audio_jobs = [
                    job for job in batch
                    if job.audio is not None
                ]


                if audio_jobs:
                    audios = [
                        job.audio
                        for job in audio_jobs
                    ]

                    async with self.lock:
                        results = await asyncio.to_thread(
                            self.model.transcribe_stream,
                            audios,
                            len(audios),
                        )

                    await self._publish_results(
                        audio_jobs,
                        results
                    )


                print(
                    f"Time spent is {time.perf_counter() - start}"
                )

            except Exception as e:

                print(
                    f"[Worker] Transcription error: {e}"
                )


                for job in batch:

                    if (
                        job.future is not None
                        and not job.future.done()
                    ):
                        job.future.set_exception(e)


            finally:

                for _ in batch:
                    self.queue.task_done()

    async def submit(self, job: AudioJob):
        await self.queue.put(job)

    async def _collect_batch(self) -> list[AudioJob]:
        batch = []
        job = await self.queue.get()
        batch.append(job)

        queue_size = self.queue.qsize()
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
                job = await asyncio.wait_for(self.queue.get(), timeout=0.15)
                batch.append(job)
        except asyncio.TimeoutError:
            pass

        return batch

    async def transcribe_file(self, path: str):
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        await self.queue.put(
            AudioJob(
                audio = None,
                future = future,
                path = path
            )
        )

        return await future

    async def _publish_results(self, batch, results):
        for job, hyp in zip(batch, results):
            if job.future is not None:
                if not job.future.done():
                    job.future.set_result(hyp)
                continue

            if job.ws is not None:
                try:
                    await job.ws.send_json({"type": "transcript", "text": hyp.text or ""})
                except Exception:
                    pass
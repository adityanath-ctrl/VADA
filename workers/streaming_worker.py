import asyncio
from dataclasses import dataclass
import numpy as np
from fastapi import WebSocket
from enginev1.prkt import BatchedTranscriptionEngine
import time


@dataclass
class AudioJob:
    audio: np.ndarray
    ws: WebSocket


class StreamingWorker:

    def __init__(
        self,
        model: BatchedTranscriptionEngine,
        queue: asyncio.Queue[AudioJob],
    ):
        self.model = model
        self.queue = queue
        self.lock = asyncio.Lock()

    async def run(self):
        print("Streaming Worker Started")

        while True:
            batch = await self._collect_batch()

            audios = [job.audio for job in batch]

            print(
                f"[Worker] Batch size: {len(batch)} | Queue left: {self.queue.qsize()}"
            )
            start = time.perf_counter()

            try:
                async with self.lock:
                    results = await asyncio.to_thread(
                        self.model.transcribe_stream,
                        audios,
                        len(audios),
                    )
                
                end = time.perf_counter()
                print(f"Time spent is {end - start}")
                await self._publish_results(batch, results)

            except Exception as e:
                print(f"[Worker] Transcription error: {e}")

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
                job = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=0.15,
                )
                batch.append(job)

        except asyncio.TimeoutError:
            pass

        return batch

    

    async def _publish_results(self, batch, results):

        for job, hyp in zip(batch, results):

            try:
                await job.ws.send_json(
                    {
                        "type": "transcript",
                        "text": hyp.text or "",
                    }
                )

            except Exception:
                pass

import ts_pipleline
import ds_pipeline
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

@dataclass
class TranscriptionJob:
    audio_buffer: bytes
    websocket: any 

@dataclass
class AudioJob:
    audio_buffer: bytes
    job_id: str
    audio_path: str

@dataclass
class DiarizationJob:
    job_id: str
    audio_path: str


def merge_words_with_speakers(words: list, speaker_turns: list) -> list:
    """
    words          : [{"start": 0.0, "end": 0.42, "word": "Hello"}, ...]
    speaker_turns  : [{"start": 0.0, "end": 2.1, "speaker": "SPEAKER_00"}, ...]
    returns        : [{"start": 0.0, "end": 0.42, "word": "Hello", "speaker": "SPEAKER_00"}, ...]
    """
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


class DiarizationManager:
    def __init__(self,model="nvidia/diar_sortformer_4spk-v1", num_workers=1):
        self.model = ds_pipeline.DiarizationPipeLine(variant=model)
        self.queue = asyncio.Queue(maxsize=20)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.workers = []
        self.results = {}
        self.num_workers = num_workers

    async def start_workers(self):
        for i in range(self.num_workers):
            self.workers.append(
                asyncio.create_task(self.worker(i))
            )


    async def worker(self, id: int):
        while True:
            job = await self.queue.get()
            loop = asyncio.get_running_loop()
            future = self.results.get(job.job_id)

            try:
                result = await loop.run_in_executor(
                    self.executor,
                    self.model.diarize,
                    job.audio_path
                )

                if future:
                    future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            finally:
                self.queue.task_done()

    async def submit(self, job: DiarizationJob) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.results[job.job_id] = future
        await self.queue.put(job)
        return future


class TranscriptionManager:
    def __init__(self,model_name="base",num_workers = 6,
                      device="cuda", compute_type='float16',
                      queue_size=50,condition_on_previous_text=False,
                      diarization_manager: DiarizationManager = None 
        ):

        """
            one centralised pipeline for GPU model
            inference 
        """
        self.pipeline    =   ts_pipleline.TranscriptionPipeline(
                                device=device,
                                compute_type=compute_type,
                                condition_on_previous_text=condition_on_previous_text
                             )

        self.diarization_manager = diarization_manager
        self.num_workers =   num_workers
        
        """ job queue for jobs max backpressure is 100 out of that trnascriotion pipeline won't accept the connection ' """
        self.job_queue   =   asyncio.Queue(maxsize=queue_size)

        """ each workers will be running and grabbing the task """
        self.workers     =   [ ]
        
        """ thread pool executor for max performance and multithreading  """
        self.executor   = ThreadPoolExecutor(max_workers=num_workers)

        """ System load  """
        self.active_connections = 0
        self.active_jobs  = 0

        self.job_events:Dict[str, asyncio.Queue] = {}


    def active_i(self):
        self.active_connections += 1

    def active_d(self):
        self.active_connections -= 1

    async def submit_job(self, job):
        if isinstance(job, AudioJob):
            self.job_events[job.job_id] = asyncio.Queue()

        await self.job_queue.put(job)

    async def wait_for_result(self, job_id: str):
        event = self.job_events.get(job_id)
        if not event:
            return None 

        result = await event.get()
        del self.job_events[job_id]
        return result

    def attach_full_pipeline(self,pipeline: ts_pipleline.TranscriptionPipeline):
        self.pipeline = pipeline

    async def start_workers(self):
        for worker_id in range(self.num_workers):
            task = asyncio.create_task(self.worker(worker_id))
            self.workers.append(task)
     
    async def worker(self, worker_id: int):
        while True:
            try:
                job = await self.job_queue.get()
                self.active_jobs += 1
                loop = asyncio.get_running_loop()
               
                if isinstance(job, AudioJob):   
                    transcription_task = loop.run_in_executor(
                        self.executor,
                        self.pipeline.transcribe_session,
                        job.audio_buffer
                    )

                    diar_future = None

                    if self.diarization_manager:
                        diar_future = await self.diarization_manager.submit(
                            DiarizationJob(job_id=job.job_id,audio_path=job.audio_path)
                        )

                    if diar_future:
                        words, speaker_turns = await asyncio.gather(
                            transcription_task, diar_future
                        )
                        merged = merge_words_with_speakers(words, speaker_turns)
                    else:
                        words = await transcription_task
                        merged = words
                    
                    await self.job_events[job.job_id].put({
                        "type": "completed",
                        "result": merged
                    })
                
                else:
                    text = await loop.run_in_executor(
                        self.executor,
                        self.pipeline.transcribe,
                        job.audio_buffer
                    )

                    if text:
                        await job.websocket.send_json({
                            "type":"transcript",
                            "text": text
                        })

            except Exception as e:
                print("worker error ", str(e))
            
            finally:
                self.active_jobs -= 1
                self.job_queue.task_done()
    


if __name__ == "__main__":
    pass
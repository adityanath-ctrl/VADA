import ts_pipleline
import ds_pipeline
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time


@dataclass
class TranscriptionJob:
    audio_buffer: bytes
    websocket: any 
    ws_lock: asyncio.Lock

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
    def __init__(self,model="nvidia/diar_streaming_sortformer_4spk-v2", num_workers=1, manager:'TranscriptionManager' = None):
        self.model = ds_pipeline.DiarizationPipeLine(variant=model)
        self.queue = asyncio.Queue(maxsize=20)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.workers = []
        self.results = {}
        self.num_workers = num_workers
        self.manager = manager

    async def start_workers(self):
        for i in range(self.num_workers):
            self.workers.append(
                asyncio.create_task(self.worker(i))
            )


    async def worker(self, id: int):
        while True:
            job = await self.queue.get()
            loop = asyncio.get_running_loop()

            try:
                result = await loop.run_in_executor(
                    self.executor,
                    self.model.diarize,
                    job.audio_path
                )
                
                future = self.manager.job_events[job.job_id]["diar"]
                future.set_result(result)
            
            except Exception as e:
                future = (
                    self.manager.job_events
                    .get(job.job_id, {})
                    .get("diar")
                )
                
                if future and not future.done():
                    future.set_exception(e)

            finally:
                self.queue.task_done()


    async def submit(self, job: DiarizationJob) -> asyncio.Future:
        await self.queue.put(job)




class TranscriptionManager:
    def __init__(self,model_name="medium",num_workers = 4,
                      device="cuda", compute_type='float16',
                      queue_size=50,condition_on_previous_text=False
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
        self.job_events = {}

    def active_i(self):
        self.active_connections += 1

    def active_d(self):
        self.active_connections -= 1

    async def submit_job(self, job):
        await self.job_queue.put(job)

    def attach_full_pipeline(self,pipeline: ts_pipleline.TranscriptionPipeline):
        self.pipeline = pipeline

    async def start_workers(self):
        for worker_id in range(self.num_workers):
            task = asyncio.create_task(self.worker(worker_id))
            self.workers.append(task)
     
    async def worker(self, worker_id: int):
        while True:
            job = None
            try:
                job = await self.job_queue.get()
                self.active_jobs += 1
                loop = asyncio.get_running_loop()
               
                if isinstance(job, AudioJob):   
                    words = await loop.run_in_executor(
                        self.executor,
                        self.pipeline.transcribe_session,
                        job.audio_buffer
                    )

                    future = self.job_events[job.job_id]["asr"]
                    if future and not future.done():
                        future.set_result(words  or [])
                
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
                if isinstance(job, AudioJob):
                    future = (
                        self.job_events
                        .get(job.job_id,{})
                        .get("asr")
                    )

                    if future and not future.done():
                        future.set_exception(e)

            finally:
                if job is not None:
                    self.active_jobs -= 1
                    self.job_queue.task_done()
    


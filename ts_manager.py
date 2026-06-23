import ts_pipleline
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

@dataclass
class TranscriptionJob:
    audio_buffer: bytes
    websocket: any 

@dataclass
class SessionTJob:
    audio_buffer: bytes
    job_id: str


class TranscriptionManager:
    def __init__(self,model_name="base",num_workers = 6,device="cuda", compute_type='float16', queue_size=50,condition_on_previous_text=False):

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

        self.jobs = {}
        self.job_events = {}

    async def submit_job(self, job: TranscriptionJob):
        await self.job_queue.put(job)
    
    def create_http_job(self,job_id):
        self.jobs[job_id] = {
            "status": "queued",
            "result": None
        }
        self.job_events[job_id] = asyncio.Queue()
    
    def set_http_processing(self,job_id):
        self.jobs[job_id]["status"] = "processing"

    def set_http_completed(self,job_id,transcript):
        self.jobs[job_id]["status"] =  "completed"
        self.jobs[job_id]["result"] = transcript


    def attach_full_pipeline(self,pipeline: ts_pipleline.TranscriptionPipeline):
        self.pipeline = pipeline

    async def start_workers(self):
        for worker_id in range(self.num_workers):
            task = asyncio.create_task(self.worker(worker_id))
            self.workers.append(task)
     
    async def worker(self, worker_id: int):
        while True:
            try:
                print("worker hanged with the worker id", worker_id)
                job = await self.job_queue.get()
                loop = asyncio.get_running_loop()
               
                if isinstance(job, SessionTJob):
                    
                    self.set_http_processing(job.job_id)

                    text = await loop.run_in_executor(
                        self.executor,
                        self.pipeline.transcribe_session,
                        job.audio_buffer
                    )
                    
                    self.set_http_completed(job.job_id, text)
                    await self.job_events[job.job_id].put({"type" : "completed"})
                
                else:
                    text = await loop.run_in_executor(
                        self.executor,
                        self.pipeline.transcribe,
                        job.audio_buffer
                    )
                    print("worker started with the worker_id ", worker_id, "and text", text)

                    if text:
                        await job.websocket.send_json({
                            "type":"transcript",
                            "text": text
                        })
                
                    self.job_queue.task_done()
            except Exception as e:
                print("worker error ", str(e))
            
    


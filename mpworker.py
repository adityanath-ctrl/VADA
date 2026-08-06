import asyncio
import json
import torch
from ds_pipeline import DiarizationPipeLine
from redis.asyncio import Redis
from redis.exceptions import TimeoutError, ConnectionError


class DiarizationWorker:
    def __init__(self):
        print("Loading Sortformer")
        self.model = DiarizationPipeLine(
            variant="nvidia/diar_streaming_sortformer_4spk-v2"
        )
        print("Sortformer loaded")


    def process(self, audio_path):
        return self.model.diarize(
            audio_path
        )




def create_redis():
    return Redis(
        host="noustalk-cache-cwzlzn.serverless.cac1.cache.amazonaws.com",
        port=6379,
        ssl=True,
        decode_responses=True,
        health_check_interval=30,
    )



async def main():
    redis_client = create_redis()

    worker = DiarizationWorker()
    while True:
        try:
            item = await redis_client.blpop(
                "diarization_jobs",
                timeout=30,
            )
            
            print("[INFO] Here comes the data in here")
            if item is None:
                continue
    
            _, payload = item
    
            job = json.loads(payload)
            
            job_id = job["job_id"]
            audio_path = job["audio_path"]
            result = worker.process(audio_path)
            
            print("[INFO]", result)
            await redis_client.set(
                f"job:{job_id}:diar",
                json.dumps(result)
            )

            await redis_client.hset(
                f"job:{job_id}",
                "diar_status",
                "done"
            )

        except (TimeoutError, ConnectionError):
            print("Redis disconnected. Reconnecting...")
    
            await redis_client.aclose()
            await asyncio.sleep(5)
    
            redis_client = await create_redis()


if __name__ == "__main__":
    asyncio.run(main())

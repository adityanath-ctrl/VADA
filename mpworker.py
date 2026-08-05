import asyncio
import json
import torch
from ds_pipeline import DiarizationPipeLine
from redis.asyncio import Redis



class DiarizationWorker:
    def __init__(self):
        print("Loading Sortformer")
        self.model = DiarizationPipeLine(
            variant="nvidia/diar_streaming_sortformer_4spk-v2"
        )
        self.model = self.model.cuda()
        print("Sortformer loaded")


    def process(self, audio_path):
        return self.model.diarize(
            audio_path
        )



async def main():
    redis = Redis(
        host="noustalk-cache-cwzlzn.serverless.cac1.cache.amazonaws.com",
        port=6379,
        ssl=True,
        decode_responses=True,
    )

    worker = DiarizationWorker()
    while True:
        item = await redis_client.blpop(
            "diarization_jobs",
            timeout=0
        )
        _, payload = item

        job = json.loads(payload)
        job_id = job["job_id"]
        audio_path = job["audio_path"]
        print(
            "Diarizing:",
            job_id
        )

        try:
            result = worker.process(
                audio_path
            )

            await redis_client.set(
                f"job:{job_id}:diar",
                json.dumps(result)
            )

            await redis_client.hset(
                f"job:{job_id}",
                "diar_status",
                "done"
            )


        except Exception as e:
            await redis_client.hset(
                f"job:{job_id}",
                mapping={
                    "diar_status":"error",
                    "error":str(e)
                }
            )


if __name__ == "__main__":
    asyncio.run(main())
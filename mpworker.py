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
        return self.model.diarize(audio_path)


def create_redis():
    return Redis(
        host="noustalk-cache-cwzlzn.serverless.cac1.cache.amazonaws.com",
        port=6379,
        ssl=True,
        decode_responses=True,
        health_check_interval=30,
        socket_keepalive=True,
        socket_timeout=35,
        socket_connect_timeout=10,
        retry_on_timeout=True,
    )


worker = DiarizationWorker()

# hash-tagged so both keys always land on the same cluster slot
QUEUE_KEY = "{diarization_jobs}:main"
PROCESSING_LIST = "{diarization_jobs}:processing"


async def main():
    redis_client = create_redis()

    while True:
        try:
            payload = await redis_client.brpoplpush(
                QUEUE_KEY,
                PROCESSING_LIST,
                timeout=30,
            )

            if payload is None:
                continue

            print("this is the item in here", payload)
            job = json.loads(payload)
            job_id = job["job_id"]
            audio_path = job["audio_path"]

            result = worker.process(audio_path)
            print("[INFO]", result)

            await redis_client.set(f"job:{job_id}:diar", json.dumps(result))
            await redis_client.hset(f"job:{job_id}", "diar_status", "done")

            await redis_client.lrem(PROCESSING_LIST, 1, payload)

        except (TimeoutError, ConnectionError) as e:
            print(f"Redis disconnected ({e}). Reconnecting...")
            try:
                await redis_client.aclose()
            except Exception:
                pass
            await asyncio.sleep(5)
            redis_client = create_redis()

        except Exception as e:
            print(f"[ERROR] job failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

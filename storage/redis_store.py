from cache.redis import redis
from cache.keys import job

from .models import Job


class RedisJobStore:

    async def create(self, j: Job):
        await redis.hset(
            job(j.id),
            mapping={
                "id": j.id,
                "status": j.status,
                "transcript": j.transcript,
                "error": j.error,
            },
        )

        await redis.expire(job(j.id), 3600)



    async def get(self, job_id: str):
        data = await redis.hgetall(job(job_id))

        if not data:
            return None

        return Job(**data)



    async def update(self, j: Job):
        await redis.hset(
            job(j.id),
            mapping={
                "status": j.status,
                "transcript": j.transcript,
                "error": j.error,
            },
        )



    async def delete(self, job_id: str):
        await redis.delete(job(job_id))






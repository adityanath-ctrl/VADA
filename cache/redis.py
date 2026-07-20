from redis.asyncio import Redis

redis = Redis(
    host="noustalk-cache-cwzlzn.serverless.cac1.cache.amazonaws.com",
    port=6379,
    ssl=True,
    decode_responses=True,
)





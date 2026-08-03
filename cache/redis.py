from redis.asyncio import Redis


redis = Redis(
    host="noustalk-cache-cwzlzn.serverless.cac1.cache.amazonaws.com",
    port=6379,
    ssl=True,
    decode_responses=True,

    socket_connect_timeout=10,
    socket_timeout=None,
    health_check_interval=30
)


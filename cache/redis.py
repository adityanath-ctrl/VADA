import os
from dotenv import load_dotenv
from redis.asyncio import Redis  # adjust import if you're using a different redis client

load_dotenv()

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_SSL = os.environ.get("REDIS_SSL", "false").lower() == "true"

print(REDIS_HOST)
# redis = Redis(
#     host=REDIS_HOST,
#     port=REDIS_PORT,
#     ssl=REDIS_SSL,
#     decode_responses=True,

#     socket_connect_timeout=10,
#     socket_timeout=None,
#     health_check_interval=30
# )

import fakeredis.aioredis as fakeredis_lib

redis = fakeredis_lib.FakeRedis(decode_responses=True)
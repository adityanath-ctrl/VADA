import asyncio
import json 
from cache.redis import redis
from utils import convert_to_wav
from enginev1.prkt import BatchedTranscriptionEngine
from enginev1.ds_pipeline import DiarizationPipeLine

def merge_words_with_speakers(words: list, speaker_turns: list) -> list:
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


class LongJobWorker:

    def __init__(self,redis,transcriber, diarizer):
        self.redis = redis
        self.transcriber = transcriber
        self.diarizer = diarizer



    async def run(self):
        print("Long job Worker started")

        while True:

            try:
                _, job_id = await redis.brpop("transcription_jobs")
                await self.process(job_id)
            except Exception as e:
                print(f"[Worker error] {e}")


    async def process(self, job_id: str):
        job = await self.redis.hgetall(f"job:{job_id}")

        if not job:
            return

        webm_path = job['path']

        wav_path = webm_path.replace(
            ".webm",
            ".wav"
        )
        
        await asyncio.to_thread(
            convert_to_wav,
            webm_path,
            wav_path
        )

        await self.update_status(
            job_id,
            status="processing",
            stage="diarization",
            progress=10,
        )

        # step 1
        # speaker diarization 
        speakers = await asyncio.to_thread(
            self.diarizer.diarize,
            path
        )

        await self.update_status(
            job_id,
            stage="transcription",
            progress=40,
        )

        ## step 2
        ## transctiption

        transcript = await asyncio.to_thread(
            self.transcriber.transcribe_file,
            path,
        )

        await self.update_status(
            job_id,
            stage="merging",
            progress=90,
        )

        ## step 3
        ## merging

        merged = merge_words_with_speakers(transcript.words,speakers)

        await self.redis.set(
            f"job:{job_id}:result",
            json.dumps(merged)
        )

        await self.update_status(
            job_id,
            status="completed",
            stage="finished",
            progress=100
        )

    async def update_status(self, job_id: str,**kwargs):
        await self.redis.hset(
            f"job:{job_id}",
            mapping=kwargs,
        )
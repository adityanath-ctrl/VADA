import asyncio
import json
from utils import convert_to_wav
from enginev1.ds_pipeline import DiarizationPipeLine
from workers.streaming_worker import StreamingWorker

def merge_words_with_speakers(words: list, speaker_turns: list) -> list:

    result = []

    for word in words:

        word_start = word["start"]
        word_end = word["end"]

        best_speaker = None
        best_overlap = 0


        # first try overlap
        for turn in speaker_turns:

            turn_start = turn["start"] - 0.2
            turn_end = turn["end"] + 0.2


            overlap_start = max(
                word_start,
                turn_start
            )

            overlap_end = min(
                word_end,
                turn_end
            )


            overlap = max(
                0,
                overlap_end - overlap_start
            )


            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]



        # fallback: nearest speaker
        if best_speaker is None:

            word_mid = (word_start + word_end) / 2

            nearest_distance = float("inf")

            for turn in speaker_turns:

                turn_mid = (
                    turn["start"] +
                    turn["end"]
                ) / 2


                distance = abs(
                    word_mid - turn_mid
                )


                if distance < nearest_distance:
                    nearest_distance = distance
                    best_speaker = turn["speaker"]



        result.append(
            {
                **word,
                "speaker": best_speaker
            }
        )


    return result


class LongJobWorker:

    def __init__(self, redis, streaming_worker: StreamingWorker, diarizer: DiarizationPipeLine):
        self.redis = redis
        self.streaming_worker = streaming_worker  
        self.diarizer = diarizer

    async def run(self):
        while True:
            job_id = None
            try:
                _, job_id = await self.redis.brpop("transcription_jobs")
                await self.process(job_id)
            except Exception as e:
                print(f"[Worker error] {e}")
                if job_id:
                    try:
                        await self.update_status(
                            job_id,
                            status="failed",
                            stage="error",
                            error=str(e),
                        )
                    except Exception as update_err:
                        print(f"[Worker error] failed to write failure status: {update_err}")
                await asyncio.sleep(5)

    async def process(self, job_id: str):
        job = await self.redis.hgetall(f"job:{job_id}")
        if not job:
            return

        webm_path = job["path"]
        wav_path = webm_path.replace(".webm", ".wav")

        await self.update_status(job_id, status="processing", stage="converting", progress=5)
        await asyncio.to_thread(convert_to_wav, webm_path, wav_path)

        await self.update_status(job_id, stage="diarization", progress=20)

        speakers = await asyncio.to_thread(self.diarizer.diarize, wav_path)

        await self.update_status(job_id, stage="transcription", progress=55)
        transcript_words = await self._transcribe_via_streaming(wav_path)

        await self.update_status(job_id, stage="merging", progress=90)

        merged = merge_words_with_speakers(transcript_words, speakers)

        await self.redis.set(f"job:{job_id}:result", json.dumps(merged))
        await self.update_status(job_id, status="completed", stage="finished", progress=100)

    async def _transcribe_via_streaming(self, wav_path):
        hyp = await self.streaming_worker.transcribe_file(
            wav_path
        )

        word_ts = hyp.timestamp.get("word")

        return [
            {
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
            }
            for w in word_ts
        ]

    async def update_status(self, job_id: str, **kwargs):
        await self.redis.hset(f"job:{job_id}", mapping=kwargs)
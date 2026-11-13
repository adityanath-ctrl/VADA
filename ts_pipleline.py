from faster_whisper import WhisperModel, BatchedInferencePipeline
import numpy as np

MAX_CUDA_WORKERS=2

class TranscriptionPipeline:
    def __init__(self, model="base", device="cuda", compute_type="float16", beam_size=1,
                       no_speech_threshold=0.5, avg_logprob_threshold=-1.0,
                       condition_on_previous_text=False, batch_size=16):

        base_model = WhisperModel(model, device=device, compute_type=compute_type, num_workers=MAX_CUDA_WORKERS)
        self.model = BatchedInferencePipeline(model=base_model)
        self.batch_size = batch_size

        self.vad_filter = True
        self.beam_size = beam_size
        self.min_time = 0.5
        self.no_speech_threshold = no_speech_threshold
        self.avg_logprob_threshold = avg_logprob_threshold
        self.condition_on_previous_text = condition_on_previous_text

        self.vad_parameters = {
            "threshold": 0.6,
            "min_speech_duration_ms": 350,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        }

    def change_vad_parameters(self, new_params):
        self.vad_parameters.update(new_params)

    def process_segments(self, segments):
        texts = []
        for s in segments:
            if s.no_speech_prob > self.no_speech_threshold:
                continue

            if s.avg_logprob < self.avg_logprob_threshold:
                continue

            text = s.text.strip()
            if text:
                texts.append(text)
        return " ".join(texts) if texts else ""

    def transcribe_session(self, wav_path: str):
        segments, info = self.model.transcribe(
            wav_path,
            batch_size=self.batch_size,
            beam_size=self.beam_size,
            temperature=0.0,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.avg_logprob_threshold,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=self.condition_on_previous_text,
            word_timestamps=True
        )

        words = []

        for segment in segments:
            if segment.words:
                for word in segment.words:
                    words.append({
                        "start": word.start,
                        "end": word.end,
                        "word": word.word.strip()
                    })

        return words

    def transcribe(self, audio_bytes: bytes) -> str:
        audio = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

        if len(audio) < 16000 * self.min_time:
            return ""

        segments, info = self.model.transcribe(
            audio,
            batch_size=self.batch_size,
            beam_size=self.beam_size,
            temperature=0.0,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.avg_logprob_threshold,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=self.condition_on_previous_text
        )
        text = self.process_segments(segments)
        return text

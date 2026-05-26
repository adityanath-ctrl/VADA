from faster_whisper import WhisperModel
import numpy as np


class TranscriptionPipeline:
    def __init__(self,model="base", device="cpu", compute_type="int8", beam_size = 2, 
                       no_speech_threshold=0.5, avg_logprob_threshold=-1.0, 
                       condition_on_previous_text=False):
                       
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
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


    def transcribe(self, audio_bytes: bytes) -> str:
        audio = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
        
        if len(audio) < 16000 * self.min_time:  
            return ""

        segments, info = self.model.transcribe(
            audio,
            beam_size=self.beam_size,
            temperature=0.0,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.avg_logprob_threshold,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=True
        )

        text = self.process_segments(segments)
        return text




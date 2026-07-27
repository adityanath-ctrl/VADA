import numpy as np
import torch
import nemo.collections.asr as nemo_asr

class TranscriptionPipeline:
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cuda",
    ):
        print(f"Loading Parakeet model: {model_name} ...")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self.model.eval()

        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        self.min_time = 0.4

    def _to_float32(self, audio_bytes: bytes) -> np.ndarray:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    def transcribe(self, audio_bytes: bytes) -> str:
        """Used by WebSocket (short chunks)"""
        audio = self._to_float32(audio_bytes)

        if len(audio) < 16000 * self.min_time:
            return ""

        with torch.inference_mode():
            output = self.model.transcribe([audio], batch_size=1)

        text = output[0].text.strip() if output else ""
        return text

    def transcribe_session(self, audio_bytes: bytes) -> list:
        """
        Used by HTTP /transcribe endpoint.
        Returns word-level timestamps (compatible with your merge logic).
        """
        audio = self._to_float32(audio_bytes)

        with torch.inference_mode():
            output = self.model.transcribe(
                [audio],
                batch_size=1,
                timestamps=True
            )

        if not output:
            return []

        word_ts = output[0].timestamp.get("word", [])

        words = []
        for w in word_ts:
            words.append({
                "start": w["start"],
                "end": w["end"],
                "word": w["word"].strip()
            })

        return words
    

    def transcribe_batch(self, audio_list: list[bytes]) -> list[str]:
        """
        Proper batching for multiple short audio clips.
        """
        audios = []
        for audio_bytes in audio_list:
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audios.append(audio)

        valid_indices = []
        valid_audios = []
        for i, audio in enumerate(audios):
            if len(audio) >= 16000 * self.min_time:
                valid_indices.append(i)
                valid_audios.append(audio)

        results = [""] * len(audio_list)

        if not valid_audios:
            return results

        with torch.inference_mode():
            outputs = self.model.transcribe(
                valid_audios,
                batch_size=len(valid_audios)
            )

        for idx, out in zip(valid_indices, outputs):
            results[idx] = out.text.strip() if out else ""

        return results
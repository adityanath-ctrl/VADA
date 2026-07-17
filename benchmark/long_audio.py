import soundfile as sf
import numpy as np

audio, sr = sf.read(
    "audio.wav",
    dtype="int16"
)


long_audio = np.tile(
    audio,
    30
)


sf.write(
    "audio_30min.wav",
    long_audio,
    sr,
    subtype="PCM_16"
)
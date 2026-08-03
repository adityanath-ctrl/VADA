from pydub import AudioSegment
import os


def convert_to_wav(
    input_path: str,
    output_path: str | None = None
):

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".wav"


    audio = AudioSegment.from_file(
        input_path,
        format="webm"
    )


    audio = (
        audio
        .set_frame_rate(16000)
        .set_channels(1)
    )


    audio.export(
        output_path,
        format="wav",
        parameters=[
            "-acodec",
            "pcm_s16le"
        ]
    )


    return output_path
import asyncio
import websockets
import soundfile as sf
import numpy as np
import time
import uuid
import json


SERVER = "wss://stt.noustalk.com/ws"
AUDIO_FILE = "audio.wav"
CLIENTS =  15

CHUNK_SIZE = 16000 * 2 * 1

def load_pcm():
    audio, sr = sf.read(
        AUDIO_FILE,
        dtype="int16"
    )
    assert sr == 16000
    pcm = audio.tobytes()
    return pcm


async def run_client(client_id, pcm):
    start = time.perf_counter()
    transcripts = []

    try:
        async with websockets.connect(
            SERVER,
            max_size=None
        ) as ws:

            print(f"[{client_id}] connected")

            # send audio
            for i in range(
                0,
                len(pcm),
                CHUNK_SIZE
            ):
                chunk = pcm[i:i+CHUNK_SIZE]

                await ws.send(chunk)

                # simulate microphone speed
                await asyncio.sleep(1)


            print(
                f"[{client_id}] audio sent"
            )


            # collect all transcripts
            while True:

                try:
                    msg = await asyncio.wait_for(
                        ws.recv(),
                        timeout=15
                    )

                    data = json.loads(msg)

                    if data.get("type") == "transcript":
                        transcripts.append(
                            data["text"]
                        )

                        print(
                            f"[{client_id}] chunk:",
                            data["text"]
                        )


                except asyncio.TimeoutError:
                    # no more transcript received
                    break


            end = time.perf_counter()
            full_transcript = " ".join(
                transcripts
            )
            print(
                "\n======================"
            )
            print(
                f"[{client_id}] FULL TRANSCRIPT:"
            )
            print(
                full_transcript
            )
            print(
                "======================\n"
            )
            return {
                "client": client_id,
                "success": True,
                "time": end-start,
                "chunks_received": len(transcripts)
            }

    except Exception as e:
        return {
            "client": client_id,
        "success": False,
        "error": str(e)
        }


async def main():
    pcm = load_pcm()
    print(
        "Audio duration:",
        len(pcm)/(16000*2),
        "seconds"
    )
    tasks = []
    for i in range(CLIENTS):
        tasks.append(
            run_client(
                i,
                pcm
            )
        )
    start = time.perf_counter()
    results = await asyncio.gather(
        *tasks
    )
    total = time.perf_counter()-start
    print("\n========= RESULTS =========")
    for r in results:
        print(r)

    successful = [
        r for r in results
        if r["success"]
    ]

    print(
        "\nSuccessful:",
        len(successful),
        "/",
        CLIENTS
    )

    print(
        "Total benchmark time:",
        total
    )


if __name__ == "__main__":
    asyncio.run(main())
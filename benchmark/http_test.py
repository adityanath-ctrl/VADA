import asyncio
import httpx
import soundfile as sf
import time
import json


SERVER = "https://localhost:8000"
AUDIO_FILE = "audio_15min.wav"
CLIENTS = 1

def load_pcm():
    audio, sr = sf.read(
        AUDIO_FILE,
        dtype="int16"
    )

    assert sr == 16000
    return audio.tobytes()



async def wait_for_result(
    client,
    job_id,
    client_id
):

    url = f"{SERVER}/transcribe/{job_id}"

    while True:

        try:

            response = await client.get(
                url,
                timeout=120
            )

            if response.status_code == 200:

                return response.json()


            elif response.status_code == 404:

                print(
                    f"[{client_id}] job not ready"
                )


            else:
                print(
                    f"[{client_id}] error:",
                    response.text
                )


        except Exception as e:

            print(
                f"[{client_id}] polling error",
                e
            )


        await asyncio.sleep(1)




async def run_client(
    client_id,
    pcm
):

    start = time.perf_counter()


    try:

        async with httpx.AsyncClient(
            verify=False,
            timeout=None
        ) as client:


            # upload audio

            print(
                f"[{client_id}] uploading"
            )


            response = await client.post(
                f"{SERVER}/transcribe",
                content=pcm,
                headers={
                    "Content-Type": "application/octet-stream"
                }
            )


            data = response.json()

            job_id = data["job_id"]


            print(
                f"[{client_id}] job:",
                job_id
            )


            result = await wait_for_result(
                client,
                job_id,
                client_id
            )


            end = time.perf_counter()


            print(
                f"\n[{client_id}] TRANSCRIPT\n"
            )

            print(
                result["text"]
            )


            return {
                "client": client_id,
                "success": True,
                "time": end-start
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


    print(
        "\n========= RESULTS ========="
    )


    for r in results:
        print(r)
    success = [
        r for r in results
        if r["success"]
    ]
    print(
        "\nSuccessful:",
        len(success),
        "/",
        CLIENTS
    )
    print(
        "Total benchmark time:",
        total
    )

if __name__ == "__main__":
    asyncio.run(main())
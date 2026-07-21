

def job(job_id: str) -> str:
    return f"job:{job_id}"


def asr_result(job_id: str) -> str:
    """Key that holds the serialised words list from ASR."""
    return f"job:{job_id}:asr"


def diar_result(job_id: str) -> str:
    """Key that holds the serialised speaker_turns list from diarization."""
    return f"job:{job_id}:diar"


def websocket(client_id: str) -> str:
    return f"ws:{client_id}"



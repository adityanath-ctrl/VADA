from dataclasses import dataclass


@dataclass
class Job:

    id: str
    status: str
    transcript: str
    error: str = ""




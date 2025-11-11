from enum import Enum


class ClientConnectionPoolType(Enum):
    SESSION = "session"
    SAMPLE = "sample"
    TRAIN = "train"
    RETRIEVE_PROMISE = "retrieve_promise"
    TELEMETRY = "telemetry"

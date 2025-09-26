from enum import Enum


class ClientConnectionPoolType(Enum):
    SAMPLE = "sample"
    TRAIN = "train"
    RETRIEVE_PROMISE = "retrieve_promise"
    TELEMETRY = "telemetry"

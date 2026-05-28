from enum import Enum


class ClientConnectionPoolType(Enum):
    SESSION = "session"
    SAMPLE = "sample"
    TRAIN = "train"
    CHECKPOINT_ARCHIVE_URL = "checkpoint_archive_url"
    RETRIEVE_PROMISE = "retrieve_promise"
    TELEMETRY = "telemetry"

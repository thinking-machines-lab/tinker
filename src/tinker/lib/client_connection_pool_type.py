from enum import Enum


class ClientConnectionPoolType(Enum):
    SESSION = "session"
    SAMPLE = "sample"
    TRAIN = "train"
    REST_SUPPORT_REDIRECT = "rest_support_redirect"
    RETRIEVE_PROMISE = "retrieve_promise"
    TELEMETRY = "telemetry"

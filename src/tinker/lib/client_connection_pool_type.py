from enum import Enum


class ClientConnectionPoolType(Enum):
    SESSION = "session"
    SAMPLE = "sample"
    TRAIN = "train"
    REST_SUPPORT_REDIRECT = "rest_support_redirect"
    RETRIEVE_PROMISE = "retrieve_promise"
    # Dedicated pool for the retrieve_futures session poller, isolated from
    # per-request retrieve_promise traffic but otherwise configured the same.
    RETRIEVE_FUTURES = "retrieve_futures"
    TELEMETRY = "telemetry"

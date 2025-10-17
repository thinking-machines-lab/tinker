import os
import logging

logger: logging.Logger = logging.getLogger("tinker")
httpx_logger: logging.Logger = logging.getLogger("httpx")


def _basic_config() -> None:
    # e.g. [2023-10-05 14:12:26 - tinker._base_client:818 - DEBUG] HTTP Request: POST http://127.0.0.1:4010/foo/bar "200 OK"
    logging.basicConfig(
        format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logging() -> None:
    # httpx prints out the HTTP requests its making out to INFO
    # log stream. The tinker API server communicates backpressure
    # to the SDK using certain 4xx error codes, which looks scary
    # to the user, but are actually harmless.
    #
    # Thus, we set the default httpx logging level to WARNING so
    # that they don't see a bunch of red herrings and get worried.
    httpx_logger.setLevel(logging.WARNING)

    env = os.environ.get("TINKER_LOG")
    if env == "debug":
        _basic_config()
        logger.setLevel(logging.DEBUG)
        httpx_logger.setLevel(logging.DEBUG)
    elif env == "info":
        _basic_config()
        logger.setLevel(logging.INFO)
        httpx_logger.setLevel(logging.INFO)

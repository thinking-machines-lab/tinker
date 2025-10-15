from typing_extensions import Literal, TypeAlias

__all__ = ["EventType"]

EventType: TypeAlias = Literal[
    "SESSION_START", "SESSION_END", "UNHANDLED_EXCEPTION", "GENERIC_EVENT"
]

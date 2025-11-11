from enum import StrEnum, auto

__all__ = ["RequestErrorCategory"]


class RequestErrorCategory(StrEnum):
    Unknown = auto()
    Server = auto()
    User = auto()

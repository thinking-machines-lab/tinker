from typing_extensions import Literal, TypeAlias

__all__ = ["StopReason"]

StopReason: TypeAlias = Literal["length", "stop"]

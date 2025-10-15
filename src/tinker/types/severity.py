from typing_extensions import Literal, TypeAlias

__all__ = ["Severity"]

Severity: TypeAlias = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

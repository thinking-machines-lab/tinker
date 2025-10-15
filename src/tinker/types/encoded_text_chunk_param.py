from __future__ import annotations

from typing import Iterable
from typing_extensions import Literal, Required, TypedDict

__all__ = ["EncodedTextChunkParam"]


class EncodedTextChunkParam(TypedDict, total=False):
    tokens: Required[Iterable[int]]
    """Array of token IDs"""

    type: Required[Literal["encoded_text"]]

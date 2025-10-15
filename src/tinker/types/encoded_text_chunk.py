from typing import Sequence

from typing_extensions import Literal

from .._models import StrictBase

__all__ = ["EncodedTextChunk"]


class EncodedTextChunk(StrictBase):
    tokens: Sequence[int]
    """Array of token IDs"""

    type: Literal["encoded_text"] = "encoded_text"

    @property
    def length(self) -> int:
        return len(self.tokens)

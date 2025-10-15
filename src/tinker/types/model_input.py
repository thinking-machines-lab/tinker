from typing import List

from .._models import StrictBase
from .model_input_chunk import ModelInputChunk
from .encoded_text_chunk import EncodedTextChunk

__all__ = ["ModelInput"]


class ModelInput(StrictBase):
    chunks: List[ModelInputChunk]
    """Sequence of input chunks (formerly TokenSequence)"""


    @classmethod
    def from_ints(cls, tokens: List[int]) -> "ModelInput":
        """
        Create a ModelInput from a list of ints (tokens).
        """
        return cls(chunks=[EncodedTextChunk(tokens=tokens)])

    def to_ints(self) -> List[int]:
        """
        Convert the ModelInput to a list of ints (tokens)
        Throws exception if there are any non-token chunks
        """
        if not all(isinstance(chunk, EncodedTextChunk) for chunk in self.chunks):
            raise ValueError(f"to_ints only supported for ModelInput with EncodedTextChunks, got {[type(chunk) for chunk in self.chunks]}")
        return [token for chunk in self.chunks for token in chunk.tokens]

    @property
    def length(self) -> int:
        """
        Return the total context length used by this ModelInput.
        """
        return sum(chunk.length for chunk in self.chunks)

    @classmethod
    def empty(cls) -> "ModelInput":
        """
        Create an empty ModelInput.
        """
        return cls(chunks=[])

    def append(self, chunk: ModelInputChunk) -> "ModelInput":
        """
        Add a new chunk, return a new ModelInput.
        """
        return ModelInput(chunks=self.chunks + [chunk])

    def append_int(self, token: int) -> "ModelInput":
        """
        Add a new token, return a new ModelInput.
        """
        return self.append(EncodedTextChunk(tokens=[token]))

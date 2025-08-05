# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Iterable
from typing_extensions import Required, TypedDict

from .model_input_chunk_param import ModelInputChunkParam

__all__ = ["ModelInputParam"]


class ModelInputParam(TypedDict, total=False):
    chunks: Required[Iterable[ModelInputChunkParam]]
    """Sequence of input chunks (formerly TokenSequence)"""

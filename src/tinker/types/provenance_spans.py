from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

__all__ = ["SampledProvenanceSpan", "PromptProvenanceSpan", "ProvenanceSpans"]


@dataclass(frozen=True, kw_only=True)
class PromptProvenanceSpan:
    """A run of tokens that were provided as sampling input.

    Claims that the covered datum positions are positions
    ``[offset, offset + length)`` of the named sequence's request prompt.
    Runs tile the field they annotate (see ``Datum.with_provenance``):
    every position is in exactly one run and position comes from order.
    """

    sequence_id: str
    """``SampledSequence.sequence_id`` of a request whose prompt contained
    this run."""

    offset: int = 0
    """Position of the run within that request's prompt."""

    length: int
    """Number of consecutive tokens in this run (must be >= 0)."""

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError(f"PromptProvenanceSpan.length must be >= 0, got {self.length}")
        if not self.sequence_id:
            raise ValueError("PromptProvenanceSpan.sequence_id must be non-empty")
        if self.offset < 0:
            raise ValueError(f"PromptProvenanceSpan.offset must be >= 0, got {self.offset}")


@dataclass(frozen=True, kw_only=True)
class SampledProvenanceSpan:
    """A run of tokens produced by one sampled sequence.

    Claims that the covered datum positions are positions
    ``[offset, offset + length)`` of the named sequence's sampled tokens.
    Two sampled sequences back to back are two adjacent runs, each naming
    its own sequence.
    """

    sequence_id: str
    """``SampledSequence.sequence_id`` of the sequence this run reproduces."""

    offset: int = 0
    """Position of the run within that sequence's sampled tokens."""

    length: int
    """Number of covered tokens (must be >= 0)."""

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError(f"SampledProvenanceSpan.length must be >= 0, got {self.length}")
        if not self.sequence_id:
            raise ValueError("SampledProvenanceSpan.sequence_id must be non-empty")
        if self.offset < 0:
            raise ValueError(f"SampledProvenanceSpan.offset must be >= 0, got {self.offset}")


ProvenanceSpans = List[Union[PromptProvenanceSpan, SampledProvenanceSpan]]
"""Consecutive runs tiling one Datum field; see ``Datum.with_provenance``."""

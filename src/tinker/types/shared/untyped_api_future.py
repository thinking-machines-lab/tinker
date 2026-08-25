from typing import List, Optional

from ..._models import BaseModel
from ..model_id import ModelID
from ..request_id import RequestID

__all__ = ["UntypedAPIFuture"]


class UntypedAPIFuture(BaseModel):
    request_id: RequestID

    model_id: Optional[ModelID] = None

    sample_sequence_ids: Optional[List[str]] = None
    """Sampling promises only: one id per requested sample, aligned with the
    final response's sequence order; each becomes the corresponding
    ``SampledSequence.sequence_id``."""

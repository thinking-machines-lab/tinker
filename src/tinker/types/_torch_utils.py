from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def _suppress_torch_sparse_csr_beta_warning() -> Iterator[None]:
    """Suppress PyTorch's unavoidable notice around supported CSR operations."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Sparse CSR tensor support is in beta state\.",
            category=UserWarning,
        )
        yield

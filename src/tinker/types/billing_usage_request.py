from datetime import datetime

from .._models import StrictBase

__all__ = ["GetBillingUsageRequest"]


class GetBillingUsageRequest(StrictBase):
    """Query parameters for GET /api/v1/billing/usage/events."""

    starting_on: datetime
    """Inclusive window start, aligned to a UTC hour boundary"""

    ending_before: datetime
    """Exclusive window end, aligned to a UTC hour boundary; at most 14 days
    after `starting_on`; the window must not start in the future"""

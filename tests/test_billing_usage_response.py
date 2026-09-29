from datetime import datetime, timezone

import pytest

from tinker.types import BillingUsageResponse


@pytest.mark.parametrize(
    "event_info",
    [
        {"type": "training", "token_count": 100},
        {"type": "sampling_prefill", "cached": False, "token_count": 100},
        {"type": "sampling_sample", "token_count": 100},
        {"type": "checkpoint", "count": 1},
        {"type": "storage", "gigabyte_hours": 1.5},
    ],
)
def test_billing_usage_response_ignores_future_fields(event_info: dict[str, object]) -> None:
    response = BillingUsageResponse.model_validate(
        {
            "data": [
                {
                    "bucket_start": "2026-09-26T00:00:00Z",
                    "bucket_end": "2026-09-26T01:00:00Z",
                    "event_info": {
                        **event_info,
                        "new_quantity": 2,
                    },
                    "estimated_cost_usd": 0.01,
                    "new_event_field": {"source": "invoice"},
                }
            ],
            "sessions": {"session-1": {"new_session_field": "value"}},
            "cost_data_through": "2026-09-26T00:00:00Z",
            "new_response_field": {"version": 2},
        }
    )

    assert response.cost_data_through == datetime(2026, 9, 26, tzinfo=timezone.utc)
    assert response.data[0].estimated_cost_usd == 0.01
    assert response.data[0].event_info.type == event_info["type"]

    serialized = response.model_dump(mode="json")
    assert "new_response_field" not in serialized
    assert "new_event_field" not in serialized["data"][0]
    assert "new_quantity" not in serialized["data"][0]["event_info"]
    assert "new_session_field" not in serialized["sessions"]["session-1"]

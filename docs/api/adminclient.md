AdminClient for Tinker API REST operations that need an admin role.

## `AdminClient` Objects

```python
class AdminClient(TelemetryProvider)
```

Client for REST API operations that need an organization admin role.

Every method here is gated on a role in your organization: billing
methods need the admin or billing-admin role, and the audit log needs
the admin role. The same credentials and session-less connections as
`RestClient` are used, and the client is constructed the same way.

Key methods:
- get_billing_usage() - hourly-bucketed billing usage events for a time window
- get_current_checkpoint_storage_usage() - current checkpoint count, size, and projected cost
- get_audit_log() - audit log of events for the organization

Args:
- `base_url`: The Tinker API URL. Defaults to the `TINKER_BASE_URL`
  environment variable, then to the production API.
- `api_key`: The Tinker API key. Defaults to the `TINKER_API_KEY`
  environment variable, then to the stored credentials.
- `_holder`: Internal use only. The client managing HTTP connections and
  async operations that a `RestClient` hands over for its deprecated admin
  methods; it cannot be combined with `base_url` or `api_key`.

Example:
```python
admin_client = tinker.AdminClient()
usage = admin_client.get_billing_usage(
    "2026-07-13T00:00:00Z", "2026-07-14T00:00:00Z"
).result()
print(f"{len(usage.data)} billing events")
```

#### `get_audit_log`

```python
def get_audit_log(
        event_type: Literal["all", "checkpoints", "projects", "teams",
                            "organizations"] = "all",
        day: date | None = None) -> ConcurrentFuture[types.AuditLogResponse]
```

Get an audit log of events for the caller's organization.

Requires the tinker-admin RBAC role (VIEW_AUDIT_LOG capability).

Args:
- `event_type`: Which resource's events to include: "checkpoints",
    "projects", "teams", "organizations", or "all". Defaults to "all".
- `day`: The date to query (default: today). The window covers
    midnight to midnight UTC.

Returns:
- A `Future` containing the `AuditLogResponse` with audit log entries

Example:
```python
from datetime import date

future = admin_client.get_audit_log()
response = future.result()
print(f"Found {len(response.entries)} audit entries")
for entry in response.entries:
    print(f"  {entry.timestamp}: {entry.event} {entry.event_details}")

# Query a specific day
future = admin_client.get_audit_log(day=date(2025, 1, 15))
```

#### `get_audit_log_async`

```python
async def get_audit_log_async(
        event_type: Literal["all", "checkpoints", "projects", "teams",
                            "organizations"] = "all",
        day: date | None = None) -> types.AuditLogResponse
```

Async version of get_audit_log.

#### `get_billing_usage`

```python
def get_billing_usage(
    starting_on: datetime | str, ending_before: datetime | str
) -> ConcurrentFuture[types.BillingUsageResponse]
```

Get detailed billing usage for your organization.

Returns hourly-bucketed usage as a list of `BillingUsageEvent`
envelopes: the shared attribution (bucket, base model, user, session,
project) lives on the envelope, and the usage-kind-specific payload
is `event_info` — a union discriminated on `.type` (training /
sampling_prefill / sampling_sample / checkpoint / storage). Each
variant carries exactly the fields that apply (token_count, gigabyte_hours,
count, the prefill cached flag). Session user_metadata comes once
per session in `response.sessions`, keyed by session_id. Token and
storage events include `estimated_cost_usd`, an estimated gross usage
cost before credits and commits, plus the applicable effective rate
per million tokens or per GB-month. These are not invoice amounts due.
Token cost is the rate per million tokens multiplied by `token_count /
1_000_000`; storage cost is the rate per GB-month multiplied by
`gigabyte_hours / 720`.
For a completed UTC day, each model/usage-category group is priced only
after its full-day quantity reconciles with finalized invoice usage
line items, or the latest draft when no finalized invoice is available.
Other groups on the same day may remain null. The current incomplete
UTC day instead uses the published Tinker rate-card snapshot, so its
estimate is not invoice-reconciled and can change after the day
completes.
Estimated costs can use either invoice-derived effective rates or
fixed published rates. `cost_data_through` marks the exclusive UTC
boundary between those pricing sources: non-null estimates before it
use invoice-derived rates, while newer non-null estimates use fixed
published rates. A cost or rate may still be null when the data
required to price that row is unavailable. It is null when the
response has no completed day with invoice pricing data.
Data lags real time by up to a few hours. Requires billing view access
in your organization.

Args:
- `starting_on`: Inclusive window start (RFC 3339 string or datetime),
  aligned to a UTC hour boundary; must not be in the future
- `ending_before`: Exclusive window end, aligned to a UTC hour
  boundary; at most 14 days after `starting_on`

Returns:
- A `Future` containing the `BillingUsageResponse`

Example:
```python
future = admin_client.get_billing_usage(
    "2026-07-13T00:00:00Z", "2026-07-14T00:00:00Z"
)
for event in future.result().data:
    match event.event_info:
        case types.StorageBillingEvent() as info:
            print(event.bucket_start, "storage", info.gigabyte_hours, "GB-h")
        case types.CheckpointBillingEvent() as info:
            print(event.bucket_start, "checkpoints", info.count)
        case info:
            print(event.bucket_start, info.type, event.base_model, info.token_count)
```

#### `get_billing_usage_async`

```python
async def get_billing_usage_async(
        starting_on: datetime | str,
        ending_before: datetime | str) -> types.BillingUsageResponse
```

Async version of get_billing_usage.

#### `get_current_checkpoint_storage_usage`

```python
def get_current_checkpoint_storage_usage(
    project_id: str | None = None
) -> ConcurrentFuture[types.CurrentCheckpointStorageUsageResponse]
```

Get current checkpoint count, size, and projected storage cost.

With no arguments, returns checkpoint storage usage for the entire authenticated
organization, grouped by project and session owner. Pass `project_id` to limit
results to one project within that organization. Checkpoint usage data can lag
by 1-2 hours. Projected costs use current publicly available rates, before credits and commits,
and are not invoice amounts due.

#### `get_current_checkpoint_storage_usage_async`

```python
async def get_current_checkpoint_storage_usage_async(
    project_id: str | None = None
) -> types.CurrentCheckpointStorageUsageResponse
```

Async version of `get_current_checkpoint_storage_usage`.

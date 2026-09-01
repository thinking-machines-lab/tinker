"""The served payload here is what `wire()` in tinker_backend produces."""

from tinker.types import AuditLogEntry

CHECKPOINT_READ = {
    "timestamp": "2026-08-19T12:00:00Z",
    "event": "checkpoint_read",
    "event_details": {
        "organization_urn": "tml:organization:bw",
        "actor": {"type": "session", "session_id": "sess-1"},
        "resource": "checkpoint",
        "model_id": "sess-1:train:2",
        "tinker_path": "tinker://sess-1:train:2/sampler_weights/0",
        "action": {"type": "read", "purpose": "sampling"},
    },
    "model_id": "sess-1:train:2",
    "tinker_path": "tinker://sess-1:train:2/sampler_weights/0",
    "purpose": "sampling",
}

PROJECT_GRANT_SET = {
    "timestamp": "2026-08-19T12:00:00Z",
    "event": "project_grant_set",
    "event_details": {
        "organization_urn": "tml:organization:bw",
        "actor": {
            "type": "user",
            "urn": "tml:organization_user:u1",
            "email": "admin@bw.com",
        },
        "resource": "project",
        "project_id": "p1",
        "project_name": "research",
        "action": {
            "type": "grant_set",
            "principal_type": "team",
            "principal_id": "t1",
            "project_role": "project_member",
        },
    },
}


def test_a_checkpoint_entry_reads_both_ways() -> None:
    """The deprecated fields still resolve, and `event_details` carries the same values."""
    entry = AuditLogEntry.model_validate(CHECKPOINT_READ)

    assert entry.event == "checkpoint_read"
    assert (entry.model_id, entry.tinker_path, entry.purpose) == (
        "sess-1:train:2",
        "tinker://sess-1:train:2/sampler_weights/0",
        "sampling",
    )
    assert entry.event_details["action"] == {"type": "read", "purpose": "sampling"}


def test_an_entry_with_no_legacy_shape_reads_from_event_details() -> None:
    """Resources added after `event_details` leave the deprecated fields unset."""
    entry = AuditLogEntry.model_validate(PROJECT_GRANT_SET)

    assert entry.event == "project_grant_set"
    assert entry.model_id is None
    assert entry.event_details["actor"] == {
        "type": "user",
        "urn": "tml:organization_user:u1",
        "email": "admin@bw.com",
    }
    assert entry.event_details["project_name"] == "research"


def test_every_entry_names_an_actor() -> None:
    """`actor` is required on the record, so both kinds carry their tag."""
    for payload, kind in ((CHECKPOINT_READ, "session"), (PROJECT_GRANT_SET, "user")):
        actor = AuditLogEntry.model_validate(payload).event_details["actor"]
        assert isinstance(actor, dict) and actor["type"] == kind

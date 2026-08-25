"""The rest client's read side: the endpoints the console and the CLI live on."""

from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def our_training_run(rest_client, training_clients, model: str):
    """This session's own run. The org's other runs belong to other people."""
    run_id = training_clients(model).model_id
    listed = rest_client.list_training_runs(limit=100).result().training_runs

    ours = [run for run in listed if run.training_run_id == run_id]
    assert ours, f"this session's run {run_id} is missing from list_training_runs"
    return ours[0]


def test_list_training_runs_returns_our_run(our_training_run, model: str) -> None:
    assert our_training_run.base_model == model
    assert our_training_run.model_owner.startswith("tml:organization_user:")
    assert our_training_run.is_lora


def test_get_training_run_matches_the_listing(rest_client, our_training_run) -> None:
    fetched = rest_client.get_training_run(our_training_run.training_run_id).result()
    assert fetched.training_run_id == our_training_run.training_run_id
    assert fetched.base_model == our_training_run.base_model
    assert not fetched.corrupted, "the run this session just created is corrupted"


def test_sessions_resolve_to_their_runs_and_samplers(rest_client) -> None:
    sessions = rest_client.list_sessions(limit=20).result().sessions
    assert sessions, "a session was opened to create the training client"

    # Any one will do: a session opened by a concurrent run may still be empty.
    resolved = [rest_client.get_session(s).result() for s in sessions]
    assert any(r.training_run_ids or r.sampler_ids for r in resolved), (
        "no session lists a training run or a sampler"
    )


def test_audit_log_records_organization_events(rest_client) -> None:
    """Needs the tinker-admin role, and covers two days: the window is UTC daily."""
    today = datetime.now(timezone.utc).date()
    entries = [
        entry
        for day in (today, today - timedelta(days=1))
        for entry in rest_client.get_audit_log(day=day).result().entries
    ]
    assert entries, "no audit events in two days of an org running this suite"

    for entry in entries:
        assert entry.event
        assert entry.timestamp

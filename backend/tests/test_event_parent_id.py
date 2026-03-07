"""Tests that experiment lifecycle events correctly set parent_id to the root experiment.created event."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from service.api import app

client = TestClient(app)

FAKE_EXPERIMENT = {
    "id": "042",
    "proposal_id": "042-test",
    "status": "created",
    "agent_id": "agent-1",
    "config": {},
    "cost_estimate": None,
}

FAKE_CREATED_EVENT = {
    "id": 1,
    "type": "experiment.created",
    "experiment_id": "042",
    "agent_id": "agent-1",
    "summary": "Experiment 042 created from proposal 042-test",
    "parent_id": None,
}

FAKE_STORE_RESULT = {
    "code_dir": "/tmp/experiments/042/code",
    "code_hash": "abc123def456",
    "manifest": {"total_files": 2},
}


@pytest.fixture(autouse=True)
def _patch_db_init():
    """Prevent actual DB init on app startup."""
    with patch("service.db.init_db"):
        yield


# ── POST /experiments ────────────────────────────────────────────────────────


@patch("service.api.event_bus")
@patch("service.api.db")
def test_create_experiment_emits_created_event_without_parent(mock_db, mock_event_bus):
    mock_db.get_experiment.return_value = None
    mock_db.create_experiment.return_value = FAKE_EXPERIMENT
    mock_event_bus.emit.return_value = FAKE_CREATED_EVENT

    resp = client.post("/experiments", json={"proposal_id": "042-test", "agent_id": "agent-1"})
    assert resp.status_code == 200

    mock_event_bus.emit.assert_called_once_with(
        "experiment.created",
        "Experiment 042 created from proposal 042-test",
        experiment_id="042",
        agent="agent-1",
    )


@patch("service.api.experiment_store")
@patch("service.api.event_bus")
@patch("service.api.db")
def test_create_experiment_with_code_sets_parent_id(mock_db, mock_event_bus, mock_store):
    mock_db.get_experiment.return_value = None
    mock_db.create_experiment.return_value = FAKE_EXPERIMENT
    mock_event_bus.emit.return_value = FAKE_CREATED_EVENT
    mock_store.store_code_from_dict.return_value = FAKE_STORE_RESULT

    resp = client.post(
        "/experiments",
        json={
            "proposal_id": "042-test",
            "agent_id": "agent-1",
            "code_files": {"train.py": "print('hi')"},
        },
    )
    assert resp.status_code == 200

    calls = mock_event_bus.emit.call_args_list
    assert len(calls) == 2

    # First call: experiment.created (no parent_id)
    assert calls[0].kwargs.get("parent_id") is None or "parent_id" not in calls[0].kwargs

    # Second call: experiment.code_written (parent_id = root event id)
    _, kwargs = calls[1]
    assert kwargs["parent_id"] == FAKE_CREATED_EVENT["id"]
    assert calls[1][0][0] == "experiment.code_written"


# ── POST /experiments/{id}/code ──────────────────────────────────────────────


@patch("service.api.experiment_store")
@patch("service.api.event_bus")
@patch("service.api.db")
def test_store_code_sets_parent_id(mock_db, mock_event_bus, mock_store):
    mock_db.get_experiment.return_value = FAKE_EXPERIMENT
    mock_store.store_code_from_dict.return_value = FAKE_STORE_RESULT
    mock_event_bus.get_root_event.return_value = 1

    resp = client.post(
        "/experiments/042/code",
        json={"files": {"train.py": "print('hi')"}},
    )
    assert resp.status_code == 200

    mock_event_bus.get_root_event.assert_called_once_with("042")
    mock_event_bus.emit.assert_called_once()
    _, kwargs = mock_event_bus.emit.call_args
    assert kwargs["parent_id"] == 1


@patch("service.api.experiment_store")
@patch("service.api.event_bus")
@patch("service.api.db")
def test_store_code_parent_id_none_when_no_root_event(mock_db, mock_event_bus, mock_store):
    mock_db.get_experiment.return_value = FAKE_EXPERIMENT
    mock_store.store_code_from_dict.return_value = FAKE_STORE_RESULT
    mock_event_bus.get_root_event.return_value = None

    resp = client.post(
        "/experiments/042/code",
        json={"files": {"train.py": "print('hi')"}},
    )
    assert resp.status_code == 200

    _, kwargs = mock_event_bus.emit.call_args
    assert kwargs["parent_id"] is None


# ── PATCH /experiments/{id} ──────────────────────────────────────────────────


@patch("service.api.event_bus")
@patch("service.api.db")
def test_update_experiment_status_sets_parent_id(mock_db, mock_event_bus):
    mock_db.get_experiment.return_value = FAKE_EXPERIMENT
    mock_db.update_experiment.return_value = {**FAKE_EXPERIMENT, "status": "completed"}
    mock_event_bus.get_root_event.return_value = 1

    resp = client.patch("/experiments/042", json={"status": "completed"})
    assert resp.status_code == 200

    mock_event_bus.get_root_event.assert_called_once_with("042")
    _, kwargs = mock_event_bus.emit.call_args
    assert kwargs["parent_id"] == 1


@patch("service.api.event_bus")
@patch("service.api.db")
def test_update_experiment_completed_event_type(mock_db, mock_event_bus):
    mock_db.get_experiment.return_value = FAKE_EXPERIMENT
    mock_db.update_experiment.return_value = {**FAKE_EXPERIMENT, "status": "completed"}
    mock_event_bus.get_root_event.return_value = 5

    client.patch("/experiments/042", json={"status": "completed"})

    args, kwargs = mock_event_bus.emit.call_args
    assert args[0] == "experiment.completed"
    assert kwargs["parent_id"] == 5


@patch("service.api.event_bus")
@patch("service.api.db")
def test_update_experiment_failed_event_type(mock_db, mock_event_bus):
    mock_db.get_experiment.return_value = FAKE_EXPERIMENT
    mock_db.update_experiment.return_value = {**FAKE_EXPERIMENT, "status": "failed"}
    mock_event_bus.get_root_event.return_value = 5

    client.patch("/experiments/042", json={"status": "failed"})

    args, kwargs = mock_event_bus.emit.call_args
    assert args[0] == "experiment.failed"
    assert kwargs["parent_id"] == 5


@patch("service.api.event_bus")
@patch("service.api.db")
def test_update_experiment_generic_status_uses_updated_event_type(mock_db, mock_event_bus):
    mock_db.get_experiment.return_value = FAKE_EXPERIMENT
    mock_db.update_experiment.return_value = {**FAKE_EXPERIMENT, "status": "running"}
    mock_event_bus.get_root_event.return_value = 5

    client.patch("/experiments/042", json={"status": "running"})

    args, kwargs = mock_event_bus.emit.call_args
    assert args[0] == "experiment.updated"
    assert kwargs["parent_id"] == 5


@patch("service.api.event_bus")
@patch("service.api.db")
def test_update_experiment_no_status_no_event(mock_db, mock_event_bus):
    """When only non-status fields are updated, no event should be emitted."""
    mock_db.get_experiment.return_value = FAKE_EXPERIMENT
    mock_db.update_experiment.return_value = {**FAKE_EXPERIMENT, "wandb_url": "https://wandb.ai/run/1"}

    client.patch("/experiments/042", json={"wandb_url": "https://wandb.ai/run/1"})

    mock_event_bus.emit.assert_not_called()


# ── event_bus.get_root_event ─────────────────────────────────────────────────


@patch("service.event_bus._db_list")
def test_get_root_event_returns_id(mock_db_list):
    mock_db_list.return_value = [{"id": 42, "type": "experiment.created"}]

    from service.event_bus import get_root_event

    result = get_root_event("exp-1")
    assert result == 42
    mock_db_list.assert_called_once_with(
        experiment_id="exp-1",
        event_type="experiment.created",
        limit=1,
    )


@patch("service.event_bus._db_list")
def test_get_root_event_returns_none_when_no_events(mock_db_list):
    mock_db_list.return_value = []

    from service.event_bus import get_root_event

    result = get_root_event("exp-nonexistent")
    assert result is None

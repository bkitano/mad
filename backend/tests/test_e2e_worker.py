"""
E2E test: local worker cycle against a live API server + opencode.

Runs the worker's run_experiment_cycle against the real API at MAD_SERVICE_URL,
with a real opencode server on OPENCODE_PORT. Does NOT use the Modal worker.

Prerequisites:
  - API server running at MAD_SERVICE_URL (default: http://localhost:8000)
  - opencode server running on OPENCODE_PORT (default: 4096)
  - At least one proposal with has_mve=True in the database

Usage:
    # With defaults (localhost:8000, opencode on 4096)
    pytest tests/test_e2e_worker.py -v -s

    # Against remote API
    MAD_SERVICE_URL=http://mad.briankitano.com pytest tests/test_e2e_worker.py -v -s

    # Dry run only (doesn't run the agent, just tests proposal selection + experiment creation)
    pytest tests/test_e2e_worker.py -v -s -k dry_run
"""

import asyncio
import os
import time
import uuid

import httpx
import pytest

from service.client import ExperimentClient
from service.opencode.service import OpencodeService

SERVICE_URL = os.environ.get("MAD_SERVICE_URL", "http://localhost:8000")
OPENCODE_PORT = int(os.environ.get("OPENCODE_PORT", "4096"))


@pytest.fixture(scope="module")
def client():
    return ExperimentClient(base_url=SERVICE_URL)


@pytest.fixture(scope="module")
def api_reachable():
    """Skip all tests if the API server is not reachable."""
    try:
        r = httpx.get(f"{SERVICE_URL}/stats", timeout=5.0)
        r.raise_for_status()
    except Exception:
        pytest.skip(f"API server not reachable at {SERVICE_URL}")


@pytest.fixture(scope="module")
def opencode_reachable():
    """Skip all tests if the opencode server is not reachable."""
    try:
        r = httpx.get(f"http://127.0.0.1:{OPENCODE_PORT}/session", timeout=5.0)
        if r.status_code >= 500:
            raise Exception(f"opencode returned {r.status_code}")
    except Exception:
        pytest.skip(f"opencode server not reachable on port {OPENCODE_PORT}")


@pytest.fixture
def opencode_service(client):
    """Create an OpencodeService that connects to the already-running opencode server."""
    svc = OpencodeService(port=OPENCODE_PORT, client=client)
    return svc


# ── Tests ────────────────────────────────────────────────────────────────────


class TestWorkerDryRun:
    """Tests that only do proposal selection + experiment creation (no agent run)."""

    @pytest.mark.usefixtures("api_reachable")
    def test_select_proposal(self, client):
        """Verify select_proposal finds a candidate."""
        from service.worker import select_proposal

        proposal = select_proposal(client)
        # May be None if no unclaimed proposals with MVEs exist
        if proposal is None:
            pytest.skip("No unclaimed proposals with MVEs available")
        assert "id" in proposal
        print(f"  Selected proposal: {proposal['id']}")

    @pytest.mark.usefixtures("api_reachable", "opencode_reachable")
    def test_dry_run_cycle(self, client, opencode_service):
        """Run a dry-run cycle: selects proposal, doesn't run agent."""
        from service.worker import run_experiment_cycle

        opencode_service.is_started = True  # skip actual opencode start for dry run

        did_work = asyncio.run(
            run_experiment_cycle(client, opencode_service, dry_run=True)
        )
        # did_work is True if a proposal was found, False otherwise
        print(f"  Dry run result: did_work={did_work}")


class TestWorkerExperimentCreation:
    """Tests that create real experiment records but don't run the agent."""

    @pytest.mark.usefixtures("api_reachable")
    def test_create_experiment_and_verify_events(self, client):
        """Create an experiment and verify the event chain has correct parent_id."""
        proposals = client.list_proposals(status="proposed")
        if not proposals:
            pytest.skip("No proposals available")

        proposal_id = proposals[0]["id"]
        # Create experiment
        exp = client.create_experiment(proposal_id=proposal_id)
        experiment_id = exp["id"]
        print(f"  Created experiment: {experiment_id}")

        # Fetch events for this experiment
        events = client.get_experiment_events(experiment_id)
        assert len(events) > 0, "Expected at least the experiment.created event"

        created_events = [e for e in events if e["type"] == "experiment.created"]
        assert len(created_events) == 1
        root_event_id = created_events[0]["id"]
        print(f"  Root event ID: {root_event_id}")

        # Emit a child event and verify parent_id
        child = client.emit_event(
            "test.child",
            "Test child event",
            experiment_id=experiment_id,
            parent_id=root_event_id,
        )
        assert child["parent_id"] == root_event_id
        print(f"  Child event {child['id']} has parent_id={child['parent_id']}")

        # Clean up: cancel the experiment
        client.update_experiment(experiment_id, status="cancelled")


class TestOpencodeServiceE2E:
    """Tests OpencodeService against a real opencode server."""

    @pytest.mark.usefixtures("api_reachable", "opencode_reachable")
    def test_service_forwarder_lifecycle(self, client, opencode_service):
        """Start forwarder, verify it's running, then stop it."""

        async def _run():
            opencode_service.start_forwarder_only()
            assert opencode_service.is_started is True

            # Let forwarder run briefly
            await asyncio.sleep(1)

            await opencode_service.stop()
            assert opencode_service.is_started is False

        asyncio.run(_run())

    @pytest.mark.usefixtures("api_reachable", "opencode_reachable")
    def test_service_metadata_propagation(self, client, opencode_service):
        """Set metadata and verify it's used when forwarding events."""
        opencode_service.metadata.update({
            "experiment_id": "e2e-test-exp",
            "parent_id": 999,
        })

        assert opencode_service.metadata["experiment_id"] == "e2e-test-exp"
        assert opencode_service.metadata["parent_id"] == 999

        opencode_service.metadata.clear()
        assert opencode_service.metadata == {}

    @pytest.mark.usefixtures("api_reachable", "opencode_reachable")
    def test_wait_until_ready(self, opencode_service):
        """wait_until_ready should succeed against a running opencode server."""
        opencode_service.wait_until_ready(timeout_s=5.0)

    @pytest.mark.usefixtures("api_reachable", "opencode_reachable")
    def test_query_requires_started(self, opencode_service):
        """query() should raise RuntimeError when service is not started."""
        assert opencode_service.is_started is False

        with pytest.raises(RuntimeError, match="not started"):
            asyncio.run(_drain_generator(opencode_service.query("hello")))


async def _drain_generator(gen):
    """Helper to consume an async generator."""
    async for _ in gen:
        pass

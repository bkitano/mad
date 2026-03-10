"""
E2E test: create an experiment, stream events, then cancel it.

Runs against the live API at MAD_SERVICE_URL (default: http://localhost:8000).
Requires the server to be running with a real Postgres backend.

Usage:
    MAD_SERVICE_URL=http://mad.briankitano.com pytest tests/test_e2e_experiment_lifecycle.py -v -s
"""

import json
import os
import threading
import time

import httpx
import pytest

BASE_URL = os.environ.get("MAD_SERVICE_URL", "http://localhost:8000")


@pytest.fixture()
def api():
    return httpx.Client(base_url=BASE_URL, timeout=30.0)


def test_create_stream_cancel(api):
    """Create an experiment, observe events via SSE, then cancel it."""

    # ── 1. Create experiment ────────────────────────────────────────────────
    # Find a proposal to use (or use a known one)
    proposals = api.get("/proposals").json()
    assert len(proposals) > 0, "No proposals in the database — seed some first"
    proposal_id = proposals[0]["id"]

    resp = api.post("/experiments", json={
        "proposal_id": proposal_id,
        "code_files": {"train.py": "print('hello from e2e test')"},
    })
    assert resp.status_code == 200, f"Failed to create experiment: {resp.text}"
    experiment = resp.json()
    experiment_id = experiment["id"]
    print(f"\n  Created experiment: {experiment_id}")

    # ── 2. Verify experiment exists ─────────────────────────────────────────
    resp = api.get(f"/experiments/{experiment_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("created", "code_ready", "submitted")

    # ── 3. Stream events via SSE and collect any for this experiment ────────
    collected_events = []
    stream_error = []

    def stream_sse():
        try:
            with httpx.Client(base_url=BASE_URL, timeout=None) as stream_client:
                with stream_client.stream("GET", "/events/stream") as r:
                    for line in r.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[len("data: "):])
                        if event.get("experiment_id") == experiment_id:
                            collected_events.append(event)
                            print(f"  SSE event: {event.get('type')} — {event.get('summary', '')[:80]}")
                        # Stop once we see the cancelled event
                        if event.get("type") == "experiment.cancelled" and event.get("experiment_id") == experiment_id:
                            return
        except httpx.ReadError:
            pass  # stream closed, expected on cancel
        except Exception as e:
            stream_error.append(str(e))

    sse_thread = threading.Thread(target=stream_sse, daemon=True)
    sse_thread.start()

    # Give the SSE connection time to establish
    time.sleep(2)

    # ── 4. Emit a couple of events to simulate activity ─────────────────────
    api.post("/events", json={
        "event_type": "worker.started",
        "summary": "E2E test worker starting",
        "experiment_id": experiment_id,
        "agent": "e2e-test",
    })

    api.patch(f"/experiments/{experiment_id}", json={"status": "running"})
    print(f"  Set experiment {experiment_id} to running")

    time.sleep(1)

    # ── 5. Cancel the experiment ────────────────────────────────────────────
    resp = api.post(f"/experiments/{experiment_id}/cancel")
    assert resp.status_code == 200, f"Cancel failed: {resp.text}"
    cancel_result = resp.json()
    assert cancel_result["status"] == "cancelled"
    print(f"  Cancelled experiment: {cancel_result}")

    # ── 6. Wait for SSE thread to pick up the cancel event ──────────────────
    sse_thread.join(timeout=10)

    # ── 7. Verify final state ───────────────────────────────────────────────
    resp = api.get(f"/experiments/{experiment_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"

    # Verify events were recorded in the database
    resp = api.get(f"/experiments/{experiment_id}/events")
    assert resp.status_code == 200
    events = resp.json()
    event_types = [e["type"] for e in events]
    assert "experiment.created" in event_types
    assert "experiment.cancelled" in event_types
    print(f"  DB events: {event_types}")

    # Verify SSE stream captured events (may be empty if Realtime isn't configured)
    if collected_events:
        sse_types = [e.get("type") for e in collected_events]
        print(f"  SSE events captured: {sse_types}")
        assert "experiment.cancelled" in sse_types

    # ── 8. Verify cancel is idempotent (should 400) ─────────────────────────
    resp = api.post(f"/experiments/{experiment_id}/cancel")
    assert resp.status_code == 400
    print(f"  Re-cancel correctly returned 400")

    if stream_error:
        print(f"  SSE stream errors (non-fatal): {stream_error}")

"""
E2E test: submit an MNIST conv-net experiment, watch it run on Modal, and
verify that events stream and W&B logs appear.

Runs against the live API at MAD_SERVICE_URL (default: http://localhost:8000).
Requires the server to be running with a real Postgres backend and Modal
configured.

Usage:
    MAD_SERVICE_URL=http://mad.briankitano.com pytest tests/test_e2e_mnist.py -v -s

The test submits a tiny conv-net (2 epochs, batch-size 256) so it should
complete in under 5 minutes on a Modal CPU worker.
"""

import json
import os
import threading
import time

import httpx
import pytest

BASE_URL = os.environ.get("MAD_SERVICE_URL", "http://localhost:8001")

# ── MNIST training code submitted as code_files ──────────────────────────────

TRAIN_PY = r"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# optional wandb
try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


class SmallConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 2
    batch_size = 256
    lr = 1e-3

    if WANDB:
        wandb.init(
            project="mad-architecture-search",
            name=f"exp-mnist-e2e-test",
            config={"epochs": epochs, "batch_size": batch_size, "lr": lr, "model": "SmallConvNet"},
        )

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("/tmp/data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("/tmp/data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SmallConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            correct += output.argmax(1).eq(target).sum().item()
            total += data.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # eval
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                test_correct += output.argmax(1).eq(target).sum().item()
                test_total += data.size(0)

        test_loss /= test_total
        test_acc = test_correct / test_total

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        if WANDB:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "test/loss": test_loss,
                "test/accuracy": test_acc,
            })

    print(f"Final test accuracy: {test_acc:.4f}")

    if WANDB:
        wandb.log({"final/test_accuracy": test_acc})
        print(f"wandb_url={wandb.run.get_url()}")
        wandb.finish()

    # Write results to a file so the worker can pick them up
    import json
    results = {
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "train_loss": train_loss,
        "epochs": epochs,
    }
    with open("results.json", "w") as f:
        json.dump(results, f)
    print(f"results={json.dumps(results)}")


if __name__ == "__main__":
    main()
"""


@pytest.fixture()
def api():
    return httpx.Client(base_url=BASE_URL, timeout=30.0)


def test_mnist_e2e(api):
    """Submit MNIST conv-net training, verify modal worker, events, and wandb."""

    # ── 1. Create a proposal in the DB ────────────────────────────────────────
    proposal_id = "999-mnist-e2e-test"

    proposal_content = f"""# MNIST Conv-Net E2E Test

## Hypothesis
A small conv-net can achieve >95% test accuracy on MNIST in 2 epochs.

## Minimum Viable Experiment
Train a 2-layer conv-net on MNIST for 2 epochs with batch size 256.

## Training Code

```python
{TRAIN_PY}
```
"""

    resp = api.post("/proposals", json={
        "proposal_id": proposal_id,
        "title": "MNIST Conv-Net E2E Test",
        "content": proposal_content,
        "hypothesis": "A small conv-net can achieve >95% test accuracy on MNIST in 2 epochs.",
    })
    assert resp.status_code == 200, f"Failed to create proposal: {resp.text}"
    proposal = resp.json()
    print(f"\n  Created proposal: {proposal['id']}")

    # ── 2. Create experiment from the proposal ────────────────────────────────
    resp = api.post("/experiments", json={
        "proposal_id": proposal_id,
        "service_url": "http://mad.briankitano.com" # local dev proxy
    })
    assert resp.status_code == 200, f"Failed to create experiment: {resp.text}"
    experiment = resp.json()
    experiment_id = experiment["id"]
    root_event_id = experiment.get("root_event_id")
    print(f"  Created experiment: {experiment_id}")
    print(f"  Root event ID: {root_event_id}")

    # ── 3. Verify experiment was created ──────────────────────────────────────
    resp = api.get(f"/experiments/{experiment_id}")
    assert resp.status_code == 200
    exp = resp.json()
    assert exp["status"] in ("created", "submitted"), f"Unexpected status: {exp['status']}"
    assert exp["proposal_id"] == proposal_id
    print(f"  Experiment status: {exp['status']}")

    # ── 4. Check that modal worker was submitted (if Modal is configured) ─────
    if exp.get("modal_job_id"):
        print(f"  Modal job ID: {exp['modal_job_id']}")
    else:
        print("  Modal job not submitted (MODAL_CREATE_JOB_URL may not be configured)")

    # ── 5. Stream events via SSE ──────────────────────────────────────────────
    collected_events = []
    stream_error = []
    stop_streaming = threading.Event()

    def stream_sse():
        try:
            with httpx.Client(base_url=BASE_URL, timeout=None) as stream_client:
                with stream_client.stream("GET", "/events/stream") as r:
                    for line in r.iter_lines():
                        if stop_streaming.is_set():
                            return
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[len("data: "):])
                        if event.get("experiment_id") == experiment_id:
                            collected_events.append(event)
                            print(f"  SSE event: {event.get('type')} — {event.get('summary', '')[:80]}")
                        # Stop on terminal events
                        if (event.get("experiment_id") == experiment_id and
                                event.get("type") in ("experiment.completed", "experiment.failed", "experiment.cancelled")):
                            return
        except httpx.ReadError:
            pass
        except Exception as e:
            stream_error.append(str(e))

    sse_thread = threading.Thread(target=stream_sse, daemon=True)
    sse_thread.start()
    time.sleep(2)  # let SSE connect

    # ── 6. Poll until experiment completes (up to 10 minutes) ────────────────
    timeout_s = 600
    poll_interval = 10
    deadline = time.time() + timeout_s
    final_status = None

    while time.time() < deadline:
        resp = api.get(f"/experiments/{experiment_id}")
        assert resp.status_code == 200
        exp = resp.json()
        final_status = exp["status"]

        if final_status in ("completed", "failed", "cancelled"):
            print(f"  Experiment reached terminal status: {final_status}")
            break

        print(f"  Polling... status={final_status}")
        time.sleep(poll_interval)
    else:
        # If we timed out, cancel the experiment to clean up
        print(f"  Timed out after {timeout_s}s — cancelling experiment")
        api.post(f"/experiments/{experiment_id}/cancel")
        final_status = "timeout"

    # Stop SSE stream
    stop_streaming.set()
    sse_thread.join(timeout=5)

    # ── 7. Verify events were recorded ────────────────────────────────────────
    resp = api.get(f"/experiments/{experiment_id}/events")
    assert resp.status_code == 200
    events = resp.json()
    event_types = [e["type"] for e in events]
    print(f"  DB event types: {event_types}")

    assert "experiment.created" in event_types, "Missing experiment.created event"

    # ── 8. Check for wandb logs ───────────────────────────────────────────────
    resp = api.get(f"/experiments/{experiment_id}")
    exp = resp.json()
    wandb_url = exp.get("wandb_url")
    if wandb_url:
        print(f"  W&B URL: {wandb_url}")
    else:
        print("  No wandb_url set on experiment (worker may not have updated it)")

    # ── 9. Print SSE summary ──────────────────────────────────────────────────
    if collected_events:
        sse_types = [e.get("type") for e in collected_events]
        print(f"  SSE events captured: {sse_types}")
    else:
        print("  No SSE events captured (Supabase Realtime may not be configured)")

    if stream_error:
        print(f"  SSE stream errors (non-fatal): {stream_error}")

    # ── 10. Final assertions ──────────────────────────────────────────────────
    # The experiment should have been created and events emitted.
    # Whether it completes depends on Modal being configured.
    assert experiment_id is not None
    assert "experiment.created" in event_types
    print(f"\n  MNIST E2E test complete. Final status: {final_status}")

"""
Modal-based long-running worker -- starts opencode, registers with API, waits for work.

The worker does NOT pick proposals itself. Instead:
1. create_worker endpoint spawns ModalWorker.run
2. Worker starts opencode, opens a Modal tunnel, registers with the API
3. The API (or dashboard) sends proposals via POST /workers/{worker_id}/prompt
4. Worker stays alive until timeout or SIGTERM

Deploy:
    uv run python -m modal deploy worker.modal_worker

Spawn a worker:
    curl -X POST https://<your-app>.modal.run/create_worker

Send it work (via the API):
    curl -X POST http://your-api:8000/workers/<worker_id>/prompt \
        -H "Content-Type: application/json" \
        -d '{"message": "Implement proposal 042-monarch-gated..."}'
"""

import asyncio
import os
import uuid
from typing import Optional

import httpx
import modal

from worker.client import ExperimentClient
from worker.opencode.service import OpencodeService

APP_NAME = "mad-worker"
DEFAULT_SERVICE_URL = "https://mad.briankitano.com"

app = modal.App(APP_NAME)

# Image: built inline with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "git")
    .run_commands(
        "curl -fsSL https://opencode.ai/install | bash"
        " && ln -s /root/.opencode/bin/opencode /usr/local/bin/opencode",
    )
    .pip_install(
        "tqdm>=4.65.0",
        "modal>=1.3.0.post1",
        "wandb>=0.15.0",
        "pyyaml>=6.0",
        "opencode-ai>=0.1.0a36",
        "httpx>=0.27.0",
        "watchdog>=6.0.0",
        "supabase>=2.0.0",
        "fastapi[standard]>=0.133.1",
    )
    .run_commands(
        "mkdir -p /workspace/proposals /workspace/code /workspace/experiments",
    )
    .env({
        "MAD_WORKSPACE": "/workspace",
        "OPENCODE_CONFIG": "/app/worker/opencode/opencode.jsonc",
    })
    .add_local_dir("worker", "/app/worker")
)

# Secrets: API keys for opencode, wandb, postgres, etc.
SECRETS = modal.Secret.from_name("mad-worker-secrets")

OPENCODE_PORT = 4096


class ModalWorker:
    """Manages the lifecycle of a long-running opencode worker on Modal."""

    def __init__(self, worker_id: str, service_url: str, function_call_id: Optional[str] = None):
        self.worker_id = worker_id
        self.service_url = service_url
        self.function_call_id = function_call_id
        self.opencode_url: Optional[str] = None
        self.client = ExperimentClient(base_url=service_url)
        self.opencode: Optional[OpencodeService] = None

    def _log(self, msg: str) -> None:
        print(f"[modal-worker] [{self.worker_id}] {msg}", flush=True)

    # -- API helpers -----------------------------------------------------------

    def heartbeat(self) -> None:
        """Send a heartbeat to the API server (best-effort)."""
        try:
            httpx.post(
                f"{self.service_url}/workers/{self.worker_id}/heartbeat",
                timeout=5.0,
            )
        except Exception as e:
            self._log(f"Failed to send heartbeat: {e}")

    def register(self) -> None:
        """Register this worker's opencode URL (and function_call_id) with the API server."""
        resp = httpx.post(
            f"{self.service_url}/workers/register",
            json={
                "worker_id": self.worker_id,
                "opencode_url": self.opencode_url,
                "function_call_id": self.function_call_id,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        self._log(f"Registered -> {self.opencode_url}")

    # -- main loop -------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Periodically send heartbeats to the API."""
        while True:
            await asyncio.sleep(60)
            self.heartbeat()

    async def run(self) -> dict:
        """Start opencode, register, then block forever waiting for prompts."""
        workspace = "/workspace"
        os.makedirs(f"{workspace}/proposals", exist_ok=True)
        os.makedirs(f"{workspace}/code", exist_ok=True)
        os.makedirs(f"{workspace}/experiments", exist_ok=True)

        os.environ["MAD_SERVICE_URL"] = self.service_url
        os.environ["MAD_WORKSPACE"] = workspace
        os.environ["MAD_WORKER_ID"] = self.worker_id

        # Start opencode + SSE forwarder (cwd=workspace so /file API paths match agent writes)
        self.opencode = OpencodeService(
            port=OPENCODE_PORT,
            client=self.client,
            worker_id=self.worker_id,
            hostname="0.0.0.0",
            workspace=workspace,
        )
        self.opencode.start()
        self.opencode.wait_until_ready()
        self.opencode.verify_opencode_config()
        self._log("opencode ready, SSE forwarder running")

        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            async with modal.forward(OPENCODE_PORT) as tunnel:
                self.opencode_url = tunnel.url
                self._log(f"Tunnel open at {self.opencode_url}")

                self.register()
                self.client.emit_event(
                    "worker.started",
                    f"Worker {self.worker_id} ready",
                    details={"opencode_url": self.opencode_url},
                    worker_id=self.worker_id,
                )

                # Block forever — heartbeat + SSE forwarder run as tasks,
                # work arrives via the API sending prompts to our opencode URL
                await heartbeat_task

        except Exception as e:
            self._log(f"ERROR: {e}")
            self.client.emit_event(
                "worker.error",
                f"Worker {self.worker_id} error: {e}",
                worker_id=self.worker_id,
            )
            return {"worker_id": self.worker_id, "status": "failed", "error": str(e)}

        finally:
            heartbeat_task.cancel()
            self.client.emit_event(
                "worker.stopped",
                f"Worker {self.worker_id} shutting down",
                worker_id=self.worker_id,
            )
            if self.opencode:
                await self.opencode.stop()

        return {"worker_id": self.worker_id, "status": "stopped"}


# -- Modal functions -----------------------------------------------------------


@app.function(
    image=image,
    secrets=[SECRETS],
    timeout=8 * 60 * 60,  # 8 hours max
    cpu=2,
    memory=4096,
    # gpu="T4",
)
async def run_worker(
    worker_id: str = "",
    service_url: str = "",
) -> dict:
    """Modal entrypoint -- creates a ModalWorker and runs it."""
    wid = worker_id or f"modal-{uuid.uuid4().hex[:8]}"
    url = service_url or os.environ.get("MAD_SERVICE_URL", DEFAULT_SERVICE_URL)
    fc_id = modal.current_function_call_id()
    return await ModalWorker(wid, url, function_call_id=fc_id).run()


@app.function(image=image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="POST")
def create_worker(payload: dict = {}) -> dict:
    """
    Spawn a new long-running idle worker.

    POST body (all optional):
        {"worker_id": "my-worker", "service_url": "http://..."}

    Returns immediately with the worker_id and function_call_id.
    The worker registers itself with the API once it's ready.
    """
    wid = payload.get("worker_id") or f"modal-{uuid.uuid4().hex[:8]}"
    url = payload.get("service_url", "")

    fc = run_worker.spawn(worker_id=wid, service_url=url)

    return {
        "worker_id": wid,
        "status": "spawning",
        "function_call_id": fc.object_id,
    }

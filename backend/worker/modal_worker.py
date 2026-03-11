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

import os
import subprocess
import time
import uuid
from typing import Optional

import httpx
import modal

APP_NAME = "mad-worker"
DEFAULT_SERVICE_URL = "https://mad.briankitano.com"

app = modal.App(APP_NAME)

# Image: pull from GHCR (built by CI) instead of building inline
WORKER_IMAGE = os.environ.get("MAD_WORKER_IMAGE", "ghcr.io/bkitano/mad-worker:latest")
image = (
    modal.Image.from_registry(WORKER_IMAGE)
    .dockerfile_commands("ENTRYPOINT []")  # clear Dockerfile ENTRYPOINT so Modal can use its own
    .run_commands(
        # Install fastapi into both system python (for Modal's deploy check)
        # and the venv (for runtime). The venv has no pip, so we use uv.
        "/usr/local/bin/python -m pip install 'fastapi[standard]'",
        "curl -LsSf https://astral.sh/uv/install.sh | sh"
        ' && /root/.local/bin/uv pip install --python /app/.venv/bin/python "fastapi[standard]"',
    )
)

# Secrets: API keys for opencode, wandb, postgres, etc.
SECRETS = modal.Secret.from_name("mad-worker-secrets")


class ModalWorker:
    """Manages the lifecycle of a long-running opencode worker on Modal."""

    def __init__(self, worker_id: str, service_url: str, function_call_id: Optional[str] = None):
        self.worker_id = worker_id
        self.service_url = service_url
        self.function_call_id = function_call_id
        self.opencode_url: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None

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

    def emit_event(self, event_type: str, summary: str, details: Optional[dict] = None) -> None:
        """Emit an event to the API server (best-effort)."""
        try:
            httpx.post(
                f"{self.service_url}/events",
                json={
                    "event_type": event_type,
                    "summary": summary,
                    "details": details or {},
                    "worker_id": self.worker_id,
                },
                timeout=5.0,
            )
        except Exception as e:
            self._log(f"Failed to emit event: {e}")

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

    # -- opencode lifecycle ----------------------------------------------------

    def start_opencode(self, workspace: str) -> None:
        """Start the opencode subprocess bound to 0.0.0.0:4096."""
        self._proc = subprocess.Popen(
            ["opencode", "serve", "--hostname", "0.0.0.0", "--port", "4096"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=workspace,
        )

    def stop_opencode(self) -> None:
        """Terminate the opencode subprocess."""
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    def wait_for_opencode(self, timeout_s: float = 30.0) -> None:
        """Block until opencode HTTP server is ready."""
        deadline = time.time() + timeout_s
        last_err = None
        while time.time() < deadline:
            try:
                r = httpx.get("http://127.0.0.1:4096/session", timeout=2.0)
                if r.status_code < 500:
                    return
            except Exception as e:
                last_err = e
            time.sleep(0.5)
        raise RuntimeError(f"opencode not ready after {timeout_s}s; last_err={last_err!r}")

    def verify_opencode_config(self) -> None:
        """Verify the opencode server has the expected provider config."""
        r = httpx.get("http://127.0.0.1:4096/config", timeout=5.0)
        r.raise_for_status()
        config = r.json()

        providers = config.get("provider", {})
        if "opencode-go" not in providers:
            raise RuntimeError(
                f"opencode config missing 'opencode-go' provider. "
                f"Got providers: {list(providers.keys())}."
            )

        api_key = providers["opencode-go"].get("options", {}).get("apiKey", "")
        if not api_key or api_key.startswith("{env:"):
            raise RuntimeError(
                f"opencode-go provider apiKey not resolved (got '{api_key}'). "
                f"Ensure OPENCODE_GO_API_KEY is set in mad-worker-secrets."
            )

        self._log(f"opencode config verified: providers={list(providers.keys())}")

    # -- main loop -------------------------------------------------------------

    def run(self) -> dict:
        """Start opencode, register, then block forever waiting for prompts."""
        workspace = "/workspace"
        os.makedirs(f"{workspace}/proposals", exist_ok=True)
        os.makedirs(f"{workspace}/code", exist_ok=True)
        os.makedirs(f"{workspace}/experiments", exist_ok=True)

        os.environ["MAD_SERVICE_URL"] = self.service_url
        os.environ["MAD_WORKSPACE"] = workspace
        os.environ["MAD_WORKER_ID"] = self.worker_id

        self.start_opencode(workspace)

        try:
            with modal.forward(4096) as tunnel:
                self.opencode_url = tunnel.url
                self.wait_for_opencode()
                self.verify_opencode_config()
                self._log(f"opencode ready at {self.opencode_url}")

                self.register()
                self.emit_event("worker.started", f"Worker {self.worker_id} ready", {
                    "opencode_url": self.opencode_url,
                })

                # Block forever -- work arrives via the API proxying to our opencode URL
                self._log("Idle, waiting for prompts...")
                while True:
                    time.sleep(60)
                    self.heartbeat()
                    self.emit_event("worker.heartbeat", f"Worker {self.worker_id} alive", {
                        "opencode_url": self.opencode_url,
                    })

        except Exception as e:
            self._log(f"ERROR: {e}")
            self.emit_event("worker.error", f"Worker {self.worker_id} error: {e}")
            return {"worker_id": self.worker_id, "status": "failed", "error": str(e)}

        finally:
            self.emit_event("worker.stopped", f"Worker {self.worker_id} shutting down")
            self.stop_opencode()

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
def run_worker(
    worker_id: str = "",
    service_url: str = "",
) -> dict:
    """Modal entrypoint -- creates a ModalWorker and runs it."""
    wid = worker_id or f"modal-{uuid.uuid4().hex[:8]}"
    url = service_url or os.environ.get("MAD_SERVICE_URL", DEFAULT_SERVICE_URL)
    fc_id = modal.current_function_call_id()
    return ModalWorker(wid, url, function_call_id=fc_id).run()


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

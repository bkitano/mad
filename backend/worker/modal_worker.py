"""
Modal-based experiment worker -- pulls worker image from GHCR.

Deploy:
    uv run python -m modal deploy worker.modal_worker

Run a single job (for testing):
    uv run python -m modal run worker.modal_worker::run_job --job-id test-001 --proposal-id 042-monarch-gated

Trigger from API (fire-and-forget):
    curl -X POST https://<your-app>.modal.run/create_job \
        -H "Content-Type: application/json" \
        -d '{"proposal_id": "042-monarch-gated"}'
"""

import asyncio
import os
import subprocess
import time
import uuid

import modal

APP_NAME = "mad-worker"
TEMPLATE_REPO = "https://github.com/bkitano/mad-experiments-template.git"

app = modal.App(APP_NAME)

# Image: pull from GHCR (built by CI) instead of building inline
WORKER_IMAGE = os.environ.get("MAD_WORKER_IMAGE", "ghcr.io/bkitano/mad-worker:latest")
image = modal.Image.from_registry(WORKER_IMAGE, add_python="3.12")

# Secrets: API keys for opencode, wandb, postgres, etc.
# Create these in Modal dashboard: `modal secret create mad-worker-secrets ...`
SECRETS = modal.Secret.from_name("mad-worker-secrets")


def _clone_template(workspace: str, use_template: bool = False) -> None:
    """Clone the experiment template repo into the workspace, or skip if use_template is False."""
    if not use_template:
        print("[modal-worker] Skipping template clone (use_template=False)")
        return
    template_dir = f"{workspace}/template"
    if os.path.exists(template_dir):
        return
    print(f"[modal-worker] Cloning template from {TEMPLATE_REPO}")
    subprocess.run(
        ["git", "clone", "--depth", "1", TEMPLATE_REPO, template_dir],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"[modal-worker] Template cloned to {template_dir}")


def _wait_for_opencode(url: str, timeout_s: float = 30.0) -> None:
    """Block until opencode HTTP server is ready (used before OpencodeService.start)."""
    import httpx

    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/session", timeout=2.0)
            if r.status_code < 500:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"opencode not ready after {timeout_s}s; last_err={last_err!r}")


def _verify_opencode_config(url: str) -> None:
    """Verify the opencode server has the expected provider config."""
    import httpx

    r = httpx.get(f"{url}/config", timeout=5.0)
    r.raise_for_status()
    config = r.json()

    providers = config.get("provider", {})
    if "opencode-go" not in providers:
        raise RuntimeError(
            f"opencode config missing 'opencode-go' provider. "
            f"Got providers: {list(providers.keys())}. "
            f"Full config: {config}"
        )

    oc_provider = providers["opencode-go"]
    api_key = oc_provider.get("options", {}).get("apiKey", "")
    if not api_key or api_key.startswith("{env:"):
        raise RuntimeError(
            f"opencode-go provider apiKey not resolved (got '{api_key}'). "
            f"Ensure OPENCODE_GO_API_KEY is set in mad-worker-secrets."
        )

    print(f"[modal-worker] opencode config verified: providers={list(providers.keys())}")


@app.function(
    image=image,
    secrets=[SECRETS],
    timeout=60 * 60,       # 1 hour max per job
    cpu=2,
    memory=4096,
    # gpu="T4",            # uncomment if experiments need GPU
)
def run_job(
    proposal_id: str,
    job_id: str = "",
    service_url: str = "",
    use_template: bool = False,
) -> dict:
    """
    One invocation = one experiment job.

    Clones the template repo (unless use_template=False), starts opencode,
    then runs the worker's experiment cycle.
    """
    import sys
    sys.path.insert(0, "/app")

    job_id = job_id or str(uuid.uuid4())
    service_url = service_url or os.environ.get("MAD_SERVICE_URL", "http://mad.briankitano.com")

    # Set up workspace
    workspace = "/workspace"
    os.makedirs(f"{workspace}/proposals", exist_ok=True)
    os.makedirs(f"{workspace}/code", exist_ok=True)
    os.makedirs(f"{workspace}/experiments", exist_ok=True)

    # Clone experiment template (tasks/, models/, etc.)
    _clone_template(workspace, use_template=use_template)

    # Set env vars the worker expects
    worker_id = f"modal-{job_id[:8]}"
    os.environ["MAD_SERVICE_URL"] = service_url
    os.environ["MAD_WORKSPACE"] = workspace
    os.environ["MAD_WORKER_ID"] = worker_id
    os.environ["OPENCODE_BASE_URL"] = "http://127.0.0.1:4096"

    # Start opencode as a subprocess -- bind 0.0.0.0 so the tunnel proxy can reach it
    opencode_proc = subprocess.Popen(
        ["opencode", "serve", "--hostname", "0.0.0.0", "--port", "4096"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=workspace,
    )

    try:
        with modal.forward(4096) as tunnel:
            _wait_for_opencode("http://127.0.0.1:4096")
            _verify_opencode_config("http://127.0.0.1:4096")
            print(f"[modal-worker] opencode ready, running proposal {proposal_id}")
            print(f"[modal-worker] opencode public URL: {tunnel.url}")

            from worker.client import ExperimentClient
            from worker.opencode import OpencodeService
            from worker.worker import run_experiment_cycle

            client = ExperimentClient(base_url=service_url)

            # Emit event with the public opencode URL so frontends can connect
            client.emit_event(
                "worker.opencode_url",
                f"OpenCode server available at {tunnel.url}",
                details={
                    "opencode_url": tunnel.url,
                    "job_id": job_id,
                    "proposal_id": proposal_id,
                },
                worker_id=worker_id,
            )

            async def _run():
                opencode = OpencodeService(port=4096, client=client, worker_id=worker_id)
                # Server already running -- just start the forwarder
                opencode._proc = opencode_proc
                opencode.start_forwarder_only()

                try:
                    did_work = await run_experiment_cycle(
                        client, opencode, specific_proposal=proposal_id,
                    )
                    await opencode.grace_period(
                        120,
                        details={
                            "opencode_url": tunnel.url,
                            "job_id": job_id,
                            "proposal_id": proposal_id,
                            "grace_seconds": 120,
                        },
                    )
                    return did_work
                finally:
                    await opencode.stop()

            did_work = asyncio.run(_run())

            return {
                "job_id": job_id,
                "proposal_id": proposal_id,
                "opencode_url": tunnel.url,
                "status": "completed" if did_work else "no_work",
            }

    except Exception as e:
        print(f"[modal-worker] ERROR: {e}")
        return {
            "job_id": job_id,
            "proposal_id": proposal_id,
            "status": "failed",
            "error": str(e),
        }

    finally:
        opencode_proc.terminate()
        try:
            opencode_proc.wait(timeout=5)
        except Exception:
            opencode_proc.kill()


@app.function(
    image=image,
    secrets=[SECRETS],
    timeout=60 * 60,
    cpu=2,
    memory=4096,
)
def run_next_job(service_url: str = "", use_template: bool = False) -> dict:
    """
    Pick the next unclaimed proposal and run it.
    Use this for continuous polling or cron-triggered runs.
    """
    import sys
    sys.path.insert(0, "/app")

    service_url = service_url or os.environ.get("MAD_SERVICE_URL", "http://mad.briankitano.com")
    job_id = str(uuid.uuid4())
    worker_id = f"modal-{job_id[:8]}"

    workspace = "/workspace"
    os.makedirs(f"{workspace}/proposals", exist_ok=True)
    os.makedirs(f"{workspace}/code", exist_ok=True)
    os.makedirs(f"{workspace}/experiments", exist_ok=True)

    # Clone experiment template
    _clone_template(workspace, use_template=use_template)

    os.environ["MAD_SERVICE_URL"] = service_url
    os.environ["MAD_WORKSPACE"] = workspace
    os.environ["MAD_WORKER_ID"] = worker_id
    os.environ["OPENCODE_BASE_URL"] = "http://127.0.0.1:4096"

    # Bind 0.0.0.0 so the tunnel proxy can reach it
    opencode_proc = subprocess.Popen(
        ["opencode", "serve", "--hostname", "0.0.0.0", "--port", "4096"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=workspace,
    )

    try:
        with modal.forward(4096) as tunnel:
            _wait_for_opencode("http://127.0.0.1:4096")
            _verify_opencode_config("http://127.0.0.1:4096")
            print(f"[modal-worker] opencode public URL: {tunnel.url}")

            from worker.client import ExperimentClient
            from worker.opencode import OpencodeService
            from worker.worker import run_experiment_cycle

            client = ExperimentClient(base_url=service_url)

            # Emit event with the public opencode URL so frontends can connect
            client.emit_event(
                "worker.opencode_url",
                f"OpenCode server available at {tunnel.url}",
                details={
                    "opencode_url": tunnel.url,
                    "job_id": job_id,
                },
                worker_id=worker_id,
            )

            async def _run():
                opencode = OpencodeService(port=4096, client=client, worker_id=worker_id)
                opencode._proc = opencode_proc
                opencode._forwarder = asyncio.create_task(opencode._event_forwarder())

                try:
                    did_work = await run_experiment_cycle(client, opencode)
                    await opencode.grace_period(
                        120,
                        details={
                            "opencode_url": tunnel.url,
                            "job_id": job_id,
                            "grace_seconds": 120,
                        },
                    )
                    return did_work
                finally:
                    await opencode.stop()

            did_work = asyncio.run(_run())

            return {
                "job_id": job_id,
                "opencode_url": tunnel.url,
                "status": "completed" if did_work else "no_work",
            }

    except Exception as e:
        return {"job_id": job_id, "status": "failed", "error": str(e)}

    finally:
        opencode_proc.terminate()
        try:
            opencode_proc.wait(timeout=5)
        except Exception:
            opencode_proc.kill()


# -- Web endpoint: trigger a job via HTTP POST ---------------------------------
@app.function(image=image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="POST")
def create_job(payload: dict) -> dict:
    """
    HTTP trigger to start a job.

    POST body:
        {"proposal_id": "042-monarch-gated"}
        or
        {"proposal_id": "auto"}  # pick next unclaimed
    """
    proposal_id = payload.get("proposal_id", "")
    service_url = payload.get("service_url", "")
    job_id = payload.get("job_id") or str(uuid.uuid4())
    use_template = payload.get("use_template", False)

    if proposal_id == "auto" or not proposal_id:
        fc = run_next_job.spawn(service_url=service_url, use_template=use_template)
        return {"job_id": job_id, "status": "queued", "mode": "auto", "function_call_id": fc.object_id}
    else:
        fc = run_job.spawn(
            proposal_id=proposal_id,
            job_id=job_id,
            service_url=service_url,
            use_template=use_template,
        )
        return {"job_id": job_id, "proposal_id": proposal_id, "status": "queued", "function_call_id": fc.object_id}

"""
Modal Sandbox worker — runs OpenCode in a Modal Sandbox with tunnel access.

Unlike the function-based modal_worker.py, this script creates an interactive
Sandbox that you can connect to from your local terminal or browser.

Deploy:
    uv run python -m modal deploy worker.modal_sandbox_worker

API usage:
    # Create a sandbox
    curl -X POST https://<app>.modal.run/create_sandbox_worker \\
        -H "Content-Type: application/json" \\
        -d '{"github_repo": "bkitano/mad-experiments-template"}'

    # Terminate a sandbox
    curl -X POST https://<app>.modal.run/terminate_sandbox \\
        -H "Content-Type: application/json" \\
        -d '{"sandbox_id": "sb-..."}'

CLI usage:
    uv run python -m worker.modal_sandbox_worker
    uv run python -m worker.modal_sandbox_worker --github-repo myorg/myrepo --github-token $(gh auth token)
    uv run python -m worker.modal_sandbox_worker --timeout 2h
"""

import json
import os
import uuid
from pathlib import Path
from typing import Optional

import modal
from pydantic import BaseModel

MINUTES = 60
HOURS = 60 * MINUTES
OPENCODE_PORT = 4096
JUPYTER_PORT = 8888
DEFAULT_GITHUB_REPO = "bkitano/mad-experiments-template"
VOLUME_NAME_PREFIX = "mad-sandbox-"
APP_NAME = "mad-sandbox-worker"
SECRETS = modal.Secret.from_name("mad-worker-secrets")

app = modal.App(APP_NAME)

# Minimal image for the endpoint functions (not the sandbox itself)
endpoint_image = modal.Image.debian_slim().pip_install("pydantic>=2.0.0", "fastapi[standard]")


HARNESS_DIR = Path(__file__).parent.parent.parent / "harness"


def define_base_image() -> modal.Image:
    image = (
        modal.Image.debian_slim()
        .apt_install("curl", "git", "gh")
        .run_commands(
            "curl -fsSL https://opencode.ai/install | bash",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
        )
        .pip_install("jupyterlab", "pydantic>=2.0.0", "pyyaml", "httpx")
        .env({
            "PATH": "/root/.opencode/bin:/root/.local/bin:${PATH}",
            "OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS": "0",
        })
    )

    if HARNESS_DIR.exists():
        image = image.add_local_dir(str(HARNESS_DIR), "/root/harness", copy=True)

    # Inline opencode config so sandbox uses the right provider/model with full permissions.
    opencode_config = json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "model": "opencode-go/glm-5.1",
        "autoupdate": True,
        "permission": "allow",
        "server": {"port": OPENCODE_PORT},
        "provider": {
            "opencode-go": {
                "options": {
                    "apiKey": "{env:OPENCODE_GO_API_KEY}"
                }
            }
        }
    })
    image = image.run_commands(
        f"mkdir -p /root/.config/opencode && echo '{opencode_config}' > /root/.config/opencode/opencode.json"
    )

    return image


def add_modal_access(image: modal.Image) -> modal.Image:
    image = image.uv_pip_install("modal", "fastapi~=0.128.0")

    modal_token_id = os.environ.get("MODAL_TOKEN_ID")
    modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET")

    if modal_token_id and modal_token_secret:
        return image.env(
            {"MODAL_TOKEN_ID": modal_token_id, "MODAL_TOKEN_SECRET": modal_token_secret}
        )

    # When running inside a Modal endpoint, ~/.modal.toml won't exist.
    # Fall back gracefully — the sandbox can still work without modal CLI access.
    MODAL_PATH = Path("~/.modal.toml").expanduser()
    if MODAL_PATH.exists():
        print("🏖️  Including Modal auth from", MODAL_PATH)
        return image.add_local_file(MODAL_PATH, "/root/.modal.toml", copy=True)

    print("⚠️  No Modal credentials found, skipping modal access for sandbox")
    return image


def create_sandbox(
    image: modal.Image,
    timeout: int,
    app: modal.App,
    secrets: list[modal.Secret],
    volume: modal.Volume,
    repo: str,
    ref: str = "main",
    token: str | None = None,
    gpu: str | None = "T4",
    volume_name: str = "",
    criteria_yaml: str | None = None,
) -> modal.Sandbox:
    print("🏖️  Creating sandbox")

    if token:
        clone_url = f"https://oauth2:{token}@github.com/{repo}.git"
    else:
        clone_url = f"https://github.com/{repo}.git"

    criteria_cmd = ""
    if criteria_yaml:
        escaped = criteria_yaml.replace("'", "'\\''")
        criteria_cmd = f"echo '{escaped}' > /root/code/criteria.yaml && "

    entrypoint = (
        f"cd /root/code && git clone --depth 1 --branch {ref} {clone_url} . && "
        f"{criteria_cmd}"
        f"uv sync && "
        f"uv pip install ipykernel && "
        f"uv run python -m ipykernel install --user --name=mad --display-name='MAD' && "
        f"jupyter lab --ip=0.0.0.0 --port={JUPYTER_PORT} --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.disable_check_xsrf=True --ServerApp.tornado_settings='{{\"headers\":{{\"Content-Security-Policy\":\"frame-ancestors *\"}}}}' &"
        f" opencode serve --hostname=0.0.0.0 --port={OPENCODE_PORT} --log-level=DEBUG --print-logs"
    )

    with modal.enable_output():
        sb = modal.Sandbox.create(
            "bash",
            "-c",
            entrypoint,
            encrypted_ports=[OPENCODE_PORT, JUPYTER_PORT],
            secrets=secrets,
            timeout=timeout,
            image=image,
            app=app,
            gpu=gpu,
            volumes={"/root/code": volume},
            workdir="/root/code",
        )
    if volume_name:
        sb.set_tags({"volume_name": volume_name})
    return sb



# -- API models ----------------------------------------------------------------


class CreateSandboxRequest(BaseModel):
    github_repo: str = DEFAULT_GITHUB_REPO
    github_ref: str = "main"
    github_token: Optional[str] = None
    timeout_hours: int = 12
    allow_modal_access: bool = True
    gpu: Optional[str] = "T4"
    volume_name: Optional[str] = None
    criteria_yaml: Optional[str] = None


class CreateSandboxResponse(BaseModel):
    sandbox_id: str
    volume_name: str
    opencode_url: str
    webui_url: str
    jupyter_url: str
    tui_command: str
    shell_command: str
    password_secret: str
    status: str


class TerminateSandboxRequest(BaseModel):
    sandbox_id: str


# -- API endpoints -------------------------------------------------------------


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="POST")
def create_sandbox_worker(payload: CreateSandboxRequest = CreateSandboxRequest()) -> dict:
    """
    Spawn a new OpenCode sandbox and return access info.

    Returns immediately once the sandbox tunnel is ready with all the
    connection parameters needed to interact with the OpenCode server.
    """
    timeout = payload.timeout_hours * HOURS

    sandbox_app = modal.App.lookup(APP_NAME, create_if_missing=True)
    image = define_base_image()

    if payload.allow_modal_access:
        image = add_modal_access(image)

    volume_name = payload.volume_name or f"{VOLUME_NAME_PREFIX}{uuid.uuid4().hex[:8]}"
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)

    password_secret_name = "mad-worker-secrets"
    password_secret = modal.Secret.from_name(password_secret_name)

    sandbox_secrets = [SECRETS, password_secret]
    if payload.github_token:
        sandbox_secrets.append(modal.Secret.from_dict({"GH_TOKEN": payload.github_token}))

    sandbox = create_sandbox(
        image, timeout, sandbox_app, sandbox_secrets, volume,
        repo=payload.github_repo, ref=payload.github_ref, token=payload.github_token,
        gpu=payload.gpu, volume_name=volume_name,
        criteria_yaml=payload.criteria_yaml,
    )

    tunnels = sandbox.tunnels()
    opencode_tunnel = tunnels[OPENCODE_PORT]
    jupyter_tunnel = tunnels[JUPYTER_PORT]

    return CreateSandboxResponse(
        sandbox_id=sandbox.object_id,
        volume_name=volume_name,
        opencode_url=opencode_tunnel.url,
        webui_url=opencode_tunnel.url,
        jupyter_url=jupyter_tunnel.url,
        tui_command=f"OPENCODE_SERVER_PASSWORD=$PASSWORD opencode attach {opencode_tunnel.url}",
        shell_command=f"modal shell {sandbox.object_id}",
        password_secret=password_secret_name,
        status="running",
    ).model_dump()


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="POST")
def terminate_sandbox(payload: TerminateSandboxRequest) -> dict:
    """Terminate a running sandbox by its ID."""
    sandbox = modal.Sandbox.from_id(payload.sandbox_id)
    sandbox.terminate()
    return {
        "sandbox_id": payload.sandbox_id,
        "status": "terminated",
    }


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="GET")
def list_volumes() -> dict:
    """List all mad-sandbox volumes."""
    volumes = []
    for vol in modal.Volume.objects.list():
        if vol.name and vol.name.startswith(VOLUME_NAME_PREFIX):
            volumes.append({"name": vol.name, "volume_id": vol.object_id})
    return {"volumes": volumes}


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="GET")
def volume_ls(volume_name: str, path: str = "/") -> dict:
    """List files in a volume at the given path."""
    volume = modal.Volume.from_name(volume_name)
    entries = []
    for entry in volume.listdir(path, recursive=False):
        entries.append({
            "path": entry.path,
            "type": "directory" if entry.type == modal.volume.FileEntryType.DIRECTORY else "file",
        })
    return {"volume_name": volume_name, "path": path, "entries": entries}


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="GET")
def volume_read(volume_name: str, path: str) -> dict:
    """Read a file from a volume. Returns the file content as text.
    curl <endpoint> | jq -r .content > out.ipynb
    """
    import base64

    volume = modal.Volume.from_name(volume_name)
    data = b""
    for chunk in volume.read_file(path):
        data += chunk

    # Try to decode as text, fall back to base64
    try:
        content = data.decode("utf-8")
        return {"volume_name": volume_name, "path": path, "encoding": "utf-8", "content": content}
    except UnicodeDecodeError:
        content = base64.b64encode(data).decode("ascii")
        return {"volume_name": volume_name, "path": path, "encoding": "base64", "content": content}


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="GET")
def list_sandboxes() -> dict:
    """List all running sandboxes for this app."""
    sandbox_app = modal.App.lookup(APP_NAME, create_if_missing=True)
    sandboxes = []
    for sb in modal.Sandbox.list(app_id=sandbox_app.app_id):
        try:
            tunnels = sb.tunnels()
            opencode_url = tunnels[OPENCODE_PORT].url if OPENCODE_PORT in tunnels else None
            jupyter_url = tunnels[JUPYTER_PORT].url if JUPYTER_PORT in tunnels else None
        except Exception:
            opencode_url = None
            jupyter_url = None
        try:
            tags = sb.get_tags()
            vol_name = tags.get("volume_name", "")
        except Exception:
            vol_name = ""
        sandboxes.append({
            "sandbox_id": sb.object_id,
            "opencode_url": opencode_url,
            "jupyter_url": jupyter_url,
            "volume_name": vol_name,
        })
    return {"sandboxes": sandboxes}



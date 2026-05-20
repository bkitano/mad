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
import time
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

# Image for the lightweight HTTP endpoints (NOT the heavy sandbox container).
# Only needs mcp for the MCP server — volume CRUD + chat moved to the FastAPI API.
endpoint_image = (
    modal.Image.debian_slim()
    .pip_install(
        "pydantic>=2.0.0",
        "fastapi[standard]",
        "mcp>=1.0",
        "websockets>=13.0",
    )
    .add_local_python_source("worker")
)


HARNESS_DIR = Path(__file__).resolve().parent.parent.parent / "harness"


def define_base_image() -> modal.Image:
    image = (
        modal.Image.debian_slim()
        .apt_install("curl", "git", "gh")
        .run_commands(
            "curl -fsSL https://opencode.ai/install | bash",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
        )
        .pip_install("jupyterlab", "pydantic>=2.0.0", "pyyaml")
        .env({
            "PATH": "/root/.opencode/bin:/root/.local/bin:${PATH}",
            "OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS": "0",
        })
    )

    # Bake the harness evaluation scripts into the image so they're
    # always available in every sandbox at /opt/harness/
    if HARNESS_DIR.exists():
        image = image.add_local_dir(str(HARNESS_DIR), "/opt/harness/harness", copy=True)
        image = image.run_commands(
            "touch /opt/harness/__init__.py",  # make it importable as a package
        )

    # opencode config is written at runtime in the entrypoint (after volume symlinks are set up)

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
    gpu: str | None = None,
    cpu: float = 4.0,
    memory: int = 8192,
    volume_name: str = "",
    user_id: str = "",
) -> modal.Sandbox:
    print("🏖️  Creating sandbox")

    if token:
        clone_url = f"https://oauth2:{token}@github.com/{repo}.git"
    else:
        clone_url = f"https://github.com/{repo}.git"

    # Write opencode config at runtime (after symlinks, so it lands on the volume)
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

    # Mount volume at /root/state, then symlink key directories into /root
    # so that code, opencode conversations, configs, etc. all persist.
    entrypoint = (
        f"mkdir -p /root/state/code /root/state/.local /root/state/.config /root/state/.cache && "
        f"rm -rf /root/.local /root/.config /root/.cache && "
        f"ln -sfn /root/state/.local /root/.local && "
        f"ln -sfn /root/state/.config /root/.config && "
        f"ln -sfn /root/state/.cache /root/.cache && "
        f"mkdir -p /root/.config/opencode && echo '{opencode_config}' > /root/.config/opencode/opencode.json && "
        f"export UV_CACHE_DIR=/tmp/.cache/uv && "
        f"cd /root/state/code && "
        f"if [ ! -d .git ]; then git clone --depth 1 --branch {ref} {clone_url} .; fi && "
        f"uv sync && "
        f"uv pip install ipykernel && "
        f"uv run python -m ipykernel install --user --name=mad --display-name='MAD' && "
        # Copy harness evaluation scripts into the workspace so the agent can run them
        f"if [ -d /opt/harness ]; then cp -r /opt/harness/harness /root/state/code/harness 2>/dev/null || true; fi && "
        # Set PYTHONPATH so harness imports work
        f"export PYTHONPATH=/root/state/code:$PYTHONPATH && "
        # Generate accelerate config that auto-uses all available GPUs
        f"uv run python -c \""
        f"import torch, yaml, os; "
        f"n = torch.cuda.device_count(); "
        f"cfg = {{'compute_environment': 'LOCAL_MACHINE', 'mixed_precision': 'fp16', 'num_machines': 1, 'num_processes': max(n, 1), 'use_cpu': n == 0}}; "
        f"cfg['distributed_type'] = 'MULTI_GPU' if n > 1 else 'NO'; "
        f"os.makedirs(os.path.expanduser('~/.cache/huggingface/accelerate'), exist_ok=True); "
        f"yaml.dump(cfg, open(os.path.expanduser('~/.cache/huggingface/accelerate/default_config.yaml'), 'w')); "
        f"print(f'Accelerate: {{n}} GPU(s), distributed={{cfg[\\\"distributed_type\\\"]}}')\" && "
        f"rm -rf /root/.local/share/jupyter/runtime/* && "
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
            cpu=cpu,
            memory=memory,
            volumes={"/root/state": volume},
            workdir="/root/state/code",
        )
    tags = {}
    if volume_name:
        tags["volume_name"] = volume_name
    if user_id:
        tags["user_id"] = user_id
    if gpu:
        tags["gpu"] = gpu
    tags["cpu"] = str(cpu)
    tags["memory"] = str(memory)
    tags["created_at"] = str(int(time.time()))
    tags["timeout_seconds"] = str(timeout)
    if tags:
        sb.set_tags(tags)
    return sb



# -- API models ----------------------------------------------------------------


class CreateSandboxRequest(BaseModel):
    github_repo: str = DEFAULT_GITHUB_REPO
    github_ref: str = "main"
    github_token: Optional[str] = None
    timeout_hours: int = 6
    allow_modal_access: bool = True
    gpu: Optional[str] = None  # T4, L4, A10G, L40S, A100, A100-80GB, H100, or "T4:4" for multi-GPU
    cpu: float = 4.0           # CPU cores
    memory: int = 8192         # Memory in MiB (default 8 GiB)
    volume_name: Optional[str] = None
    user_id: Optional[str] = None


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
        gpu=payload.gpu, cpu=payload.cpu, memory=payload.memory, volume_name=volume_name,
        user_id=payload.user_id or "",
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


# -- MCP server ----------------------------------------------------------------
# Exposes the volume-inspection tools to any MCP client (Claude Desktop, etc.)
# over the Streamable HTTP transport at /mcp.


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.asgi_app()
def mcp_server():
    from mcp.server.fastmcp import FastMCP

    from worker import volume_tools

    # Alias module-level create_sandbox to avoid name collision with the MCP tool
    _create_sandbox = create_sandbox

    server = FastMCP("mad-volume", stateless_http=True)

    def _get_jupyter_url_mcp(sandbox_id: str) -> str:
        import modal as _modal
        sb = _modal.Sandbox.from_id(sandbox_id)
        tunnels = sb.tunnels()
        if JUPYTER_PORT not in tunnels:
            raise ValueError(f"Sandbox {sandbox_id} has no Jupyter tunnel")
        return tunnels[JUPYTER_PORT].url

    @server.tool(description="List every Modal volume in the workspace.")
    def list_volumes() -> list[dict]:
        return volume_tools.list_volumes()

    @server.tool(description="List the immediate contents of a path on a volume (non-recursive).")
    def list_files(volume_name: str, path: str = "/") -> list[dict]:
        return volume_tools.list_files(volume_name, path)

    @server.tool(description="Read a file from a volume. Returns UTF-8 text or base64 if binary.")
    def read_file(volume_name: str, path: str, max_bytes: int = 200_000) -> dict:
        return volume_tools.read_file(volume_name, path, max_bytes)

    @server.tool(description="Read a Jupyter notebook (.ipynb) with image outputs stripped.")
    def read_notebook(volume_name: str, path: str) -> dict:
        return volume_tools.read_notebook(volume_name, path)

    @server.tool(description="Recursively grep a volume for a Python regex. Skips binary files.")
    def grep(volume_name: str, pattern: str, path: str = "/", max_matches: int = 200) -> dict:
        return volume_tools.grep(volume_name, pattern, path, max_matches)

    @server.tool(description="Create a new sandbox with a volume attached (creates the volume if it doesn't exist). Returns sandbox URLs (OpenCode, Jupyter) for interactive access. Use this to spin up compute for running code, editing files, or executing experiments.")
    def create_sandbox(
        volume_name: str,
        github_repo: str = DEFAULT_GITHUB_REPO,
        github_ref: str = "main",
        gpu: str = "",
        cpu: float = 4.0,
        memory: int = 8192,
        timeout_hours: int = 6,
    ) -> dict:
        """Create a sandbox with the given volume mounted."""
        import modal as _modal

        timeout = timeout_hours * HOURS
        sandbox_app = _modal.App.lookup(APP_NAME, create_if_missing=True)
        image = define_base_image()
        image = add_modal_access(image)

        volume = _modal.Volume.from_name(volume_name, create_if_missing=True)
        password_secret = _modal.Secret.from_name("mad-worker-secrets")

        sb = _create_sandbox(
            image, timeout, sandbox_app, [SECRETS, password_secret], volume,
            repo=github_repo, ref=github_ref, token=None,
            gpu=gpu, cpu=cpu, memory=memory, volume_name=volume_name,
        )

        tunnels = sb.tunnels()
        opencode_tunnel = tunnels[OPENCODE_PORT]
        jupyter_tunnel = tunnels[JUPYTER_PORT]

        return {
            "sandbox_id": sb.object_id,
            "volume_name": volume_name,
            "opencode_url": opencode_tunnel.url,
            "jupyter_url": jupyter_tunnel.url,
            "status": "running",
        }

    @server.tool(description="Terminate a running sandbox by its ID.")
    def terminate_sandbox_tool(sandbox_id: str) -> dict:
        """Stop a running sandbox."""
        import modal as _modal
        sb = _modal.Sandbox.from_id(sandbox_id)
        sb.terminate()
        return {"sandbox_id": sandbox_id, "status": "terminated"}

    @server.tool(description="List all running sandboxes with their URLs and attached volumes.")
    def list_sandboxes() -> list[dict]:
        """List active sandboxes."""
        import modal as _modal
        sandbox_app = _modal.App.lookup(APP_NAME, create_if_missing=True)
        sandboxes = []
        for sb in _modal.Sandbox.list(app_id=sandbox_app.app_id):
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
        return sandboxes

    @server.tool(description="Rename a volume. Useful for labeling experiments after they complete.")
    def rename_volume(old_name: str, new_name: str) -> dict:
        """Rename a Modal volume."""
        import modal as _modal
        _modal.Volume.rename(old_name, new_name)
        return {"old_name": old_name, "new_name": new_name, "status": "renamed"}

    @server.tool(description="Permanently delete a volume. This is irreversible.")
    def delete_volume(volume_name: str) -> dict:
        """Delete a Modal volume."""
        import modal as _modal
        _modal.Volume.objects.delete(volume_name)
        return {"volume_name": volume_name, "status": "deleted"}

    @server.tool(description="Check if the Jupyter server in a sandbox is up and ready. Call before kernel operations on recently created sandboxes.")
    def check_jupyter_health(sandbox_id: str) -> dict:
        """Health check for Jupyter server."""
        import httpx as _httpx
        try:
            jupyter_url = _get_jupyter_url_mcp(sandbox_id)
        except Exception as e:
            return {"sandbox_id": sandbox_id, "healthy": False, "reason": f"No Jupyter tunnel: {e}"}
        try:
            resp = _httpx.get(f"{jupyter_url}/api/status", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            return {
                "sandbox_id": sandbox_id, "healthy": True,
                "started": data.get("started"),
                "last_activity": data.get("last_activity"),
                "connections": data.get("connections", 0),
                "kernels": data.get("kernels", 0),
            }
        except _httpx.ConnectError:
            return {"sandbox_id": sandbox_id, "healthy": False, "reason": "Jupyter not reachable (may still be starting)"}
        except _httpx.TimeoutException:
            return {"sandbox_id": sandbox_id, "healthy": False, "reason": "Jupyter timed out (may still be starting)"}
        except Exception as e:
            return {"sandbox_id": sandbox_id, "healthy": False, "reason": str(e)}

    @server.tool(description="List available Jupyter kernel specs in a running sandbox.")
    def list_jupyter_kernels(sandbox_id: str) -> dict:
        """Discover which kernels are installed in a sandbox."""
        import httpx as _httpx
        jupyter_url = _get_jupyter_url_mcp(sandbox_id)
        resp = _httpx.get(f"{jupyter_url}/api/kernelspecs", timeout=10.0)
        resp.raise_for_status()
        specs = resp.json()
        return {
            "default": specs.get("default", ""),
            "kernelspecs": {
                name: {
                    "display_name": spec["spec"]["display_name"],
                    "language": spec["spec"].get("language", ""),
                }
                for name, spec in specs.get("kernelspecs", {}).items()
            },
        }

    @server.tool(description="Execute code in a Jupyter kernel inside a sandbox. The kernel persists state between calls (variables, imports, loaded data). Use kernel_name='mad' for the project environment.")
    def execute_in_jupyter(sandbox_id: str, code: str, kernel_name: str = "mad", timeout: float = 300.0) -> dict:
        """Run code on a Jupyter kernel and return output."""
        import httpx as _httpx
        import json as _json
        import uuid as _uuid
        import time as _time
        import websockets.sync.client as _ws_sync

        jupyter_url = _get_jupyter_url_mcp(sandbox_id)

        # Find or start kernel
        resp = _httpx.get(f"{jupyter_url}/api/kernels", timeout=10.0)
        resp.raise_for_status()
        kernel_id = None
        for k in resp.json():
            if k.get("name") == kernel_name and k.get("execution_state") != "dead":
                kernel_id = k["id"]
                break
        if not kernel_id:
            resp = _httpx.post(f"{jupyter_url}/api/kernels", json={"name": kernel_name}, timeout=10.0)
            resp.raise_for_status()
            kernel_id = resp.json()["id"]

        # Execute via websocket
        ws_url = jupyter_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/api/kernels/{kernel_id}/channels"
        msg_id = _uuid.uuid4().hex

        execute_request = {
            "header": {"msg_id": msg_id, "msg_type": "execute_request", "username": "", "session": _uuid.uuid4().hex, "version": "5.3"},
            "parent_header": {}, "metadata": {},
            "content": {"code": code, "silent": False, "store_history": True, "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
            "buffers": [], "channel": "shell",
        }

        outputs, errors, status = [], [], "ok"
        with _ws_sync.connect(ws_url, close_timeout=5) as ws:
            ws.send(_json.dumps(execute_request))
            deadline = _time.monotonic() + timeout
            while _time.monotonic() < deadline:
                try:
                    ws.settimeout(min(deadline - _time.monotonic(), 5.0))
                    raw = ws.recv()
                except TimeoutError:
                    continue
                msg = _json.loads(raw)
                msg_type = msg.get("msg_type") or msg.get("header", {}).get("msg_type", "")
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                if msg_type == "stream":
                    outputs.append(msg["content"].get("text", ""))
                elif msg_type in ("execute_result", "display_data"):
                    data = msg["content"].get("data", {})
                    if "text/plain" in data:
                        outputs.append(data["text/plain"])
                elif msg_type == "error":
                    errors.extend(msg["content"].get("traceback", []))
                    status = "error"
                elif msg_type == "execute_reply":
                    break

        return {
            "sandbox_id": sandbox_id, "kernel_id": kernel_id, "kernel_name": kernel_name,
            "status": status,
            "output": "\n".join(outputs)[:50_000],
            "errors": "\n".join(errors)[:10_000] if errors else None,
        }

    @server.tool(description="Run all code cells of a .ipynb notebook on a Jupyter kernel in a sandbox. Stops on first error.")
    def run_notebook_in_jupyter(sandbox_id: str, notebook_path: str, kernel_name: str = "mad", timeout_per_cell: float = 300.0) -> dict:
        """Execute a notebook end-to-end."""
        import httpx as _httpx
        jupyter_url = _get_jupyter_url_mcp(sandbox_id)

        # Read notebook via Jupyter contents API
        resp = _httpx.get(f"{jupyter_url}/api/contents/{notebook_path}", timeout=30.0)
        resp.raise_for_status()
        nb_data = resp.json()
        if nb_data.get("type") != "notebook":
            return {"error": f"Not a notebook: {notebook_path}"}

        cells = nb_data.get("content", {}).get("cells", [])
        code_cells = [(i, c) for i, c in enumerate(cells) if c.get("cell_type") == "code"]

        results = []
        for i, cell in code_cells:
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if not source.strip():
                continue
            # Reuse execute_in_jupyter for each cell
            cell_result = execute_in_jupyter(sandbox_id, source, kernel_name, timeout_per_cell)
            results.append({"cell_index": i, "source": source[:500], **{k: cell_result[k] for k in ("status", "output", "errors")}})
            if cell_result["status"] == "error":
                break

        return {
            "sandbox_id": sandbox_id, "notebook_path": notebook_path,
            "kernel_name": kernel_name,
            "cells_executed": len(results),
            "cells_total": len(code_cells),
            "results": results,
        }

    return server.streamable_http_app()



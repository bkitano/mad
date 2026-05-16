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

# Default model for the volume chat agent. Overridable per-request and via the
# OPENROUTER_MODEL_DEFAULT env var on the mad-worker-secrets Modal secret.
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
MAX_AGENT_STEPS = 12
CHAT_SESSIONS_DICT = "mad-volume-chat-sessions"

app = modal.App(APP_NAME)

# Image for the lightweight HTTP endpoints (NOT the heavy sandbox container).
# Includes mcp + openai for the MCP server and chat agent endpoints.
endpoint_image = (
    modal.Image.debian_slim()
    .pip_install(
        "pydantic>=2.0.0",
        "fastapi[standard]",
        "mcp>=1.0",
        "openai>=1.50",
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
    gpu: str | None = "T4",
    cpu: float = 4.0,
    memory: int = 32768,
    volume_name: str = "",
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
    gpu: Optional[str] = "T4"  # T4, L4, A10G, L40S, A100, A100-80GB, H100, or "T4:4" for multi-GPU
    cpu: float = 4.0           # CPU cores
    memory: int = 32768        # Memory in MiB (default 32 GiB)
    volume_name: Optional[str] = None


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
    from worker import volume_tools

    return {"volumes": volume_tools.list_volumes()}


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="GET")
def volume_ls(volume_name: str, path: str = "/") -> dict:
    """List files in a volume at the given path."""
    from worker import volume_tools

    entries = volume_tools.list_files(volume_name, path)
    return {"volume_name": volume_name, "path": path, "entries": entries}


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="GET")
def volume_read(volume_name: str, path: str) -> dict:
    """Read a file from a volume. Returns the file content as text.
    curl <endpoint> | jq -r .content > out.ipynb
    """
    from worker import volume_tools

    # max_bytes=None preserves the historical "no truncation" behavior.
    result = volume_tools.read_file(volume_name, path, max_bytes=None)
    return {
        "volume_name": volume_name,
        "path": result["path"],
        "encoding": result["encoding"],
        "content": result["content"],
    }


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


class RenameVolumeRequest(BaseModel):
    old_name: str
    new_name: str


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="POST")
def rename_volume(payload: RenameVolumeRequest) -> dict:
    """Rename a volume."""
    modal.Volume.rename(payload.old_name, payload.new_name)
    return {
        "old_name": payload.old_name,
        "new_name": payload.new_name,
        "status": "renamed",
    }


class DeleteVolumeRequest(BaseModel):
    volume_name: str


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="POST")
def delete_volume(payload: DeleteVolumeRequest) -> dict:
    """Delete a volume by name."""
    modal.Volume.objects.delete(payload.volume_name)
    return {
        "volume_name": payload.volume_name,
        "status": "deleted",
    }


# -- MCP server ----------------------------------------------------------------
# Exposes the volume-inspection tools to any MCP client (Claude Desktop, etc.)
# over the Streamable HTTP transport at /mcp.


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.asgi_app()
def mcp_server():
    from mcp.server.fastmcp import FastMCP

    from worker import volume_tools

    server = FastMCP("mad-volume", stateless_http=True)

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

    return server.streamable_http_app()


# -- Volume chat ---------------------------------------------------------------
# Chat-with-the-volume endpoint. Runs an OpenRouter-powered agent loop that
# calls the same volume_tools functions in-process. State is stashed in a
# modal.Dict so sessions survive container recycles.


class VolumeChatRequest(BaseModel):
    volume_name: str
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None


SYSTEM_PROMPT_TMPL = (
    "You are a helpful assistant with tool-access to files on the Modal volume "
    "'{volume_name}'. Use the tools to explore the volume and answer questions "
    "about the experiment(s) that ran on it.\n\n"
    "Guidelines:\n"
    "- Prefer `grep` and targeted `read_file` calls over walking the directory tree.\n"
    "- Use `read_notebook` for .ipynb files — never `read_file`.\n"
    "- When the user asks about training results, look for wandb run URLs, "
    "metric logs, and notebook outputs.\n"
    "- Be concise. Quote short excerpts rather than dumping whole files."
)


def _serialize_assistant_message(msg) -> dict:
    """Strip the OpenAI ChatCompletionMessage to what OpenRouter expects on replay."""
    out: dict = {"role": "assistant"}
    if msg.content:
        out["content"] = msg.content
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="POST")
def volume_chat(req: VolumeChatRequest) -> dict:
    """Chat with the contents of a Modal volume via an LLM + tool-use loop."""
    from fastapi import HTTPException
    from openai import OpenAI

    from worker import volume_tools

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY missing — add it to the mad-worker-secrets Modal secret.",
        )

    sessions = modal.Dict.from_name(CHAT_SESSIONS_DICT, create_if_missing=True)
    session_id = req.session_id or uuid.uuid4().hex
    history: list[dict] = sessions.get(session_id, [])

    if not history:
        history.append(
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TMPL.format(volume_name=req.volume_name),
            }
        )
    history.append({"role": "user", "content": req.message})

    model = req.model or os.environ.get("OPENROUTER_MODEL_DEFAULT") or DEFAULT_OPENROUTER_MODEL
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    final_text = ""
    for _ in range(MAX_AGENT_STEPS):
        resp = client.chat.completions.create(
            model=model,
            messages=history,
            tools=volume_tools.TOOLS_SCHEMA,
        )
        msg = resp.choices[0].message
        history.append(_serialize_assistant_message(msg))

        if not msg.tool_calls:
            final_text = msg.content or ""
            break

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = volume_tools.dispatch_tool(tc.function.name, req.volume_name, args)
                content = json.dumps(result, default=str)
            except Exception as e:
                content = json.dumps({"error": f"{type(e).__name__}: {e}"})
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
            )
    else:
        # Hit the step budget without a clean assistant-only reply. Surface the
        # last assistant content (if any) so the user sees something.
        for h in reversed(history):
            if h.get("role") == "assistant" and h.get("content"):
                final_text = h["content"]
                break

    sessions[session_id] = history

    return {
        "session_id": session_id,
        "response": final_text,
        "model": model,
        "volume_name": req.volume_name,
    }


@app.function(image=endpoint_image, secrets=[SECRETS])
@modal.fastapi_endpoint(method="POST")
def volume_chat_reset(session_id: str) -> dict:
    """Drop a chat session's history."""
    sessions = modal.Dict.from_name(CHAT_SESSIONS_DICT, create_if_missing=True)
    sessions.pop(session_id, None)
    return {"session_id": session_id, "status": "reset"}



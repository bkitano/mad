"""
Standalone experiment worker -- runs in any container, talks to the API server.

Uses the opencode agent SDK to give the agent full tool access (Read, Write,
Glob, Grep, Bash) for implementing and running experiments end-to-end.

Requirements:
  - MAD_SERVICE_URL env var (e.g. http://your-server:8000)
  - opencode installed and configured (ANTHROPIC_API_KEY or OPENAI_API_KEY)
  - Modal CLI configured (modal token set)
  - wandb configured (WANDB_API_KEY)

Usage:
    # Run one experiment cycle
    python -m worker.worker --once

    # Run continuously (poll every 5 minutes)
    python -m worker.worker

    # Run a specific proposal
    python -m worker.worker --proposal 042-monarch-gated

    # Dry run (claim + read proposal, don't run agent)
    python -m worker.worker --dry-run
"""

import argparse
import asyncio
import json
import os
import re
import signal
import uuid
from pathlib import Path
from typing import Optional

from worker.client import ExperimentClient
from worker.opencode import OpenCodeAgentOptions, OpencodeService
from worker.opencode.types import EventMessagePartUpdated, TextPart, ToolPart

WORKER_ID = os.environ.get("MAD_WORKER_ID", f"worker-{uuid.uuid4().hex[:8]}")
POLL_INTERVAL = int(os.environ.get("MAD_POLL_INTERVAL", "300"))  # seconds

# Workspace dirs inside this container
WORKSPACE = Path(os.environ.get("MAD_WORKSPACE", "/workspace"))
PROPOSALS_DIR = WORKSPACE / "proposals"
CODE_DIR = WORKSPACE / "code"
EXPERIMENTS_DIR = WORKSPACE / "experiments"


# -- System Prompt -------------------------------------------------------------

EXPERIMENT_AGENT_SYSTEM_PROMPT = """You are the Experiment Agent, an expert in implementing and running
Minimum Viable Experiments (MVEs) to validate architectural ideas quickly and cheaply.

Your goal is to implement the given proposal's MVE, run it on Modal, and report results.

## Your Knowledge Base

- **Proposals folder**: {proposals_dir} -- Contains experiment proposals with MVE sections
- **Code folder**: {code_dir} -- Where you create experiment implementations
- **Experiments folder**: {experiments_dir} -- Where you log experiment results

## What You Do

1. **Create experiment directory**: Make a new numbered directory in code/ (e.g., code/042)
2. **Start experiment log**: Create experiments/experiment-log-{exp_id}.md and begin logging
3. **Implement MVE**: Write minimal but complete code to run the experiment:
   - Model implementation (models/model_name.py)
   - Training script (train.py)
   - Config file (config.yaml)
   - Requirements (pyproject.toml)
   - Modal deployment config (modal_config.py) -- REQUIRED
   - README with setup instructions
4. **Log everything**: Update experiment-log.md throughout with attempts, bugs, fixes, decisions
5. **Run experiment**: Submit to Modal with `modal run --detach modal_config.py`
6. **Report results**: Create experiments/{exp_id}_results.md with findings

## Code Structure

```
code/{exp_id}/
├── README.md
├── pyproject.toml
├── config.yaml
├── modal_config.py        <- REQUIRED
├── models/
│   ├── __init__.py
│   └── model_name.py
├── train.py
└── evaluate.py            <- optional
```

## Modal Deployment (REQUIRED)

**ALL experiments MUST run on Modal, NOT locally.**

1. **Always create `modal_config.py`** -- configure GPU (T4 default), timeout, image, volumes
2. **Run with**: `modal run --detach modal_config.py --config config.yaml`
   - `--detach` is CRITICAL -- runs async, returns job ID immediately
3. **Log the Modal job ID and URL** from the command output
4. NEVER run `python train.py` directly

## Weights & Biases (REQUIRED)

ALL experiments MUST log to wandb:
- Project: "mad-architecture-search"
- Run name: "exp-{exp_id}"
- Log: loss curves, final metrics, hyperparams, hardware info

## Experiment Log (CRITICAL)

Maintain `experiments/experiment-log-{exp_id}.md`:
- Implementation attempts and decisions
- Bugs encountered and fixes
- Training run details (Modal job ID, URL, metrics)
- Final results and verdict

After experiments complete, create `experiments/{exp_id}_results.md` with a JSON block:

{json_block}

The `wandb_run_id` MUST be in `entity/project/run_id` format — this is used for verification.

If the experiment fails at any point:
- Write full error details to the log
- Create experiments/{exp_id}_results.md documenting the failure
- Never leave a silent failure

## Human Messages

Human operators may inject additional prompts into this session at any time. Treat these as high-priority guidance that overrides or augments your current plan.
"""


# -- Helpers -------------------------------------------------------------------


def log(msg: str):
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{WORKER_ID}] {msg}", flush=True)


def select_proposal(client: ExperimentClient) -> Optional[dict]:
    """Pick the best unimplemented proposal."""
    proposals = client.list_proposals(status="proposed")

    candidates = [
        p for p in proposals
        if p.get("has_mve")
    ]
    if not candidates:
        return None

    priority_order = {"high": 3, "medium": 2, "low": 1}
    candidates.sort(key=lambda p: -priority_order.get(p.get("priority", "low").lower(), 0))
    return candidates[0]


async def _heartbeat_loop(client: ExperimentClient, proposal_id: str, experiment_id: str, detail: str, stop: asyncio.Event, root_event_id: Optional[int] = None):
    """Send heartbeats every 30 seconds until stop is set."""
    while not stop.is_set():
        try:
            await asyncio.sleep(30)
            if not stop.is_set():
                client.emit_event(
                    "worker.heartbeat",
                    f"Worker {WORKER_ID} heartbeat on {proposal_id}",
                    experiment_id=experiment_id,
                    details={"proposal_id": proposal_id, "detail": detail},
                    parent_id=root_event_id,
                    worker_id=WORKER_ID,
                )
        except Exception:
            pass


# -- Agent Execution -----------------------------------------------------------


async def run_agent_on_proposal(
    proposal_id: str,
    proposal_content: str,
    experiment_id: str,
    opencode: OpencodeService,
) -> dict:
    """
    Run the full experiment agent on a proposal via OpencodeService.query().

    The agent has Bash/Read/Write/Glob/Grep access and handles everything:
    code generation, Modal submission, experiment logging.
    """
    # Write proposal to workspace so the agent can reference it by path
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    proposal_file = PROPOSALS_DIR / f"{proposal_id}.md"
    proposal_file.write_text(proposal_content)

    prompt = f"""Implement and run the following experiment proposal.

Proposal ID: {proposal_id}
Experiment ID: {experiment_id}

The full proposal is at: {proposal_file}

Your tasks:
1. Read the proposal to understand the MVE
2. Create code/{experiment_id}/ with a complete implementation
3. Submit to Modal with `modal run --detach modal_config.py`
4. Log the Modal job ID and URL
5. Create experiments/experiment-log-{experiment_id}.md with full details
6. Create experiments/{experiment_id}_results.md with outcomes

Work in {WORKSPACE}. All code goes under {CODE_DIR}/{experiment_id}/.
"""

    json_block = json.dumps(
        {
            "wandb_run_id": "mad-lab/mad-architecture-search/abc123",
            "modal_job_id": "fc-xyz789",
            "final_metrics": {"val_loss": 0.12, "val_acc": 0.95}
        },
        ensure_ascii=False,
        separators=(',', ':'),
    )
    system_prompt = EXPERIMENT_AGENT_SYSTEM_PROMPT.format(
        proposals_dir=str(PROPOSALS_DIR),
        code_dir=str(CODE_DIR),
        experiments_dir=str(EXPERIMENTS_DIR),
        exp_id=experiment_id,
        json_block=json_block,
    )

    messages = []
    log(f"Starting agent for {proposal_id} (experiment {experiment_id})")

    async for event in opencode.query(
        prompt=prompt,
        options=OpenCodeAgentOptions(
            model="glm5",
            system_prompt=system_prompt,
            allowed_tools=["Read", "Write", "Glob", "Grep", "Bash"],
            permission_mode="acceptEdits",
            cwd=str(WORKSPACE),
        ),
    ):
        try:
            summary = event.type
            if isinstance(event, EventMessagePartUpdated):
                part = event.part
                if isinstance(part, TextPart):
                    summary = part.text[:200].replace("\n", " ")
                    messages.append(part.text)
                elif isinstance(part, ToolPart):
                    summary = f"{part.tool} ({part.state.status})"

            log(f"  {event.type}: {summary[:200]}")
        except Exception:
            pass

    # Check for results file written by agent
    results_file = EXPERIMENTS_DIR / f"{experiment_id}_results.md"
    results_text = results_file.read_text() if results_file.exists() else ""

    return {
        "experiment_id": experiment_id,
        "messages": messages,
        "results_text": results_text,
        "status": "completed" if results_file.exists() else "unknown",
    }


# -- Main Cycle ----------------------------------------------------------------


async def run_experiment_cycle(
    client: ExperimentClient,
    opencode: OpencodeService,
    specific_proposal: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """Run one experiment cycle. Returns True if work was done."""

    # 1. Select proposal
    if specific_proposal:
        proposal = client.get_proposal(specific_proposal)
    else:
        proposal = select_proposal(client)

    if proposal is None:
        log("No available proposals to work on")
        return False

    proposal_id = proposal["id"]
    log(f"Selected proposal: {proposal_id}")

    try:
        # 2. Fetch full proposal content
        full_proposal = client.get_proposal(proposal_id)
        proposal_content = full_proposal.get("content", "")

        if dry_run:
            log(f"DRY RUN -- proposal {proposal_id} has {len(proposal_content)} chars")
            log(f"Has MVE: {full_proposal.get('has_mve')}")
            return True

        # 3. Derive experiment ID
        match = re.match(r"(\d+)", proposal_id)
        experiment_id = match.group(1).zfill(3) if match else proposal_id[:3]

        # 4. Create experiment record in API (server assigns run suffix if needed)
        exp_record = client.create_experiment(proposal_id=proposal_id, worker_id=WORKER_ID)
        experiment_id = exp_record["id"]
        root_event_id = exp_record.get("root_event_id")
        log(f"Created experiment {experiment_id}")

        # Track current experiment on client so signal handlers can reference it
        client._current_experiment_id = experiment_id
        client._current_root_event_id = root_event_id

        # Tag the forwarder so all subsequent opencode events get this context
        opencode.metadata.update({
            "experiment_id": experiment_id,
            "parent_id": root_event_id,
        })

        client.emit_event(
            "info",
            f"Worker {WORKER_ID} starting agent on {proposal_id}",
            experiment_id=experiment_id,
            parent_id=root_event_id,
            worker_id=WORKER_ID,
        )
        client.update_experiment(experiment_id, status="running")

        # 5. Run agent with background heartbeat and message relay
        stop_heartbeat = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(client, proposal_id, experiment_id, "running agent", stop_heartbeat, root_event_id=root_event_id)
        )

        try:
            result = await run_agent_on_proposal(
                proposal_id, proposal_content, experiment_id, opencode,
            )
        finally:
            stop_heartbeat.set()
            heartbeat_task.cancel()

        # 6. Store code from disk and upload artifacts
        code_src = CODE_DIR / experiment_id
        if code_src.exists():
            try:
                client.store_code_from_disk(experiment_id, str(code_src))
                log(f"Code stored for {experiment_id}")
            except Exception as e:
                log(f"Warning: code storage failed: {e}")

        # Upload experiment artifacts to Supabase Storage
        artifacts_url = None
        if os.environ.get("SUPABASE_URL"):
            try:
                from worker.artifact_upload import upload_artifacts
                artifacts_url = upload_artifacts(experiment_id, EXPERIMENTS_DIR)
                client.emit_event(
                    "experiment.artifacts_ready",
                    "Artifacts uploaded to Supabase Storage",
                    experiment_id=experiment_id,
                    details={"artifacts_url": artifacts_url},
                    parent_id=root_event_id,
                    worker_id=WORKER_ID,
                )
                log(f"Artifacts uploaded: {artifacts_url[:80]}...")
            except Exception as e:
                log(f"Warning: artifact upload failed: {e}")

        # 7. Report outcome
        log(f"Agent finished. Status: {result['status']}")
        client.update_experiment(
            experiment_id,
            status="completed" if result["status"] == "completed" else "failed",
        )
        client.emit_event(
            "completed" if result["status"] == "completed" else "failed",
            f"Agent finished: {result['status']}",
            experiment_id=experiment_id,
            details={"results": result},
            parent_id=root_event_id,
            worker_id=WORKER_ID,
        )

        return True

    except Exception as e:
        log(f"ERROR: {e}")
        client.emit_event(
            "error",
            f"Worker {WORKER_ID} error: {str(e)[:200]}",
            details={"error": str(e)},
            worker_id=WORKER_ID,
        )
        return False

    finally:
        # Clear metadata so stale context doesn't leak into the next cycle
        opencode.metadata.clear()
        client._current_experiment_id = None
        client._current_root_event_id = None
        log(f"Finished working on {proposal_id}")


# -- Entry Point ---------------------------------------------------------------


def _install_signal_handlers(client: ExperimentClient):
    """Install handlers for SIGTERM/SIGINT that emit a termination event before exiting."""
    loop = asyncio.get_running_loop()

    def _handle_signal(sig: signal.Signals):
        sig_name = sig.name
        log(f"Received {sig_name}, emitting termination event")
        try:
            experiment_id = getattr(client, "_current_experiment_id", None)
            parent_id = getattr(client, "_current_root_event_id", None)
            client.emit_event(
                "worker.terminated",
                f"Worker {WORKER_ID} terminated by {sig_name}",
                experiment_id=experiment_id,
                details={"signal": sig_name, "worker_id": WORKER_ID},
                parent_id=parent_id,
                worker_id=WORKER_ID,
            )
        except Exception as e:
            log(f"Failed to emit termination event: {e}")
        # Re-raise as KeyboardInterrupt so the asyncio loop shuts down cleanly
        raise SystemExit(128 + sig.value)

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal, sig)


async def main_async():
    parser = argparse.ArgumentParser(description="MAD Experiment Worker")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--proposal", help="Specific proposal ID to implement")
    parser.add_argument("--dry-run", action="store_true", help="Claim + read, don't run agent")
    parser.add_argument("--service-url", default=None, help="API server URL")
    parser.add_argument("--opencode-port", type=int, default=4096, help="Port for opencode server")
    parser.add_argument("--grace-seconds", type=int, default=120, help="Grace period after experiment")
    parser.add_argument("--idle", action="store_true", help="Start opencode server and sit idle (no proposals)")
    args = parser.parse_args()

    service_url = args.service_url or os.environ.get("MAD_SERVICE_URL", "http://localhost:8000")
    client = ExperimentClient(base_url=service_url)

    opencode = OpencodeService(port=args.opencode_port, client=client, worker_id=WORKER_ID)
    opencode.start()

    log(f"Worker starting, service={service_url}, workspace={WORKSPACE}")

    # Register with the API server
    try:
        _httpx = __import__("httpx")
        _httpx.post(
            f"{service_url}/workers/register",
            json={"worker_id": WORKER_ID, "opencode_url": opencode.url},
            timeout=5.0,
        )
        log(f"Registered as worker {WORKER_ID} -> {opencode.url}")
    except Exception as e:
        log(f"Warning: failed to register with API server: {e}")

    _install_signal_handlers(client)

    try:
        if args.idle:
            log("Idle mode -- opencode server running, no proposals will be processed")
            log(f"  opencode URL: {opencode.url}")
            while True:
                await asyncio.sleep(3600)

        if args.once or args.proposal:
            # Track current experiment context on client for signal handler access
            client._current_experiment_id = None
            client._current_root_event_id = None
            await run_experiment_cycle(
                client, opencode,
                specific_proposal=args.proposal, dry_run=args.dry_run,
            )
            await opencode.grace_period(
                args.grace_seconds,
                details={"opencode_url": opencode.url, "grace_seconds": args.grace_seconds},
            )
            return

        log(f"Running continuously, poll interval={POLL_INTERVAL}s")
        client._current_experiment_id = None
        client._current_root_event_id = None
        while True:
            try:
                did_work = await run_experiment_cycle(client, opencode, dry_run=args.dry_run)
                if not did_work:
                    log(f"No work available, sleeping {POLL_INTERVAL}s")
                    await asyncio.sleep(POLL_INTERVAL)
                else:
                    await asyncio.sleep(10)
            except KeyboardInterrupt:
                log("Shutting down")
                break
            except Exception as e:
                log(f"Cycle error: {e}, retrying in 60s")
                await asyncio.sleep(60)
    finally:
        await opencode.stop()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

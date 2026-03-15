"""
FastAPI service exposing the experiment execution system.

All state is in Postgres (Supabase). Events table supports Realtime SSE.

Run:
    uvicorn api.api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import httpx as _httpx
import wandb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from supabase import acreate_client

from src import event_bus
from src import experiment_store
from src.db import DatabaseManager
from src.stores import EventsStore, ExperimentsStore, ProposalsStore, WorkersStore

logger = logging.getLogger(__name__)

db = DatabaseManager()
experiments = ExperimentsStore(db)
events = EventsStore(db)
proposals = ProposalsStore(db)
workers_store = WorkersStore(db)

MODAL_CREATE_JOB_URL = os.environ.get("MODAL_CREATE_JOB_URL", "")

PROJECT_ROOT = Path(__file__).parent.parent

app = FastAPI(
    title="MAD Experiment Service",
    description="API for managing autonomous ML experiment execution",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    pass


# -- Pydantic models ----------------------------------------------------------


class CreateProposalRequest(BaseModel):
    proposal_id: str
    title: str
    content: str
    priority: Optional[str] = None
    hypothesis: Optional[str] = None
    based_on: Optional[str] = None


class CreateExperimentRequest(BaseModel):
    proposal_id: str
    cost_estimate: Optional[float] = None
    worker_id: Optional[str] = None
    service_url: Optional[str] = None  # override MAD_SERVICE_URL for Modal worker


class StoreCodeRequest(BaseModel):
    files: dict[str, str]


class UpdateExperimentRequest(BaseModel):
    status: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_url: Optional[str] = None
    results: Optional[dict] = None
    error: Optional[str] = None
    error_class: Optional[str] = None
    cost_actual: Optional[float] = None


class EmitEventRequest(BaseModel):
    event_type: str
    summary: str
    experiment_id: Optional[str] = None
    details: Optional[dict] = None
    parent_id: Optional[int] = None
    worker_id: Optional[str] = None


class DispatchTaskRequest(BaseModel):
    agent_type: str
    prompt: str
    priority: int = 5
    experiment_id: Optional[str] = None


class DirectiveUpdateRequest(BaseModel):
    agent_name: str
    directive: str


class SendMessageRequest(BaseModel):
    content: str
    sender: str = "human"


class WorkerRegisterRequest(BaseModel):
    worker_id: str
    opencode_url: str
    function_call_id: Optional[str] = None


class WorkerPromptRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    experiment_id: Optional[str] = None


# -- GET /experiments ----------------------------------------------------------


@app.get("/experiments")
def list_experiments(
    status: Optional[str] = Query(None, description="Filter by status"),
    proposal_id: Optional[str] = Query(None, description="Filter by proposal_id"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return experiments.list(status=status, proposal_id=proposal_id, limit=limit, offset=offset)


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    exp = experiments.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return exp


@app.get("/experiments/{experiment_id}/code")
def get_experiment_code(experiment_id: str):
    manifest = experiment_store.get_manifest(experiment_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"No stored code for experiment {experiment_id}")

    files = {}
    for f in manifest["files"]:
        content = experiment_store.get_file(experiment_id, f["path"])
        files[f["path"]] = content

    return {
        "experiment_id": experiment_id,
        "code_hash": manifest["code_hash"],
        "stored_at": manifest["stored_at"],
        "total_files": manifest["total_files"],
        "files": files,
    }


@app.get("/experiments/{experiment_id}/code/tree")
def get_experiment_code_tree(experiment_id: str):
    manifest = experiment_store.get_manifest(experiment_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"No stored code for experiment {experiment_id}")

    total_size = sum(f["size"] for f in manifest["files"])
    return {
        "experiment_id": experiment_id,
        "code_hash": manifest["code_hash"],
        "total_files": manifest["total_files"],
        "total_size_bytes": total_size,
        "files": [{"path": f["path"], "size": f["size"]} for f in manifest["files"]],
    }


@app.get("/experiments/{experiment_id}/code/download")
def download_experiment_code(experiment_id: str):
    exp_dir = experiment_store._exp_dir(experiment_id)
    tar_path = exp_dir / "code.tar.gz"
    if not tar_path.exists():
        raise HTTPException(status_code=404, detail=f"No stored code for experiment {experiment_id}")
    return FileResponse(
        path=str(tar_path),
        media_type="application/gzip",
        filename=f"experiment-{experiment_id}-code.tar.gz",
    )


@app.get("/experiments/{experiment_id}/code/raw/{file_path:path}")
def get_experiment_file_raw(experiment_id: str, file_path: str):
    content = experiment_store.get_file(experiment_id, file_path)
    if content is None:
        raise HTTPException(status_code=404, detail=f"File {file_path} not found in experiment {experiment_id}")

    ext = Path(file_path).suffix
    media_types = {
        ".py": "text/x-python", ".yaml": "text/yaml", ".yml": "text/yaml",
        ".json": "application/json", ".toml": "text/plain", ".md": "text/markdown",
        ".sh": "text/x-shellscript", ".txt": "text/plain",
    }
    return PlainTextResponse(content=content, media_type=media_types.get(ext, "text/plain"))


@app.get("/experiments/{experiment_id}/code/review")
def review_experiment_code(experiment_id: str):
    manifest = experiment_store.get_manifest(experiment_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"No stored code for experiment {experiment_id}")

    parts = [f"# Experiment {experiment_id} — Code Review\n"]
    parts.append(f"# Hash: {manifest['code_hash']}")
    parts.append(f"# Files: {manifest['total_files']}\n")

    for f in manifest["files"]:
        content = experiment_store.get_file(experiment_id, f["path"])
        separator = "=" * 70
        parts.append(f"\n{separator}")
        parts.append(f"# {f['path']} ({f['size']} bytes)")
        parts.append(separator)
        parts.append(content or "<empty>")

    return PlainTextResponse(content="\n".join(parts), media_type="text/plain")


@app.get("/experiments/{experiment_id}/code/{file_path:path}")
def get_experiment_file(experiment_id: str, file_path: str):
    content = experiment_store.get_file(experiment_id, file_path)
    if content is None:
        raise HTTPException(status_code=404, detail=f"File {file_path} not found in experiment {experiment_id}")
    return {"path": file_path, "content": content}


@app.get("/experiments/{experiment_id}/events")
def get_experiment_events(
    experiment_id: str,
    limit: int = Query(100, ge=1, le=1000),
):
    exp = experiments.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return event_bus.query(experiment_id=experiment_id, limit=limit)


def _verify_wandb(wandb_url: str | None, wandb_run_id: str | None) -> dict:
    """Call W&B API to validate the run actually exists and has metrics."""
    if not wandb_url and not wandb_run_id:
        return {"verified": False, "reason": "no wandb_url or wandb_run_id"}
    run_ref = wandb_run_id or wandb_url
    try:
        api = wandb.Api()
        run = api.run(run_ref)
        summary = dict(run.summary)
        return {
            "verified": run.state == "finished",
            "state": run.state,
            "duration_seconds": run.summary.get("_runtime"),
            "metrics": {k: v for k, v in summary.items() if not k.startswith("_")},
            "created_at": str(run.created_at),
        }
    except Exception as e:
        logger.error("wandb verify failed: %s: %s", type(e).__name__, e)
        return {"verified": False, "reason": str(e)}


@app.get("/experiments/{experiment_id}/verify")
def verify_experiment(experiment_id: str):
    exp = experiments.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    wandb_details = _verify_wandb(exp.get("wandb_url"), exp.get("wandb_run_id"))

    # Compare claimed metrics vs W&B actuals if both exist
    metrics_match = None
    claimed_results = exp.get("results") or {}
    claimed_metrics = claimed_results.get("final_metrics") if isinstance(claimed_results, dict) else None
    actual_metrics = wandb_details.get("metrics")
    if claimed_metrics and actual_metrics:
        discrepancies = {}
        for k in claimed_metrics:
            if k not in actual_metrics:
                continue
            cv, av = claimed_metrics[k], actual_metrics[k]
            if (isinstance(cv, bool) or isinstance(av, bool)
                    or not isinstance(cv, (int, float))
                    or not isinstance(av, (int, float))):
                continue  # skip non-numeric — don't add to discrepancies
            elif abs(cv - av) > 0.01:
                discrepancies[k] = {"claimed": cv, "actual": av}
        metrics_match = {"match": len(discrepancies) == 0, "discrepancies": discrepancies}

    checks = {
        "code_stored": experiment_store.get_code_dir(experiment_id) is not None,
        "code_integrity": experiment_store.verify_integrity(experiment_id),
        "modal_submitted": bool(exp.get("modal_job_id")),
        "modal_url": bool(exp.get("modal_url")),
        "wandb_details": wandb_details,
        "has_results": bool(exp.get("results")),
        "status": exp.get("status"),
    }
    if metrics_match is not None:
        checks["metrics_match"] = metrics_match
        # Tighten wandb_details["verified"] to require metric validation too
        wandb_details["verified"] = (
            wandb_details.get("verified", False)
            and bool(actual_metrics)
            and metrics_match["match"]
        )

    checks["fully_verified"] = all([
        checks["code_stored"],
        checks["code_integrity"],
        checks["modal_submitted"],
        wandb_details.get("verified"),
        checks["has_results"],
        checks["status"] == "completed",
        metrics_match is not None and metrics_match["match"],
    ])

    return {"experiment_id": experiment_id, "checks": checks, "experiment": exp}


@app.get("/experiments/{experiment_id}/artifacts")
def get_experiment_artifacts(experiment_id: str):
    exp = experiments.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    # Fetch artifacts_url from event bus
    artifacts_url = None
    artifacts_events = event_bus.query(
        experiment_id=experiment_id,
        event_type="experiment.artifacts_ready",
        limit=1,
    )
    if artifacts_events:
        artifacts_url = artifacts_events[0].get("details", {}).get("artifacts_url")

    code_files = experiment_store.list_files(experiment_id) if exp.get("code_hash") else []

    results = exp.get("results")
    if not results:
        completed_events = event_bus.query(
            experiment_id=experiment_id,
            event_type="completed",
            limit=1,
        )
        if completed_events:
            results = completed_events[0].get("details", {}).get("results")

    return {
        "id": experiment_id,
        "proposal_id": exp.get("proposal_id"),
        "status": exp.get("status"),
        "artifacts_url": artifacts_url,
        "code_files": code_files,
        "wandb_url": exp.get("wandb_url"),
        "results": results,
    }


# -- GET /events --------------------------------------------------------------


@app.get("/events")
def get_events(
    experiment_id: Optional[str] = None,
    event_type: Optional[str] = None,
    since: Optional[str] = None,
    parent_id: Optional[int] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return event_bus.query(
        experiment_id=experiment_id,
        event_type=event_type,
        since=since,
        parent_id=parent_id,
        limit=limit,
        offset=offset,
    )


@app.get("/events/stream")
async def stream_events():
    """SSE stream backed by Supabase Realtime."""
    supabase = await acreate_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"],
    )
    q: asyncio.Queue = asyncio.Queue()

    def on_insert(payload):
        data = payload.get("data", {})
        record = data.get("record") or payload.get("new") or payload
        q.put_nowait(record)

    channel = supabase.channel("events-sse")
    channel.on_postgres_changes(
        event="INSERT",
        schema="public",
        table="events",
        callback=on_insert,
    )
    await channel.subscribe()

    async def generate():
        try:
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await supabase.remove_channel(channel)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/events/{event_id}")
def get_event(event_id: int):
    event = events.get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
    return event


@app.get("/events/{event_id}/children")
def get_event_children(
    event_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return event_bus.query(parent_id=event_id, limit=limit, offset=offset)


# -- GET /proposals ------------------------------------------------------------


@app.get("/proposals")
def list_proposals(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    rows = proposals.list(limit=limit, offset=offset)

    return [
        {
            "id": row["proposal_id"],
            "title": row["title"],
            "priority": row["priority"],
            "created": str(row["created"]) if row.get("created") else None,
            "based_on": row["based_on"],
            "has_mve": "## Minimum Viable Experiment" in (row["content"] or ""),
        }
        for row in rows
    ]


@app.get("/proposals/{proposal_id}")
def get_proposal(proposal_id: str):
    row = proposals.get(proposal_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")

    return {
        "id": row["proposal_id"],
        "title": row["title"],
        "priority": row["priority"],
        "created": str(row["created"]) if row.get("created") else None,
        "based_on": row["based_on"],
        "hypothesis": row.get("hypothesis"),
        "has_mve": "## Minimum Viable Experiment" in (row["content"] or ""),
        "content": row["content"],
    }


# -- POST /proposals -----------------------------------------------------------


@app.post("/proposals")
def create_proposal(req: CreateProposalRequest):
    """Create or upsert a proposal in the database."""
    row = proposals.create(
        proposal_id=req.proposal_id,
        title=req.title,
        content=req.content,
        priority=req.priority,
        hypothesis=req.hypothesis,
        based_on=req.based_on,
    )
    return {
        "id": row["proposal_id"],
        "title": row["title"],
        "priority": row["priority"],
        "hypothesis": row.get("hypothesis"),
        "content": row["content"],
    }


# -- GET /stats ----------------------------------------------------------------


@app.get("/stats")
def get_stats():
    all_exps = experiments.list(limit=10000)
    status_counts = {}
    for exp in all_exps:
        s = exp.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    total_cost = sum(exp.get("cost_actual") or 0 for exp in all_exps)
    verified = sum(
        1 for exp in all_exps
        if exp.get("status") == "completed" and exp.get("wandb_url")
    )

    return {
        "total_experiments": len(all_exps),
        "by_status": status_counts,
        "total_cost": round(total_cost, 2),
        "verified_complete": verified,
    }


# -- POST /experiments ---------------------------------------------------------


@app.post("/experiments")
def create_experiment(req: CreateExperimentRequest):
    """Create a new experiment from a proposal. Auto-suffixes (e.g. 042-r2) on reruns."""
    match = re.match(r"(\d+)", req.proposal_id)
    if not match:
        raise HTTPException(status_code=400, detail="proposal_id must start with a number")
    experiment_id = match.group(1).zfill(3)

    # Auto-suffix for reruns
    run_num = 1
    if experiments.get(experiment_id):
        run_num = experiments.get_next_run_number(experiment_id)
        experiment_id = f"{experiment_id}-r{run_num}"

    # Spawn a Modal worker for this experiment if endpoint is configured
    worker_id = req.worker_id
    if not worker_id and MODAL_CREATE_JOB_URL:
        try:
            modal_resp = _httpx.post(
                MODAL_CREATE_JOB_URL,
                json={
                    "worker_id": f"exp-{experiment_id}",
                    "service_url": req.service_url or os.environ.get("MAD_SERVICE_URL", "http://mad.briankitano.com"),
                },
                timeout=15.0,
            )
            modal_resp.raise_for_status()
            modal_result = modal_resp.json()
            worker_id = modal_result.get("worker_id")
            fc_id = modal_result.get("function_call_id")
        except Exception:
            modal_result = None
            fc_id = None

    exp = experiments.create(
        experiment_id=experiment_id,
        proposal_id=req.proposal_id,
        cost_estimate=req.cost_estimate,
        worker_id=worker_id,
        run_number=run_num,
    )

    created_event = event_bus.emit(
        "experiment.created",
        f"Experiment {experiment_id} created from proposal {req.proposal_id}",
        experiment_id=experiment_id,
        worker_id=worker_id,
    )
    root_event_id = created_event.get("id")

    if not req.worker_id and MODAL_CREATE_JOB_URL:
        if worker_id and fc_id:
            experiments.update(experiment_id, modal_job_id=fc_id, status="submitted")
            event_bus.emit(
                "experiment.submitted",
                f"Modal worker {worker_id} spawned for experiment {experiment_id}",
                experiment_id=experiment_id,
                details=modal_result,
                parent_id=root_event_id,
            )
        elif not worker_id:
            event_bus.emit(
                "experiment.submit_error",
                f"Failed to spawn Modal worker for experiment {experiment_id}",
                experiment_id=experiment_id,
                parent_id=root_event_id,
            )

    exp["root_event_id"] = root_event_id

    return exp


@app.post("/experiments/{experiment_id}/code")
def store_experiment_code(experiment_id: str, req: StoreCodeRequest):
    exp = experiments.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    store_result = experiment_store.store_code_from_dict(experiment_id, req.files)
    experiments.update(
        experiment_id,
        code_dir=store_result["code_dir"],
        code_hash=store_result["code_hash"],
        status="code_ready",
    )
    event_bus.emit(
        "experiment.code_written",
        f"Code stored: {store_result['manifest']['total_files']} files",
        experiment_id=experiment_id,
        parent_id=event_bus.get_root_event(experiment_id),
    )
    return store_result


@app.post("/experiments/{experiment_id}/store-from-disk")
def store_code_from_disk(experiment_id: str, source_dir: str = Query(...)):
    exp = experiments.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    source = Path(source_dir)
    if not source.exists():
        raise HTTPException(status_code=400, detail=f"Source directory does not exist: {source_dir}")

    store_result = experiment_store.store_code(experiment_id, source)
    experiments.update(
        experiment_id,
        code_dir=store_result["code_dir"],
        code_hash=store_result["code_hash"],
        status="code_ready",
    )
    event_bus.emit(
        "experiment.code_written",
        f"Code stored from {source_dir}: {store_result['manifest']['total_files']} files",
        experiment_id=experiment_id,
        parent_id=event_bus.get_root_event(experiment_id),
    )
    return store_result


@app.patch("/experiments/{experiment_id}")
def update_experiment(experiment_id: str, req: UpdateExperimentRequest):
    exp = experiments.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated = experiments.update(experiment_id, **updates)

    if req.status:
        event_bus.emit(
            f"experiment.{req.status}" if req.status in ("completed", "failed", "cancelled") else "experiment.updated",
            f"Experiment {experiment_id} status \u2192 {req.status}",
            experiment_id=experiment_id,
            details=updates,
            parent_id=event_bus.get_root_event(experiment_id),
        )

    return updated


# -- Worker proxy --------------------------------------------------------------


def _get_worker(worker_id: str) -> dict:
    entry = workers_store.get(worker_id)
    if not entry or entry["status"] == "stopped":
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not registered")
    return entry


def _get_worker_url(worker_id: str) -> str:
    return _get_worker(worker_id)["opencode_url"]


@app.get("/workers")
def list_workers(
    include_stopped: bool = False,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List all registered workers. Stale workers are auto-detected via heartbeat TTL."""
    return workers_store.list(include_stopped=include_stopped, limit=limit, offset=offset)


@app.post("/workers/register")
async def register_worker(req: WorkerRegisterRequest):
    """Register a worker's opencode URL (and optional Modal function_call_id)."""
    opencode_url = req.opencode_url.rstrip("/")
    row = workers_store.register(
        worker_id=req.worker_id,
        opencode_url=opencode_url,
        function_call_id=req.function_call_id,
    )

    # Emit event for new worker registration
    event_bus.emit(
        "worker.registered",
        f"Worker {req.worker_id} registered with opencode URL {opencode_url}",
        worker_id=req.worker_id,
    )

    asyncio.get_event_loop().create_task(check_worker_ready(req.worker_id))
    asyncio.get_event_loop().create_task(assign_submitted_proposals_to_worker(req.worker_id))

    return row

@app.post("/workers/{worker_id}/readyz")
async def check_worker_ready(worker_id: str):

    worker = workers_store.get(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

    # check the opencode_url is responsive
    try:
        async with _httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get(f"{worker['opencode_url']}/doc")
            resp.raise_for_status()
    except Exception as e:
        event_bus.emit(
            "worker.readyz_failed",
            f"Worker {worker_id} registration succeeded but opencode URL is not responsive: {e}",
            worker_id=worker_id,
        )

@app.post("/workers/{worker_id}/assign_submitted_proposals")
async def assign_submitted_proposals_to_worker(worker_id: str):
    # send it the proposal if we have an experiment waiting for that worker
    experiments_waiting = experiments.list(status="submitted", worker_id=worker_id)

    worker = workers_store.get(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")
    
    session_ids = {}
    for exp in experiments_waiting:
        event_bus.emit(
            "worker.proposal_sent",
            f"Sending proposal {exp['proposal_id']} to worker {worker_id}",
            experiment_id=exp["id"],
            worker_id=worker_id,
        )
        proposal = proposals.get(exp.proposal_id)
        if not proposal:
            continue
        try:
            async with _httpx.AsyncClient(base_url=worker['opencode_url'], timeout=30.0) as http:
                sess_resp = await http.post("/session", json={})
                sess_resp.raise_for_status()
                session_id = sess_resp.json()["id"]

                session_ids[exp.proposal_id] = session_id

                await http.post(
                    f"/session/{session_id}/prompt_async",
                    json={"parts": [{"type": "text", "text": proposal["content"]}]},
                )
            experiments.update(exp.id, status="running")
            event_bus.emit(
                "worker.proposal_accepted",
                f"Worker {worker_id} accepted proposal {exp.proposal_id} with session_id {session_id}",
                experiment_id=exp.id,
                worker_id=worker_id,
            )
        except Exception as e:
            event_bus.emit(
                "worker.proposal_error",
                f"Failed to send proposal {exp['proposal_id']} to worker {worker_id}: {e}",
                experiment_id=exp["id"],
                worker_id=worker_id,
            )
    
    return {"worker_id": worker_id, "session_ids": session_ids}

@app.post("/workers/{worker_id}/heartbeat")
def worker_heartbeat(worker_id: str):
    """Update a worker's heartbeat timestamp."""
    row = workers_store.heartbeat(worker_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")
    return row


@app.delete("/workers/{worker_id}")
def kill_worker(worker_id: str):
    """Stop a worker and terminate its Modal container if possible."""
    entry = workers_store.get(worker_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not registered")

    modal_killed = False
    fc_id = entry.get("function_call_id")
    if fc_id:
        try:
            from modal.functions import FunctionCall

            fc = FunctionCall.from_id(fc_id)
            fc.cancel(terminate_containers=True)
            modal_killed = True
        except Exception as e:
            event_bus.emit(
                "worker.kill_error",
                f"Failed to kill Modal container for {worker_id}: {e}",
                details={"worker_id": worker_id, "function_call_id": fc_id},
            )

    workers_store.remove(worker_id)

    event_bus.emit(
        "worker.killed",
        f"Worker {worker_id} stopped" + (" (Modal container terminated)" if modal_killed else ""),
        details={"worker_id": worker_id, "modal_killed": modal_killed},
    )

    return {"worker_id": worker_id, "status": "stopped", "modal_killed": modal_killed}


@app.get("/workers/{worker_id}/sessions")
async def list_worker_sessions(worker_id: str):
    """List sessions on a worker's opencode server."""
    url = _get_worker_url(worker_id)
    async with _httpx.AsyncClient(base_url=url, timeout=10.0) as http:
        resp = await http.get("/session")
        resp.raise_for_status()
        return resp.json()


@app.post("/workers/{worker_id}/sessions")
async def create_worker_session(worker_id: str):
    """Create a new session on a worker's opencode server."""
    url = _get_worker_url(worker_id)
    async with _httpx.AsyncClient(base_url=url, timeout=10.0) as http:
        resp = await http.post("/session", json={})
        resp.raise_for_status()
        return resp.json()


@app.post("/workers/{worker_id}/prompt")
async def worker_prompt(worker_id: str, req: WorkerPromptRequest):
    """Send a message to a worker (fire-and-forget). Auto-creates session if needed."""
    url = _get_worker_url(worker_id)
    async with _httpx.AsyncClient(base_url=url, timeout=30.0) as http:
        session_id = req.session_id
        if not session_id:
            resp = await http.post("/session", json={})
            resp.raise_for_status()
            session_id = resp.json()["id"]

        resp = await http.post(
            f"/session/{session_id}/prompt_async",
            json={"parts": [{"type": "text", "text": req.message}]},
        )
        resp.raise_for_status()
        return {"session_id": session_id, "status": "sent"}


@app.post("/workers/{worker_id}/prompt/sync")
async def worker_prompt_sync(worker_id: str, req: WorkerPromptRequest):
    """Send a message and wait for the full response."""
    url = _get_worker_url(worker_id)
    async with _httpx.AsyncClient(base_url=url, timeout=None) as http:
        session_id = req.session_id
        if not session_id:
            resp = await http.post("/session", json={})
            resp.raise_for_status()
            session_id = resp.json()["id"]

        resp = await http.post(
            f"/session/{session_id}/message",
            json={"parts": [{"type": "text", "text": req.message}]},
        )
        resp.raise_for_status()
        return {"session_id": session_id, "response": resp.json()}


@app.post("/experiments/{experiment_id}/cancel")
def cancel_experiment(experiment_id: str):
    """Cancel a running experiment and terminate its Modal job."""
    exp = experiments.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    if exp.get("status") in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Experiment already {exp['status']}")

    # Cancel Modal function call if we have a reference
    modal_job_id = exp.get("modal_job_id")
    modal_cancelled = False
    if modal_job_id:
        try:
            from modal.functions import FunctionCall

            fc = FunctionCall.from_id(modal_job_id)
            fc.cancel(terminate_containers=True)
            modal_cancelled = True
        except Exception as e:
            event_bus.emit(
                "experiment.cancel_error",
                f"Failed to cancel Modal job {modal_job_id}: {e}",
                experiment_id=experiment_id,
                parent_id=event_bus.get_root_event(experiment_id),
            )

    experiments.update(experiment_id, status="cancelled")
    event_bus.emit(
        "experiment.cancelled",
        f"Experiment {experiment_id} cancelled" + (" (Modal job terminated)" if modal_cancelled else ""),
        experiment_id=experiment_id,
        parent_id=event_bus.get_root_event(experiment_id),
    )

    return {
        "experiment_id": experiment_id,
        "status": "cancelled",
        "modal_cancelled": modal_cancelled,
    }


# -- POST /events -------------------------------------------------------------


@app.post("/events")
def post_event(req: EmitEventRequest):
    experiment_id = req.experiment_id
    parent_id = req.parent_id
    # Backfill experiment_id from the experiment assigned to this worker
    if not experiment_id and req.worker_id:
        worker_exps = experiments.list(worker_id=req.worker_id, limit=1)
        if worker_exps:
            experiment_id = worker_exps[0].id
    # Backfill parent_id from the root event if we have an experiment but no parent
    if experiment_id and not parent_id:
        parent_id = event_bus.get_root_event(experiment_id)
    result = event_bus.emit(
        event_type=req.event_type,
        summary=req.summary,
        experiment_id=experiment_id,
        details=req.details,
        parent_id=parent_id,
        worker_id=req.worker_id,
    )
    # Auto-complete experiment when opencode session goes idle
    if req.event_type == "session.idle" and experiment_id:
        exp = experiments.get(experiment_id)
        if exp and exp.get("status") == "running":
            experiments.update(experiment_id, status="completed")
            event_bus.emit(
                "experiment.completed",
                f"Experiment {experiment_id} completed (session idle)",
                experiment_id=experiment_id,
                parent_id=event_bus.get_root_event(experiment_id),
            )
    return result


# -- POST /dispatch ------------------------------------------------------------


TASK_QUEUE_DIR = Path(__file__).parent.parent / ".data" / "tasks"


@app.post("/dispatch")
def dispatch_task(req: DispatchTaskRequest):
    TASK_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    import uuid
    from datetime import datetime, timezone

    task_id = uuid.uuid4().hex[:12]
    task = {
        "id": task_id,
        "agent_type": req.agent_type,
        "prompt": req.prompt,
        "priority": req.priority,
        "experiment_id": req.experiment_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    task_file = TASK_QUEUE_DIR / f"{task_id}.json"
    task_file.write_text(json.dumps(task, indent=2))

    event_bus.emit(
        "task.dispatched",
        f"Task dispatched: {req.agent_type} \u2014 {req.prompt[:100]}",
        experiment_id=req.experiment_id,
        details=task,
    )
    return task


@app.get("/dispatch")
def list_dispatched_tasks(
    status: str = Query("pending", description="Filter by status"),
    agent_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    TASK_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for fp in sorted(TASK_QUEUE_DIR.glob("*.json")):
        task = json.loads(fp.read_text())
        if task.get("status") != status:
            continue
        if agent_type and task.get("agent_type") != agent_type:
            continue
        tasks.append(task)

    tasks.sort(key=lambda t: -t.get("priority", 0))
    return tasks[offset:offset + limit]


@app.patch("/dispatch/{task_id}")
def update_task(task_id: str, status: str = Query(...)):
    task_file = TASK_QUEUE_DIR / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = json.loads(task_file.read_text())
    task["status"] = status
    task_file.write_text(json.dumps(task, indent=2))
    return task


# -- POST /directives ---------------------------------------------------------


DIRECTIVES_DIR = Path(__file__).parent.parent / "agent_directives"


@app.post("/directives")
def update_directive(req: DirectiveUpdateRequest):
    DIRECTIVES_DIR.mkdir(parents=True, exist_ok=True)

    if req.agent_name == "all":
        feedback_file = PROJECT_ROOT / "human_feedback.md"
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n\n## {timestamp} - Directive Update\n\n{req.directive}\n"

        with open(feedback_file, "a") as f:
            f.write(entry)

        event_bus.emit("directive.updated", f"Macro directive updated: {req.directive[:100]}")
        return {"target": "all", "file": str(feedback_file)}
    else:
        directive_file = DIRECTIVES_DIR / f"{req.agent_name}.md"
        directive_file.write_text(req.directive)

        event_bus.emit(
            "directive.updated",
            f"Micro directive for {req.agent_name}: {req.directive[:100]}",
        )
        return {"target": req.agent_name, "file": str(directive_file)}


@app.get("/directives")
def get_directives():
    result = {}

    feedback_file = PROJECT_ROOT / "human_feedback.md"
    if feedback_file.exists():
        result["macro"] = feedback_file.read_text()

    DIRECTIVES_DIR.mkdir(parents=True, exist_ok=True)
    micro = {}
    for fp in DIRECTIVES_DIR.glob("*.md"):
        micro[fp.stem] = fp.read_text()
    result["micro"] = micro

    return result


@app.get("/directives/{agent_name}")
def get_agent_directive(agent_name: str):
    parts = []

    feedback_file = PROJECT_ROOT / "human_feedback.md"
    if feedback_file.exists():
        parts.append(f"# Global Directives\n\n{feedback_file.read_text()}")

    directive_file = DIRECTIVES_DIR / f"{agent_name}.md"
    if directive_file.exists():
        parts.append(f"# Agent-Specific Directives\n\n{directive_file.read_text()}")

    if not parts:
        return {"agent_name": agent_name, "directive": ""}

    return {"agent_name": agent_name, "directive": "\n\n---\n\n".join(parts)}

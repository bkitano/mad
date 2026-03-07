"""
FastAPI service exposing the experiment execution system.

All state is in Postgres (Supabase). Events table supports Realtime SSE.

Run:
    uvicorn service.api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from service import db, event_bus, experiment_store

PROJECT_ROOT = Path(__file__).parent.parent

app = FastAPI(
    title="MAD Experiment Service",
    description="API for managing autonomous ML experiment execution",
    version="0.2.0",
)


@app.on_event("startup")
def startup():
    db.init_db()


# ── Pydantic models ─────────────────────────────────────────────────────────


class CreateExperimentRequest(BaseModel):
    proposal_id: str
    agent_id: str = ""
    config: Optional[dict] = None
    cost_estimate: Optional[float] = None
    code_files: Optional[dict[str, str]] = None


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
    agent: str = ""
    details: Optional[dict] = None
    parent_id: Optional[int] = None


class DispatchTaskRequest(BaseModel):
    agent_type: str
    prompt: str
    priority: int = 5
    experiment_id: Optional[str] = None


class DirectiveUpdateRequest(BaseModel):
    agent_name: str
    directive: str


class ClaimRequest(BaseModel):
    agent_id: str
    proposal_id: str


class HeartbeatRequest(BaseModel):
    agent_id: str
    proposal_id: str
    details: Optional[str] = None


class ReleaseRequest(BaseModel):
    agent_id: str
    proposal_id: str
    status: str = "completed"


# ── GET /experiments ─────────────────────────────────────────────────────────


@app.get("/experiments")
def list_experiments(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return db.list_experiments(status=status, limit=limit, offset=offset)


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    exp = db.get_experiment(experiment_id)
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
    exp = db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return event_bus.query(experiment_id=experiment_id, limit=limit)


@app.get("/experiments/{experiment_id}/verify")
def verify_experiment(experiment_id: str):
    exp = db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    checks = {
        "code_stored": experiment_store.get_code_dir(experiment_id) is not None,
        "code_integrity": experiment_store.verify_integrity(experiment_id),
        "modal_submitted": bool(exp.get("modal_job_id")),
        "modal_url": bool(exp.get("modal_url")),
        "wandb_tracked": bool(exp.get("wandb_url")),
        "has_results": bool(exp.get("results")),
        "status": exp.get("status"),
    }
    checks["fully_verified"] = all([
        checks["code_stored"], checks["code_integrity"],
        checks["modal_submitted"], checks["wandb_tracked"],
        checks["has_results"], checks["status"] == "completed",
    ])

    return {"experiment_id": experiment_id, "checks": checks, "experiment": exp}


# ── GET /events ──────────────────────────────────────────────────────────────


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
    q = event_bus.subscribe()

    async def generate():
        try:
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            event_bus.unsubscribe(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/events/{event_id}")
def get_event(event_id: int):
    rows = db._fetch("SELECT * FROM events WHERE id = %s", (event_id,))
    if not rows:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
    return rows[0]


@app.get("/events/{event_id}/children")
def get_event_children(
    event_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return event_bus.query(parent_id=event_id, limit=limit, offset=offset)


# ── GET /proposals ───────────────────────────────────────────────────────────


@app.get("/proposals")
def list_proposals(
    status: Optional[str] = Query(None, description="Filter by status"),
):
    if status:
        rows = db._fetch(
            "SELECT filename, experiment_number, title, status, priority, created, based_on, results_file, content"
            " FROM proposals WHERE lower(status) = lower(%s)"
            " ORDER BY experiment_number NULLS LAST, filename",
            (status,),
        )
    else:
        rows = db._fetch(
            "SELECT filename, experiment_number, title, status, priority, created, based_on, results_file, content"
            " FROM proposals ORDER BY experiment_number NULLS LAST, filename"
        )

    return [
        {
            "id": row["filename"].removesuffix(".md"),
            "filename": row["filename"],
            "experiment_number": row["experiment_number"],
            "title": row["title"],
            "status": row["status"],
            "priority": row["priority"],
            "created": str(row["created"]) if row["created"] else None,
            "based_on": row["based_on"],
            "results_file": row["results_file"],
            "has_mve": "## Minimum Viable Experiment" in (row["content"] or ""),
        }
        for row in rows
    ]


@app.get("/proposals/{proposal_id}")
def get_proposal(proposal_id: str):
    fname = proposal_id if proposal_id.endswith(".md") else f"{proposal_id}.md"
    row = db._fetch_one(
        "SELECT * FROM proposals WHERE filename = %s OR filename LIKE %s"
        " ORDER BY filename LIMIT 1",
        (fname, f"{proposal_id}%"),
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")

    return {
        "id": row["filename"].removesuffix(".md"),
        "filename": row["filename"],
        "experiment_number": row["experiment_number"],
        "title": row["title"],
        "status": row["status"],
        "priority": row["priority"],
        "created": str(row["created"]) if row["created"] else None,
        "based_on": row["based_on"],
        "results_file": row["results_file"],
        "hypothesis": row["hypothesis"],
        "has_mve": "## Minimum Viable Experiment" in (row["content"] or ""),
        "content": row["content"],
    }


# ── GET /stats ───────────────────────────────────────────────────────────────


@app.get("/stats")
def get_stats():
    all_exps = db.list_experiments(limit=10000)
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


# ── Claims ───────────────────────────────────────────────────────────────────


@app.get("/claims")
def get_claims(status: Optional[str] = Query("active")):
    return db.list_claims(status=status)


@app.post("/claims")
def claim(req: ClaimRequest):
    success = db.claim_proposal(req.proposal_id, req.agent_id)
    if not success:
        return {"claimed": False, "proposal_id": req.proposal_id, "reason": "already claimed"}

    event_bus.emit(
        "claim.acquired",
        f"Proposal {req.proposal_id} claimed by {req.agent_id}",
        agent=req.agent_id,
        details={"proposal_id": req.proposal_id},
    )
    return {"claimed": True, "proposal_id": req.proposal_id, "agent_id": req.agent_id}


@app.post("/claims/heartbeat")
def claim_heartbeat(req: HeartbeatRequest):
    success = db.heartbeat_claim(req.proposal_id, req.agent_id, req.details)
    if not success:
        raise HTTPException(status_code=404, detail="Claim not found or wrong agent")
    return {"ok": True}


@app.post("/claims/release")
def release(req: ReleaseRequest):
    success = db.release_claim(req.proposal_id, req.agent_id, req.status)

    event_bus.emit(
        "claim.released",
        f"Proposal {req.proposal_id} released by {req.agent_id} ({req.status})",
        agent=req.agent_id,
        details={"proposal_id": req.proposal_id, "status": req.status},
    )
    return {"released": success}


@app.get("/claims/{proposal_id}")
def check_claim(proposal_id: str):
    return {"proposal_id": proposal_id, "claimed": db.is_proposal_claimed(proposal_id)}


# ── POST /experiments ────────────────────────────────────────────────────────


@app.post("/experiments")
def create_experiment(req: CreateExperimentRequest):
    """Create a new experiment from a proposal. Auto-suffixes (e.g. 042-r2) on reruns."""
    match = re.match(r"(\d+)", req.proposal_id)
    if not match:
        raise HTTPException(status_code=400, detail="proposal_id must start with a number")
    experiment_id = match.group(1).zfill(3)

    # Auto-suffix for reruns
    if db.get_experiment(experiment_id):
        run_num = 2
        while db.get_experiment(f"{experiment_id}-r{run_num}"):
            run_num += 1
        experiment_id = f"{experiment_id}-r{run_num}"

    exp = db.create_experiment(
        experiment_id=experiment_id,
        proposal_id=req.proposal_id,
        agent_id=req.agent_id,
        config=req.config,
        cost_estimate=req.cost_estimate,
    )

    created_event = event_bus.emit(
        "experiment.created",
        f"Experiment {experiment_id} created from proposal {req.proposal_id}",
        experiment_id=experiment_id,
        agent=req.agent_id,
    )
    root_event_id = created_event.get("id")

    if req.code_files:
        store_result = experiment_store.store_code_from_dict(experiment_id, req.code_files)
        db.update_experiment(
            experiment_id,
            code_dir=store_result["code_dir"],
            code_hash=store_result["code_hash"],
            status="code_ready",
        )
        event_bus.emit(
            "experiment.code_written",
            f"Code stored: {store_result['manifest']['total_files']} files, hash={store_result['code_hash'][:12]}",
            experiment_id=experiment_id,
            agent=req.agent_id,
            parent_id=root_event_id,
        )

    exp["root_event_id"] = root_event_id
    return exp


@app.post("/experiments/{experiment_id}/code")
def store_experiment_code(experiment_id: str, req: StoreCodeRequest):
    exp = db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    store_result = experiment_store.store_code_from_dict(experiment_id, req.files)
    db.update_experiment(
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
    exp = db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    source = Path(source_dir)
    if not source.exists():
        raise HTTPException(status_code=400, detail=f"Source directory does not exist: {source_dir}")

    store_result = experiment_store.store_code(experiment_id, source)
    db.update_experiment(
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
    exp = db.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated = db.update_experiment(experiment_id, **updates)

    if req.status:
        event_bus.emit(
            f"experiment.{req.status}" if req.status in ("completed", "failed", "cancelled") else "experiment.updated",
            f"Experiment {experiment_id} status → {req.status}",
            experiment_id=experiment_id,
            details=updates,
            parent_id=event_bus.get_root_event(experiment_id),
        )

    return updated


# ── POST /events ─────────────────────────────────────────────────────────────


@app.post("/events")
def post_event(req: EmitEventRequest):
    return event_bus.emit(
        event_type=req.event_type,
        summary=req.summary,
        experiment_id=req.experiment_id,
        agent=req.agent,
        details=req.details,
        parent_id=req.parent_id,
    )


# ── POST /dispatch ───────────────────────────────────────────────────────────


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
        f"Task dispatched: {req.agent_type} — {req.prompt[:100]}",
        experiment_id=req.experiment_id,
        details=task,
    )
    return task


@app.get("/dispatch")
def list_dispatched_tasks(
    status: str = Query("pending", description="Filter by status"),
    agent_type: Optional[str] = None,
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
    return tasks


@app.patch("/dispatch/{task_id}")
def update_task(task_id: str, status: str = Query(...)):
    task_file = TASK_QUEUE_DIR / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = json.loads(task_file.read_text())
    task["status"] = status
    task_file.write_text(json.dumps(task, indent=2))
    return task


# ── POST /directives ────────────────────────────────────────────────────────


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
            agent=req.agent_name,
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

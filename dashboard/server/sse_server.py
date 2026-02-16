#!/usr/bin/env python3
"""
MAD Architecture Search - FastAPI SSE Dashboard Server

Watches experiment files and streams updates to connected clients via Server-Sent Events.
"""

import json
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROPOSALS_DIR = PROJECT_ROOT / "proposals"
TRICKS_DIR = PROJECT_ROOT / "tricks"
NOTES_DIR = PROJECT_ROOT / "notes"
RESEARCH_LOGS_DIR = NOTES_DIR / "research_logs"
CODE_DIR = PROJECT_ROOT / "code"
RUNNER_LOG = PROJECT_ROOT / "runner.log"
ACTIVE_WORK_FILE = EXPERIMENTS_DIR / "active_work.json"
AGENT_STATUS_FILE = EXPERIMENTS_DIR / "agent_status.json"
RESEARCH_LOG = NOTES_DIR / "log.md"

# Server settings
MAX_CONNECTIONS = 100
HEARTBEAT_INTERVAL = 30  # seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
connected_clients: List[asyncio.Queue] = []
file_observer: Optional[Observer] = None


class ExperimentWatcher(FileSystemEventHandler):
    """Watches experiment directory for file changes."""

    def __init__(self):
        self.last_update = {}

    def on_modified(self, event):
        if event.is_directory:
            return

        # Debounce rapid updates
        now = time.time()
        if event.src_path in self.last_update:
            if now - self.last_update[event.src_path] < 1.0:
                return

        self.last_update[event.src_path] = now

        # Broadcast update
        file_path = Path(event.src_path)
        logger.info(f"File changed: {file_path.name}")

        if file_path.name == "active_work.json":
            asyncio.create_task(broadcast_update("active_work", get_active_work()))
        elif file_path.name.endswith("_results.md"):
            asyncio.create_task(broadcast_update("results", {"file": file_path.name}))
        elif file_path.name.endswith(".log"):
            asyncio.create_task(broadcast_update("logs", {"file": file_path.name}))


def get_active_work() -> Dict:
    """Read active work status."""
    try:
        if ACTIVE_WORK_FILE.exists():
            with open(ACTIVE_WORK_FILE) as f:
                data = json.load(f)
                data["server_timestamp"] = datetime.now().isoformat()
                return data
        return {"active_work": {}, "history": [], "server_timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error reading active_work.json: {e}")
        return {"error": str(e), "server_timestamp": datetime.now().isoformat()}


def get_recent_logs(n: int = 50) -> List[str]:
    """Get last N lines from runner.log."""
    try:
        if RUNNER_LOG.exists():
            with open(RUNNER_LOG) as f:
                lines = f.readlines()
                return lines[-n:]
        return []
    except Exception as e:
        logger.error(f"Error reading runner.log: {e}")
        return [f"Error: {e}"]


def get_experiment_results() -> List[Dict]:
    """List all experiment results."""
    try:
        results = []
        for result_file in sorted(EXPERIMENTS_DIR.glob("*_results.md")):
            results.append({
                "filename": result_file.name,
                "experiment_id": result_file.stem.replace("_results", ""),
                "modified": result_file.stat().st_mtime,
            })
        return results
    except Exception as e:
        logger.error(f"Error listing results: {e}")
        return []


def get_proposals() -> List[Dict]:
    """List all proposals with metadata."""
    try:
        proposals = []
        if not PROPOSALS_DIR.exists():
            return proposals

        for proposal_file in sorted(PROPOSALS_DIR.glob("*.md")):
            content = proposal_file.read_text()
            metadata = {
                "id": proposal_file.stem,
                "filename": proposal_file.name,
                "modified": proposal_file.stat().st_mtime,
            }

            lines = content.split('\n')

            # Try parsing YAML frontmatter first
            if lines and lines[0].strip() == '---':
                in_frontmatter = True
                frontmatter_end = None

                for i, line in enumerate(lines[1:], start=1):
                    if line.strip() == '---':
                        frontmatter_end = i
                        break
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        metadata[key] = value

                # Extract title from content after frontmatter
                if frontmatter_end:
                    for line in lines[frontmatter_end + 1:]:
                        if line.startswith('# '):
                            metadata['title'] = line.lstrip('# ').strip()
                            break
            else:
                # Fallback: parse inline markdown metadata
                for line in lines[:20]:
                    if line.startswith('**Status**:'):
                        metadata['status'] = line.split(':', 1)[1].strip()
                    elif line.startswith('**Priority**:'):
                        metadata['priority'] = line.split(':', 1)[1].strip()
                    elif line.startswith('**Created**:'):
                        metadata['created'] = line.split(':', 1)[1].strip()
                    elif line.startswith('**Based on**:'):
                        metadata['based_on'] = line.split(':', 1)[1].strip()
                    elif line.startswith('# '):
                        metadata['title'] = line.lstrip('# ').strip()

            # Check if experiment artifacts exist
            experiment_number = metadata.get('experiment_number')
            if experiment_number:
                # Check for experiment log
                log_file = EXPERIMENTS_DIR / f"experiment-log-{experiment_number}.md"
                if log_file.exists():
                    metadata['experiment_log'] = True

                # Check for results file
                results_file = EXPERIMENTS_DIR / f"{experiment_number}_results.md"
                if results_file.exists():
                    metadata['results_file'] = True

                # Check if code directory exists
                code_dir = CODE_DIR / experiment_number
                if code_dir.exists() and code_dir.is_dir():
                    metadata['code_available'] = True

            proposals.append(metadata)

        return proposals
    except Exception as e:
        logger.error(f"Error listing proposals: {e}")
        return []


def get_tricks() -> List[Dict]:
    """List all tricks with metadata."""
    try:
        tricks = []
        if not TRICKS_DIR.exists():
            return tricks

        for trick_file in sorted(TRICKS_DIR.glob("*.md")):
            content = trick_file.read_text()
            metadata = {
                "id": trick_file.stem,
                "filename": trick_file.name,
                "modified": trick_file.stat().st_mtime,
            }

            # Extract title
            for line in content.split('\n')[:10]:
                if line.startswith('# '):
                    metadata['title'] = line.lstrip('# ').strip()
                    break

            tricks.append(metadata)

        return tricks
    except Exception as e:
        logger.error(f"Error listing tricks: {e}")
        return []


def get_research_logs() -> List[Dict]:
    """List all timestamped research log files in reverse chronological order."""
    try:
        logs = []
        if not RESEARCH_LOGS_DIR.exists():
            return logs

        for log_file in RESEARCH_LOGS_DIR.glob("*.md"):
            # Parse frontmatter to get metadata
            content = log_file.read_text()
            lines = content.split('\n')

            metadata = {
                "filename": log_file.name,
                "modified": log_file.stat().st_mtime,
            }

            # Extract frontmatter
            if lines and lines[0] == '---':
                in_frontmatter = True
                for line in lines[1:]:
                    if line == '---':
                        break
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        metadata[key] = value

            logs.append(metadata)

        # Sort by filename (which is timestamp) in reverse chronological order
        logs.sort(key=lambda x: x['filename'], reverse=True)
        return logs
    except Exception as e:
        logger.error(f"Error listing research logs: {e}")
        return []


async def broadcast_update(event_type: str, data: Dict):
    """Send update to all connected clients."""
    if not connected_clients:
        return

    message = {"event": event_type, "data": data}
    disconnected = []

    for i, queue in enumerate(connected_clients):
        try:
            await asyncio.wait_for(queue.put(message), timeout=1.0)
        except (asyncio.TimeoutError, Exception):
            disconnected.append(i)

    # Remove disconnected clients
    for i in reversed(disconnected):
        connected_clients.pop(i)

    if disconnected:
        logger.info(f"Removed {len(disconnected)} disconnected clients. Active: {len(connected_clients)}")


def start_file_watcher():
    """Start watching experiment directory."""
    global file_observer
    event_handler = ExperimentWatcher()
    file_observer = Observer()
    file_observer.schedule(event_handler, str(EXPERIMENTS_DIR), recursive=False)
    file_observer.start()
    logger.info(f"Started watching: {EXPERIMENTS_DIR}")


async def periodic_broadcast():
    """Periodically broadcast active work status."""
    while True:
        await asyncio.sleep(10)
        try:
            await broadcast_update("active_work", get_active_work())
        except Exception as e:
            logger.error(f"Error in periodic broadcast: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting MAD Dashboard Server")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Watching: {EXPERIMENTS_DIR}")

    start_file_watcher()

    # Start periodic broadcast task
    broadcast_task = asyncio.create_task(periodic_broadcast())

    yield

    # Shutdown
    broadcast_task.cancel()
    if file_observer:
        file_observer.stop()
        file_observer.join()
    logger.info("Server stopped")


# Create FastAPI app
app = FastAPI(
    title="MAD Architecture Search Dashboard API",
    description="Real-time experiment monitoring and knowledge base",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "MAD Architecture Search Dashboard API",
        "version": "1.0.0",
        "endpoints": {
            "/stream": "SSE stream for real-time updates",
            "/api/status": "Current experiment status",
            "/api/logs": "Recent log lines",
            "/api/result/{id}": "Get experiment result",
            "/api/experiment-log/{id}": "Get experiment log",
            "/api/experiment-code/{id}": "Get experiment code structure",
            "/api/experiment-code/{id}/file?path=...": "Get specific file from experiment",
            "/api/proposals": "List all proposals",
            "/api/proposal/{id}": "Get specific proposal",
            "/api/tricks": "List all tricks",
            "/api/trick/{id}": "Get specific trick",
            "/api/research-logs": "List all timestamped research log files",
            "/api/research-log/{filename}": "Get specific research log file",
            "/api/agent-status": "Get status of all agents",
            "/health": "Health check",
            "/docs": "API documentation (Swagger UI)",
        },
        "active_connections": len(connected_clients),
    }


@app.get("/stream")
async def stream():
    """SSE endpoint for real-time updates."""
    if len(connected_clients) >= MAX_CONNECTIONS:
        raise HTTPException(status_code=503, detail="Max connections reached")

    async def event_generator():
        queue = asyncio.Queue(maxsize=10)
        connected_clients.append(queue)
        logger.info(f"New client connected. Active connections: {len(connected_clients)}")

        try:
            # Send initial state
            yield {
                "event": "connected",
                "data": json.dumps({"message": "Connected to MAD dashboard"})
            }
            yield {
                "event": "active_work",
                "data": json.dumps(get_active_work())
            }

            # Stream updates
            last_heartbeat = time.time()

            while True:
                try:
                    # Wait for update with timeout
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield {
                        "event": message["event"],
                        "data": json.dumps(message["data"])
                    }
                except asyncio.TimeoutError:
                    # Send heartbeat if needed
                    now = time.time()
                    if now - last_heartbeat > HEARTBEAT_INTERVAL:
                        yield {"comment": "heartbeat"}
                        last_heartbeat = now

        except asyncio.CancelledError:
            pass
        finally:
            if queue in connected_clients:
                connected_clients.remove(queue)
                logger.info(f"Client disconnected. Active connections: {len(connected_clients)}")

    return EventSourceResponse(event_generator())


@app.get("/api/status")
async def status():
    """Get current status."""
    return {
        "active_work": get_active_work(),
        "results": get_experiment_results(),
        "server_time": datetime.now().isoformat(),
        "active_connections": len(connected_clients),
    }


@app.get("/api/logs")
async def logs(n: int = 50):
    """Get recent logs."""
    return {"logs": get_recent_logs(n)}


@app.get("/api/result/{experiment_id}")
async def get_result(experiment_id: str):
    """Get specific experiment result."""
    result_file = EXPERIMENTS_DIR / f"{experiment_id}_results.md"
    if result_file.exists():
        return {
            "experiment_id": experiment_id,
            "content": result_file.read_text(),
        }
    raise HTTPException(status_code=404, detail="Not found")


@app.get("/api/experiment-log/{experiment_id}")
async def get_experiment_log(experiment_id: str):
    """Get specific experiment log."""
    log_file = EXPERIMENTS_DIR / f"experiment-log-{experiment_id}.md"
    if log_file.exists():
        return {
            "experiment_id": experiment_id,
            "content": log_file.read_text(),
        }

    # Log doesn't exist yet - return helpful message
    raise HTTPException(
        status_code=404,
        detail=f"Experiment log not found for {experiment_id}. The experiment may not have started yet or hasn't created its log file."
    )


@app.get("/api/proposals")
async def proposals():
    """List all proposals."""
    proposal_list = get_proposals()
    return {
        "proposals": proposal_list,
        "count": len(proposal_list),
    }


@app.get("/api/proposal/{proposal_id}")
async def get_proposal(proposal_id: str):
    """Get specific proposal content."""
    proposal_file = PROPOSALS_DIR / f"{proposal_id}.md"
    if proposal_file.exists():
        return {
            "id": proposal_id,
            "content": proposal_file.read_text(),
            "modified": proposal_file.stat().st_mtime,
        }
    raise HTTPException(status_code=404, detail="Not found")


@app.get("/api/tricks")
async def tricks():
    """List all tricks."""
    trick_list = get_tricks()
    return {
        "tricks": trick_list,
        "count": len(trick_list),
    }


@app.get("/api/trick/{trick_id}")
async def get_trick(trick_id: str):
    """Get specific trick content."""
    trick_file = TRICKS_DIR / f"{trick_id}.md"
    if trick_file.exists():
        return {
            "id": trick_id,
            "content": trick_file.read_text(),
            "modified": trick_file.stat().st_mtime,
        }
    raise HTTPException(status_code=404, detail="Not found")


@app.get("/api/research-log")
async def research_log():
    """Get research activity log (legacy monolithic file)."""
    if RESEARCH_LOG.exists():
        return {
            "content": RESEARCH_LOG.read_text(),
            "modified": RESEARCH_LOG.stat().st_mtime,
        }
    raise HTTPException(status_code=404, detail="Not found")


@app.get("/api/research-logs")
async def research_logs():
    """List all timestamped research log files in reverse chronological order."""
    log_list = get_research_logs()
    return {
        "logs": log_list,
        "count": len(log_list),
    }


@app.get("/api/research-log/{filename}")
async def get_research_log_file(filename: str):
    """Get specific research log file."""
    log_file = RESEARCH_LOGS_DIR / filename
    if log_file.exists() and log_file.suffix == ".md":
        return {
            "filename": filename,
            "content": log_file.read_text(),
            "modified": log_file.stat().st_mtime,
        }
    raise HTTPException(status_code=404, detail="Not found")


@app.get("/api/agent-status")
async def get_agent_status():
    """Get status of all agents (trick search, research, experiment, log, scaler)."""
    try:
        if AGENT_STATUS_FILE.exists():
            with open(AGENT_STATUS_FILE) as f:
                status_data = json.load(f)
            return {
                "agents": status_data,
                "timestamp": datetime.now().isoformat(),
            }
        return {
            "agents": {},
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error reading agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/experiment-code/{experiment_id}")
async def get_experiment_code(experiment_id: str):
    """Get experiment code structure and files."""
    code_dir = CODE_DIR / experiment_id
    if not code_dir.exists() or not code_dir.is_dir():
        raise HTTPException(status_code=404, detail="Code directory not found")

    def get_file_tree(path: Path, relative_to: Path) -> Dict:
        """Recursively build file tree structure."""
        result = {
            "name": path.name,
            "path": str(path.relative_to(relative_to)),
            "type": "directory" if path.is_dir() else "file",
        }

        if path.is_dir():
            # Skip venv, __pycache__, and hidden directories
            if path.name.startswith('.') or path.name in ['__pycache__', 'wandb', '.venv', 'venv']:
                return None

            children = []
            for child in sorted(path.iterdir()):
                child_node = get_file_tree(child, relative_to)
                if child_node:
                    children.append(child_node)
            result["children"] = children
        else:
            # Add file size
            result["size"] = path.stat().st_size

        return result

    file_tree = get_file_tree(code_dir, code_dir)

    return {
        "experiment_id": experiment_id,
        "file_tree": file_tree,
    }


@app.get("/api/experiment-code/{experiment_id}/file")
async def get_experiment_file(experiment_id: str, path: str):
    """Get specific file content from experiment code directory."""
    code_dir = CODE_DIR / experiment_id
    file_path = code_dir / path

    # Security: Ensure file is within code directory
    try:
        file_path = file_path.resolve()
        code_dir = code_dir.resolve()
        if not str(file_path).startswith(str(code_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Limit file size to 1MB for safety
    if file_path.stat().st_size > 1_000_000:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        content = file_path.read_text()
        return {
            "experiment_id": experiment_id,
            "path": path,
            "content": content,
            "size": file_path.stat().st_size,
        }
    except UnicodeDecodeError:
        raise HTTPException(status_code=415, detail="Binary file not supported")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_connections": len(connected_clients),
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
    )

"""
HTTP client for agents to interact with the experiment service.

Agents should use this instead of importing service modules directly.
This ensures all interactions go through the API and are properly tracked.

Usage:
    from service.client import ExperimentClient

    client = ExperimentClient()  # defaults to http://localhost:8000

    # Create experiment
    exp = client.create_experiment("042-monarch-gated")

    # Store code
    client.store_code(exp["id"], {"train.py": "...", "config.yaml": "..."})

    # Submit to Modal
    result = client.submit(exp["id"])

    # Report results
    client.update_experiment(exp["id"], status="completed", results={...}, wandb_url="...")

    # Emit events (with optional parent_id for event chaining)
    event = client.emit_event("error", "Training failed", experiment_id=exp["id"])
    client.emit_event("debug.started", "Debugger investigating", parent_id=event["id"])

    # Check what to work on
    tasks = client.get_pending_tasks("experiment")
"""

import os
from typing import Optional

import httpx

DEFAULT_BASE_URL = os.environ.get("MAD_SERVICE_URL", "http://localhost:8000")


class ExperimentClient:
    """HTTP client for the experiment service API."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: Optional[dict] = None) -> dict | list:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}{path}", params=params)
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, json: Optional[dict] = None, params: Optional[dict] = None) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}{path}", json=json, params=params)
            resp.raise_for_status()
            return resp.json()

    def _patch(self, path: str, json: Optional[dict] = None, params: Optional[dict] = None) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.patch(f"{self.base_url}{path}", json=json, params=params)
            resp.raise_for_status()
            return resp.json()

    # ── Experiments ──────────────────────────────────────────────────────────

    def list_experiments(self, status: Optional[str] = None, limit: int = 100) -> list:
        params = {"limit": limit}
        if status:
            params["status"] = status
        return self._get("/experiments", params=params)

    def get_experiment(self, experiment_id: str) -> dict:
        return self._get(f"/experiments/{experiment_id}")

    def create_experiment(
        self,
        proposal_id: str,
        cost_estimate: Optional[float] = None,
    ) -> dict:
        return self._post("/experiments", json={
            "proposal_id": proposal_id,
            "cost_estimate": cost_estimate,
        })

    def store_code(self, experiment_id: str, files: dict[str, str]) -> dict:
        return self._post(f"/experiments/{experiment_id}/code", json={"files": files})

    def store_code_from_disk(self, experiment_id: str, source_dir: str) -> dict:
        return self._post(
            f"/experiments/{experiment_id}/store-from-disk",
            params={"source_dir": source_dir},
        )

    def get_code(self, experiment_id: str) -> dict:
        return self._get(f"/experiments/{experiment_id}/code")

    def get_file(self, experiment_id: str, file_path: str) -> dict:
        return self._get(f"/experiments/{experiment_id}/code/{file_path}")

    def submit(
        self,
        experiment_id: str,
        config_file: str = "config.yaml",
    ) -> dict:
        return self._post(f"/experiments/{experiment_id}/submit", json={
            "config_file": config_file,
        })

    def retry(self, experiment_id: str) -> dict:
        return self._post(f"/experiments/{experiment_id}/retry")

    def update_experiment(self, experiment_id: str, **fields) -> dict:
        return self._patch(f"/experiments/{experiment_id}", json=fields)

    def cancel_experiment(self, experiment_id: str) -> dict:
        """Cancel a running experiment and its Modal job."""
        return self._post(f"/experiments/{experiment_id}/cancel")

    def verify(self, experiment_id: str) -> dict:
        return self._get(f"/experiments/{experiment_id}/verify")

    def get_experiment_events(self, experiment_id: str, limit: int = 100) -> list:
        return self._get(f"/experiments/{experiment_id}/events", params={"limit": limit})

    # ── Events ───────────────────────────────────────────────────────────────

    def emit_event(
        self,
        event_type: str,
        summary: str,
        experiment_id: Optional[str] = None,
        details: Optional[dict] = None,
        parent_id: Optional[int] = None,
    ) -> dict:
        return self._post("/events", json={
            "event_type": event_type,
            "summary": summary,
            "experiment_id": experiment_id,
            "details": details,
            "parent_id": parent_id,
        })

    def get_events(
        self,
        experiment_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[str] = None,
        parent_id: Optional[int] = None,
        limit: int = 100,
    ) -> list:
        params = {"limit": limit}
        if experiment_id:
            params["experiment_id"] = experiment_id
        if event_type:
            params["event_type"] = event_type
        if since:
            params["since"] = since
        if parent_id is not None:
            params["parent_id"] = parent_id
        return self._get("/events", params=params)

    def get_event(self, event_id: int) -> dict:
        return self._get(f"/events/{event_id}")

    def get_event_children(self, event_id: int, limit: int = 100) -> list:
        return self._get(f"/events/{event_id}/children", params={"limit": limit})

    # ── Proposals ────────────────────────────────────────────────────────────

    def list_proposals(self, status: Optional[str] = None) -> list:
        params = {}
        if status:
            params["status"] = status
        return self._get("/proposals", params=params)

    def get_proposal(self, proposal_id: str) -> dict:
        """Get full proposal content + metadata."""
        return self._get(f"/proposals/{proposal_id}")

    def create_proposal(
        self,
        filename: str,
        title: str,
        content: str,
        experiment_number: Optional[int] = None,
        status: str = "draft",
        priority: Optional[str] = None,
        hypothesis: Optional[str] = None,
        based_on: Optional[str] = None,
    ) -> dict:
        return self._post("/proposals", json={
            "filename": filename,
            "title": title,
            "content": content,
            "experiment_number": experiment_number,
            "status": status,
            "priority": priority,
            "hypothesis": hypothesis,
            "based_on": based_on,
        })

    def get_code_review(self, experiment_id: str) -> str:
        """Get all code as plaintext for review."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/experiments/{experiment_id}/code/review")
            resp.raise_for_status()
            return resp.text

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return self._get("/stats")

    # ── Tasks ────────────────────────────────────────────────────────────────

    def dispatch_task(
        self,
        agent_type: str,
        prompt: str,
        priority: int = 5,
        experiment_id: Optional[str] = None,
    ) -> dict:
        return self._post("/dispatch", json={
            "agent_type": agent_type,
            "prompt": prompt,
            "priority": priority,
            "experiment_id": experiment_id,
        })

    def get_pending_tasks(self, agent_type: Optional[str] = None) -> list:
        params = {"status": "pending"}
        if agent_type:
            params["agent_type"] = agent_type
        return self._get("/dispatch", params=params)

    def claim_task(self, task_id: str) -> dict:
        return self._patch(f"/dispatch/{task_id}", params={"status": "running"})

    def complete_task(self, task_id: str) -> dict:
        return self._patch(f"/dispatch/{task_id}", params={"status": "completed"})

    # ── Directives ───────────────────────────────────────────────────────────

    def get_directives(self) -> dict:
        return self._get("/directives")

    def get_my_directive(self, agent_name: str) -> dict:
        return self._get(f"/directives/{agent_name}")

    def update_directive(self, agent_name: str, directive: str) -> dict:
        return self._post("/directives", json={
            "agent_name": agent_name,
            "directive": directive,
        })

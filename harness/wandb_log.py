"""
Log an EvaluationVerdict to Weights & Biases so eval results from independent
sandboxes can be compared in one place.

Convention:
    project   = "mad"               (override via --wandb-project / WANDB_PROJECT)
    group     = verdict.proposal_id (compare reruns of the same idea)
    job_type  = verdict.experiment_id
    tags      = [proposal_id, experiment_id, "pass"|"fail"]

The agent's training run is preferred — we resume it by id so verdict.* keys
land on the same W&B run as the training curves. If no run id is discoverable,
a fresh run is created so the verdict still gets recorded.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from harness.schema import EvaluationVerdict

logger = logging.getLogger(__name__)


def _resolve_run_id(
    explicit: Optional[str],
    results: Optional[dict],
    cwd: Optional[Path],
) -> Optional[str]:
    """Find the W&B run id in priority order: explicit > env > results.json > wandb/latest-run."""
    if explicit:
        return explicit
    env_id = os.environ.get("WANDB_RUN_ID")
    if env_id:
        return env_id
    if results:
        for key in ("wandb_run_id", "wandb/run_id", "wandb"):
            val = results.get(key)
            if isinstance(val, str) and val:
                return val
            if isinstance(val, dict):
                rid = val.get("run_id") or val.get("id")
                if rid:
                    return rid
    # Last resort: wandb writes `wandb/latest-run -> run-<ts>-<id>`
    if cwd is not None:
        latest = cwd / "wandb" / "latest-run"
        try:
            if latest.is_symlink() or latest.exists():
                target = os.readlink(latest) if latest.is_symlink() else latest.name
                # Expect "run-<timestamp>-<id>"
                parts = Path(target).name.split("-")
                if len(parts) >= 3:
                    return parts[-1]
        except OSError:
            pass
    return None


def log_verdict_to_wandb(
    verdict: EvaluationVerdict,
    project: str = "mad",
    entity: Optional[str] = None,
    run_id: Optional[str] = None,
    results: Optional[dict] = None,
    cwd: Optional[Path] = None,
) -> dict:
    """Push the verdict onto a W&B run. Returns a small status dict; never raises."""
    try:
        import wandb
    except ImportError:
        return {"logged": False, "reason": "wandb not installed"}

    resolved_id = _resolve_run_id(run_id, results, cwd)
    tags = [
        verdict.proposal_id,
        verdict.experiment_id,
        "pass" if verdict.overall_pass else "fail",
    ]

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            id=resolved_id,
            resume="allow" if resolved_id else None,
            group=verdict.proposal_id or None,
            job_type=verdict.experiment_id or None,
            tags=tags,
            reinit=True,
        )
    except Exception as e:  # network down, missing creds, etc.
        logger.warning("wandb.init failed: %s: %s", type(e).__name__, e)
        return {"logged": False, "reason": f"wandb.init failed: {e}"}

    try:
        # Top-line summary keys — show up in W&B run table columns.
        wandb.summary["verdict/overall_pass"] = verdict.overall_pass
        wandb.summary["verdict/training_completed"] = verdict.training_completed
        wandb.summary["verdict/path_violations"] = len(verdict.path_violations)
        wandb.summary["verdict/required_failed"] = sum(
            1 for r in verdict.criteria_results if r.required and not r.passed
        )
        if verdict.wall_time_seconds is not None:
            wandb.summary["verdict/wall_time_seconds"] = verdict.wall_time_seconds
        if verdict.error:
            wandb.summary["verdict/error"] = verdict.error

        # Per-criterion summary keys so they're sortable / chartable across runs.
        for r in verdict.criteria_results:
            if r.achieved is not None:
                wandb.summary[f"verdict/{r.metric_key}"] = r.achieved
            wandb.summary[f"verdict/{r.metric_key}/passed"] = r.passed

        # Criteria table — usable in W&B Reports for side-by-side comparison.
        columns = ["name", "metric_key", "target", "achieved", "comparator", "required", "passed", "detail"]
        rows = [
            [r.name, r.metric_key, r.target, r.achieved, r.comparator.value, r.required, r.passed, r.detail]
            for r in verdict.criteria_results
        ]
        wandb.log({"verdict/criteria": wandb.Table(columns=columns, data=rows)})

        # Stamp config so it's filterable.
        wandb.config.update(
            {
                "experiment_id": verdict.experiment_id,
                "proposal_id": verdict.proposal_id,
            },
            allow_val_change=True,
        )
        url = run.url
    except Exception as e:
        logger.warning("wandb logging failed: %s: %s", type(e).__name__, e)
        wandb.finish(exit_code=1)
        return {"logged": False, "reason": f"wandb logging failed: {e}"}

    wandb.finish()
    return {"logged": True, "run_id": resolved_id or run.id, "url": url}

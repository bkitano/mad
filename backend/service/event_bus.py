"""
Event bus for experiment lifecycle tracking.

Persists events to Postgres (events table) and optionally forwards to
webhook sinks. Use Supabase Realtime on the events table for live
SSE/websocket subscriptions.
"""

import logging
import os
from typing import Optional

import httpx

from service.db import emit_event as _db_emit, list_events as _db_list

logger = logging.getLogger(__name__)

# Webhook config
WEBHOOK_URL = os.environ.get("MAD_WEBHOOK_URL", "")
WEBHOOK_EVENTS = set(
    os.environ.get("MAD_WEBHOOK_EVENTS", "failed,completed,error,cancelled").split(",")
)


def emit(
    event_type: str,
    summary: str,
    experiment_id: Optional[str] = None,
    agent: str = "",
    details: Optional[dict] = None,
    parent_id: Optional[int] = None,
) -> dict:
    """Emit an event: persist to Postgres, optionally webhook."""
    event = _db_emit(
        event_type=event_type,
        summary=summary,
        experiment_id=experiment_id,
        agent=agent,
        details=details,
        parent_id=parent_id,
    )

    if WEBHOOK_URL and event_type in WEBHOOK_EVENTS:
        _fire_webhook(event)

    return event


def _fire_webhook(event: dict) -> None:
    try:
        icon = {
            "completed": "✅",
            "failed": "❌",
            "error": "⚠️",
            "cancelled": "🚫",
        }.get(event.get("type", ""), "📋")

        title = f"{icon} [{event.get('type', '').upper()}] Experiment {event.get('experiment_id', '?')}"
        body = event.get("summary", "")

        with httpx.Client(timeout=5) as client:
            client.post(
                WEBHOOK_URL,
                content=body,
                headers={"Title": title, "Tags": event.get("type", "")},
            )
    except Exception as e:
        logger.warning(f"Webhook failed: {e}")


def get_root_event(experiment_id: str) -> Optional[int]:
    """Return the id of the experiment.created event for the given experiment, or None."""
    rows = _db_list(
        experiment_id=experiment_id,
        event_type="experiment.created",
        limit=1,
    )
    if rows:
        return rows[0].get("id")
    return None


def query(
    experiment_id: Optional[str] = None,
    event_type: Optional[str] = None,
    since: Optional[str] = None,
    parent_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    return _db_list(
        experiment_id=experiment_id,
        event_type=event_type,
        since=since,
        parent_id=parent_id,
        limit=limit,
        offset=offset,
    )

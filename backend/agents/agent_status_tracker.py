"""
Agent Status Tracker

Tracks the status of all running agents (trick search, research, experiment, log, scaler)
and provides a unified view for the dashboard.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Literal

PROJECT_ROOT = Path(__file__).parent.parent
STATUS_FILE = PROJECT_ROOT / "experiments" / "agent_status.json"

AgentType = Literal["trick_search", "research", "experiment", "log", "scaler"]
AgentStatus = Literal["running", "waiting", "idle", "error"]


def update_agent_status(
    agent_type: AgentType,
    status: AgentStatus,
    details: Optional[Dict] = None
):
    """
    Update the status of a specific agent type.

    Args:
        agent_type: Type of agent (trick_search, research, experiment, log, scaler)
        status: Current status (running, waiting, idle, error)
        details: Optional dict with additional info (iteration, next_run, message, etc.)
    """
    try:
        # Load existing status
        if STATUS_FILE.exists():
            with open(STATUS_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # Update this agent's status
        data[agent_type] = {
            "status": status,
            "last_update": datetime.now().isoformat(),
            "details": details or {}
        }

        # Write back
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"Error updating agent status: {e}")


def get_all_agent_statuses() -> Dict:
    """
    Get the current status of all agents.

    Returns:
        Dict mapping agent type to status info
    """
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error reading agent status: {e}")
        return {}


def get_agent_status(agent_type: AgentType) -> Optional[Dict]:
    """
    Get the status of a specific agent type.

    Args:
        agent_type: Type of agent to query

    Returns:
        Status dict or None if not found
    """
    all_statuses = get_all_agent_statuses()
    return all_statuses.get(agent_type)

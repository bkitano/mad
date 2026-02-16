"""
Agent State Management

Provides state persistence for agents so they can resume work after failures,
token limits, or interruptions.

Each agent maintains a state file in .state/ with:
- What task they were working on
- Progress made so far
- Retry count
- Timestamps

This enables graceful recovery from failures.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any


PROJECT_ROOT = Path(__file__).parent.parent
STATE_DIR = PROJECT_ROOT / ".state"


def ensure_state_dir():
    """Ensure the .state directory exists."""
    STATE_DIR.mkdir(exist_ok=True)


def save_agent_state(agent_name: str, state: Dict[str, Any]):
    """
    Save agent state to disk.

    Args:
        agent_name: Name of the agent (e.g., "trick_search", "research", "experiment", "log")
        state: Dictionary containing agent state. Should include:
            - task_description: What the agent is working on
            - progress: Dict describing what's been done
            - context: Any data needed to resume (queries, proposal IDs, etc.)
    """
    ensure_state_dir()

    state_file = STATE_DIR / f"{agent_name}_state.json"

    # Add metadata
    full_state = {
        "agent": agent_name,
        "task": state.get("task_description", ""),
        "progress": state.get("progress", {}),
        "context": state.get("context", {}),
        "started_at": state.get("started_at", datetime.now().isoformat()),
        "last_updated": datetime.now().isoformat(),
        "attempt_count": state.get("attempt_count", 0) + 1,
    }

    state_file.write_text(json.dumps(full_state, indent=2))
    print(f"[STATE] Saved state for {agent_name} (attempt {full_state['attempt_count']})")


def load_agent_state(agent_name: str, max_age_hours: int = 24) -> Optional[Dict[str, Any]]:
    """
    Load agent state from disk if it exists and is recent.

    Args:
        agent_name: Name of the agent
        max_age_hours: Maximum age of state to consider valid (default: 24 hours)

    Returns:
        State dict if valid state exists, None otherwise
    """
    ensure_state_dir()

    state_file = STATE_DIR / f"{agent_name}_state.json"

    if not state_file.exists():
        return None

    try:
        state = json.loads(state_file.read_text())

        # Check if state is too old
        last_updated = datetime.fromisoformat(state["last_updated"])
        age = datetime.now() - last_updated

        if age > timedelta(hours=max_age_hours):
            print(f"[STATE] State for {agent_name} is {age.total_seconds()/3600:.1f}h old, discarding")
            state_file.unlink()
            return None

        print(f"[STATE] Loaded state for {agent_name} (attempt {state['attempt_count']}, age: {age.total_seconds()/60:.1f}m)")
        return state

    except Exception as e:
        print(f"[STATE] Error loading state for {agent_name}: {e}")
        return None


def clear_agent_state(agent_name: str):
    """
    Clear agent state after successful completion.

    Args:
        agent_name: Name of the agent
    """
    ensure_state_dir()

    state_file = STATE_DIR / f"{agent_name}_state.json"

    if state_file.exists():
        state_file.unlink()
        print(f"[STATE] Cleared state for {agent_name}")


def update_agent_progress(agent_name: str, progress_update: Dict[str, Any]):
    """
    Update progress in existing state without changing other fields.

    Args:
        agent_name: Name of the agent
        progress_update: Dict with progress updates to merge
    """
    state = load_agent_state(agent_name, max_age_hours=24)

    if not state:
        print(f"[STATE] No existing state for {agent_name}, cannot update progress")
        return

    # Merge progress updates
    state["progress"].update(progress_update)
    state["last_updated"] = datetime.now().isoformat()

    state_file = STATE_DIR / f"{agent_name}_state.json"
    state_file.write_text(json.dumps(state, indent=2))
    print(f"[STATE] Updated progress for {agent_name}")


def should_retry(agent_name: str, max_attempts: int = 3) -> bool:
    """
    Check if agent should retry based on attempt count.

    Args:
        agent_name: Name of the agent
        max_attempts: Maximum number of retry attempts (default: 3)

    Returns:
        True if agent should retry, False if max attempts reached
    """
    state = load_agent_state(agent_name, max_age_hours=24)

    if not state:
        return True  # No state = first attempt

    attempt_count = state.get("attempt_count", 0)

    if attempt_count >= max_attempts:
        print(f"[STATE] {agent_name} reached max attempts ({max_attempts}), giving up")
        clear_agent_state(agent_name)
        return False

    return True


def get_retry_delay(attempt_count: int) -> int:
    """
    Calculate exponential backoff delay in seconds.

    Args:
        attempt_count: Current attempt number (1-indexed)

    Returns:
        Delay in seconds (5min, 10min, 20min)
    """
    delays = [300, 600, 1200]  # 5min, 10min, 20min
    return delays[min(attempt_count - 1, len(delays) - 1)]


def get_all_agent_states(max_age_hours: int = 24) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Get state of all agents for coordination.

    Args:
        max_age_hours: Maximum age of state to consider valid (default: 24 hours)

    Returns:
        Dict mapping agent names to their states (or None if no state)
    """
    agents = ["trick_search", "research", "experiment", "log"]
    states = {}

    for agent in agents:
        states[agent] = load_agent_state(agent, max_age_hours=max_age_hours)

    return states


def is_agent_active(agent_name: str, max_age_minutes: int = 30) -> bool:
    """
    Check if an agent is currently active (has recent state).

    Args:
        agent_name: Name of the agent
        max_age_minutes: Consider active if state updated within this many minutes

    Returns:
        True if agent appears to be actively working
    """
    state = load_agent_state(agent_name, max_age_hours=24)

    if not state:
        return False

    # Check if state was updated recently
    last_updated = datetime.fromisoformat(state["last_updated"])
    age = datetime.now() - last_updated

    return age.total_seconds() < (max_age_minutes * 60)

"""
Work Tracker

Coordinates work across multiple experiment agents to prevent duplicate work.

The Work Tracker maintains a shared JSON file that tracks:
- Which proposals are currently being worked on
- Which agent is working on each
- When work started
- Last heartbeat timestamp

Agents must:
1. Claim a proposal before starting work
2. Send heartbeat updates every ~5 minutes
3. Release the claim when done
4. Check for stale claims (no heartbeat in >30 minutes)

Usage:
    from agents.work_tracker import claim_proposal, heartbeat, release_proposal, get_active_work

    # Claim a proposal
    if claim_proposal(proposal_id="006-monarch-gated", agent_id="agent-1"):
        try:
            # Do work...
            heartbeat(proposal_id)
        finally:
            release_proposal(proposal_id)
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import fcntl
import time


PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
WORK_TRACKER_FILE = EXPERIMENTS_DIR / "active_work.json"

# Configuration
HEARTBEAT_TIMEOUT_MINUTES = 30  # Consider work stale after 30 min without heartbeat
HEARTBEAT_INTERVAL_SECONDS = 300  # 5 minutes


def _lock_file(f):
    """Acquire exclusive lock on file."""
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)


def _unlock_file(f):
    """Release lock on file."""
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _read_tracker() -> Dict:
    """
    Read the work tracker file with file locking.

    Returns:
        Dict with active work data
    """
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    if not WORK_TRACKER_FILE.exists():
        return {"active_work": {}, "history": []}

    with open(WORK_TRACKER_FILE, 'r') as f:
        _lock_file(f)
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {"active_work": {}, "history": []}
        finally:
            _unlock_file(f)

    return data


def _write_tracker(data: Dict):
    """
    Write the work tracker file with file locking.

    Args:
        data: Dict with active work data
    """
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    # Write to temp file first, then atomic rename
    temp_file = WORK_TRACKER_FILE.with_suffix('.json.tmp')

    with open(temp_file, 'w') as f:
        _lock_file(f)
        try:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        finally:
            _unlock_file(f)

    # Atomic rename
    temp_file.replace(WORK_TRACKER_FILE)


def _clean_stale_work(data: Dict) -> Dict:
    """
    Remove stale work entries (no heartbeat in >30 minutes).

    Args:
        data: Work tracker data

    Returns:
        Cleaned data
    """
    now = datetime.now()
    cutoff = now - timedelta(minutes=HEARTBEAT_TIMEOUT_MINUTES)

    active_work = data.get("active_work", {})
    stale_proposals = []

    for proposal_id, work_info in active_work.items():
        last_heartbeat_str = work_info.get("last_heartbeat", work_info.get("started_at"))
        try:
            last_heartbeat = datetime.fromisoformat(last_heartbeat_str)
            if last_heartbeat < cutoff:
                stale_proposals.append(proposal_id)
        except (ValueError, TypeError):
            # Invalid timestamp, consider stale
            stale_proposals.append(proposal_id)

    # Move stale work to history
    for proposal_id in stale_proposals:
        work_info = active_work.pop(proposal_id)
        work_info["status"] = "stale"
        work_info["staled_at"] = now.isoformat()
        data.setdefault("history", []).append(work_info)

    if stale_proposals:
        print(f"[WORK TRACKER] Cleaned {len(stale_proposals)} stale work entries: {stale_proposals}")

    return data


def get_active_work() -> Dict[str, Dict]:
    """
    Get all currently active work (after cleaning stale entries).

    Returns:
        Dict mapping proposal_id -> work_info
    """
    data = _read_tracker()
    data = _clean_stale_work(data)
    _write_tracker(data)
    return data.get("active_work", {})


def is_proposal_claimed(proposal_id: str) -> bool:
    """
    Check if a proposal is currently being worked on.

    Args:
        proposal_id: Proposal ID (e.g., "006-monarch-gated-state-transition")

    Returns:
        True if claimed by an active agent
    """
    active_work = get_active_work()
    return proposal_id in active_work


def claim_proposal(proposal_id: str, agent_id: Optional[str] = None) -> bool:
    """
    Claim a proposal for work.

    Args:
        proposal_id: Proposal ID to claim
        agent_id: Optional agent ID (auto-generated if not provided)

    Returns:
        True if claim successful, False if already claimed
    """
    if agent_id is None:
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"

    data = _read_tracker()
    data = _clean_stale_work(data)

    active_work = data.get("active_work", {})

    # Check if already claimed
    if proposal_id in active_work:
        existing = active_work[proposal_id]
        print(f"[WORK TRACKER] Proposal {proposal_id} already claimed by {existing.get('agent_id')}")
        return False

    # Claim it
    now = datetime.now().isoformat()
    active_work[proposal_id] = {
        "agent_id": agent_id,
        "proposal_id": proposal_id,
        "started_at": now,
        "last_heartbeat": now,
        "status": "in_progress",
    }

    data["active_work"] = active_work
    _write_tracker(data)

    print(f"[WORK TRACKER] ✓ Claimed {proposal_id} for {agent_id}")
    return True


def heartbeat(proposal_id: str, agent_id: Optional[str] = None, status: str = "in_progress"):
    """
    Update heartbeat for a proposal being worked on.

    Args:
        proposal_id: Proposal ID
        agent_id: Agent ID (for verification)
        status: Current status (e.g., "in_progress", "implementing", "training")
    """
    data = _read_tracker()
    active_work = data.get("active_work", {})

    if proposal_id not in active_work:
        print(f"[WORK TRACKER] Warning: Heartbeat for unclaimed proposal {proposal_id}")
        return

    work_info = active_work[proposal_id]

    # Verify agent ID if provided
    if agent_id and work_info.get("agent_id") != agent_id:
        print(f"[WORK TRACKER] Warning: Agent ID mismatch for {proposal_id}")
        return

    # Update heartbeat
    work_info["last_heartbeat"] = datetime.now().isoformat()
    work_info["status"] = status

    _write_tracker(data)
    print(f"[WORK TRACKER] ♥ Heartbeat for {proposal_id} ({status})")


def release_proposal(proposal_id: str, agent_id: Optional[str] = None, status: str = "completed"):
    """
    Release a proposal after work is done.

    Args:
        proposal_id: Proposal ID
        agent_id: Agent ID (for verification)
        status: Final status (e.g., "completed", "failed", "abandoned")
    """
    data = _read_tracker()
    active_work = data.get("active_work", {})

    if proposal_id not in active_work:
        print(f"[WORK TRACKER] Warning: Release for unclaimed proposal {proposal_id}")
        return

    work_info = active_work.pop(proposal_id)

    # Verify agent ID if provided
    if agent_id and work_info.get("agent_id") != agent_id:
        print(f"[WORK TRACKER] Warning: Agent ID mismatch for {proposal_id}")
        # Still release it
        pass

    # Move to history
    work_info["completed_at"] = datetime.now().isoformat()
    work_info["status"] = status

    data.setdefault("history", []).append(work_info)
    _write_tracker(data)

    print(f"[WORK TRACKER] ✓ Released {proposal_id} ({status})")


def get_claimed_proposals() -> List[str]:
    """
    Get list of proposal IDs currently claimed by active agents.

    Returns:
        List of proposal IDs
    """
    active_work = get_active_work()
    return list(active_work.keys())


# =============================================================================
# Experiment Scaling Coordination
# =============================================================================

def is_experiment_claimed(exp_num: str) -> bool:
    """
    Check if an experiment is currently being scaled.

    Args:
        exp_num: Experiment number (e.g., "003", "008")

    Returns:
        True if claimed by an active scaler agent
    """
    exp_id = f"exp-{exp_num}-scaling"
    active_work = get_active_work()
    return exp_id in active_work


def claim_experiment(exp_num: str, agent_id: Optional[str] = None) -> bool:
    """
    Claim an experiment for scaling work.

    Args:
        exp_num: Experiment number to claim (e.g., "003", "008")
        agent_id: Optional agent ID (auto-generated if not provided)

    Returns:
        True if claim successful, False if already claimed
    """
    exp_id = f"exp-{exp_num}-scaling"
    return claim_proposal(exp_id, agent_id)


def heartbeat_experiment(exp_num: str, agent_id: Optional[str] = None, status: str = "scaling"):
    """
    Update heartbeat for an experiment being scaled.

    Args:
        exp_num: Experiment number
        agent_id: Agent ID (for verification)
        status: Current status (e.g., "scaling", "implementing", "writing_code")
    """
    exp_id = f"exp-{exp_num}-scaling"
    heartbeat(exp_id, agent_id, status)


def release_experiment(exp_num: str, agent_id: Optional[str] = None, status: str = "completed"):
    """
    Release an experiment after scaling is done.

    Args:
        exp_num: Experiment number
        agent_id: Agent ID (for verification)
        status: Final status (e.g., "completed", "failed", "skipped")
    """
    exp_id = f"exp-{exp_num}-scaling"
    release_proposal(exp_id, agent_id, status)


def get_claimed_experiments() -> List[str]:
    """
    Get list of experiment numbers currently claimed by scaler agents.

    Returns:
        List of experiment numbers (e.g., ["003", "008"])
    """
    active_work = get_active_work()
    exp_ids = [k for k in active_work.keys() if k.startswith("exp-") and k.endswith("-scaling")]
    return [exp_id.replace("exp-", "").replace("-scaling", "") for exp_id in exp_ids]


def get_work_status(proposal_id: str) -> Optional[Dict]:
    """
    Get work status for a specific proposal.

    Args:
        proposal_id: Proposal ID

    Returns:
        Work info dict if active, None otherwise
    """
    active_work = get_active_work()
    return active_work.get(proposal_id)


def print_active_work():
    """Print a summary of all active work."""
    active_work = get_active_work()

    if not active_work:
        print("[WORK TRACKER] No active work")
        return

    print("\n" + "=" * 80)
    print("Active Work")
    print("=" * 80)

    for proposal_id, work_info in active_work.items():
        agent_id = work_info.get("agent_id", "unknown")
        status = work_info.get("status", "unknown")
        started = work_info.get("started_at", "unknown")
        heartbeat = work_info.get("last_heartbeat", "unknown")

        # Calculate time since start
        try:
            start_time = datetime.fromisoformat(started)
            elapsed = datetime.now() - start_time
            elapsed_str = f"{elapsed.total_seconds() / 60:.1f} min"
        except:
            elapsed_str = "unknown"

        print(f"\n{proposal_id}")
        print(f"  Agent: {agent_id}")
        print(f"  Status: {status}")
        print(f"  Elapsed: {elapsed_str}")
        print(f"  Last heartbeat: {heartbeat}")

    print("\n" + "=" * 80)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI for managing work tracker."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m agents.work_tracker [list|clean|claim|release] [args...]")
        print("\nCommands:")
        print("  list                    - Show active work")
        print("  clean                   - Clean stale work entries")
        print("  claim <proposal_id>     - Claim a proposal")
        print("  release <proposal_id>   - Release a proposal")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        print_active_work()

    elif command == "clean":
        data = _read_tracker()
        data = _clean_stale_work(data)
        _write_tracker(data)
        print("[WORK TRACKER] ✓ Cleaned stale entries")

    elif command == "claim":
        if len(sys.argv) < 3:
            print("Usage: python -m agents.work_tracker claim <proposal_id>")
            sys.exit(1)
        proposal_id = sys.argv[2]
        success = claim_proposal(proposal_id)
        sys.exit(0 if success else 1)

    elif command == "release":
        if len(sys.argv) < 3:
            print("Usage: python -m agents.work_tracker release <proposal_id>")
            sys.exit(1)
        proposal_id = sys.argv[2]
        release_proposal(proposal_id)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()

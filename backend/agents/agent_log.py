"""
Agent Log Management

Each agent maintains its own detailed log file for notes, observations, and decisions.
These logs serve as persistent memory that agents can reference across sessions.

Logs vs State:
- State files (.state/): Short-term resumption data (cleared on success)
- Log files (notes/): Long-term memory and observations (permanent)

Each agent has a log at notes/{agent_name}_log.md
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


PROJECT_ROOT = Path(__file__).parent.parent
NOTES_DIR = PROJECT_ROOT / "notes"


def ensure_notes_dir():
    """Ensure the notes directory exists."""
    NOTES_DIR.mkdir(exist_ok=True)


def get_agent_log_path(agent_name: str) -> Path:
    """
    Get the path to an agent's log file.

    Args:
        agent_name: Name of the agent (e.g., "trick_search", "research", "experiment")

    Returns:
        Path to the agent's log file
    """
    return NOTES_DIR / f"{agent_name}_log.md"


def write_agent_log(agent_name: str, entry: str, entry_type: str = "note"):
    """
    Append an entry to an agent's log.

    Args:
        agent_name: Name of the agent
        entry: The log entry text
        entry_type: Type of entry (note, decision, observation, error)
    """
    ensure_notes_dir()

    log_path = get_agent_log_path(agent_name)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize log if it doesn't exist
    if not log_path.exists():
        header = f"# {agent_name.replace('_', ' ').title()} - Agent Log\n\n"
        header += "Internal notes and observations from the agent's work.\n\n"
        header += "---\n\n"
        log_path.write_text(header)

    # Append entry
    existing_log = log_path.read_text()

    entry_header = f"## {timestamp} - {entry_type.title()}\n\n"
    formatted_entry = entry_header + entry.strip() + "\n\n---\n\n"

    updated_log = existing_log + formatted_entry
    log_path.write_text(updated_log)

    print(f"[{agent_name.upper()}] Logged {entry_type} to {log_path.name}")


def read_agent_log(agent_name: str, last_n_entries: Optional[int] = None) -> str:
    """
    Read an agent's log file.

    Args:
        agent_name: Name of the agent
        last_n_entries: If specified, return only the last N entries

    Returns:
        Log content (or empty string if no log exists)
    """
    log_path = get_agent_log_path(agent_name)

    if not log_path.exists():
        return ""

    content = log_path.read_text()

    if last_n_entries is not None:
        # Split by separator and get last N entries
        entries = content.split("---\n\n")
        if len(entries) > 1:
            # Keep header + last N entries
            header = entries[0]
            recent = entries[-last_n_entries:] if last_n_entries > 0 else []
            content = header + "---\n\n".join(recent)

    return content


def read_all_agent_logs(last_n_entries: Optional[int] = 3) -> Dict[str, str]:
    """
    Read logs from all agents.

    Args:
        last_n_entries: Number of recent entries to read from each log

    Returns:
        Dict mapping agent names to their log content
    """
    agents = ["trick_search", "research", "experiment"]
    logs = {}

    for agent in agents:
        logs[agent] = read_agent_log(agent, last_n_entries=last_n_entries)

    return logs


def get_agent_log_summary(agent_name: str, last_n_entries: int = 5) -> str:
    """
    Get a summary of recent entries from an agent's log.

    Args:
        agent_name: Name of the agent
        last_n_entries: Number of recent entries to include

    Returns:
        Summary string with recent entries
    """
    log_content = read_agent_log(agent_name, last_n_entries=last_n_entries)

    if not log_content:
        return f"No log found for {agent_name}"

    # Count total entries
    entry_count = log_content.count("##")

    summary = f"**{agent_name.replace('_', ' ').title()} Log**\n"
    summary += f"- Total entries: {entry_count}\n"
    summary += f"- Recent entries (last {last_n_entries}):\n\n"
    summary += log_content

    return summary

"""
Human Feedback Management

Provides utilities for reading and appending human feedback that guides agent behavior.
All agents should reference human_feedback.md to learn from ongoing instructions.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict


PROJECT_ROOT = Path(__file__).parent.parent
FEEDBACK_FILE = PROJECT_ROOT / "human_feedback.md"


def read_human_feedback(last_n_entries: Optional[int] = None) -> str:
    """
    Read human feedback from the feedback file.

    Args:
        last_n_entries: If specified, return only the last N entries

    Returns:
        Feedback content (or empty string if no feedback exists)
    """
    if not FEEDBACK_FILE.exists():
        return ""

    content = FEEDBACK_FILE.read_text()

    if last_n_entries is not None and last_n_entries > 0:
        # Split by "## " to get individual entries (skip the header)
        parts = content.split("## ")
        if len(parts) > 1:
            # Keep header + last N entries
            header = parts[0]
            entries = parts[1:]  # Skip header part

            # Filter out the "Example Entry" and "Feedback Entries" sections
            actual_entries = [e for e in entries
                            if not e.startswith("Example Entry")
                            and not e.startswith("Feedback Entries")
                            and not e.startswith("Instructions")]

            if actual_entries:
                recent = actual_entries[-last_n_entries:]
                content = header + "## " + "## ".join(recent)

    return content


def has_recent_feedback(hours: int = 24) -> bool:
    """
    Check if there's feedback from the last N hours.

    Args:
        hours: Time window to check

    Returns:
        True if recent feedback exists
    """
    if not FEEDBACK_FILE.exists():
        return False

    # Simple heuristic: check if file was modified recently
    mtime = datetime.fromtimestamp(FEEDBACK_FILE.stat().st_mtime)
    age_hours = (datetime.now() - mtime).total_seconds() / 3600

    return age_hours < hours


def append_feedback(topic: str, feedback: str):
    """
    Append a new feedback entry (for programmatic use).

    Args:
        topic: Topic/category of the feedback
        feedback: The feedback text
    """
    if not FEEDBACK_FILE.exists():
        FEEDBACK_FILE.write_text("# Human Feedback & Instructions\n\n---\n\n")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    existing_content = FEEDBACK_FILE.read_text()

    new_entry = f"## {timestamp} - {topic}\n\n{feedback.strip()}\n\n---\n\n"

    # Append to the end
    updated_content = existing_content + new_entry

    FEEDBACK_FILE.write_text(updated_content)

    print(f"[FEEDBACK] Added entry: {topic}")


def get_feedback_summary(max_entries: int = 10) -> str:
    """
    Get a concise summary of recent feedback for agent context.

    Args:
        max_entries: Maximum number of recent entries to include

    Returns:
        Summary string suitable for agent prompts
    """
    feedback = read_human_feedback(last_n_entries=max_entries)

    if not feedback or len(feedback.strip()) < 100:
        return "No human feedback yet."

    # Count entries
    entry_count = feedback.count("## ") - feedback.count("## Example") - feedback.count("## Instructions") - feedback.count("## Feedback Entries")

    summary = f"**Human Feedback Available** ({entry_count} entries):\n\n{feedback}"

    return summary

#!/usr/bin/env python3
"""
Split legacy log.md into atomic timestamped files.

Parses the monolithic log.md file and creates individual timestamped files
in notes/research_logs/ with proper frontmatter.
"""

import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
LOG_FILE = PROJECT_ROOT / "notes" / "log.md"
RESEARCH_LOGS_DIR = PROJECT_ROOT / "notes" / "research_logs"

def parse_log_entries(content: str):
    """Parse log.md and extract individual entries."""
    lines = content.split('\n')
    entries = []
    current_entry = None
    current_lines = []

    # Skip header content
    in_header = True

    for line in lines:
        # Match date headers like "## 2026-02-15 — 03:52 UTC"
        match = re.match(r'^##\s+(\d{4})-(\d{2})-(\d{2})\s+—\s+(\d{2}):(\d{2})\s+UTC', line)

        if match:
            # Save previous entry
            if current_entry and current_lines:
                entries.append({
                    'timestamp': current_entry,
                    'content': '\n'.join(current_lines).strip()
                })

            # Start new entry
            year, month, day, hour, minute = match.groups()
            current_entry = datetime(int(year), int(month), int(day), int(hour), int(minute))
            current_lines = [line]
            in_header = False

        elif not in_header and current_entry:
            # Skip separator lines
            if line.strip() != '---':
                current_lines.append(line)

    # Add last entry
    if current_entry and current_lines:
        entries.append({
            'timestamp': current_entry,
            'content': '\n'.join(current_lines).strip()
        })

    return entries


def count_items_in_entry(content: str):
    """Count tricks, proposals, and experiments mentioned in entry."""
    tricks = len(re.findall(r'### 📚 New Discoveries', content))
    proposals = content.count('**Proposal')
    experiments = content.count('**Experiment')

    return tricks, proposals, experiments


def main():
    print(f"Reading {LOG_FILE}...")

    if not LOG_FILE.exists():
        print(f"Error: {LOG_FILE} not found")
        return

    content = LOG_FILE.read_text()
    entries = parse_log_entries(content)

    print(f"Found {len(entries)} log entries")

    # Create research_logs directory if needed
    RESEARCH_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Write each entry as a separate file
    for entry in entries:
        timestamp = entry['timestamp']
        content = entry['content']

        # Count items
        tricks, proposals, experiments = count_items_in_entry(content)

        # Create filename: YYYY-MM-DDTHH-MM-SS.md
        filename = timestamp.strftime("%Y-%m-%dT%H-%M-00.md")
        filepath = RESEARCH_LOGS_DIR / filename

        # Skip if file already exists
        if filepath.exists():
            print(f"  Skipping {filename} (already exists)")
            continue

        # Create frontmatter
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        frontmatter = f"""---
title: Research Log Update
date: {timestamp.isoformat()}
timestamp: {timestamp_str}
tricks: {tricks}
proposals: {proposals}
experiments: {experiments}
source: legacy_migration
---

"""

        # Write file
        full_content = frontmatter + content
        filepath.write_text(full_content)

        print(f"  Created {filename}")

    print(f"\n✓ Migration complete! Created {len(entries)} files in {RESEARCH_LOGS_DIR}")


if __name__ == "__main__":
    main()

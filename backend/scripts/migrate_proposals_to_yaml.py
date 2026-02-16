#!/usr/bin/env python3
"""
Migrate proposal files from inline markdown metadata to YAML frontmatter.

Converts:
    # Title
    **Status**: proposed
    **Priority**: high

To:
    ---
    status: proposed
    priority: high
    ---

    # Title
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROPOSALS_DIR = PROJECT_ROOT / "proposals"


def migrate_proposal(filepath: Path):
    """Migrate a single proposal file to YAML frontmatter."""
    content = filepath.read_text()
    lines = content.split('\n')

    # Check if already has YAML frontmatter
    if lines and lines[0].strip() == '---':
        return False, "Already has YAML frontmatter"

    # Extract metadata from inline markdown
    metadata = {}
    title_line = None
    content_start = 0

    for i, line in enumerate(lines[:20]):
        # Extract title
        if line.startswith('# ') and title_line is None:
            title_line = i
            continue

        # Extract inline metadata
        if line.startswith('**Status**:'):
            metadata['status'] = line.split(':', 1)[1].strip()
            continue
        elif line.startswith('**Priority**:'):
            metadata['priority'] = line.split(':', 1)[1].strip()
            continue
        elif line.startswith('**Created**:'):
            metadata['created'] = line.split(':', 1)[1].strip()
            continue
        elif line.startswith('**Based on**:'):
            # Clean up the based_on field
            based_on = line.split(':', 1)[1].strip()
            metadata['based_on'] = based_on
            continue

        # Find where actual content starts (after metadata block)
        if metadata and line.strip() == '':
            continue
        elif metadata and not line.startswith('**'):
            content_start = i
            break

    if not metadata:
        return False, "No metadata found"

    # Construct new file with YAML frontmatter
    new_lines = ['---']

    # Add metadata in consistent order
    for key in ['status', 'priority', 'created', 'based_on']:
        if key in metadata:
            new_lines.append(f"{key}: {metadata[key]}")

    new_lines.append('---')
    new_lines.append('')

    # Add title if it exists
    if title_line is not None:
        new_lines.append(lines[title_line])
        new_lines.append('')

    # Add remaining content (skip old title and metadata lines)
    skip_until = max(content_start, title_line + 1 if title_line else 0)
    remaining_content = '\n'.join(lines[skip_until:]).lstrip('\n')
    new_lines.append(remaining_content)

    # Write back
    new_content = '\n'.join(new_lines)
    filepath.write_text(new_content)

    return True, f"Migrated with {len(metadata)} metadata fields"


def main():
    print(f"Scanning {PROPOSALS_DIR}...")

    if not PROPOSALS_DIR.exists():
        print(f"Error: {PROPOSALS_DIR} not found")
        return

    proposal_files = sorted(PROPOSALS_DIR.glob("*.md"))
    print(f"Found {len(proposal_files)} proposals\n")

    migrated = 0
    skipped = 0

    for proposal_file in proposal_files:
        success, message = migrate_proposal(proposal_file)

        if success:
            print(f"✓ {proposal_file.name}: {message}")
            migrated += 1
        else:
            print(f"  {proposal_file.name}: {message}")
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Migration complete!")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped:  {skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

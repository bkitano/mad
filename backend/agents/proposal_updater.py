"""
Utility for updating proposal YAML frontmatter.

Provides functions to safely update proposal metadata without corrupting content.
"""

from pathlib import Path
from typing import Dict, Optional


def update_proposal_status(
    proposal_path: Path,
    new_status: str,
    additional_fields: Optional[Dict[str, str]] = None
) -> bool:
    """
    Update the status field in a proposal's YAML frontmatter.

    Args:
        proposal_path: Path to the proposal markdown file
        new_status: New status value (e.g., 'ongoing', 'completed', 'abandoned')
        additional_fields: Optional dict of other fields to update/add

    Returns:
        True if successful, False otherwise
    """
    try:
        if not proposal_path.exists():
            print(f"Error: Proposal file not found: {proposal_path}")
            return False

        content = proposal_path.read_text()
        lines = content.split('\n')

        # Check if file has YAML frontmatter
        if not lines or lines[0].strip() != '---':
            print(f"Error: Proposal {proposal_path.name} has no YAML frontmatter")
            return False

        # Find frontmatter boundaries
        frontmatter_end = None
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == '---':
                frontmatter_end = i
                break

        if frontmatter_end is None:
            print(f"Error: Could not find end of YAML frontmatter in {proposal_path.name}")
            return False

        # Parse frontmatter
        frontmatter_lines = lines[1:frontmatter_end]
        updated_frontmatter = []
        status_found = False

        # Merge additional fields if provided
        fields_to_update = {'status': new_status}
        if additional_fields:
            fields_to_update.update(additional_fields)

        # Update existing fields
        for line in frontmatter_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()

                if key in fields_to_update:
                    updated_frontmatter.append(f"{key}: {fields_to_update[key]}")
                    if key == 'status':
                        status_found = True
                    # Mark as processed
                    fields_to_update.pop(key, None)
                else:
                    updated_frontmatter.append(line)
            else:
                updated_frontmatter.append(line)

        # Add any new fields that weren't in the original frontmatter
        for key, value in fields_to_update.items():
            updated_frontmatter.append(f"{key}: {value}")

        # Reconstruct file
        new_lines = ['---'] + updated_frontmatter + lines[frontmatter_end:]
        new_content = '\n'.join(new_lines)

        # Write back
        proposal_path.write_text(new_content)

        print(f"✓ Updated {proposal_path.name}: status={new_status}")
        return True

    except Exception as e:
        print(f"Error updating proposal {proposal_path.name}: {e}")
        return False


def get_proposal_status(proposal_path: Path) -> Optional[str]:
    """
    Get the current status from a proposal's YAML frontmatter.

    Args:
        proposal_path: Path to the proposal markdown file

    Returns:
        Status string if found, None otherwise
    """
    try:
        if not proposal_path.exists():
            return None

        content = proposal_path.read_text()
        lines = content.split('\n')

        if not lines or lines[0].strip() != '---':
            return None

        for line in lines[1:]:
            if line.strip() == '---':
                break
            if ':' in line:
                key, value = line.split(':', 1)
                if key.strip() == 'status':
                    return value.strip()

        return None

    except Exception as e:
        print(f"Error reading proposal status: {e}")
        return None

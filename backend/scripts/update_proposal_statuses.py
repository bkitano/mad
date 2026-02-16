#!/usr/bin/env python3
"""
Update proposal statuses based on experiment implementation state.

Rules:
- If experiment has results file → status = "completed"
- If experiment code exists but no results → status = "ongoing"
- Otherwise → status = "proposed"
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.proposal_updater import update_proposal_status, get_proposal_status

PROJECT_ROOT = Path(__file__).parent.parent
CODE_DIR = PROJECT_ROOT / "code"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROPOSALS_DIR = PROJECT_ROOT / "proposals"


def get_experiment_numbers():
    """Get all experiment numbers from code/ directory."""
    if not CODE_DIR.exists():
        return []

    exp_nums = []
    for exp_dir in CODE_DIR.iterdir():
        if exp_dir.is_dir():
            try:
                # Extract number from directory name (e.g., "002" or "026_scaled")
                num = exp_dir.name.split('_')[0]
                exp_nums.append(num)
            except:
                continue

    return sorted(set(exp_nums))  # Remove duplicates and sort


def has_results(exp_num):
    """Check if experiment has a results file."""
    results_file = EXPERIMENTS_DIR / f"{exp_num}_results.md"
    return results_file.exists()


def get_experiment_files(exp_num):
    """Get paths to experiment log and results files if they exist."""
    files = {}

    # Check for experiment log
    exp_log = EXPERIMENTS_DIR / f"experiment-log-{exp_num}.md"
    if exp_log.exists():
        files['experiment_log'] = f"experiment-log-{exp_num}.md"

    # Check for results
    results = EXPERIMENTS_DIR / f"{exp_num}_results.md"
    if results.exists():
        files['results_file'] = f"{exp_num}_results.md"

    return files


def get_proposal_for_experiment(exp_num):
    """Find the proposal file corresponding to an experiment number."""
    # Try common patterns
    patterns = [
        f"{exp_num}-*.md",
        f"0{exp_num}-*.md" if len(exp_num) == 2 else None,
        f"00{exp_num}-*.md" if len(exp_num) == 1 else None,
    ]

    for pattern in patterns:
        if pattern:
            matches = list(PROPOSALS_DIR.glob(pattern))
            if matches:
                return matches[0]

    return None


def main():
    print("Scanning experiments and updating proposal statuses...\n")

    exp_nums = get_experiment_numbers()
    print(f"Found {len(exp_nums)} experiments: {', '.join(exp_nums)}\n")

    updated = []
    not_found = []
    skipped = []

    for exp_num in exp_nums:
        proposal_file = get_proposal_for_experiment(exp_num)

        if not proposal_file:
            not_found.append(exp_num)
            print(f"⚠ Experiment {exp_num}: No matching proposal found")
            continue

        current_status = get_proposal_status(proposal_file)

        # Determine new status
        if has_results(exp_num):
            new_status = "completed"
        else:
            new_status = "ongoing"

        # Get experiment file paths
        exp_files = get_experiment_files(exp_num)
        additional_fields = {
            'experiment_number': exp_num,
        }
        if 'experiment_log' in exp_files:
            additional_fields['experiment_log'] = exp_files['experiment_log']
        if 'results_file' in exp_files:
            additional_fields['results_file'] = exp_files['results_file']

        # Update status and add experiment file links
        success = update_proposal_status(proposal_file, new_status, additional_fields)
        if success:
            updated.append((exp_num, current_status or "unknown", new_status))
            files_info = ", ".join([f"{k}={v}" for k, v in exp_files.items()])
            log_info = f" [{files_info}]" if files_info else ""
            print(f"✓ Experiment {exp_num} ({proposal_file.stem}): {current_status or 'unknown'} → {new_status}{log_info}")
        else:
            print(f"✗ Experiment {exp_num} ({proposal_file.stem}): Failed to update")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Updated: {len(updated)}")
    print(f"  Skipped (already correct): {len(skipped)}")
    print(f"  Not found: {len(not_found)}")
    print(f"{'='*60}")

    if updated:
        print("\nUpdated proposals:")
        for exp_num, old, new in updated:
            print(f"  {exp_num}: {old} → {new}")


if __name__ == "__main__":
    main()
